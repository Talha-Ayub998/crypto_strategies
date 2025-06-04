"""
Top‑20 SHORT Futures Strategy (Full Flowchart Implementation)
"""
import sys
import os
import schedule, time, json
from datetime import datetime, timedelta

# Add parent directory to sys.path so 'utils' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    client,
    log,
    tg_send,
    fetch_klines,
    crossed_below,
    btc_below_50ma,
    adjust_qty_price,
    place_order,
    handle_exceptions,
    moving_average,
    get_margin_ratio,
    manage_margin,
    DRY_RUN,
    MAX_PER_COIN_PCT,
    TOP_COINS,
    ALLOC_PCT
)

# === Candidate Selection ===
def futures_universe():
    try:
        info = client.futures_exchange_info()
        return [s["symbol"] for s in info["symbols"] if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT"]
    except Exception as e:
        log(f"Universe error: {e}")
        return []

def bottom20_candidates():
    candidates = []
    symbols = futures_universe()
    for sym in symbols:
        try:
            df = fetch_klines(sym)
            if df.empty or len(df) < 21:
                continue

            df = df.iloc[:-1]  # remove current day
            dvol = (df["close"] * df["volume"]).rolling(20).mean().iloc[-1]

            candidates.append({"symbol": sym, "dvol": dvol})
        except Exception as e:
            log(f"{sym} error: {e}")

    # ✅ Sort by dollar volume (top 50), then pick bottom 20
    top50 = sorted(candidates, key=lambda x: x["dvol"], reverse=True)[:50]
    bottom20 = sorted(top50, key=lambda x: x["dvol"])[:TOP_COINS]

    log(f"Selected short symbols: {[c['symbol'] for c in bottom20]}")
    return bottom20

# === Exit Logic ===
def close_all_shorts():
    try:
        tg_send("BTC > 50MA — Closing all shorts ❌")
        positions = [p for p in client.futures_position_information() if float(p["positionAmt"]) < 0 and float(p["entryPrice"]) > float(p["markPrice"])]
        for p in positions:
            try:
                sym = p["symbol"]
                amt = abs(float(p["positionAmt"]))
                price = float(p["markPrice"])
                _, qty = adjust_qty_price(sym, price, amt)
                if qty == 0:
                    continue
                client.futures_create_order(symbol=sym, side="BUY", type="MARKET", quantity=qty, reduceOnly=True)
                log(f"Closed short {sym}")
            except Exception as e:
                log(f"Error closing {sym}: {e}")
        prefix = "[DRY-RUN] " if DRY_RUN else ""
        tg_send(f"{prefix}All short positions closed ✅")
    except Exception as e:
        log(f"close_all_shorts error: {e}")
        tg_send(f"Short close error: {e}")

# === Daily Execution ===
@handle_exceptions
def rebalance_shorts():
    try:
        if not btc_below_50ma():
            close_all_shorts()
            return

        balance = client.futures_account_balance()
        usdt = sum(float(a["balance"]) for a in balance if a["asset"] == "USDT")
        cap = usdt * ALLOC_PCT  # 30% total allocation
        if cap <= 0:
            tg_send("No USDT balance 😐")
            return

        bottom20 = bottom20_candidates()
        if not bottom20:
            tg_send("No short candidates 💤")
            return

        # 🔒 LOCK per-coin allocation
        alloc_each = usdt * MAX_PER_COIN_PCT
        alloc_map = {c["symbol"]: alloc_each for c in bottom20}

        with open("alloc_map_short.json", "w") as f:
            json.dump(alloc_map, f)

        summary = []
        for c in bottom20:
            symbol = c["symbol"]
            # ⏩ Skip already spent allocations
            if isinstance(alloc_map.get(symbol), str) and alloc_map[symbol] == "spent":
                log(f"{symbol} skipped — already spent allocation")
                summary.append(f"{symbol} ⏩ skipped — already spent allocation")
                continue

            place_order(symbol, alloc_map[symbol], "SELL", summary, alloc_map_path="alloc_map_short.json")

        ratio = get_margin_ratio()
        prefix = "[DRY-RUN] " if DRY_RUN else ""
        msg = f"{prefix}📊 Daily Summary\nMargin Ratio: {ratio:.2f}%\n\n"

        success = [m for m in summary if "✅" in m]
        fail = [m for m in summary if "❌" in m]
        skipped = [m for m in summary if "skipped" in m]


        if success:
            msg += "✅ Success:\n" + "\n".join(success) + "\n\n"
        if fail:
            msg += "❌ Failed:\n" + "\n".join(fail) + "\n\n"
        if skipped:
            msg += "⏩ Skipped:\n" + "\n".join(skipped)

        tg_send(msg.strip())


    except Exception as e:
        log(f"rebalance_shorts error: {e}")
        tg_send(f"Rebalance error: {e}")

# === Noon Order Conversion ===
@handle_exceptions
def noon_fill_check():
    try:
        open_orders = client.futures_get_open_orders()
        for o in open_orders:
            try:
                client.futures_cancel_order(symbol=o["symbol"], orderId=o["orderId"])
                remaining = float(o["origQty"]) - float(o["executedQty"])
                if remaining > 0:
                    client.futures_create_order(symbol=o["symbol"], side="SELL", type="MARKET", quantity=remaining)
            except Exception as e:
                log(f"Noon fill error {o['symbol']}: {e}")
    except Exception as e:
        log(f"noon_fill_check error: {e}")

# === Exit Condition Monitoring ===
@handle_exceptions
def check_exit_conditions():
    try:
        active_positions = [p for p in client.futures_position_information() if float(p["positionAmt"]) < 0]
        bottom20 = [c["symbol"] for c in bottom20_candidates()]
        exit_list = []
        for p in active_positions:
            sym = p["symbol"]
            try:
                df = fetch_klines(sym)
                if df.empty or len(df) < 2:
                    continue
                close_below_5ma = df.close.iloc[-1] < moving_average(df.close, 5).iloc[-1]
                still_bottom20 = sym in bottom20
                btc_below = btc_below_50ma()

                if not (close_below_5ma and still_bottom20 and btc_below):
                    amt = abs(float(p["positionAmt"]))
                    price = float(p["markPrice"])
                    _, qty = adjust_qty_price(sym, price, amt)
                    if qty > 0:
                        prefix = "[DRY-RUN] " if DRY_RUN else ""
                        if DRY_RUN:
                            log(f"{prefix}Skipping exit order for {sym} | BUY | Qty: {qty}")
                            exit_msg = f"{prefix}{sym} would be closed: 5MA: {close_below_5ma}, In B20: {still_bottom20}, BTC<50: {btc_below}"
                        else:
                            client.futures_create_order(symbol=sym, side="BUY", type="MARKET", quantity=qty, reduceOnly=True)
                            exit_msg = f"{prefix}{sym} EXIT | Side: BUY | Qty: {qty:.4f} | Price: {price:.4f}"

                        exit_list.append(exit_msg)
                        tg_send(exit_msg)

            except Exception as e:
                log(f"Exit check error {sym}: {e}")
        if exit_list:
            ratio = get_margin_ratio()
            prefix = "[DRY-RUN] " if DRY_RUN else ""
            tg_send(f"{prefix}🚨 Exit Signals Triggered\nMargin Ratio: {ratio:.2f}%\n" + "\n".join(exit_list))

    except Exception as e:
        log(f"check_exit_conditions error: {e}")

# === Scheduler ===
def main():
    schedule.every().day.at("00:01").do(rebalance_shorts)
    schedule.every().day.at("12:00").do(noon_fill_check)
    schedule.every().day.at("23:59").do(check_exit_conditions) #adjust time if you want to run it slightly earlier
    schedule.every(5).minutes.do(manage_margin)
    tg_send("✅ Short Strategy Bot Running")
    while True:
        schedule.run_pending()
        time.sleep(10)

if __name__ == "__main__":
    main()
