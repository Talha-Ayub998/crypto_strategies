"""
Binance USDT‑M Futures Top‑20 Long Strategy — Full Flowchart Implementation
"""
import sys
import os
import schedule, time
import json

# Add parent directory to sys.path so 'utils' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    client,
    log,
    tg_send,
    fetch_klines,
    roc,
    crossed_above,
    btc_above_50ma,
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


# === Universe & Filtering ===
def futures_universe():
    try:
        info = client.futures_exchange_info()
        return [s["symbol"] for s in info["symbols"] if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT"]
    except Exception as e:
        log(f"Universe error: {e}")
        return []

def top20_candidates():
    candidates = []
    symbols = futures_universe()
    for sym in symbols:
        try:
            df = fetch_klines(sym)
            if df.empty or len(df) < 21:
                continue

            df = df.iloc[:-1]  # remove current day
            dvol = (df["close"] * df["volume"]).rolling(20).mean().iloc[-1]

            ma20 = moving_average(df["close"], 20)
            if not crossed_above(df["close"], ma20):
                continue

            r = roc(df["close"], 20).iloc[-1]
            candidates.append({"symbol": sym, "dvol": dvol, "roc": r})
        except Exception as e:
            log(f"{sym} error: {e}")

    # ✅ Sort dynamically by dollar volume (top 50), then by ROC (top 20)
    top50 = sorted(candidates, key=lambda x: x["dvol"], reverse=True)[:50]
    top20 =  sorted(top50, key=lambda x: x["roc"], reverse=True)[:TOP_COINS]

    log(f"Selected long symbols: {[c['symbol'] for c in top20]}")
    return top20


# === Close All ===
def close_all_longs():
    try:
        tg_send("BTC < 50MA — Closing all longs ❌")
        positions = [p for p in client.futures_position_information() if float(p["positionAmt"]) > 0 and float(p["entryPrice"]) < float(p["markPrice"])]
        for p in positions:
            sym = p["symbol"]
            amt = abs(float(p["positionAmt"]))
            price = float(p["markPrice"])
            _, qty = adjust_qty_price(sym, price, amt)
            if qty == 0:
                continue
            client.futures_create_order(
                symbol=sym, side="SELL", type="MARKET",
                quantity=qty, reduceOnly=True
            )
        tg_send("✅ All long positions closed")
    except Exception as e:
        log(f"close_all_longs error: {e}")
        tg_send(f"Close error: {e}")

# === Main Daily Routine ===
@handle_exceptions
def rebalance_longs():
    try:
        if not btc_above_50ma():
            close_all_longs()
            return

        balance = client.futures_account_balance()
        usdt = sum(float(a["balance"]) for a in balance if a["asset"] == "USDT")
        cap = usdt * ALLOC_PCT
        if cap <= 0:
            tg_send("No USDT balance 😐")
            return

        top20 = top20_candidates()
        if not top20:
            tg_send("No long candidates 💤")
            return

        # ✅ LOCK per-coin allocation
        alloc_each = min(cap / len(top20), usdt * MAX_PER_COIN_PCT)
        alloc_map = {c["symbol"]: alloc_each for c in top20}

        with open("alloc_map_long.json", "w") as f:
            json.dump(alloc_map, f)

        summary = []
        for c in top20:
            symbol = c["symbol"]
            if isinstance(alloc_map.get(symbol), str) and alloc_map[symbol] == "spent":
                log(f"{symbol} skipped — already spent allocation")
                continue
            place_order(symbol, alloc_map[symbol], "BUY", summary, alloc_map_path="alloc_map_long.json")

        ratio = get_margin_ratio()
        prefix = "[DRY-RUN] " if DRY_RUN else ""
        header = f"{prefix}🟢 Top20 LONG Strategy Entry\nMargin Ratio: {ratio:.2f}%\n"

        success = [m for m in summary if "✅" in m]
        fail = [m for m in summary if "❌" in m]
        skipped = [m for m in summary if "skipped" in m]

        msg = f"{prefix}📊 Daily Summary\nMargin Ratio: {ratio:.2f}%\n\n"

        if success:
            msg += "✅ Success:\n" + "\n".join(success) + "\n\n"
        if fail:
            msg += "❌ Failed:\n" + "\n".join(fail) + "\n\n"
        if skipped:
            msg += "⏩ Skipped:\n" + "\n".join(skipped)

        tg_send(msg.strip())


    except Exception as e:
        log(f"rebalance_longs error: {e}")
        tg_send(f"Rebalance error: {e}")

# === Market Fallback ===
@handle_exceptions
def noon_fill_check():
    try:
        # ✅ Do nothing to preserve capital allocation
        prefix = "[DRY-RUN] " if DRY_RUN else ""
        log(f"{prefix}⏳ Skipping fallback fill — preserving original VWAP limit orders.")
        tg_send(f"{prefix}⏳ Noon check: Skipped fallback fill to preserve fixed allocations and risk control.")
    except Exception as e:
        log(f"noon_fill_check error: {e}")

# === Exit Monitor ===
@handle_exceptions
def check_exit_conditions():
    try:
        active_positions = [p for p in client.futures_position_information() if float(p["positionAmt"]) > 0]
        top20 = [c["symbol"] for c in top20_candidates()]
        exit_list = []
        for p in active_positions:
            sym = p["symbol"]
            try:
                df = fetch_klines(sym)
                if df.empty or len(df) < 2:
                    continue
                close_above_20ma = df.close.iloc[-1] > moving_average(df.close, 20).iloc[-1]
                still_top20 = sym in top20

                btc_ok = btc_above_50ma()
                if not (close_above_20ma and still_top20 and btc_ok):
                    amt = abs(float(p["positionAmt"]))
                    price = float(p["markPrice"])

                    _, qty = adjust_qty_price(sym, price, amt)
                    if qty > 0:
                        prefix = "[DRY-RUN] " if DRY_RUN else ""
                        if DRY_RUN:
                            log(f"{prefix}Skipping exit order for {sym} | SELL | Qty: {qty}")
                            exit_msg = f"{prefix}{sym} would be closed: 20MA: {close_above_20ma}, In T20: {still_top20}, BTC>50: {btc_ok}"
                        else:
                            client.futures_create_order(symbol=sym, side="SELL", type="MARKET", quantity=qty, reduceOnly=True)
                            exit_msg = f"{prefix}{sym} EXIT | Side: SELL | Qty: {qty:.4f} | Price: {price:.4f}"

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
    schedule.every().day.at("00:01").do(rebalance_longs)
    schedule.every().day.at("12:00").do(noon_fill_check)
    schedule.every().day.at("23:59").do(check_exit_conditions) #adjust time if you want to run it slightly earlier
    schedule.every(5).minutes.do(manage_margin)
    tg_send("✅ Long Strategy Bot Running")
    while True:
        schedule.run_pending()
        time.sleep(10)

if __name__ == "__main__":
    main()