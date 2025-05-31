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
    client, log, tg_send, fetch_klines,
    crossed_below, btc_below_50ma,
    adjust_qty_price, place_order, handle_exceptions,
    moving_average,  get_margin_ratio, liquidate_all, reduce_positions, 
)

# === Constants from Flowchart ===
ALLOC_PCT        = 0.30
TOP_COINS        = 20
MAX_PER_COIN_PCT = 0.05
VWAP_DISCOUNT    = 1.02

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

            df = df.iloc[:-1]  # Remove current  day
            dvol = (df["close"] * df["volume"]).rolling(20).mean().iloc[-1]

            ma10 = moving_average(df["close"], 10)
            if not crossed_below(df["close"], ma10):
                continue

            candidates.append({"symbol": sym, "dvol": dvol})
        except Exception as e:
            log(f"{sym} error: {e}")

    # ✅ Sort by dollar volume, take top 20
    top20 = sorted(candidates, key=lambda x: x["dvol"], reverse=True)[:TOP_COINS]
    log(f"Selected short symbols: {[c['symbol'] for c in top20]}")

    return top20


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
        tg_send("All short positions closed ✅")
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
        cap = usdt * ALLOC_PCT
        if cap <= 0:
            tg_send("No USDT balance 😐")
            return

        bottom20 = bottom20_candidates()
        if not bottom20:
            tg_send("No short candidates 💤")
            return

        alloc_each = min(cap / len(bottom20), usdt * MAX_PER_COIN_PCT)
        summary = []
        for c in bottom20:
            place_order(c["symbol"], alloc_each, "SELL", summary, discount=VWAP_DISCOUNT)

        tg_send("🔻 Short Orders Summary:\n" + "\n".join(summary))

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
                        client.futures_create_order(symbol=sym, side="BUY", type="MARKET", quantity=qty, reduceOnly=True)
                        exit_list.append(f"{sym} closed: 5MA: {close_below_5ma}, In B20: {still_bottom20}, BTC<50: {btc_below}")
                            
            except Exception as e:
                log(f"Exit check error {sym}: {e}")
        if exit_list:
            tg_send("🚨 Exit Signals Short Triggered:\n" + "\n".join(exit_list))
    except Exception as e:
        log(f"check_exit_conditions error: {e}")

@handle_exceptions
def manage_margin():
    try:
        ratio = get_margin_ratio()
        log(f"⚠️ Margin Ratio: {ratio:.2f}%")

        if ratio < 5:
            tg_send("🔻 Margin < 5% — LIQUIDATE ALL POSITIONS ⚠️")
            liquidate_all()
        elif 5 <= ratio < 10:
            tg_send("🔸 Margin 5–10% — Sell/buy to cover 50% of portfolio")
            reduce_positions(0.50)
        elif 10 <= ratio < 20:
            tg_send("🔸 Margin 10–20% — Sell/buy to cover 30% of portfolio")
            reduce_positions(0.30)
        elif 20 <= ratio < 30:
            tg_send("🔸 Margin 20–30% — Sell/buy to cover 30% of portfolio")
            reduce_positions(0.30)
        else:
            pass  # Healthy margin, no action needed

    except Exception as e:
        log(f"manage_margin error: {e}")

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
