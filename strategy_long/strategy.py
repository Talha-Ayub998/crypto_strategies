"""
Binance USDT‑M Futures Top‑20 Long Strategy — Full Flowchart Implementation
"""
import sys
import os
import schedule, time

# Add parent directory to sys.path so 'utils' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    client, log, tg_send, fetch_klines, roc,
    crossed_above, btc_above_50ma, adjust_qty_price, place_order,
    handle_exceptions, moving_average, get_margin_ratio, liquidate_all, reduce_positions,
)

# === Strategy Constants ===
ALLOC_PCT        = 0.30
TOP_COINS        = 20
MAX_PER_COIN_PCT = 0.05
VWAP_DISCOUNT    = 0.98

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
    return sorted(top50, key=lambda x: x["roc"], reverse=True)[:TOP_COINS]


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

        alloc_each = min(cap / len(top20), usdt * MAX_PER_COIN_PCT)
        summary = []
        for c in top20:
            place_order(c["symbol"], alloc_each, "BUY", summary, discount=VWAP_DISCOUNT)

        tg_send("🟢 Long Orders Summary:\n" + "\n".join(summary))

    except Exception as e:
        log(f"rebalance_longs error: {e}")
        tg_send(f"Rebalance error: {e}")

# === Market Fallback ===
@handle_exceptions
def noon_fill_check():
    try:
        open_orders = client.futures_get_open_orders()
        for o in open_orders:
            try:
                client.futures_cancel_order(symbol=o["symbol"], orderId=o["orderId"])
                remaining = float(o["origQty"]) - float(o["executedQty"])
                if remaining > 0:
                    client.futures_create_order(symbol=o["symbol"], side="BUY", type="MARKET", quantity=remaining)
            except Exception as e:
                log(f"Noon fallback error {o['symbol']}: {e}")
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
                        client.futures_create_order(symbol=sym, side="SELL", type="MARKET", quantity=qty, reduceOnly=True)
                        exit_list.append(f"{sym} closed: 20MA: {close_above_20ma}, In T20: {still_top20}, BTC>50: {btc_ok}")
            except Exception as e:
                log(f"Exit check error {sym}: {e}")

        if exit_list:
            tg_send("🚨 Exit Signals Long Triggered:\n" + "\n".join(exit_list))
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