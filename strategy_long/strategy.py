"""
Binance USDT‑M Futures Top‑20 Long Strategy — Full Flowchart Implementation
"""
import schedule, time
from utils import (
    client, log, tg_send, fetch_klines, vwap, roc,
    crossed_above, btc_above_50ma, symbol_info, adjust_qty_price, place_order
)

# === Strategy Constants ===
ALLOC_PCT        = 0.30
TOP_COINS        = 20
MAX_VOL_LIST     = 50
VOL_LIMIT        = 10_000_000
VWAP_DISCOUNT    = 0.98
CHECK_INTERVAL   = 60 * 30  # every 30 minutes

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
            if df.empty or len(df) < 31:
                continue
            df = df.iloc[:-1]  # remove current day
            dvol = (df.volume * df.close).rolling(20).mean().iloc[-1]
            if dvol < VOL_LIMIT:
                continue
            ma20 = df.close.rolling(20).mean()
            if not crossed_above(df.close, ma20):
                continue
            r = roc(df.close, 20).iloc[-1]
            candidates.append({"symbol": sym, "dvol": dvol, "roc": r})
        except Exception as e:
            log(f"{sym} error: {e}")
    top50 = sorted(candidates, key=lambda x: x["dvol"], reverse=True)[:MAX_VOL_LIST]
    return sorted(top50, key=lambda x: x["roc"], reverse=True)[:TOP_COINS]

# === Entry ===
def place_entry(symbol, usdt_alloc, summary):
    try:
        df = fetch_klines(symbol)
        if df.empty:
            summary.append(f"{symbol} ❌ No data")
            return
        vw = vwap(df.iloc[-1:])
        price = vw * VWAP_DISCOUNT
        qty = usdt_alloc / price
        price, qty = adjust_qty_price(symbol, price, qty)
        if qty == 0:
            summary.append(f"{symbol} ❌ qty=0")
            return
        order = client.futures_create_order(
            symbol=symbol, side="BUY", type="LIMIT",
            timeInForce="GTC", quantity=qty, price=str(price)
        )
        summary.append(f"{symbol} | BUY | Qty: {qty} | Price: {price:.2f} | ID: {order['orderId']} | {order['status']}")
    except Exception as e:
        log(f"{symbol} entry error: {e}")
        summary.append(f"{symbol} ❌ Error: {e}")

# === Close All ===
def close_all_longs():
    try:
        tg_send("BTC < 50MA — Closing all longs ❌")
        positions = [p for p in client.futures_position_information() if float(p["positionAmt"]) > 0]
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

        alloc_each = cap / len(top20)
        summary = []
        for c in top20:
            place_order(c["symbol"], alloc_each, "BUY", summary, discount=0.98)

        tg_send("🟢 Long Orders Summary:\n" + "\n".join(summary))

    except Exception as e:
        log(f"rebalance_longs error: {e}")
        tg_send(f"Rebalance error: {e}")

# === Market Fallback ===
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
                close_above_20ma = df.close.iloc[-1] > df.close.rolling(20).mean().iloc[-1]
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
            tg_send("🚨 Exit Signals Triggered:\n" + "\n".join(exit_list))
    except Exception as e:
        log(f"check_exit_conditions error: {e}")

# === Scheduler ===
def main():
    schedule.every().day.at("00:01").do(rebalance_longs)
    schedule.every().day.at("12:00").do(noon_fill_check)
    schedule.every(CHECK_INTERVAL).seconds.do(check_exit_conditions)
    tg_send("✅ Long Strategy Bot Running")
    while True:
        schedule.run_pending()
        time.sleep(10)

if __name__ == "__main__":
    main()