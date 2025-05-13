"""
Top-20 SHORT Futures Strategy
• Runs daily at 00:01 UTC
• Allocates 30% of USDT futures balance
• Filters:
    - USDT-M PERPETUAL contracts
    - Top 50 by 20-day avg dollar volume
    - Crossed BELOW 10MA yesterday
    - Only futures
• Ranks by BOTTOM 20 ROC20 (momentum)
• Enters SHORT via LIMIT ≥ VWAP, fallback MARKET at 12:00
• Closes all if BTC closes ABOVE 50MA
"""
import os, math, time, logging, functools, threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd, numpy as np, requests, schedule
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# === Setup ==================================================================
load_dotenv()
API_KEY  = os.getenv("FUTURE_API_KEY")
API_SEC  = os.getenv("FUTURE_API_SECRET")
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID")

client = Client(API_KEY, API_SEC, testnet=True)
client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"

ALLOC_PCT        = 0.30
TOP_COINS        = 20
MAX_VOL_LIST     = 50
VOL_LIMIT        = 10_000_000
DATA_LOOKBACK    = 60
VWAP_DISCOUNT    = 1.02  # LIMIT at or above VWAP for shorts

# === Logging & Telegram =====================================================
def get_logger():
    path = Path.cwd() / f"{Path.cwd().name}.log"
    logger = logging.getLogger("futures_short")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(path)
        fh.setFormatter(logging.Formatter('%(asctime)s — %(levelname)s — %(message)s'))
        logger.addHandler(fh)
    return logger
logger = get_logger()
def log(msg): print(msg); logger.info(msg)

def tg_send(text):
    if not text: return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try: requests.post(url, json={"chat_id": TG_CHAT, "text": text}, timeout=10)
    except Exception as e: log(f"Telegram error: {e}")

def retry(max_try=5):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **kw):
            delay = 1
            for _ in range(max_try):
                try: return fn(*a, **kw)
                except BinanceAPIException as e:
                    if e.code == -1003: time.sleep(delay); delay *= 2
                    else: raise
                except Exception as e:
                    log(f"Retry error: {e}")
                    time.sleep(delay)
                    delay *= 2
            raise Exception("Retry limit exceeded")
        return wrap
    return deco

# === Data Access ============================================================
FAPI_BASE = "https://fapi.binance.com/fapi/v1/klines"
@retry()
def fetch_klines(symbol, interval="1d", lookback_days=DATA_LOOKBACK):
    try:
        start = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp()*1000)
        params = {"symbol": symbol, "interval": interval, "startTime": start}
        r = requests.get(FAPI_BASE, params=params, timeout=10)
        if r.status_code != 200:
            log(f"Kline fail {symbol}: {r.text}"); return pd.DataFrame()
        df = pd.DataFrame(r.json(), columns=[
            "open_time","open","high","low","close","volume","close_time","qav",
            "trades","tbbav","tbqav","ignore"])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("open_time")[["open","high","low","close","volume"]].astype(float)
        return df
    except Exception as e:
        log(f"fetch_klines error {symbol}: {e}")
        return pd.DataFrame()

# === Filters ================================================================
def roc(series, period=20): return series.pct_change(periods=period) * 100
def vwap(df): return (df.close * df.volume).sum() / df.volume.sum()
def crossed_below(close, ma): return close.iloc[-2] > ma.iloc[-2] and close.iloc[-1] < ma.iloc[-1]

@retry()
def futures_universe():
    try:
        info = client.futures_exchange_info()
        return [s["symbol"] for s in info["symbols"] if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT"]
    except Exception as e:
        log(f"futures_universe error: {e}")
        return []

def bottom20_candidates():
    candidates = []
    for sym in futures_universe():
        try:
            df = fetch_klines(sym)
            if df.empty or len(df) < 31:
                log(f"{sym}: insufficient data"); continue
            df = df.iloc[:-1]
            dvol = (df.volume * df.close).rolling(20).mean().iloc[-1]
            if dvol < VOL_LIMIT:
                log(f"{sym}: low volume"); continue
            ma10 = df.close.rolling(10).mean()
            if not crossed_below(df.close, ma10):
                log(f"{sym}: no 10MA cross down"); continue
            roc20 = roc(df.close, 20).iloc[-1]
            candidates.append({"symbol": sym, "dvol": dvol, "roc": roc20})
        except Exception as e:
            log(f"{sym}: error — {e}")
            continue
    top50 = sorted(candidates, key=lambda x: x["dvol"], reverse=True)[:MAX_VOL_LIST]
    bottom20 = sorted(top50, key=lambda x: x["roc"])[:TOP_COINS]
    return bottom20

@retry()
def symbol_info(symbol):
    try:
        info = client.futures_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                return s
    except Exception as e:
        log(f"symbol_info error {symbol}: {e}")
    return None

def adjust_qty_price(symbol, price, qty):
    try:
        info = symbol_info(symbol)
        if not info: return price, 0
        step = float([f for f in info["filters"] if f["filterType"] == "LOT_SIZE"][0]["stepSize"])
        tick = float([f for f in info["filters"] if f["filterType"] == "PRICE_FILTER"][0]["tickSize"])
        prec_q = int(round(-math.log(step, 10)))
        prec_p = int(round(-math.log(tick, 10)))
        adj_qty = round(math.floor(qty / step) * step, prec_q)
        adj_prc = round(math.floor(price / tick) * tick, prec_p)
        return adj_prc, adj_qty
    except Exception as e:
        log(f"adjust_qty_price error {symbol}: {e}")
        return price, 0

def place_short(symbol, usdt_alloc):
    try:
        df_d = fetch_klines(symbol)
        vw = vwap(df_d.iloc[-1:]) if not df_d.empty else 0
        price = vw * VWAP_DISCOUNT
        qty = usdt_alloc / price
        price, qty = adjust_qty_price(symbol, price, qty)
        if qty == 0: return None
        order = client.futures_create_order(
            symbol=symbol,
            side="SELL",
            type="LIMIT",
            timeInForce="GTC",
            quantity=qty,
            price=str(price)
        )
        return order
    except Exception as e:
        log(f"place_short error {symbol}: {e}")
        return None

def close_all_shorts():
    try:
        tg_send("BTC > 50MA — Closing shorts ❌")
        positions = [p for p in client.futures_position_information() if float(p["positionAmt"]) < 0]
        for p in positions:
            sym = p["symbol"]
            amt = abs(float(p["positionAmt"]))
            price = float(p["markPrice"])
            _, qty = adjust_qty_price(sym, price, amt)
            if qty == 0: continue
            try:
                client.futures_create_order(symbol=sym, side="BUY", type="MARKET", quantity=qty, reduceOnly=True)
                log(f"Closed short {sym}")
            except Exception as e:
                log(f"Close error {sym}: {e}")
        tg_send("All short positions closed ✅")
    except Exception as e:
        log(f"close_all_shorts error: {e}")
        tg_send(f"Error closing shorts: {e}")

def btc_below_50ma():
    try:
        df = fetch_klines("BTCUSDT")
        if len(df) < 51: return False
        df = df.iloc[:-1]
        ma = df.close.rolling(50).mean().iloc[-1]
        return df.close.iloc[-1] < ma
    except Exception as e:
        log(f"btc_below_50ma error: {e}")
        return False

# === Rebalance & Noon fallback =============================================
def rebalance_shorts():
    try:
        if not btc_below_50ma():
            close_all_shorts(); return

        balance = client.futures_account_balance()
        usdt = sum(float(a["balance"]) for a in balance if a["asset"] == "USDT")
        cap = usdt * ALLOC_PCT
        if cap <= 0:
            tg_send("No USDT balance 😐"); return

        bottom20 = bottom20_candidates()
        if not bottom20:
            tg_send("No short candidates 💤"); return

        alloc_each = cap / len(bottom20)
        tg_send(f"Shorting Bottom-20 🔻: {len(bottom20)} coins")

        for coin in bottom20:
            try:
                place_short(coin["symbol"], alloc_each)
                log(f"Short entry {coin['symbol']}")
            except Exception as e:
                log(f"Short entry error {coin['symbol']}: {e}")
    except Exception as e:
        log(f"rebalance_shorts error: {e}")
        tg_send(f"Rebalance error: {e}")

def noon_fill_check():
    try:
        open_orders = client.futures_get_open_orders()
        for o in open_orders:
            try:
                client.futures_cancel_order(symbol=o["symbol"], orderId=o["orderId"])
                remaining = float(o["origQty"]) - float(o["executedQty"])
                if remaining <= 0: continue
                client.futures_create_order(symbol=o["symbol"], side="SELL", type="MARKET", quantity=remaining)
            except Exception as e:
                log(f"Noon fallback {o['symbol']}: {e}")
    except Exception as e:
        log(f"noon_fill_check error: {e}")

# === Main loop =============================================================
def main():
    schedule.every().day.at("00:01").do(rebalance_shorts)
    schedule.every().day.at("12:00").do(noon_fill_check)
    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == "__main__":
    tg_send("Futures SHORT bot started 🔻")
    main()
