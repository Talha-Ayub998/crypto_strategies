"""
Binance USDT‑M Futures *Top‑20 Long* strategy
=================================================
•   Runs daily at 00:01 UTC
•   Allocates 30 % of available USDT balance to the strategy
•   Universe: USDT‑margined PERPETUAL contracts that
    – are in the *top‑50* by 20‑day average dollar‑volume
    – closed *above* their 20‑day SMA yesterday (first cross)
•   Ranks the filtered list by 20‑day ROC (momentum) and keeps the top 20.
•   Opens/adjusts **LONG** positions using LIMIT orders at 98 % of VWAP
    (fallback to MARKET at 12:00 UTC if not filled).
•   If BTCUSDT closes *below* its 50‑day SMA → close all open strategy positions.
"""

# === Standard libs ===========================================================
from typing import Optional
import os, json, math, time, logging, threading, functools
from datetime import datetime, timedelta, timezone
from pathlib import Path

# === 3rd‑party ===============================================================
import pandas as pd
import numpy  as np
import requests
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# === Load ENV ================================================================
load_dotenv()
API_KEY  = os.getenv("FUTURE_API_KEY")
API_SEC  = os.getenv("FUTURE_API_SECRET")
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID")

# === Binance client (USDT‑M Futures test‑net) ================================
client               = Client(API_KEY, API_SEC, testnet=True)
client.FUTURES_URL   = "https://testnet.binancefuture.com/fapi"  # ensure futures test‑net

# === Strategy constants ======================================================
ALLOC_PCT        = 0.30           # 30 % of futures USDT balance
TOP_COINS        = 20             # hold max 20
MAX_VOL_LIST     = 50             # top‑50 by $‑volume pre‑filter
VOL_LIMIT        = 10_000_000     # minimum avg $‑volume as extra safety
DATA_LOOKBACK    = 60             # candles to pull (need ≥ 31)
VWAP_DISCOUNT    = 0.98           # limit price = 98 % of VWAP

# === Helpers ================================================================

def get_logger() -> logging.Logger:
    folder = Path.cwd()
    log_f  = folder / f"{folder.name}.log"
    logger = logging.getLogger("futures_long")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_f)
        fh.setFormatter(logging.Formatter('%(asctime)s — %(levelname)s — %(message)s'))
        logger.addHandler(fh)
    return logger
logger = get_logger()

def log(msg: str):
    print(msg)
    logger.info(msg)

# --- Telegram ---------------------------------------------------------------

def tg_send(text: str = None):
    if not text: return
    url  = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    data = {"chat_id": TG_CHAT, "text": text}
    try:
        requests.post(url, json=data, timeout=10)
    except Exception as e:
        log(f"TG error: {e}")

# --- Rate‑limit / retry decorator ------------------------------------------

def retry(max_try=5):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **kw):
            delay = 1
            for _ in range(max_try):
                try:
                    return fn(*a, **kw)
                except BinanceAPIException as e:
                    if e.code == -1003:  # rate‑limit
                        time.sleep(delay)
                        delay *= 2
                    else:
                        raise
            raise Exception("Retry limit exceeded")
        return wrap
    return deco

# === Data access ============================================================
FAPI_BASE = "https://fapi.binance.com/fapi/v1/klines"

@retry()
def fetch_klines(symbol: str, interval="1d", lookback_days: int = DATA_LOOKBACK) -> pd.DataFrame:
    # start = int((datetime.utcnow() - timedelta(days=lookback_days)).timestamp()*1000)
    start = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000)
    params = {"symbol": symbol, "interval": interval, "startTime": start}
    r = requests.get(FAPI_BASE, params=params, timeout=10)
    if r.status_code != 200:
        log(f"Kline fetch fail {symbol}: {r.text}"); return pd.DataFrame()
    df = pd.DataFrame(r.json(), columns=[
        "open_time","open","high","low","close","volume","close_time","qav",
        "trades","tbbav","tbqav","ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")[["open","high","low","close","volume"]].astype(float)
    return df

# === Technicals =============================================================

def roc(series: pd.Series, period: int = 20):
    return series.pct_change(periods=period) * 100

def vwap(df: pd.DataFrame):
    return (df.close * df.volume).sum() / df.volume.sum()

def crossed_above(close: pd.Series, ma: pd.Series) -> bool:
    return close.iloc[-2] < ma.iloc[-2] and close.iloc[-1] > ma.iloc[-1]

# === Universe build =========================================================

@retry()
def futures_universe():
    try:
        exinfo = client.futures_exchange_info()
        symbols = [
            s["symbol"]
            for s in exinfo["symbols"]
            if s["contractType"] == "PERPETUAL"
            and s["quoteAsset"] == "USDT"
            and s["status"] == "TRADING"
        ]
        return symbols
    except Exception as e:
        log(f"Failed to fetch futures universe: {e}")
        return []

# --- Candidate selection ----------------------------------------------------

# def top20_candidates():
#     candidates = []
#     for sym in futures_universe():
#         df = fetch_klines(sym)
#         if len(df) < 31:            # need ≥ 31 full candles
#             continue
#         df = df.iloc[:-1]           # drop today (partial) candle
#         dollar_vol = (df.volume * df.close).rolling(20).mean().iloc[-1]
#         if dollar_vol < VOL_LIMIT:
#             continue
#         ma20       = df.close.rolling(20).mean()
#         if not crossed_above(df.close, ma20):
#             continue
#         roc20_val  = roc(df.close, 20).iloc[-1]
#         candidates.append({"symbol": sym, "dvol": dollar_vol, "roc": roc20_val})

#     # rank by dollar volume -> top 50
#     top50 = sorted(candidates, key=lambda x: x["dvol"], reverse=True)[:MAX_VOL_LIST]
#     # rank by roc -> top 20
#     top20 = sorted(top50, key=lambda x: x["roc"], reverse=True)[:TOP_COINS]
#     return top20

def top20_candidates():
    candidates = []
    for sym in futures_universe():
        try:
            df = fetch_klines(sym)
            if df.empty or len(df) < 31:
                log(f"{sym}: insufficient data")
                continue
            df = df.iloc[:-1]  # remove current partial candle

            # Dollar volume
            dollar_vol = (df.volume * df.close).rolling(20).mean().iloc[-1]
            if dollar_vol < VOL_LIMIT:
                log(f"{sym}: volume too low")
                continue

            # 20MA crossover
            ma20 = df.close.rolling(20).mean()
            if not crossed_above(df.close, ma20):
                log(f"{sym}: no 20MA cross")
                continue

            # ROC20
            roc20_val = roc(df.close, 20).iloc[-1]
            candidates.append({
                "symbol": sym,
                "dvol": dollar_vol,
                "roc": roc20_val
            })

        except Exception as e:
            log(f"{sym}: error during processing — {e}")
            continue

    # Sort top 50 by volume
    top50 = sorted(candidates, key=lambda x: x["dvol"], reverse=True)[:MAX_VOL_LIST]
    # Sort top 20 by ROC
    top20 = sorted(top50, key=lambda x: x["roc"], reverse=True)[:TOP_COINS]
    return top20


# === Order utilities ========================================================

@retry()
def symbol_info(symbol: str):
    try:
        info = client.futures_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                return s
    except Exception as e:
        log(f"Symbol info fetch error for {symbol}: {e}")
    return None

# LOT_SIZE / PRICE_FILTER helpers -------------------------------------------

def adjust_qty_price(symbol: str, price: float, qty: float):
    try:
        info = symbol_info(symbol)
        if not info:
            return price, 0
        step = float([f for f in info["filters"] if f["filterType"] == "LOT_SIZE"][0]["stepSize"])
        tick = float([f for f in info["filters"] if f["filterType"] == "PRICE_FILTER"][0]["tickSize"])
        prec_q = int(round(-math.log(step, 10)))
        prec_p = int(round(-math.log(tick, 10)))
        adj_qty = math.floor(qty / step) * step
        adj_qty = round(adj_qty, prec_q)
        adj_prc = math.floor(price / tick) * tick
        adj_prc = round(adj_prc, prec_p)
        return adj_prc, adj_qty
    except Exception as e:
        log(f"Adjust price/qty error for {symbol}: {e}")
        return price, 0

# Place LIMIT at 98 % VWAP ----------------------------------------------------

def place_entry(symbol: str, usdt_alloc: float) -> Optional[dict]:
    try:
        df = fetch_klines(symbol, "1h", 1)
        df_d = fetch_klines(symbol)
        vw = vwap(df_d.iloc[-1:]) if not df_d.empty else vwap(df)
        price = vw * VWAP_DISCOUNT
        qty = usdt_alloc / price
        price, qty = adjust_qty_price(symbol, price, qty)
        if qty == 0:
            log(f"Skipped {symbol}: qty=0 after adjust")
            return None
        return client.futures_create_order(
            symbol=symbol,
            side="BUY",
            type="LIMIT",
            timeInForce="GTC",
            quantity=qty,
            price=str(price)
        )
    except Exception as e:
        log(f"Entry placement failed for {symbol}: {e}")
        return None

# Close ALL positions --------------------------------------------------------

def close_all():
    try:
        tg_send("BTC < 50MA — closing all strategy positions…")
        positions = [p for p in client.futures_position_information() if float(p["positionAmt"]) != 0]
        for p in positions:
            sym = p["symbol"]
            amt = float(p["positionAmt"])
            side = "SELL" if amt > 0 else "BUY"
            qty = abs(amt)
            price = float(p["markPrice"])
            _, qty = adjust_qty_price(sym, price, qty)
            if qty == 0:
                continue
            try:
                client.futures_create_order(symbol=sym, side=side, type="MARKET", quantity=qty, reduceOnly=True)
                log(f"Closed {sym} {qty}")
            except Exception as e:
                log(f"Close error for {sym}: {e}")
        tg_send("All futures positions closed ✅")
    except Exception as e:
        log(f"close_all(): Failed — {e}")
        tg_send(f"Error closing positions: {e}")

# === BTC 50MA check =========================================================

def btc_above_50ma() -> bool:
    try:
        df = fetch_klines("BTCUSDT")
        if len(df) < 51:
            log("BTCUSDT: insufficient data for 50MA check")
            return False
        df = df.iloc[:-1]
        ma = df.close.rolling(50).mean().iloc[-1]
        return df.close.iloc[-1] > ma
    except Exception as e:
        log(f"BTC 50MA check error: {e}")
        return False

# === Rebalance routine ======================================================

# === Safe Rebalancing Routine ===============================================
def rebalance():
    try:
        if not btc_above_50ma():
            close_all()
            return

        bal_info = client.futures_account_balance()
        usdt_bal = sum(float(a["balance"]) for a in bal_info if a["asset"] == "USDT")
        alloc_cap = usdt_bal * ALLOC_PCT
        if alloc_cap <= 0:
            tg_send("No USDT balance available 🙁")
            return

        top20 = top20_candidates()
        if not top20:
            tg_send("No candidates today")
            return

        tg_send("Opening Top‑20 long basket…")
        alloc_each = alloc_cap / len(top20)
        for c in top20:
            try:
                res = place_entry(c["symbol"], alloc_each)
                if res:
                    log(f"Entry placed {c['symbol']}")
            except Exception as e:
                log(f"Entry error for {c['symbol']}: {e}")
    except Exception as e:
        log(f"rebalance(): error — {e}")
        tg_send(f"Rebalance failed: {e}")

# === 12:00 UTC fill‑check (convert to market) ===============================

def noon_fill_check():
    try:
        open_orders = client.futures_get_open_orders()
        for o in open_orders:
            try:
                client.futures_cancel_order(symbol=o["symbol"], orderId=o["orderId"])
                side = o["side"]
                qty = float(o["origQty"]) - float(o["executedQty"])
                if qty <= 0:
                    continue
                client.futures_create_order(symbol=o["symbol"], side=side, type="MARKET", quantity=qty)
            except Exception as e:
                log(f"Noon fill convert failed for {o['symbol']}: {e}")
    except Exception as e:
        log(f"noon_fill_check(): error — {e}")


# === Scheduler ==============================================================
import schedule, time as _time

def main():
    schedule.every().day.at("00:01").do(rebalance)
    schedule.every().day.at("12:00").do(noon_fill_check)
    while True:
        schedule.run_pending(); _time.sleep(30)

if __name__ == "__main__":
    tg_send("Futures Long bot started ✅")
    main()



