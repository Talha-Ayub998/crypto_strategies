import traceback
import os, math, time, logging, functools, json
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd, requests
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# === Load ENV and client ====================================================
load_dotenv()
API_KEY  = os.getenv("FUTURE_API_KEY")
API_SEC  = os.getenv("FUTURE_API_SECRET")
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID")

client = Client(API_KEY, API_SEC, testnet=True)
client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"



#Constant
FARID_EXCEPTION_CHANEL = os.getenv("EXCEPTION_CHANEL")

def handle_exceptions(func):
    # Decorator to handle all exceptions
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log(f"Alert: handle_exceptions ({e})")
            error_type = e.__class__.__name__
            environment = "local"
            send_slack_validator_notification(
                error_type, environment)
    return wrapper  # Return the wrapped function

@handle_exceptions
def handle_imports_error(e):
    try:
        error_type = e.__class__.__name__
        environment = "local"
        send_slack_validator_notification(
            error_type, environment)
    except Exception as e:
        log("Error on handle_imports_error", e)


def format_slack_message(
    title,
    message,
    emoji='red_circle',
    print_stack_trace=False,
    module=None,
    environment=None,
) -> dict:
    header = f":{emoji}: :{emoji}:   *{title}*   :{emoji}: :{emoji}:"

    blocks = [{
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": header
        }
    }]

    if module:
        module_block = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":gear: :gear:  {module}  :gear: :gear:"
            }
        }
        blocks.append(module_block)

    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f":red_circle: :red_circle:   {environment}   :red_circle: :red_circle:"
        }
    })

    if print_stack_trace:
        stack_trace = traceback.format_exc()
        stack_trace_lines = stack_trace.splitlines()
        formatted_stack_trace = "\n".join(
            [f"{line}" for line in stack_trace_lines])
        stack_trace_block = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Stack Trace:*\n```\n{formatted_stack_trace}\n```"
            }
        }
        blocks.append(stack_trace_block)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    message.append(f"-Timestamp: {timestamp}")

    details_text = f"*Details:*\n" + '\n'.join(message)
    details_block = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": details_text
        }
    }
    blocks.append(details_block)

    blocks.append({
        "type": "divider"
    })

    return {"blocks": blocks}


def send_slack_validator_notification(error_type, environment ,stack_trace=False):
    url = FARID_EXCEPTION_CHANEL
    title = f":red_circle: :red_circle:   Error: {error_type}   :red_circle: :red_circle:"
    module = f"Default Service"


    if not stack_trace:
        stack_trace = traceback.format_exc()
        stack_trace_lines = stack_trace.splitlines()
        print(f'Alert: send_slack_validator_notification (AuthorizationError: {stack_trace_lines})')

    """Call format_slack_message to create the Slack message"""
    slack_message = format_slack_message(
        title=title,
        message=["Additional details..."],
        print_stack_trace=stack_trace,
        module=module,
        environment=environment,
    )

    """Convert the Slack message to JSON"""
    json_data = json.dumps(slack_message)

    response = requests.post(url, data=json_data)
    if response.status_code != 200:
        log(f"Alert: send_slack_validator_notification ({response.status_code} {response.text})")
        raise Exception(response.status_code, response.text)

# === Logger =================================================================
def get_logger(name="futures_bot"):
    path = Path.cwd() / f"{Path.cwd().name}.log"
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(path)
        fh.setFormatter(logging.Formatter('%(asctime)s — %(levelname)s — %(message)s'))
        logger.addHandler(fh)
    return logger

logger = get_logger()

def log(msg): print(msg); logger.info(msg)

# === Telegram ==============================================================

def tg_send(text):
    if not text: return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try: requests.post(url, json={"chat_id": TG_CHAT, "text": text}, timeout=10)
    except Exception as e: log(f"Telegram error: {e}")

# === Retry Decorator ========================================================
def retry(max_try=5):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **kw):
            delay = 1
            for _ in range(max_try):
                try:
                    return fn(*a, **kw)
                except BinanceAPIException as e:
                    if e.code == -1003: time.sleep(delay); delay *= 2
                    else:
                        raise
                except Exception as e:
                    log(f"Retry error: {e}")
                    time.sleep(delay); delay *= 2
            raise Exception("Retry limit exceeded")
        return wrap
    return deco

# === Data Helpers ===========================================================
FAPI_BASE = "https://fapi.binance.com/fapi/v1/klines"

@retry()
def fetch_klines(symbol, interval="1d", lookback_days=60):
    try:
        start = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp()*1000)
        params = {"symbol": symbol, "interval": interval, "startTime": start}
        r = requests.get(FAPI_BASE, params=params, timeout=10)
        if r.status_code != 200:
            log(f"Kline fetch fail {symbol}: {r.text}")
            return pd.DataFrame()
        df = pd.DataFrame(r.json(), columns=[
            "open_time","open","high","low","close","volume","close_time","qav",
            "trades","tbbav","tbqav","ignore"])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("open_time")[["open","high","low","close","volume"]].astype(float)
        return df
    except Exception as e:
        log(f"fetch_klines error {symbol}: {e}")
        return pd.DataFrame()

def roc(series, period=20):
    return series.pct_change(periods=period) * 100

def vwap(df):
    return (df.close * df.volume).sum() / df.volume.sum()

def crossed_above(close, ma):
    return close.iloc[-2] < ma.iloc[-2] and close.iloc[-1] > ma.iloc[-1]

def crossed_below(close, ma):
    return close.iloc[-2] > ma.iloc[-2] and close.iloc[-1] < ma.iloc[-1]

# === Binance Symbol Info ====================================================
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
        if not info:
            return price, 0
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

# === BTC MA checks ==========================================================
def btc_above_50ma():
    df = fetch_klines("BTCUSDT")
    if len(df) < 51:
        return False
    df = df.iloc[:-1]
    ma =  moving_average(df.close, 50).iloc[-1]
    return df.close.iloc[-1] > ma

def btc_below_50ma():
    df = fetch_klines("BTCUSDT")
    if len(df) < 51:
        return False
    df = df.iloc[:-1]
    ma =  moving_average(df.close, 50).iloc[-1]
    return df.close.iloc[-1] < ma

# === Place an order ==========================================================

def place_order(symbol, usdt_alloc, side, summary, discount):
    try:
        df = fetch_klines(symbol)
        if df.empty:
            summary.append(f"{symbol} ❌ No data")
            return None

        vw = vwap(df.iloc[-1:])
        price = vw * discount
        qty = usdt_alloc / price
        price, qty = adjust_qty_price(symbol, price, qty)
        if qty == 0:
            summary.append(f"{symbol} ❌ qty=0")
            return None

        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="LIMIT",
            timeInForce="GTC",
            quantity=qty,
            price=str(price)
        )

        summary.append(
            f"{symbol} | {side} | Qty: {qty} | Price: {price:.4f} | OrderID: {order['orderId']} | {order.get('status', '-')}"
        )
        return order
    except Exception as e:
        log(f"{symbol} order failed: {e}")
        summary.append(f"{symbol} ❌ Error: {e}")
        return None

def moving_average(series, period=20):
    """
    Calculate the simple moving average (SMA) of a pandas Series.
    """
    try:
        return series.rolling(period).mean()
    except Exception as e:
        log(f"Error on moving average {e}")

