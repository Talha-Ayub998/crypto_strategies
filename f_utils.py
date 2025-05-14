import os
import ast
import json
import logging
import requests
import traceback
import pandas as pd
from pathlib import Path
from dateutil import parser
from datetime import datetime, timedelta, timezone

from constant import ASSET_TRADES, JPY_PAIRS, GOLD , INDICES, OTHER_PAIRS, ASSETS, PIP_VALUES
#Constant
FARID_EXCEPTION_CHANEL = "https://hooks.slack.com/services/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"
#https://hooks.slack.com/services/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"

def handle_exceptions(func):
    # Decorator to handle all exceptions
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print_log(f"Alert: handle_exceptions ({e})")
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
        print_log("Error on handle_imports_error", e)


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
        print_log(f"Alert: send_slack_validator_notification ({response.status_code} {response.text})")
        raise Exception(response.status_code, response.text)

@handle_exceptions
def get_folder_logger():
    # Use the current working directory directly for logging
    folder_path = Path(os.getcwd())  # cwd is set in master.py's subprocess call
    log_file = folder_path / f"{folder_path.name}.log"

    # Use a unique logger name based on the folder path
    logger_name = f"{folder_path.name}_logger"

    # Check if the logger already exists
    if logger_name not in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        # File handler for this specific folder's log file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Define the formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handler to the logger if not already added
        if not logger.hasHandlers():
            logger.addHandler(file_handler)
    else:
        logger = logging.getLogger(logger_name)

    return logger

@handle_exceptions
def print_log(message):
    print(message)
    logger = get_folder_logger()
    logger.info(message)
 

@handle_exceptions
def fetch_latest_date_atr(df):
    return round(df['atr'].iloc[0],5), df['datetime'].iloc[0]

@handle_exceptions
def reset_index(df):
    # df['datetime'] = pd.to_datetime(df['time'])
    df = df.iloc[::-1].reset_index(drop=True)
    return df

@handle_exceptions
def round_oil_volume_ic(volume, ic_symbol):
    if ic_symbol == 'XTIUSD':
        volume = round(volume * 2) / 2
        if volume > 50:
            volume = 50
        return volume
    return volume

@handle_exceptions
def is_trading_window_open(start_hour, end_hour, current_hour):

    # (e.g., 05 to 23)
    if start_hour <= end_hour:
        return start_hour <= current_hour < end_hour
    # (e.g., 23 to 05) or 5 AM to 2 AM
    else:
        return current_hour >= start_hour or current_hour < end_hour


@handle_exceptions
def is_gap_within_atr_threshold(gap, current_day_atr, gap_threshold):
    return gap < gap_threshold * current_day_atr

@handle_exceptions
def calculate_fvg_gap(high, low):
    return high - low

@handle_exceptions
def check_mitigations(direction, high, low, max_high, min_low, gap, current_day_atr, gap_threshold):
    if direction == 'Bearish':
        return high <= max_high or is_gap_within_atr_threshold(gap, current_day_atr, gap_threshold)
    elif direction == 'Bullish':
        return min_low <= low or is_gap_within_atr_threshold(gap, current_day_atr, gap_threshold)
    return False

@handle_exceptions
def get_fvg_values(df, j, direction):
    if direction == 'Bearish':
        return df['low'].iloc[j], df['high'].iloc[j - 2] #j-2 is latest candle
    elif direction == 'Bullish':
        return df['low'].iloc[j - 2], df['high'].iloc[j]
    return None, None

@handle_exceptions
def check_fvg_and_mitigation(df, j, current_day_atr, max_high, min_low, direction_candidate,gap_threshold, check_mitigation=False):
    """
    Checks if there is a Fair Value Gap (FVG) and whether it is mitigated.
    Returns the FVG flag, candle, high, low, date, and direction.
    """
    date = df['datetime'].iloc[j]
    high, low = get_fvg_values(df, j, direction_candidate)
    
    if not (high or low):
        return False, None, None, None, None, None

    gap = calculate_fvg_gap(high, low)

    if not gap or gap < 0:
        return False, None, None, None, None, None

    if check_mitigation:
        mitigation_detail = check_mitigations(direction_candidate, high, low, max_high, min_low, gap, current_day_atr, gap_threshold)
        if mitigation_detail:
            return False, None, None, None, None, None
    
    return True, df.iloc[j - 2], high, low, date, direction_candidate


@handle_exceptions
def is_bullish_candle(open_price, close_price):
    """Returns True if the candle is bullish (close > open)."""
    return close_price > open_price

@handle_exceptions
def is_bearish_candle(open_price, close_price):
    return open_price > close_price

def process_fvg(df, j, current_day_atr, max_high, min_low, direction, gap_threshold, check_mitigation):
        return check_fvg_and_mitigation(df, j, current_day_atr, max_high, min_low, direction, gap_threshold, check_mitigation)



@handle_exceptions
def calculate_pips(high, low, pair): 

    pip_value = None    
    if pair in JPY_PAIRS:
        pip_value =  PIP_VALUES["JPY_PAIRS"]
    elif pair in INDICES:
        pip_value = PIP_VALUES["INDICES"]
    elif pair in GOLD:
        pip_value = PIP_VALUES["GOLD"]
    elif pair in OTHER_PAIRS:
        pip_value = PIP_VALUES["OTHER_PAIRS"]

    return round(((high - low) / pip_value), 0) if pip_value else None


def get_pair_value(symbol):
    for k, v in ASSETS.items():
        if v == symbol:
            return k
      
def calculate_candlestick_body_ratio(latest_candle, second_candle, first_candle, symbol):
    """
    This function calculates the ratio of the third candlestick body (latest_candle)
    to the average of the first two candlestick bodies.
    """
    # Calculate the body of each candlestick


    first_body = abs(first_candle['close'] - first_candle['open'])
    second_body = abs(second_candle['close'] - second_candle['open'])
    third_body = abs(latest_candle['close'] - latest_candle['open'])

    average_body_1_and_2 = (first_body + second_body) / 2
    

    if average_body_1_and_2 == 0:
        return None  
    
    ratio = third_body / average_body_1_and_2
    return ratio


@handle_exceptions
def identify_fvg(df, current_day_atr, gap_threshold, check_mitigation, symbol):
    ''' Return the FVG Flag if a FVG is occured in given Candles
    Different Scenarios for FVG in Order 1st Candle - 2nd Candle - 3rd Candle
    # 1. Bearish - Bearish - Bullish
    # 2. Bearish - Bearish - Bearish
    # 3. Bearish - Bullish - Bearish
    # 4. Bearish - Bullish - Bullish
    # 5. Bullish - Bullish - Bullish
    # 6. Bullish - Bearish - Bearish
    # 7. Bullish - Bullish - Bearish
    # 8. Bullish - Bearish - Bullish
    Indexes : j-2 Candle is Latest candle , j-1 Second Candle, j Third Candle
    Bullish and Bearish : if open-close < 0 then Bullish, open-close > 0 Bearish
    
    Index [0] is the latest candle
    '''

    fvg_flag = False
    fvg_candle = df.iloc[0]
    high = 0
    low = 0
    direction = ''
    date = ''

    max_high = df['high'].iloc[0]
    min_low = df['low'].iloc[0]

    pair = get_pair_value(symbol)
    
    
    fvg_candles = []

    for j in range(2, len(df)):
        latest_candle_open, latest_candle_close = df['open'].iloc[j - 2], df['close'].iloc[j - 2]
        second_candle_open, second_candle_close = df['open'].iloc[j - 1], df['close'].iloc[j - 1]
        first_candle_open, first_candle_close = df['open'].iloc[j], df['close'].iloc[j]

        max_high = max(max_high, df['high'].iloc[j - 2])
        min_low = min(min_low, df['low'].iloc[j - 2])
        

        conditions = [
            (is_bearish_candle(first_candle_open, first_candle_close), is_bearish_candle(second_candle_open, second_candle_close), is_bullish_candle(latest_candle_open, latest_candle_close), 'Bearish'),
            (is_bearish_candle(first_candle_open, first_candle_close), is_bearish_candle(second_candle_open, second_candle_close), is_bearish_candle(latest_candle_open, latest_candle_close), 'Bearish'),
            (is_bearish_candle(first_candle_open, first_candle_close), is_bullish_candle(second_candle_open, second_candle_close), is_bearish_candle(latest_candle_open, latest_candle_close), 'Bullish'),
            (is_bearish_candle(first_candle_open, first_candle_close), is_bullish_candle(second_candle_open, second_candle_close), is_bullish_candle(latest_candle_open, latest_candle_close), 'Bullish'),
            (is_bullish_candle(first_candle_open, first_candle_close), is_bullish_candle(second_candle_open, second_candle_close), is_bullish_candle(latest_candle_open, latest_candle_close), 'Bullish'),
            (is_bullish_candle(first_candle_open, first_candle_close), is_bearish_candle(second_candle_open, second_candle_close), is_bearish_candle(latest_candle_open, latest_candle_close), 'Bearish'),
            (is_bullish_candle(first_candle_open, first_candle_close), is_bullish_candle(second_candle_open, second_candle_close), is_bearish_candle(latest_candle_open, latest_candle_close), 'Bullish'),
            (is_bullish_candle(first_candle_open, first_candle_close), is_bearish_candle(second_candle_open, second_candle_close), is_bullish_candle(latest_candle_open, latest_candle_close), 'Bearish'),
        ]

        for condition in conditions:
            if all(condition[:3]):
                fvg_flag, fvg_candle, high, low, date, direction = process_fvg(df, j, current_day_atr, max_high, min_low, condition[3], gap_threshold, check_mitigation)
             
                if not fvg_flag:
                    continue
                
                latest_candle =  df.iloc[j - 2]
                second_candle = df.iloc[j - 1]
                first_candle = df.iloc[j]
                if direction == 'Bullish':
                    fvg_pips = calculate_pips(latest_candle['high'], first_candle['low'], pair)
                else:
                    fvg_pips = calculate_pips(first_candle['high'], latest_candle['low'], pair)
                
                try:
                
                    fvg_move_size = fvg_pips / current_day_atr
                
                except Exception as e:
                    print(e)
               
                fvg_third_candle_body_ratio = calculate_candlestick_body_ratio(latest_candle, second_candle, first_candle, symbol)
                fvg_candles.append(
                    {'fvg_flag':fvg_flag,  
                     'fvg_low': low,
                     'fvg_high': high,
                     'signal':direction, 
                     'date':df['datetime'].iloc[j - 2], 
                     'fvg_move_size': round(fvg_move_size, 4), 
                     'fvg_third_candle_body_ratio': round(fvg_third_candle_body_ratio, 4),
                     })
                
    return fvg_candles


@handle_exceptions
def get_order_type(order_type, current_price, entry_price):
    if order_type == 'Buy':
        mt5_order_type = 'Limit' if current_price > entry_price else 'Stop'
    elif order_type == 'Sell':
        mt5_order_type = 'Limit' if current_price < entry_price else 'Stop'
    print_log(f'Order type is {order_type}, {current_price}, {entry_price}')
    return mt5_order_type




# Get the current date in string format (e.g., '2024-09-27')
def get_current_date():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')

# Load trades from the file
def load_trades(filename='trades_count.txt'):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return default trades with the current date if file doesn't exist
        return {"date": get_current_date(), "trades": ASSET_TRADES.copy()}


def load_asset_trades_from_parameters(strategy_path):
    """Loads the ASSET variable from the parameters.py file in the given strategy folder."""
    import importlib.util
    
    parameters_path = strategy_path / 'parameters.py'
    
    # Check if the parameters.py file exists
    if parameters_path.exists():
        # Dynamically load the module from the parameters.py file
        spec = importlib.util.spec_from_file_location('parameters', parameters_path)
        parameters = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parameters)
        
        # Retrieve the ASSET variable and ensure it's a dictionary
        if hasattr(parameters, 'ASSET') and isinstance(parameters.ASSET, dict):
            # Set all values in the dictionary to 0
            return {key: 0 for key in parameters.ASSET}
    
    # Return None if ASSET is not found or parameters.py does not exist
    return None

def save_trades(trades, date, strategy_path, filename='trades_count.txt'):
    """Saves trades to a JSON file in the given strategy folder."""
    data = {"date": date, "trades": trades}
    trades_file = strategy_path / filename  # Save file in the strategy's folder

    # Ensure the directory exists
    strategy_path.mkdir(parents=True, exist_ok=True)
    
    with trades_file.open('w') as file:
        json.dump(data, file)


def reset_trades_for_new_day(asset_trades, strategy_path):
    """Resets trades for a new day and saves them in the strategy folder."""
    if 'strategy_2_account_48' in str(strategy_path):
        asset_trades = load_asset_trades_from_parameters(strategy_path)

    new_trades = asset_trades.copy()  # Reset to default trades
    save_trades(new_trades, get_current_date(), strategy_path)
    print(f"Trades reset for new day in {strategy_path}")
    return new_trades

# Check if a trade can be executed for a given ticker
def can_execute_trade(ticker, trade_data, max_trades_per_day):
    trade_data_date = trade_data['date']
    trades = trade_data['trades']
    if trades[ticker] >= max_trades_per_day and trade_data_date == get_current_date():
        print(f"Skipping trade for {ticker}: Max trades for today reached.")
        return False
    return True

# Execute trade and update the trades count
def execute_trade(ticker, trade_data, strategy_path):
    trades = trade_data['trades']
    trades[ticker] += 1
    save_trades(trades, get_current_date(), strategy_path)
    print(f"Trade executed for {ticker}. Total trades for today: {trades[ticker]}")

def is_all_bullish(candles):
    return all(candle['close'] > candle['open'] for candle in candles)

def is_all_bearish(candles):
    return all(candle['close'] < candle['open'] for candle in candles)


def calculate_body_size(candle):
    return abs(candle['open'] - candle['close'])

def calculate_average_body_of_first_two_candles(candle_1, candle_2):
    body_candle_1 = calculate_body_size(candle_1)
    body_candle_2 = calculate_body_size(candle_2)
    return (body_candle_1 + body_candle_2) / 2

def check_conditions(candle_1, candle_2, candle_3, buy_price_point):
    average_body_1_and_2 = calculate_average_body_of_first_two_candles(candle_1, candle_2)
    body_candle_3 = calculate_body_size(candle_3)
    
    average_body_condition = body_candle_3 >= 0.8 * average_body_1_and_2
    price_point_condition = candle_3['close'] < buy_price_point

    return average_body_condition, price_point_condition


def send_message_to_slack(url, message):
    """
    Send a message to Slack using a specified URL.

    Returns:
        None
    """    
    response = requests.post(url, json=message)

    if response.status_code != 200:
        print(f"Alert: send_message_to_slack ({response.status_code} {response.text})")


def get_percentage_of_value(percent, daily_atr):
    """
    Calculate a percentage of the daily ATR value.
    """
    return round((percent / 100) * daily_atr, 4)



def is_bearish_pattern_s2(candle_1, candle_2, candle_3):
    """
    Determines if a bearish pattern is present based on the conditions for candle_1, candle_2, and candle_3.

    """
    return (
        candle_2['high'] > candle_1['high'] and
        candle_3['close'] < min(candle_2['close'], candle_2['open']) and
        candle_3['close'] < min(candle_1['close'], candle_1['open']) and
        candle_2['close'] < max(candle_1['open'], candle_1['close']) and
        candle_3['open'] > candle_3['close']
    )


def is_bullish_pattern_s2(candle_1, candle_2, candle_3):
    """
    Determines if a bullish pattern is present based on the conditions for candle_1, candle_2, and candle_3.
    """
    return (
        candle_2['low'] < candle_1['low'] and
        candle_3['close'] > max(candle_2['close'], candle_2['open']) and
        candle_3['close'] > max(candle_1['close'], candle_1['open']) and
        candle_2['close'] > min(candle_1['close'], candle_1['open']) and
        candle_3['open'] < candle_3['close']
    )


def extract_first_three_candles(df):
    """
    Extract the first, second, and latest (third) candles from the DataFrame.
    
    """
    candle_1 = df.iloc[2]
    candle_2 = df.iloc[1]
    latest_candle = df.iloc[0]

    return candle_1, candle_2, latest_candle



def calculate_oil_prices(current_price, apply_difference_of_oil):
    """
    Calculate the buy limit price and buy limit take profit (TP) based on the current price 
    and the specified difference.
    """
    upper_oil_price = current_price + apply_difference_of_oil
    lower_oil_price = current_price - apply_difference_of_oil
    return upper_oil_price, lower_oil_price

def identify_ha_trend(ticker , candle, m3_ha_trend):
    if candle['HA_low'] == candle['HA_open'] and candle['HA_close'] > candle['HA_open']:  
        return 'Bullish'
    elif candle['HA_high'] == candle['HA_open'] and candle['HA_close'] < candle['HA_open']:
        return 'Bearish'
    elif  m3_ha_trend.get(ticker) and m3_ha_trend[ticker] == "Bullish" and candle['HA_close'] > candle['HA_open']:
        return 'Bullish'
    elif m3_ha_trend.get(ticker) and m3_ha_trend[ticker] == "Bearish" and candle['HA_close'] < candle['HA_open']:
        return 'Bearish'
    else:
        return 'Neutral'
    
def identify_db_ha_trend(candle):
    score = 0
 
    # If HA is bullish (HA_close > HA_open),
    if candle['HA_close'] > candle['HA_open']:
        score += 1
    # If there is no lower wick (HA_low == HA_open),
    if candle['HA_low'] == candle['HA_open']:
        score += 1

    # If HA is bearish (HA_close < HA_open),
    if candle['HA_close'] < candle['HA_open']:
        score -= 1
    # If there is no upper wick (HA_high == HA_open),
    if candle['HA_high'] == candle['HA_open']:
        score -= 1


    return score


def get_broker_expiry_unix(target_hour_utc=23, target_minute_utc=0, tick_time=None, utc_now=None):

    if tick_time is None:
        raise ValueError("tick_time is required to estimate broker's local time.")

    # Current UTC time
    utc_now = utc_now or datetime.now(timezone.utc)

    # Step 1: Convert tick time to a UTC-aware datetime to approximate broker's local time
    broker_local_time = datetime.fromtimestamp(tick_time, tz=timezone.utc)

    # Step 2: Estimate broker's UTC offset by comparing with UTC now
    broker_offset = broker_local_time - utc_now
    print(f"Broker's UTC offset estimated: {broker_offset}")

    # Step 3: Define target expiry in UTC
    target_expiry_utc = utc_now.replace(hour=target_hour_utc, minute=target_minute_utc, second=0, microsecond=0)
    
    # Adjust target expiry to the next day if the current UTC time is past the target
    if utc_now >= target_expiry_utc:
        target_expiry_utc += timedelta(days=1)

    # Step 4: Convert target UTC expiry to broker's local time and obtain Unix timestamp
    target_expiry_broker_time = target_expiry_utc + broker_offset
    expiry_time_unix = int(target_expiry_broker_time.timestamp())
    
    # print_log(f"Current UTC time: {utc_now}")
    # print_log(f"Broker time approximation: {broker_local_time}")
    # print_log(f"Target expiry (UTC): {target_expiry_utc} (11:00 PM UTC)")
    # print_log(f"Broker-adjusted expiry: {target_expiry_broker_time} (local broker time)")
    # print_log(f"Expiry timestamp (Unix): {expiry_time_unix}")

    return expiry_time_unix


def check_and_clean_candle(df, timeframe, tick_time):
    """
    Check if the last candle is incomplete based on the timeframe, and drop it if necessary.
    """
    
    current_time = datetime.fromtimestamp(tick_time, tz=timezone.utc)

    if timeframe in ['M1', 'M3', 'M5', 'M15', 'M30', 'H1', 'H4', 'H12', 'D1', 'W1']:
        
        if timeframe == 'M1':
            current_minute = current_time.minute
            if df.iloc[-1]['datetime'].minute == current_minute:
                df = df.iloc[:-1]

        elif timeframe == 'M3':
            current_minute = current_time.minute
            # Checking if current minute aligns with a multiple of 3
            if current_minute % 3 == df.iloc[-1]['datetime'].minute % 3:
                df = df.iloc[:-1]

        elif timeframe == 'M5':
            current_minute = current_time.minute
            # Checking if current minute aligns with a multiple of 5
            if current_minute % 5 == df.iloc[-1]['datetime'].minute % 5:
                df = df.iloc[:-1]

        elif timeframe == 'M15':
            current_minute = current_time.minute
            # Checking if current minute aligns with a multiple of 15
            if current_minute % 15 == df.iloc[-1]['datetime'].minute % 15:
                df = df.iloc[:-1]

        elif timeframe == 'M30':
            current_minute = current_time.minute
            # Checking if current minute aligns with a multiple of 30
            if current_minute % 30 == df.iloc[-1]['datetime'].minute % 30:
                df = df.iloc[:-1]

        elif timeframe == 'H1':
            current_hour = current_time.hour
            if df.iloc[-1]['datetime'].hour == current_hour:
                df = df.iloc[:-1]
        
        elif timeframe == 'H4':
            current_hour = current_time.hour
            # Checking if current hour aligns with a multiple of 4
            if current_hour % 4 == df.iloc[-1]['datetime'].hour % 4:
                df = df.iloc[:-1]
        
        elif timeframe == 'H12':
            current_half_day = current_time.hour // 12
            if df.iloc[-1]['datetime'].hour // 12 == current_half_day:
                df = df.iloc[:-1]
        
        elif timeframe == 'D1':
            current_date = current_time.date()
            if df.iloc[-1]['datetime'].date() == current_date:
                df = df.iloc[:-1]
        
        elif timeframe == 'W1':
            current_week = current_time.isocalendar()[1]  # ISO week number
            if df.iloc[-1]['datetime'].isocalendar()[1] == current_week:
                df = df.iloc[:-1]

    return df

@handle_exceptions
def calculate_trade_parameters_s3_fvg(direction, high, low, latest_day_atr, gap_threshold):
    """
    Determine trade parameters based on direction.
    """
    if direction == 'Bullish':
        entry_type = 'Buy'
        entry_price = high
        stop_loss = low - gap_threshold * latest_day_atr
    elif direction == 'Bearish':
        entry_type = 'Sell'
        entry_price = low
        stop_loss = high + gap_threshold * latest_day_atr

    
    return entry_type, entry_price, stop_loss

@handle_exceptions
def is_ticker_daily_bias_nuetral(daily_bias_dict, symbol):
    return daily_bias_dict[symbol] == 'Neutral'

@handle_exceptions
def is_ticker_both_bias_is_neutral(bias_bias, sell_bias):
    NEUTRAL = ['Neutral', 'Pause', 'Opposite']
    return bias_bias in NEUTRAL and sell_bias in  NEUTRAL

@handle_exceptions
def adjust_sl_of_s7_depend_on_pips(entry_price, stop_loss, ticker, is_bearish):
    pips = calculate_pips(stop_loss, entry_price, ticker) if is_bearish else calculate_pips(entry_price, stop_loss, ticker)

    print_log(f"Pips are {pips}, pair: {ticker}")
    
    # Determine the correct pip value based on the ticker group
    if ticker in JPY_PAIRS:
        pip_value = PIP_VALUES["JPY_PAIRS"]
    elif ticker in INDICES:
        pip_value = PIP_VALUES["INDICES"]
    elif ticker in GOLD:
        pip_value = PIP_VALUES["GOLD"]
    elif ticker in OTHER_PAIRS:
        pip_value = PIP_VALUES["OTHER_PAIRS"]
    else:
        return stop_loss  # Return original stop_loss if ticker is unrecognized

    # Adjust stop loss if the calculated pips are less than 5
    if pips and pips < 5:
        print_log(f"Pips has been adjusted in stoploss , current stoploss is {stop_loss} and entry price is {entry_price}")
        stop_loss = entry_price + (pip_value * 5) if is_bearish else entry_price - (pip_value * 5)
        print_log(f"Adjusted stoploss is {stop_loss}")
    
    return stop_loss


@handle_exceptions
def calculate_move_of_s7(candle_1, latest_candle, is_bearish, is_bullish):
    """
    Calculate the move based on the type of candle (bearish or bullish).
    """
    move = None
    if is_bearish:
        move = abs(candle_1['high'] - latest_candle['close'])
    
    elif is_bullish:
        move = abs(candle_1['low'] - latest_candle['close'])
    
    return move  


@handle_exceptions
def check_fvg(candle_1, candle_3, is_bearish):
    if is_bearish:

        fvg_gap = candle_1['low'] - candle_3['high']
    else:
   
        fvg_gap = candle_3['low'] - candle_1['high']

    return fvg_gap > 0


@handle_exceptions
def aggressive_entry_for_s7(candle, is_bearish):
    """
    Handles aggressive entry setup.
    """
    if is_bearish:
        entry_price = min(candle['close'], candle['open']) 
        stop_loss = candle['high']  # Stop loss at the high of the candle
    else:
        entry_price = max(candle['close'], candle['open']) 
        stop_loss = candle['low'] 
    return entry_price, stop_loss


@handle_exceptions
def conservative_entry_for_s7(fvg_gap, candle_3, candle_2, is_bearish):
    """
    Handles conservative entry setup with FVG gap adjustment.
    """
    fvg_gap *= 0.25
    if is_bearish:
        entry_price = candle_3['high'] + fvg_gap 
        stop_loss = candle_2['high'] 
    else:
        entry_price = candle_3['low'] - fvg_gap 
        stop_loss = candle_2['low']  
    return entry_price, stop_loss


# def get_ema_daily_bias_test():
#     daily_bias_dict = {"EURUSD.sd": "Sell", "GBPUSD.sd": "Sell", "USDCAD.sd": "Buy", "USDJPY.sd": "Buy", "AUDUSD.sd": "Sell", "USDCHF.sd": "Buy", "NZDUSD.sd": "Sell", "GBPCHF.sd": "Nuetral", "CADCHF.sd": "Buy", "AUDCHF.sd": "Nuetral", "NZDCHF.sd": "Nuetral", "NZDJPY.sd": "Buy", "AUDJPY.sd": "Nuetral", "GBPJPY.sd": "Nuetral", "EURJPY.sd": "Nuetral", "EURCHF.sd": "Nuetral", "GBPCAD.sd": "Sell", "EURCAD.sd": "Sell", "EURGBP.sd": "Nuetral", "EURAUD.sd": "Nuetral", "EURNZD.sd": "Nuetral", "GBPAUD.sd": "Sell", "GBPNZD.sd": "Sell", "AUDCAD.sd": "Sell", "NZDCAD.sd": "Sell", "AUDNZD.sd": "Sell", "US500Roll": "Nuetral", "UT100Roll": "Nuetral", "US30Roll": "Nuetral", "DE40Roll": "Sell", "UK100Roll": "Sell", "XAUUSD.sd": "Sell", "XAGUSD.sd": "Nuetral", "USOILRoll": "Sell"}

#     return daily_bias_dict

@handle_exceptions
def get_ema_daily_bias():
    with open('E:/Ahmad/daily_bias.txt') as f: 
            data = f.read() 
    daily_bias_dict = ast.literal_eval(data)

    return daily_bias_dict

def is_valid_hour(start_hour, end_hour):
    current_hour = datetime.now(timezone.utc).hour
    return start_hour <= current_hour <= end_hour

@handle_exceptions
def get_bias_3(symbol):
    # Farid system file path
    # file_path = r"C:\Users\ADMIN\Desktop\Forex Projects\Refactor Code\refactor_strategies_deployment\daily_bias_3_5M\symbol_bias_data.json"

    # shared system file path
    file_path = r"F:\Farid\refactor_strategies_deployment\daily_bias_3_1H\symbol_bias_data.json"
    with open(file_path, "r") as json_file:
        symbol_data = json.load(json_file)
        
    for entry in symbol_data:
        if entry["Symbol"] == symbol:
            print(f"Buy Bias for {symbol}: {entry['buybias']}")
            print(f"Sell Bias for {symbol}: {entry['sellbias']}")
            return entry

@handle_exceptions
def get_bias_4(symbol):
    # Farid system file path
    # file_path = r"C:\Users\ADMIN\Desktop\Forex Projects\Refactor Code\refactor_strategies_deployment\daily_bias_3_5M\symbol_bias_data.json"

    # shared system file path
    file_path = r"F:\Farid\refactor_strategies_deployment\daily_bias_4_1H\symbol_bias_data.json"
    with open(file_path, "r") as json_file:
        symbol_data = json.load(json_file)
        
    for entry in symbol_data:
        if entry["Symbol"] == symbol:
            print(f"Buy Bias for {symbol}: {entry['buybias']}")
            print(f"Sell Bias for {symbol}: {entry['sellbias']}")
            return entry
        
@handle_exceptions
def get_bias_2(symbol):
    # Farid system file path
    # file_path = r"C:\Users\ADMIN\Desktop\Forex Projects\Refactor Code\refactor_strategies_deployment\daily_bias_3_5M\symbol_bias_data.json"

    # shared system file path
    file_path = r"F:\Farid\refactor_strategies_deployment\daily_bias_2_1H\symbol_bias_data.json"
    with open(file_path, "r") as json_file:
        symbol_data = json.load(json_file)
        
    for entry in symbol_data:
        if entry["Symbol"] == symbol:
            print(f"Buy Bias for {symbol}: {entry['buybias']}")
            print(f"Sell Bias for {symbol}: {entry['sellbias']}")
            return entry
        
@handle_exceptions
def process_trade_logic_s7(move, daily_atr, candle_1, candle_2, latest_candle, ticker, is_bearish):
    if is_bearish:
        order_type = 'Sell'
    else:
        order_type = 'Buy'

    entry_type, entry_price, stop_loss = None, 0, 0
    fvg_gap = check_fvg(candle_1, latest_candle, is_bearish)
    print_log(f"is fvg present {fvg_gap}")

    if move <= get_percentage_of_value(20, daily_atr):
        entry_type = 'Aggressive'
        print_log(f"Entry Type: {entry_type}")
        entry_price, stop_loss = aggressive_entry_for_s7(candle_2, is_bearish)
        
    elif get_percentage_of_value(20, daily_atr) < move <= get_percentage_of_value(50, daily_atr):
        entry_type = 'Conservative'
        print_log(f"Entry Type: {entry_type}")
        if fvg_gap:
            entry_price, stop_loss = conservative_entry_for_s7(fvg_gap, latest_candle, candle_2, is_bearish)
            stop_loss = adjust_sl_of_s7_depend_on_pips(entry_price, stop_loss, ticker, is_bearish)

    # Print or return the trade details based on your system's requirements
    print_log(f"Entry Type: {entry_type}, Entry Price: {entry_price}, Stop Loss: {stop_loss}, Order Type: {order_type}")
    return entry_type, entry_price, stop_loss, order_type


#sample_icot_signal_file_data = {"DE40Roll": ["Sell", "2024-12-23T16:30:05.580260", "Exit Sell"], "EURCAD.sd": ["Buy", "2024-12-31T01:15:01.315617", "Buy"], "EURUSD.sd": ["Sell", "2024-12-30T15:00:04.750587", "Exit Sell"], "GBPJPY.sd": ["Buy", "2024-12-31T10:45:02.443461", "Exit Buy"], "AUDJPY.sd": ["Buy", "2024-12-31T11:00:06.783683", "Exit Buy"], "EURJPY.sd": ["Buy", "2024-12-31T07:00:00.286452", "Buy"], "GBPUSD.sd": ["Sell", "2024-12-30T12:45:00.866427", "Sell"], "GBPCAD.sd": ["Buy", "2024-12-31T08:45:04.454985", "Exit Buy"], "USDCHF.sd": ["Buy", "2024-12-30T09:45:03.703821", "Exit Buy"], "XAGUSD.sd": ["Sell", "2024-11-14T06:44:59.970785", "Exit Sell"], "US30Roll": ["Buy", "2024-12-30T17:45:06.314052", "Exit Buy"], "AUDUSD.sd": ["Sell", "2024-12-27T11:00:05.551057", "Exit Sell"], "NZDUSD.sd": ["Buy", "2024-12-31T07:45:12.694405", "Buy"], "UT100Roll": ["Buy", "2024-12-30T16:30:07.399406", "Buy"], "US500Roll": ["Buy", "2024-12-30T16:30:07.355575", "Buy"], "AUDCHF.sd": ["Buy", "2024-11-13T18:00:10.604898", "Buy"], "XAUUSD.sd": ["Buy", "2024-12-30T18:14:59.688501", "Buy"], "USDJPY.sd": ["Sell", "2024-12-31T02:30:04.306731", "Exit Sell"], "UK100Roll": ["Sell", "2024-12-30T15:45:07.692080", "Exit Sell"], "CADCHF.sd": ["Sell", "2024-12-30T18:45:13.902205", "Sell"], "NZDCHF.sd": ["Buy", "2024-11-13T19:15:17.132249", "Buy"], "NZDJPY.sd": ["Buy", "2024-11-12T14:15:03.663791", "Exit Buy"], "USOILRoll": ["Buy", "2024-12-30T13:00:05.633457", "Exit Buy"], "EURCHF.sd": ["Sell", "2024-11-05T13:00:05.784285", "Exit Sell"], "GBPCHF.sd": ["Sell", "2024-12-30T15:00:08.179161", "Sell"], "USDCAD.sd": ["Buy", "2024-12-31T03:45:02.866098", "Buy"], "{{ticker}}": ["Buy", "2024-10-21T12:15:02.828849", "Exit Buy"], "AUDCAD.sd": ["Buy", "2024-12-31T01:30:13.118669", "Buy"], "GBPAUD.sd": ["Buy", "2024-12-30T10:00:03.003554", "Exit Buy"], "EURAUD.sd": ["Buy", "2024-12-30T10:30:02.882831", "Exit Buy"], "NZDCAD.sd": ["Buy", "2024-12-31T08:45:11.647998", "Exit Buy"]}

@handle_exceptions
def get_5M_ICOT_signal():
    with open('E:/Ahmad/5M_Signal.txt') as f: 
            data = f.read() 
    m5_icot_signal_dict = ast.literal_eval(data)
    return m5_icot_signal_dict

@handle_exceptions
def get_15M_ICOT_signal():
    with open('E:/Ahmad/15M_Signal.txt') as f: 
            data = f.read() 
    m15_icot_signal_dict = ast.literal_eval(data)
    return m15_icot_signal_dict

@handle_exceptions
def get_1M_JC_ICOT_signal():
    with open('E:/Ahmad/1M_JC_Signal.txt') as f: 
            data = f.read() 
    m1_jc_icot_signal_dict = ast.literal_eval(data)
    return m1_jc_icot_signal_dict

@handle_exceptions
def calculate_candle_move(candle_1, candle_2, latest_candle):
    high_max = max(candle_1['high'], candle_2['high'], latest_candle['high'])
    low_min = min(candle_1['low'], candle_2['low'], latest_candle['low'])
    candle_move = high_max - low_min
    return round(candle_move, 5)

@handle_exceptions
def is_time_within_window(timestamp_dt, current_time, window_size_minutes=None):

    timestamp_dt_aware = timestamp_dt.replace(tzinfo=timezone.utc)
    time_difference = abs(current_time - timestamp_dt_aware)
    return time_difference <= timedelta(minutes=window_size_minutes)

@handle_exceptions
def save_trade_data(csv_file, **kwargs):
    """
    Save the header and values of the given parameters to a CSV file using pandas.

    If the header is already available in the CSV, it will only append the new row of data.
    Otherwise, it will add the header and data.

    Parameters:
    - csv_file (str): Path to the CSV file.
    - kwargs: Key-value pairs representing headers and values.

    Returns:
    None
    """
    # Create a DataFrame from the provided parameters
    new_data = pd.DataFrame([kwargs])

    # Check if the file exists
    if os.path.exists(csv_file):
        # Load the existing file
        existing_data = pd.read_csv(csv_file)

        # Check if the headers match
        if list(existing_data.columns) == list(new_data.columns):
            # Append the new data to the file
            new_data.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            raise ValueError("The headers of the new data do not match the existing file.")
    else:
        # Save the new data with headers
        new_data.to_csv(csv_file, mode='w', header=True, index=False)

@handle_exceptions
def extract_order_data(order_result, data_dict, base_price, entry_type, break_even_sl):
    """
    Extracts specific data from the OrderSendResult object.
    """
    # Map order type and action for clarity
    order_action_map = {0: "Buy", 1: "Sell", 2: "Buy Limit", 3: "Sell Limit", 4: "Buy Stop", 5: "Sell Stop"}
    
    data_dict.update({
        "order_dt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Assuming order datetime is now
        "trade_execution": "Yes",
        "order_no": order_result.order,
        "order_type": entry_type,
        "order_action": order_action_map.get(order_result.request.type, "Unknown"),
        "base_price": base_price,
        "order_price": order_result.request.price,
        "stop_loss": order_result.request.sl,
        "tp": order_result.request.tp,
        "lot": order_result.request.volume,
        "stop_loss_break_even" : break_even_sl
    })
    
    return data_dict

  
@handle_exceptions
def round_value(value):
    return round(value, 5)

@handle_exceptions
def get_trade_filename(strategy_path):
    """
    Formats the trade file name from the strategy path
    """
    file_name = f'{strategy_path}_trades.csv'
    file_name = file_name.split('\\')[-1]
    return file_name


@handle_exceptions
def parse_icot_signal(m15_icot_signal, symbol):
    """
    Parse ICOT signal data for a given symbol and extract key values.
    """
    try:
        # Extract raw values from signal data
        timestamp_str = m15_icot_signal[symbol][1]
        timestamp_dt = parser.isoparse(timestamp_str)
        m5_signal = m15_icot_signal[symbol][0]
        entry_type = m15_icot_signal[symbol][2]
        
        return m5_signal, timestamp_dt, entry_type
        
    except (KeyError, IndexError) as e:
        raise Exception(f"Error parsing signal data for symbol {symbol}: {str(e)}")

@handle_exceptions
def validate_entry(entry_type, entry_price, buy_above_price, sell_below_price):
    if entry_type == 'Buy' and entry_price < buy_above_price:
        print_log(f'Entry type is {entry_type}, but entry_price: {entry_price} is below buy_above_price {buy_above_price}')
        return False
    if entry_type == 'Sell' and entry_price > sell_below_price:
        print_log(f'Entry type is {entry_type}, but entry_price: {entry_price} is above sell_below_price {sell_below_price}')
        return False
    return True

def calculate_one_pip(pair): 
    pip_value = None    
    if pair in JPY_PAIRS:
        pip_value =  PIP_VALUES["JPY_PAIRS"]
    elif pair in INDICES:
        pip_value = PIP_VALUES["INDICES"]
    elif pair in GOLD:
        pip_value = PIP_VALUES["GOLD"]
    elif pair in OTHER_PAIRS:
        pip_value = PIP_VALUES["OTHER_PAIRS"]

    print_log(f'Pip value is {pip_value}')
    return pip_value if pip_value else 0


def determine_ema_alignment(ema_20, ema_50, ema_100, is_buy_case=True):
    """
    Determines EMA alignment based on the given EMAs and trading direction.
    """
    if is_buy_case:
        return "Aligned" if ema_20 > ema_50 > ema_100 else \
               "Reverse-Aligned" if ema_20 < ema_50 < ema_100 else \
               "Unaligned"
    else:
        return "Aligned" if ema_20 < ema_50 < ema_100 else \
               "Reverse-Aligned" if ema_20 > ema_50 > ema_100 else \
               "Unaligned"


