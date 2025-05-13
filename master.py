import os
import json
import schedule
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import time

# __file__ = r'C:\\Users\\ADMIN\\Desktop\\Forex Projects\\Refactor Code\\refactor_strategies_deployment\\master.py'

# Set the root path for strategies
STRATEGIES_ROOT = Path(__file__).resolve().parent
from utils import reset_trades_for_new_day, print_log, handle_exceptions

from constant import ASSET_TRADES
# Specify the Python executable path explicitly
PYTHON_EXECUTABLE = "F:\\Farid\\Strategies-deployments\\ml_env\\Scripts\\python.exe"

# Path of Farid System
#PYTHON_EXECUTABLE = "C:\\Users\\ADMIN\\Desktop\\Forex Projects\\ml_env\\Scripts\\python.exe"
@handle_exceptions
def run_function(strategy_folder, strategy_path, function_name):
    """Executes a strategy function in a separate process."""
    print_log(f"Running {function_name} from {strategy_folder} at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    subprocess.Popen(
        [PYTHON_EXECUTABLE, str(strategy_path), function_name],
        cwd=str(strategy_path.parent),
        shell=True
    )

@handle_exceptions
def schedule_strategy_function(strategy_folder, function_name, config):
    """Schedules the strategy and reset functions based on config parameters."""

    strategy_path = STRATEGIES_ROOT / strategy_folder / "strategy.py"
    interval = config.get("interval")
    unit = config.get("unit", "minutes")
    time_window = config.get("time_window", {})
    reset_time_window = config.get("reset_trades", {})
    offset = config.get("offset", 0)
    exact_time = config.get("exact_time")  # Fixed time for running only once daily

    def run_reset_trades(strategy_folder):
        """Runs the reset trades logic within the context of a specific strategy folder."""
        strategy_path = STRATEGIES_ROOT / strategy_folder
        print_log(f"Resetting trades for strategy in {strategy_path} at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        reset_trades_for_new_day(ASSET_TRADES, strategy_path)

    def threaded_run():
        """Runs the main strategy based on interval or exact time and resets within a reset window if specified."""
        current_time_utc = datetime.now(timezone.utc)
        if current_time_utc.weekday() >= 5:  # Skip weekends
            # print_log(f"Skipping execution of {function_name} from {strategy_folder} (Weekend).")
            return

        current_hour = current_time_utc.hour
        current_minute = current_time_utc.minute

        # Handle exact time execution
        if exact_time:
            exact_hour = exact_time.get("hour")
            exact_minute = exact_time.get("minute", 0)
            if current_hour == exact_hour and current_minute == exact_minute:
                thread = threading.Thread(target=run_function, args=(strategy_folder, strategy_path, function_name))
                thread.start()
            else:
                print_log(f"Skipping {function_name} from {strategy_folder} as it only runs at {exact_hour}:{exact_minute}.")
            return

        # Check if current time is within the time window
        start_hour = time_window.get("start", 0)
        end_hour = time_window.get("end", 24)

        if not (start_hour <= current_hour < end_hour):
            print_log(f"Skipping {function_name} from {strategy_folder} due to time window.")
            return

        # Run the function
        thread = threading.Thread(target=run_function, args=(strategy_folder, strategy_path, function_name))
        thread.start()

    # Function to calculate schedule times
    def get_schedule_times(interval, unit, offset, time_window):
        start_hour = time_window.get("start", 0)
        end_hour = time_window.get("end", 24)
        times = []

        if unit != "minutes":
            print_log(f"Unsupported unit '{unit}' for {function_name} in {strategy_folder}")
            return times

        start_total_minutes = start_hour * 60
        end_total_minutes = end_hour * 60

        # Find the first total_minutes >= start_total_minutes such that (total_minutes - offset) % interval == 0
        first_total_minutes = ((start_total_minutes - offset + interval - 1) // interval) * interval + offset

        for total_minutes in range(first_total_minutes, end_total_minutes, interval):
            hour = total_minutes // 60
            minute = total_minutes % 60
            if hour >= start_hour and hour < end_hour and minute < 60:
                times.append(f"{hour:02}:{minute:02}")

        return times
    
    if reset_time_window:
        reset_start = reset_time_window.get("start", 0)
        reset_time = f"{reset_start:02}:00"
        schedule.every().day.at(reset_time).do(
            lambda: threading.Thread(target=run_reset_trades, args=(strategy_folder,)).start()
        )

    # Schedule based on interval or exact time
    if exact_time:
        schedule.every().day.at(f"{exact_time['hour']:02}:{exact_time.get('minute', 0):02}").do(threaded_run)
    elif unit == "seconds":
        schedule.every(interval).seconds.do(threaded_run)
    elif unit == "minutes":
        # For interval of 1 minute, schedule every minute within the time window
        if interval == 1:
            for hour in range(time_window.get("start", 0), time_window.get("end", 24)):
                for minute in range(60):
                    time_str = f"{hour:02}:{minute:02}"
                    schedule.every().day.at(time_str).do(threaded_run)
        else:
            times = get_schedule_times(interval, unit, offset, time_window)
            for time_str in times:
                schedule.every().day.at(time_str).do(threaded_run)
    else:
        print_log(f"Unsupported time unit '{unit}' for {function_name} in {strategy_folder}")

# Detect each strategy folder and schedule functions as per config.json
for folder_name in os.listdir(STRATEGIES_ROOT):
    folder_path = STRATEGIES_ROOT / folder_name
    config_path = folder_path / "config.json"
    
    if folder_path.is_dir() and (folder_path / "strategy.py").exists() and config_path.exists():
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
            for func in config.get("functions", []):
                func_name = func.get("name")
                schedule_strategy_function(folder_name, func_name, func)

while True:
    schedule.run_pending()
    time.sleep(1)  # Check every second for high precision