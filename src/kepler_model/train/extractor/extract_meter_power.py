
"""
Extracts meter power data from the file with 
name of "$(date -u "+%Y-%m-%dT%H_%M_%SZ").csv"

calculates the power from current and voltage data

transforms the time to timestamp

"""

from datetime import datetime, timezone
import pandas as pd
import argparse
from kepler_model.util import (    assure_path,
    load_csv,
    save_csv,
)


def meter_time_to_utc(meter_time, saved_time, saved_timestamp):

    meter_timestamp = meter_time + (saved_timestamp - saved_time)
    return meter_timestamp

def extract_meter_power(saved_path, saved_date):
    power_data_path = f"{saved_path}/{saved_date}.xlsx"
    power_data = pd.read_excel(power_data_path, index_col=0, skiprows=4)

    # Convert the meter index time to UTC timestamp
    print(f"Converting meter time to UTC for file: {power_data_path}...")
    meter_time = power_data.index[-1]
    saved_timestamp = int(datetime.strptime(saved_date, "%Y-%m-%dT%H_%M_%SZ").replace(tzinfo=timezone.utc).timestamp())
    power_data.index = power_data.index.map(lambda x: meter_time_to_utc(x, meter_time, saved_timestamp))
    power_data.index.name = "timestamp"
    # Calculate power from current and voltage
    print("Calculating platform power from voltage and current...")
    power_data["platform_power"] = power_data["  Voltage  (V)"] * power_data["  Current  (A)"]
    power_data.to_csv(power_data_path.replace('.xlsx', '-timestamp.csv'))
    return power_data

def parse_args():
    parser = argparse.ArgumentParser(description="Extract meter power data from CSV file.")
    parser.add_argument("--saved_path", type=str, required=True, help="Path to the saved CSV file.")
    parser.add_argument("--saved_date", type=str, required=True, help="Date of the saved CSV file in 'YYYY-MM-DDTHH_MM_SSZ' format.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    extracted_data = extract_meter_power(args.saved_path, args.saved_date)
    print("Extracted data:")
    print(extracted_data)
