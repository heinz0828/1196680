import pandas as pd
import numpy as np
import os


# Local CSV file mapping
EXTERNAL_FILES = {
    'WTI': 'Crude Oil WTI Futures Historical Data.csv',
    'Gold': 'Gold Futures Historical Data.csv',
    'Silver': 'Silver Futures Historical Data.csv',
    'USD': 'US Dollar Index Historical Data.csv',
}


def _parse_csv(filepath: str) -> pd.DataFrame:
    """Parse an Investing.com-style CSV file."""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

    # Parse Price (may contain commas)
    df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)

    # Sort ascending
    df = df.sort_values('Date').reset_index(drop=True)
    return df[['Date', 'Price']]


def download_external_data(start_date: str, end_date: str,
                           cache_path: str = None) -> pd.DataFrame:
    """
    Load external financial data from local CSV files.

    Returns DataFrame with columns:
    Date, WTI_Close, WTI_Return, Gold_Close, Gold_Return,
    Silver_Close, Silver_Return, USD_Close, USD_Return
    """
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw')

    all_dfs = []
    for name, filename in EXTERNAL_FILES.items():
        filepath = os.path.join(raw_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping {name}")
            continue

        df = _parse_csv(filepath)
        df = df.rename(columns={'Price': f'{name}_Close'})

        # Compute daily return
        df[f'{name}_Return'] = df[f'{name}_Close'].pct_change().fillna(0)

        all_dfs.append(df)

    if not all_dfs:
        print("Warning: No external data files found.")
        return pd.DataFrame(columns=['Date'])

    # Merge all on Date
    result = all_dfs[0]
    for df in all_dfs[1:]:
        result = pd.merge(result, df, on='Date', how='outer')

    # Sort and filter date range
    result = result.sort_values('Date').reset_index(drop=True)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    result = result[(result['Date'] >= start) & (result['Date'] <= end)]

    # Forward fill then backward fill
    for col in result.columns:
        if col != 'Date':
            result[col] = result[col].ffill().bfill()

    result = result.reset_index(drop=True)
    print(f"External data loaded: {len(result)} rows, "
          f"{result['Date'].min().date()} to {result['Date'].max().date()}")

    return result
