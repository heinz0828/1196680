import pandas as pd
import numpy as np
import os


def load_copper_data(csv_path: str) -> pd.DataFrame:
    """Load and clean copper futures CSV data."""
    df = pd.read_csv(csv_path)

    # Parse date
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

    # Parse Price, Open, High, Low (remove commas if any)
    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)

    # Parse Volume: strip K/M suffix
    def parse_volume(v):
        v = str(v).strip()
        if v == '-' or v == '':
            return np.nan
        if v.endswith('K'):
            return float(v[:-1]) * 1000
        elif v.endswith('M'):
            return float(v[:-1]) * 1e6
        elif v.endswith('B'):
            return float(v[:-1]) * 1e9
        return float(v)

    df['Volume'] = df['Vol.'].apply(parse_volume)

    # Parse Change %
    df['Change'] = df['Change %'].str.replace('%', '').astype(float) / 100.0

    # Drop original columns
    df = df.drop(columns=['Vol.', 'Change %'])

    # Sort by date ascending
    df = df.sort_values('Date').reset_index(drop=True)

    return df
