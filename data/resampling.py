"""Resample daily OHLCV data to weekly frequency."""
import pandas as pd


def resample_copper_ohlcv(df: pd.DataFrame, freq: str = 'W-FRI') -> pd.DataFrame:
    """Resample daily copper OHLCV to weekly bars.

    Args:
        df: Daily DataFrame with columns [Date, Price, Open, High, Low, Volume, Change]
        freq: Pandas offset alias (default W-FRI = week ending Friday)
    Returns:
        Weekly DataFrame with same column schema.
    """
    df = df.copy()
    df = df.set_index('Date')

    weekly = pd.DataFrame()
    weekly['Open'] = df['Open'].resample(freq).first()
    weekly['High'] = df['High'].resample(freq).max()
    weekly['Low'] = df['Low'].resample(freq).min()
    weekly['Price'] = df['Price'].resample(freq).last()
    weekly['Volume'] = df['Volume'].resample(freq).sum()

    # Recompute Change from weekly closes (not sum of daily changes)
    weekly['Change'] = weekly['Price'].pct_change()

    weekly = weekly.dropna(subset=['Price'])
    weekly = weekly.reset_index()
    return weekly


def resample_external_prices(df: pd.DataFrame, freq: str = 'W-FRI') -> pd.DataFrame:
    """Resample daily external data to weekly.

    For Close columns: take last (weekly close).
    For Return columns: recompute from weekly closes.

    Args:
        df: Daily external DataFrame with Date + *_Close + *_Return columns
        freq: Pandas offset alias
    Returns:
        Weekly external DataFrame with same column schema.
    """
    df = df.copy()
    df = df.set_index('Date')

    close_cols = [c for c in df.columns if c.endswith('_Close')]
    return_cols = [c for c in df.columns if c.endswith('_Return')]

    weekly = pd.DataFrame()
    for col in close_cols:
        weekly[col] = df[col].resample(freq).last()

    # Recompute returns from weekly closes
    for col in return_cols:
        base = col.replace('_Return', '_Close')
        if base in weekly.columns:
            weekly[col] = weekly[base].pct_change(fill_method=None).fillna(0)

    weekly = weekly.dropna(subset=close_cols)
    weekly = weekly.reset_index()
    return weekly
