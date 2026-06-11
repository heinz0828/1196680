import pandas as pd
import numpy as np


def add_technical_indicators(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """计算技术指标 (MA, RSI, MACD, BB, ATR, ROC, Williams%R, CCI, OBV)"""
    if params is None:
        params = {
            'ma_periods': [5, 10, 20, 60],
            'rsi_period': 14,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'bb_period': 20,
            'atr_period': 14,
            'roc_period': 10,
            'williams_period': 14,
            'cci_period': 20,
        }

    df = df.copy()
    price = df['Price']
    high = df['High']
    low = df['Low']
    close = price
    volume = df['Volume']

    # Moving Averages (always 4 columns: MA_1 .. MA_4)
    for i, w in enumerate(params['ma_periods'], 1):
        df[f'MA_{i}'] = close.rolling(window=w).mean()

    # RSI
    rsi_p = params['rsi_period']
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=rsi_p).mean()
    avg_loss = loss.rolling(window=rsi_p).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = close.ewm(span=params['macd_fast'], adjust=False).mean()
    ema_slow = close.ewm(span=params['macd_slow'], adjust=False).mean()
    df['MACD_line'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD_line'].ewm(span=params['macd_signal'], adjust=False).mean()
    df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']

    # Bollinger Bands
    bb_p = params['bb_period']
    sma_bb = close.rolling(window=bb_p).mean()
    std_bb = close.rolling(window=bb_p).std()
    df['BB_upper'] = sma_bb + 2 * std_bb
    df['BB_middle'] = sma_bb
    df['BB_lower'] = sma_bb - 2 * std_bb

    # ATR
    atr_p = params['atr_period']
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': (high - close.shift(1)).abs(),
        'lc': (low - close.shift(1)).abs()
    }).max(axis=1)
    df['ATR'] = tr.rolling(window=atr_p).mean()

    # Rate of Change
    df['ROC'] = close.pct_change(periods=params['roc_period'])

    # Williams %R
    w_p = params['williams_period']
    highest_high = high.rolling(window=w_p).max()
    lowest_low = low.rolling(window=w_p).min()
    df['Williams_R'] = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100

    # CCI
    cci_p = params['cci_period']
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=cci_p).mean()
    mad_tp = tp.rolling(window=cci_p).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df['CCI'] = (tp - sma_tp) / (0.015 * mad_tp + 1e-10)

    # OBV (no period parameter)
    vol_filled = volume.fillna(0)
    obv = np.zeros(len(df))
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv[i] = obv[i - 1] + vol_filled.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv[i] = obv[i - 1] - vol_filled.iloc[i]
        else:
            obv[i] = obv[i - 1]
    df['OBV'] = obv

    return df
