import pandas as pd
import numpy as np
from typing import Tuple, Dict


def preprocess_data(
    copper_df: pd.DataFrame,
    external_df: pd.DataFrame = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    dropna_thresh: int = 60,
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """合并外部数据, Z-score标准化, 时序分割 -> (features, prices, split_indices, norm_params)"""
    df = copper_df.copy()

    # Merge external data if available
    if external_df is not None and len(external_df) > 0:
        df['Date'] = pd.to_datetime(df['Date'])
        external_df['Date'] = pd.to_datetime(external_df['Date'])
        df = pd.merge(df, external_df, on='Date', how='left')
        ext_cols = [c for c in external_df.columns if c != 'Date']
        for col in ext_cols:
            df[col] = df[col].ffill().bfill()

    # Select feature columns (exclude Date)
    feature_cols = [c for c in df.columns if c != 'Date']

    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0)

    df[feature_cols] = df[feature_cols].ffill()
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    prices = df['Price'].values.copy().astype(np.float32)
    features = df[feature_cols].values.astype(np.float32)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(features)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    split_indices = {
        'train': (0, train_end),
        'val': (train_end, val_end),
        'test': (val_end, n),
    }

    # Z-score normalize features using training set statistics
    train_data = features[:train_end]
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std[std < 1e-8] = 1.0

    features = (features - mean) / std

    norm_params = {
        'mean': mean,
        'std': std,
        'feature_cols': feature_cols,
        'price_col_idx': feature_cols.index('Price'),
    }

    return features, prices, split_indices, norm_params
