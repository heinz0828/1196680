import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple


class CopperFuturesDataset(Dataset):
    """Sliding-window dataset with return targets relative to P_{t-1}.

    For horizon h, the target vector is:
    [(P_t - P_{t-1}) / P_{t-1}, ...,
     (P_{t+h-1} - P_{t-1}) / P_{t-1}].
    """

    def __init__(self, features: np.ndarray, prices: np.ndarray,
                 start_idx: int, end_idx: int,
                 window_size: int = 60, horizon: int = 1):
        self.features = features
        self.prices = prices
        self.window_size = window_size
        self.horizon = horizon

        # 有效索引范围
        self.valid_start = max(start_idx, window_size)
        self.valid_end = min(end_idx, len(prices) - horizon)

        self.indices = list(range(self.valid_start, self.valid_end))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        x = self.features[t - self.window_size: t]  # (W, F)

        base_price = self.prices[t - 1]
        target_prices = self.prices[t: t + self.horizon]
        returns = (target_prices - base_price) / (base_price + 1e-8)

        return torch.FloatTensor(x), torch.FloatTensor(returns), torch.FloatTensor([base_price])


def create_dataloaders(
    features: np.ndarray,
    prices: np.ndarray,
    split_indices: Dict,
    window_size: int = 60,
    horizon: int = 1,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建 train/val/test DataLoader"""
    loaders = {}
    for split_name in ['train', 'val', 'test']:
        start, end = split_indices[split_name]
        ds = CopperFuturesDataset(features, prices, start, end,
                                  window_size, horizon)
        shuffle = (split_name == 'train')
        loaders[split_name] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=False
        )

    return loaders['train'], loaders['val'], loaders['test']
