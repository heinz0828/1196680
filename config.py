from dataclasses import dataclass, field
from typing import Dict, List


# Indicator period presets per frequency
DAILY_INDICATOR_PARAMS: Dict[str, object] = {
    'ma_periods': [5, 10, 20, 60],
    'rsi_period': 14,
    'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
    'bb_period': 20,
    'atr_period': 14,
    'roc_period': 10,
    'williams_period': 14,
    'cci_period': 20,
}

WEEKLY_INDICATOR_PARAMS: Dict[str, object] = {
    'ma_periods': [4, 8, 13, 26],
    'rsi_period': 7,
    'macd_fast': 6, 'macd_slow': 13, 'macd_signal': 5,
    'bb_period': 10,
    'atr_period': 7,
    'roc_period': 4,
    'williams_period': 7,
    'cci_period': 10,
}


@dataclass
class Config:
    # Data
    csv_path: str = 'data/raw/Copper Futures Historical Data.csv'
    external_cache: str = 'data/raw/external_data.csv'
    frequency: str = 'weekly'
    window_size: int = 20
    horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    trading_periods_per_year: int = 52

    # Model
    d_model: int = 64
    n_hgnn_layers: int = 2
    k_neigs: List[int] = field(default_factory=lambda: [3, 5, 8])
    n_adaptive_edges: int = 12
    gru_hidden: int = 64
    dropout: float = 0.25

    # Training
    batch_size: int = 32
    max_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 40
    grad_clip: float = 1.0
    seed: int = 42
    print_freq: int = 10

    # Paths
    checkpoint_dir: str = 'results/checkpoints'
    figure_dir: str = 'results/figures'
    table_dir: str = 'results/tables'

    @property
    def indicator_params(self) -> Dict[str, object]:
        if self.frequency == 'weekly':
            return WEEKLY_INDICATOR_PARAMS
        return DAILY_INDICATOR_PARAMS

    def apply_frequency(self, freq: str):
        """Apply frequency-specific defaults."""
        self.frequency = freq
        if freq == 'daily':
            self.window_size = 60
            self.batch_size = 64
            self.patience = 30
            self.trading_periods_per_year = 252
        else:  # weekly
            self.window_size = 12
            self.batch_size = 32
            self.d_model = 64
            self.gru_hidden = 64
            self.n_hgnn_layers = 2
            self.n_adaptive_edges = 8
            self.dropout = 0.20
            self.lr = 1e-3
            self.weight_decay = 1e-4
            self.max_epochs = 200
            self.patience = 40
            self.trading_periods_per_year = 52
