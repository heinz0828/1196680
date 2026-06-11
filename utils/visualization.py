import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def _save_figure(save_path, dpi=600):
    """Save each figure as high-resolution PNG plus PDF/SVG vector copies."""
    if not save_path:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    base, _ = os.path.splitext(save_path)
    plt.savefig(base + '.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(base + '.pdf', bbox_inches='tight')
    plt.savefig(base + '.svg', bbox_inches='tight')


def plot_predictions(y_true, y_pred, model_name='MDHGNN', save_path=None):
    """Plot true vs predicted prices."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true.flatten(), label='True', alpha=0.8, linewidth=1)
    ax.plot(y_pred.flatten(), label=f'{model_name} Predicted', alpha=0.8, linewidth=1)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Copper Price')
    ax.set_title(f'{model_name} - True vs Predicted Copper Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure(save_path)
    plt.close()


def plot_loss_curves(history, model_name='MDHGNN', save_path=None):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{model_name} - Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure(save_path)
    plt.close()


def plot_comparison_bar(results_dict, metric='RMSE', save_path=None):
    """Bar chart comparing models on a given metric."""
    models = list(results_dict.keys())
    values = [results_dict[m][metric] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#FF5722' if m == 'MDHGNN' else '#2196F3' for m in models]
    bars = ax.bar(models, values, color=colors)

    ax.set_ylabel(metric)
    ax.set_title(f'Model Comparison - {metric}')
    ax.grid(True, axis='y', alpha=0.3)

    for bar, val in zip(bars, values):
        y_pos = bar.get_height() if val >= 0 else bar.get_height() - 0.01
        va = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                f'{val:.4f}', ha='center', va=va, fontsize=9)

    plt.tight_layout()
    _save_figure(save_path)
    plt.close()


def plot_multi_horizon_comparison(multi_horizon_data, metric, model_names=None,
                                   save_path=None):
    """Grouped bar chart: x = horizon, grouped bars = models.

    Args:
        multi_horizon_data: {horizon: {model: {metric: value}}}
        metric: which metric to plot
        model_names: list of models to include (default: all)
    """
    horizons = sorted(multi_horizon_data.keys())
    if model_names is None:
        model_names = list(multi_horizon_data[horizons[0]].keys())

    n_models = len(model_names)
    x = np.arange(len(horizons))
    width = 0.8 / n_models

    model_colors = {
        'MDHGNN': '#FF5722', 'LSTM': '#2196F3', 'GRU': '#4CAF50',
        'MLP': '#FF9800', 'TCN': '#9C27B0', 'Transformer': '#607D8B',
        'Naive': '#BDBDBD',
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(model_names):
        vals = [multi_horizon_data[h].get(name, {}).get(metric, 0) for h in horizons]
        color = model_colors.get(name, '#999999')
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.85)

    ax.set_xlabel('Prediction Horizon')
    ax.set_ylabel(metric)
    ax.set_title(f'Multi-Horizon Comparison - {metric}')
    ax.set_xticks(x)
    ax.set_xticklabels([f'h={h}' for h in horizons])
    ax.legend(loc='best')
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    _save_figure(save_path)
    plt.close()


def plot_all_predictions(results_preds, y_true, save_path=None):
    """Overlay predictions from all models."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(y_true.flatten(), label='True', color='black', linewidth=1.5, alpha=0.9)

    colors = {'MDHGNN': '#FF5722', 'LSTM': '#2196F3', 'GRU': '#4CAF50', 'Transformer': '#9C27B0'}
    for name, preds in results_preds.items():
        color = colors.get(name, '#999999')
        ax.plot(preds.flatten(), label=name, alpha=0.7, linewidth=1, color=color)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Copper Price')
    ax.set_title('All Models - Copper Price Prediction Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure(save_path)
    plt.close()
