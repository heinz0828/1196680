import numpy as np


def compute_metrics(y_true, y_pred):
    """价格指标: RMSE, MAE, MAPE, R2"""
    y_true, y_pred = y_true.flatten(), y_pred.flatten()

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    mask = np.abs(y_true) > 1e-8
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else float('inf')

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}


def compute_return_metrics(pred_returns, true_returns):
    """收益率指标: Ret_RMSE(bp), Ret_R2, 方向准确率"""
    pr, tr = pred_returns.flatten(), true_returns.flatten()

    ret_rmse = np.sqrt(np.mean((tr - pr) ** 2)) * 1e4
    ret_mae = np.mean(np.abs(tr - pr)) * 1e4

    ss_res = np.sum((tr - pr) ** 2)
    ss_tot = np.sum((tr - tr.mean()) ** 2)
    ret_r2 = 1 - ss_res / (ss_tot + 1e-10)

    true_dir = (tr > 0).astype(int)
    pred_dir = (pr > 0).astype(int)
    da = np.mean(true_dir == pred_dir) * 100

    up_mask, down_mask = tr > 0, tr < 0
    da_up = np.mean(pred_dir[up_mask] == 1) * 100 if up_mask.sum() > 0 else 0.0
    da_down = np.mean(pred_dir[down_mask] == 0) * 100 if down_mask.sum() > 0 else 0.0

    return {'Ret_RMSE': ret_rmse, 'Ret_MAE': ret_mae, 'Ret_R2': ret_r2,
            'DA': da, 'DA_up': da_up, 'DA_down': da_down}


def print_metrics(metrics, model_name=''):
    header = f"[{model_name}] " if model_name else ""
    parts = []
    if 'RMSE' in metrics: parts.append(f"RMSE={metrics['RMSE']:.4f}")
    if 'MAE' in metrics: parts.append(f"MAE={metrics['MAE']:.4f}")
    if 'MAPE' in metrics: parts.append(f"MAPE={metrics['MAPE']:.2f}%")
    if 'R2' in metrics: parts.append(f"R2={metrics['R2']:.4f}")
    print(f"{header}{' | '.join(parts)}")


def compute_strategy_return(pred_returns, true_returns, trading_periods_per_year=52):
    """策略收益: 按预测方向做多/做空"""
    pr, tr = pred_returns.flatten(), true_returns.flatten()
    position = np.sign(pr)
    strategy_ret = position * tr

    n = len(strategy_ret)
    ann_factor = trading_periods_per_year / max(n, 1)

    cum_pnl = np.cumsum(strategy_ret)
    running_max = np.maximum.accumulate(cum_pnl)
    max_dd = np.max(running_max - cum_pnl) * 100 if n > 0 else 0.0

    return {
        'CumReturn': np.sum(strategy_ret) * 100,
        'AnnReturn': np.sum(strategy_ret) * ann_factor * 100,
        'Sharpe': (np.mean(strategy_ret) / (np.std(strategy_ret) + 1e-10)) * np.sqrt(trading_periods_per_year),
        'MaxDrawdown': max_dd,
    }


def diebold_mariano_test(e1, e2, horizon=1):
    """DM检验: 比较两个模型预测精度是否有显著差异"""
    from scipy import stats
    e1, e2 = e1.flatten(), e2.flatten()
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    d_bar = np.mean(d)

    # Newey-West 方差估计
    gamma_0 = np.var(d, ddof=1)
    nw_var = gamma_0
    for k in range(1, horizon):
        gamma_k = np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / (n - 1)
        nw_var += 2 * (1 - k / horizon) * gamma_k

    dm_stat = d_bar / np.sqrt(max(nw_var, 1e-12) / n)
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
    return {'DM_stat': dm_stat, 'p_value': p_value}


def print_return_metrics(metrics, model_name=''):
    header = f"[{model_name}] " if model_name else ""
    print(f"{header}Ret_RMSE={metrics['Ret_RMSE']:.2f}bp | "
          f"Ret_R2={metrics['Ret_R2']:.4f} | "
          f"DA={metrics['DA']:.1f}% | "
          f"Sharpe={metrics.get('Sharpe', 0):.3f}")
