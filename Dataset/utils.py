from typing import Any, Dict

import math
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal.windows import triang
from tabulate import tabulate


def resolve_binning_params(max_age, bin_step=None, num_bins=None):
    if max_age <= 0:
        raise ValueError("max_age must be greater than 0")

    if bin_step is not None:
        if bin_step <= 0:
            raise ValueError("bin_step must be greater than 0")
        num_bins = math.ceil(max_age / bin_step)

    elif num_bins is not None:
        if num_bins <= 0:
            raise ValueError("num_bins must be greater than 0")
        bin_step = max_age / num_bins

    else:
        bin_step = 1
        num_bins = max_age

    return {"max_age": max_age, "bin_step": bin_step, "num_bins": num_bins}

def prepare_ages(df, label_col='label'):
    ages = df[label_col].values.astype(np.float32)
    return ages

def prepare_weights(df, max_age=121, label_col='label',
                    method='none', max_norm=None,
                    lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2, lds_norm='sum'):
    value_dict = {x: 0 for x in range(max_age)}
    labels = df[label_col].values
    for label in labels:
        value_dict[min(max_age-1, int(label))] += 1

    if method == 'sqrt_inv':
        value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
    elif method == 'inverse':
        value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}
    else:
        raise ValueError(f"Unknown re-weighting method: {method}")
    num_per_label = [value_dict[min(max_age-1, int(label))] for label in labels]

    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma, lds_norm)
        smoothed_value = convolve1d(
            np.asarray([v for _, v in value_dict.items()]),
            weights=lds_kernel_window,
            mode='constant'
        )
        num_per_label = [smoothed_value[min(max_age - 1, int(label))] for label in labels]
    weights = np.array([np.float32(1.0 / x) for x in num_per_label])

    scaling = len(weights) / np.sum(weights)
    weights = scaling * weights
    if max_norm is not None:
        weights = np.clip(weights, a_min=None, a_max=max_norm)
    return weights

def get_lds_kernel_window(kernel, ks, sigma, norm):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma)
    elif kernel == 'triang':
        kernel_window = triang(ks)
    elif kernel == 'laplace':
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1)))
    else:
        raise NotImplementedError(f"Unknown LDS kernel: {kernel}")
    if norm == 'max':
        kernel_window = kernel_window / max(kernel_window)
    elif norm == 'sum':
        kernel_window = kernel_window / sum(kernel_window)
    else:
        raise NotImplementedError(f"Unknown normalization method: {norm}")
    return kernel_window

def log_weight_stats(logger, weights, labels, max_age, many_shot_thr=100, low_shot_thr=20, prefix="train"):
    w = np.asarray(weights, dtype=np.float64)
    y = np.asarray(labels).astype(np.int64)
    y = np.clip(y, 0, max_age - 1)

    counts = np.bincount(y, minlength=max_age)
    freq = counts[y]

    m_many = freq >= many_shot_thr
    m_few  = freq <= low_shot_thr
    m_med  = ~(m_many | m_few)

    groups = {
        "All":  np.ones_like(m_many, dtype=bool),
        "Many": m_many,
        "Med.": m_med,
        "Few":  m_few,
    }
    rows = ['All', 'Many', 'Med.', 'Few']
    items = ['n', 'mean', 'std', 'min', 'p50', 'p90', 'p95', 'p99', 'max', 'Neff', 'Neff/N']

    table: Dict[str, Dict[str, Any]] = {r: {} for r in rows}
    def _summ(name, mask):
        ww = w[mask]
        if ww.size == 0:
            logger.info(f"[weights/{prefix}/{name}] n=0 (skip)")
            table[name] = {}
            return
        q = np.quantile(ww, [0, 0.5, 0.9, 0.95, 0.99, 1.0])
        neff = (ww.sum() ** 2) / (np.square(ww).sum() + 1e-12)

        table[name] = {
            'n': f"{ww.size}",
            'mean': f"{ww.mean():.4f}",
            'std': f"{ww.std():.4f}",
            'min': f"{q[0]:.4f}",
            'p50': f"{q[1]:.4f}",
            'p90': f"{q[2]:.4f}",
            'p95': f"{q[3]:.4f}",
            'p99': f"{q[4]:.4f}",
            'max': f"{q[5]:.4f}",
            'Neff': f"{neff:.0f}",
            'Neff/N': f"{neff/ww.size:.4f}",
        }

    for k, m in groups.items():
        _summ(k, m)

    all_cols = sorted(set().union(*(table[r].keys() for r in rows)))
    df = pd.DataFrame(index=rows, columns=all_cols)[items]
    for r in rows:
        for c in all_cols:
            df.at[r, c] = table[r].get(c, "-")
    df.index.name = f"Weights"

    table_str = tabulate(
        df,
        headers='keys',
        tablefmt='fancy_grid',
        numalign="right",
        stralign="left",
        floatfmt=".4f",
        showindex=True,
    )
    logger.info("\n" + table_str)
