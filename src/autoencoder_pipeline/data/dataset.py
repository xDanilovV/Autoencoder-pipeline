"""
PyTorch dataset classes for GC-IMS pipeline.
"""
import torch
import numpy as np
from torch.utils.data import Dataset


class TimeseriesDataset(Dataset):
    def __init__(self, timeseries: np.ndarray):
        self.timeseries = torch.FloatTensor(timeseries)

    def __len__(self):
        return len(self.timeseries)

    def __getitem__(self, idx):
        return self.timeseries[idx]


def extract_columns_as_timeseries(X: np.ndarray) -> np.ndarray:
    n_samples, M, N = X.shape
    # Take each column (RT slice)
    timeseries = X.transpose(0, 2, 1).reshape(-1, M)
    return timeseries


def extract_rows_as_timeseries(Z: np.ndarray) -> np.ndarray:
    n_samples, D, N = Z.shape
    timeseries = Z.reshape(-1, N)
    return timeseries


def undersample_flat_timeseries(timeseries: np.ndarray, prob: float = 0.25) -> np.ndarray:
    stds = np.std(timeseries, axis=1)
    median_std = np.median(stds)

    mask = np.ones(len(timeseries), dtype=bool)

    below = stds < median_std
    n_below = np.sum(below)

    # number to keep
    keep_n = int(n_below * prob)

    # which ones to drop
    drop_indices = np.random.choice(
        np.where(below)[0],
        size=n_below - keep_n,
        replace=False
    )

    mask[drop_indices] = False

    print(
        f"Undersampling: kept {mask.sum()}/{len(timeseries)} "
        f"({len(timeseries) - mask.sum()} removed)"
    )

    return timeseries[mask]