from dataclasses import dataclass
from pathlib import Path

import ims
import numpy as np

from config import config


@dataclass
class DatasetPreprocessor:
    method: str = "log"
    align_rip: bool = True
    cut_rip: bool = True
    rip_search_fraction: tuple[float, float] | None = (0.25, 0.55)
    rip_cut_half_width: int = 28
    rip_detection_max_rows: int = 1024
    roi_threshold: float = 0.02
    roi_background_percentile: float = 60
    roi_profile_smooth: int = 31
    roi_margin: int = 8
    roi_min_size: int = 32
    max_rows: int | None = None
    max_cols: int | None = None
    common_shape: tuple[int, int] | None = None
    rip_target_col: int | None = None
    row_slice: tuple[int, int] | None = None
    col_slice: tuple[int, int] | None = None
    compressed_shape: tuple[int, int] | None = None
    global_min: float | None = None
    global_max: float | None = None
    progress_interval: int = 25

    def fit(self, X: list[np.ndarray] | np.ndarray) -> "DatasetPreprocessor":
        matrices = [np.asarray(mat, dtype=np.float32) for mat in X]
        self._fit_common_shape(matrices)
        self._fit_rip_alignment(matrices)
        self.row_slice, self.col_slice = self._fit_roi(matrices)

        mins = []
        maxes = []
        total = len(matrices)
        print("  Measuring preprocessing scale...", flush=True)
        for idx, mat in enumerate(matrices, start=1):
            common = self._prepare_common(mat)
            window = self._apply_window(common)
            window = self._compress(window)
            transformed = self._scale(window)
            mins.append(float(transformed.min()))
            maxes.append(float(transformed.max()))
            self._print_progress("scale", idx, total)

        self.global_min = min(mins)
        self.global_max = max(maxes)
        return self

    def transform(self, X: list[np.ndarray] | np.ndarray) -> np.ndarray:
        if self.common_shape is None or self.row_slice is None or self.col_slice is None:
            raise ValueError("DatasetPreprocessor must be fitted before transform().")
        if self.global_min is None or self.global_max is None:
            raise ValueError("DatasetPreprocessor must be fitted before transform().")

        transformed_list = []
        denom = max(self.global_max - self.global_min, 1e-8)
        total = len(X)

        for idx, mat in enumerate(X, start=1):
            common = self._prepare_common(mat)
            window = self._apply_window(common)
            window = self._compress(window)
            transformed = self._scale(window)
            transformed = (transformed - self.global_min) / denom
            transformed_list.append(np.clip(transformed, 0.0, 1.0).astype(np.float32))
            self._print_progress("transform", idx, total)

        return np.stack(transformed_list, axis=0)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.global_min is None or self.global_max is None:
            raise ValueError("DatasetPreprocessor must be fitted before inverse_transform().")

        restored = X.astype(np.float32) * (self.global_max - self.global_min) + self.global_min
        if self.method == "log":
            restored = np.expm1(restored)
        return np.maximum(restored, 0.0).astype(np.float32)

    @property
    def output_shape(self) -> tuple[int, int]:
        if self.compressed_shape is not None:
            return self.compressed_shape
        if self.row_slice is None or self.col_slice is None:
            raise ValueError("DatasetPreprocessor has not been fitted.")
        return self.row_slice[1] - self.row_slice[0], self.col_slice[1] - self.col_slice[0]

    def _fit_common_shape(self, X: list[np.ndarray] | np.ndarray) -> None:
        matrices = [np.asarray(mat, dtype=np.float32) for mat in X]
        min_rows = min(mat.shape[0] for mat in matrices)
        min_cols = min(mat.shape[1] for mat in matrices)
        self.common_shape = (min_rows, min_cols)

    def _fit_rip_alignment(self, X: list[np.ndarray] | np.ndarray) -> None:
        if self.common_shape is None:
            raise ValueError("Common shape must be fitted before RIP alignment.")

        common_rows, common_cols = self.common_shape
        rip_cols = []
        total = len(X)
        print("  Detecting RIP columns...", flush=True)
        for idx, mat in enumerate(X, start=1):
            common = center_crop_or_pad(mat, common_rows, common_cols)
            rip_cols.append(self._detect_rip_column(common))
            self._print_progress("rip", idx, total)

        if not rip_cols:
            self.rip_target_col = common_cols // 2
            return

        self.rip_target_col = int(np.median(rip_cols))

    def _fit_roi(self, X: list[np.ndarray] | np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
        common_rows, common_cols = self.common_shape
        reference = np.zeros((common_rows, common_cols), dtype=np.float32)
        count = 0

        total = len(X)
        print("  Building aligned ROI reference...", flush=True)
        for idx, mat in enumerate(X, start=1):
            common = self._prepare_common(mat)
            reference += np.maximum(common, 0.0)
            count += 1
            self._print_progress("roi", idx, total)

        reference /= max(count, 1)
        reference = np.maximum(
            reference - np.percentile(reference, self.roi_background_percentile),
            0.0,
        )
        max_intensity = float(reference.max())

        if max_intensity <= 0:
            self._set_compressed_shape(common_rows, common_cols)
            return (0, common_rows), (0, common_cols)

        row_profile = smooth_profile(reference.sum(axis=1), self.roi_profile_smooth)
        col_profile = smooth_profile(reference.sum(axis=0), self.roi_profile_smooth)
        row_active = row_profile >= (self.roi_threshold * float(row_profile.max()))
        col_active = col_profile >= (self.roi_threshold * float(col_profile.max()))

        if row_active.sum() < self.roi_min_size or col_active.sum() < self.roi_min_size:
            self._set_compressed_shape(common_rows, common_cols)
            return (0, common_rows), (0, common_cols)

        row_start, row_end = self._bounds_with_margin(row_active, common_rows)
        col_start, col_end = self._bounds_with_margin(col_active, common_cols)
        self._set_compressed_shape(row_end - row_start, col_end - col_start)
        return (row_start, row_end), (col_start, col_end)

    def _print_progress(self, stage: str, idx: int, total: int) -> None:
        interval = max(int(self.progress_interval), 1)
        if idx == 1 or idx == total or idx % interval == 0:
            print(f"    {stage}: {idx}/{total}", flush=True)

    def _prepare_common(self, mat: np.ndarray) -> np.ndarray:
        if self.common_shape is None:
            raise ValueError("DatasetPreprocessor must be fitted before preparing spectra.")

        common = center_crop_or_pad(mat, self.common_shape[0], self.common_shape[1])
        rip_col = self._detect_rip_column(common)

        if self.align_rip and self.rip_target_col is not None:
            common = shift_columns(common, self.rip_target_col - rip_col)
            rip_col = self.rip_target_col

        if self.cut_rip:
            common = suppress_column_band(common, rip_col, self.rip_cut_half_width)

        return common

    def _detect_rip_column(self, mat: np.ndarray) -> int:
        cols = mat.shape[1]
        stride = max(mat.shape[0] // max(int(self.rip_detection_max_rows), 1), 1)
        sampled = mat[::stride]
        profile = np.mean(np.abs(sampled), axis=0)
        profile = smooth_profile(profile, self.roi_profile_smooth)

        start, end = 0, cols
        if self.rip_search_fraction is not None:
            frac_start, frac_end = self.rip_search_fraction
            start = int(np.clip(frac_start, 0.0, 1.0) * cols)
            end = int(np.clip(frac_end, 0.0, 1.0) * cols)
            if end <= start:
                start, end = 0, cols

        local = profile[start:end]
        if local.size == 0 or not np.isfinite(local).any():
            return cols // 2

        return start + int(np.nanargmax(local))

    def _bounds_with_margin(self, active: np.ndarray, limit: int) -> tuple[int, int]:
        indices = np.where(active)[0]
        start = max(int(indices[0]) - self.roi_margin, 0)
        end = min(int(indices[-1]) + self.roi_margin + 1, limit)

        if end - start < self.roi_min_size:
            midpoint = (start + end) // 2
            half = self.roi_min_size // 2
            start = max(midpoint - half, 0)
            end = min(start + self.roi_min_size, limit)
            start = max(end - self.roi_min_size, 0)

        return start, end

    def _apply_window(self, X: np.ndarray) -> np.ndarray:
        row_start, row_end = self.row_slice
        col_start, col_end = self.col_slice
        if X.ndim == 2:
            return X[row_start:row_end, col_start:col_end]
        return X[:, row_start:row_end, col_start:col_end]

    def _set_compressed_shape(self, rows: int, cols: int) -> None:
        target_rows = rows if self.max_rows is None else min(rows, self.max_rows)
        target_cols = cols if self.max_cols is None else min(cols, self.max_cols)
        self.compressed_shape = (target_rows, target_cols)

    def _compress(self, X: np.ndarray) -> np.ndarray:
        if self.compressed_shape is None:
            self._set_compressed_shape(X.shape[-2], X.shape[-1])
        target_rows, target_cols = self.compressed_shape

        if X.ndim == 2:
            return resize_mean(X, target_rows, target_cols)
        return np.stack([resize_mean(mat, target_rows, target_cols) for mat in X], axis=0)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X = np.maximum(X, 0.0)
        if self.method == "log":
            X = np.log1p(X)
        return X.astype(np.float32)


def collect_spectral_records(data_root, selected_classes=None):
    data_root = Path(data_root)
    selected = set(selected_classes) if selected_classes is not None else None
    records = []

    for file_path in sorted(data_root.rglob("*.mea")):
        label = infer_label(data_root, file_path)
        if selected is not None and label not in selected:
            continue
        records.append((file_path, label))

    return records


def infer_label(data_root: Path, file_path: Path) -> str:
    parts = file_path.relative_to(data_root).parts
    if len(parts) >= 3 and parts[0].startswith("GCIMS_"):
        return parts[1]
    return parts[0]


def discover_classes(data_root, selected_classes=None):
    labels = sorted({label for _, label in collect_spectral_records(data_root)})
    if selected_classes is None:
        return labels
    selected = set(selected_classes)
    return [label for label in labels if label in selected]


def load_spectral_data(data_root, selected_classes=None, verbose=True):
    records = collect_spectral_records(data_root, selected_classes=selected_classes)
    if not records:
        raise ValueError(f"No .mea files found under {data_root}.")

    X_list = []
    y_list = []

    class_counts = {}
    for file_path, label in records:
        class_counts[label] = class_counts.get(label, 0) + 1

        spectrum = ims.Spectrum.read_mea(str(file_path))
        spectrum = spectrum.sub_first_rows(config.RIP_CROP_ROWS)
        spectrum.savgol()
        spectrum.rip_scaling()

        X_list.append(spectrum.values.astype(np.float32))
        y_list.append(label)

    if verbose:
        print("\nLoaded raw vendor-preprocessed spectra:")
        for label in sorted(class_counts):
            print(f"  {label}: {class_counts[label]} files")
        shapes = np.array([mat.shape for mat in X_list])
        print(
            f"  Shape range: rows {shapes[:, 0].min()}-{shapes[:, 0].max()}, "
            f"cols {shapes[:, 1].min()}-{shapes[:, 1].max()}"
        )
        print("  Cropping/window fitting will happen after the train/validation split.")

    return X_list, np.array(y_list, dtype=object)


def build_preprocessor(X_train: list[np.ndarray] | np.ndarray, method: str = "log") -> DatasetPreprocessor:
    return DatasetPreprocessor(
        method=method,
        align_rip=config.ALIGN_RIP,
        cut_rip=config.CUT_RIP,
        rip_search_fraction=config.RIP_SEARCH_FRACTION,
        rip_cut_half_width=config.RIP_CUT_HALF_WIDTH,
        rip_detection_max_rows=config.RIP_DETECTION_MAX_ROWS,
        roi_threshold=config.ROI_INTENSITY_THRESHOLD,
        roi_background_percentile=config.ROI_BACKGROUND_PERCENTILE,
        roi_profile_smooth=config.ROI_PROFILE_SMOOTH,
        roi_margin=config.ROI_MARGIN,
        roi_min_size=config.ROI_MIN_SIZE,
        max_rows=config.MAX_MODEL_ROWS,
        max_cols=config.MAX_MODEL_COLS,
        progress_interval=config.PREPROCESS_PROGRESS_INTERVAL,
    ).fit(X_train)


def center_crop_or_pad(mat, target_rows, target_cols):
    mat = np.asarray(mat, dtype=np.float32)
    rows, cols = mat.shape
    if rows == target_rows and cols == target_cols:
        return mat.astype(np.float32, copy=False)

    row_start = max((rows - target_rows) // 2, 0)
    col_start = max((cols - target_cols) // 2, 0)
    cropped = mat[row_start:row_start + min(rows, target_rows), col_start:col_start + min(cols, target_cols)]

    out = np.zeros((target_rows, target_cols), dtype=np.float32)
    out_row_start = max((target_rows - cropped.shape[0]) // 2, 0)
    out_col_start = max((target_cols - cropped.shape[1]) // 2, 0)
    out[
        out_row_start:out_row_start + cropped.shape[0],
        out_col_start:out_col_start + cropped.shape[1],
    ] = cropped
    return out


def smooth_profile(profile, window):
    profile = np.asarray(profile, dtype=np.float32)
    window = int(window)
    if window <= 1 or profile.size < 3:
        return profile

    window = min(window, profile.size)
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return profile

    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(profile, kernel, mode="same").astype(np.float32)


def shift_columns(mat, shift):
    mat = np.asarray(mat, dtype=np.float32)
    shift = int(shift)
    if shift == 0:
        return mat.astype(np.float32, copy=True)

    out = np.zeros_like(mat, dtype=np.float32)
    if abs(shift) >= mat.shape[1]:
        return out

    if shift > 0:
        out[:, shift:] = mat[:, :-shift]
    else:
        out[:, :shift] = mat[:, -shift:]
    return out


def suppress_column_band(mat, center_col, half_width):
    mat = np.asarray(mat, dtype=np.float32)
    out = mat.astype(np.float32, copy=True)
    cols = out.shape[1]
    center_col = int(np.clip(center_col, 0, cols - 1))
    half_width = max(int(half_width), 0)
    start = max(center_col - half_width, 0)
    end = min(center_col + half_width + 1, cols)

    if start >= end:
        return out

    if start == 0 and end >= cols:
        out[:, :] = 0.0
        return out

    if start == 0:
        fill = out[:, end:end + 1] if end < cols else np.zeros((out.shape[0], 1), dtype=np.float32)
        out[:, start:end] = fill
        return out

    if end >= cols:
        out[:, start:end] = out[:, start - 1:start]
        return out

    left = out[:, start - 1:start]
    right = out[:, end:end + 1]
    alpha = np.linspace(0.0, 1.0, end - start + 2, dtype=np.float32)[1:-1]
    out[:, start:end] = left * (1.0 - alpha) + right * alpha
    return out


def resize_mean(mat, target_rows, target_cols):
    mat = np.asarray(mat, dtype=np.float32)
    rows, cols = mat.shape

    if rows == target_rows and cols == target_cols:
        return mat.astype(np.float32, copy=False)

    if rows < target_rows or cols < target_cols:
        return center_crop_or_pad(mat, target_rows, target_cols)

    row_edges = np.linspace(0, rows, target_rows + 1, dtype=np.int64)
    row_reduced = np.empty((target_rows, cols), dtype=np.float32)
    for i in range(target_rows):
        row_start, row_end = row_edges[i], row_edges[i + 1]
        row_reduced[i] = mat[row_start:row_end].mean(axis=0)

    col_edges = np.linspace(0, cols, target_cols + 1, dtype=np.int64)
    out = np.empty((target_rows, target_cols), dtype=np.float32)
    for j in range(target_cols):
        col_start, col_end = col_edges[j], col_edges[j + 1]
        out[:, j] = row_reduced[:, col_start:col_end].mean(axis=1)

    return out


def match_synthetic_to_real_distribution(X_synth, X_real, method="statistics"):
    if method == "statistics":
        synth_mean = X_synth.mean()
        synth_std = X_synth.std()

        real_mean = X_real.mean()
        real_std = X_real.std()

        X_synth_adjusted = (X_synth - synth_mean) / (synth_std + 1e-8)
        X_synth_adjusted = X_synth_adjusted * real_std + real_mean

        real_min = X_real.min()
        real_max = X_real.max()
        X_synth_adjusted = np.clip(X_synth_adjusted, real_min, real_max)

        return X_synth_adjusted.astype(np.float32)

    if method == "histogram":
        from scipy import interpolate

        real_flat = X_real.flatten()
        synth_flat = X_synth.flatten()

        real_values, real_counts = np.unique(real_flat, return_counts=True)
        real_cdf = np.cumsum(real_counts).astype(float)
        real_cdf /= real_cdf[-1]

        synth_values, synth_counts = np.unique(synth_flat, return_counts=True)
        synth_cdf = np.cumsum(synth_counts).astype(float)
        synth_cdf /= synth_cdf[-1]

        interp_func = interpolate.interp1d(
            synth_cdf,
            synth_values,
            bounds_error=False,
            fill_value=(synth_values[0], synth_values[-1]),
        )

        real_quantiles = np.interp(synth_flat, real_values, real_cdf)
        matched_values = interp_func(real_quantiles)

        return matched_values.reshape(X_synth.shape).astype(np.float32)

    return X_synth.astype(np.float32)
