from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import os
from scipy.ndimage import shift as scipy_shift, map_coordinates
from scipy.interpolate import interp1d
import ims


class Config:
    ROOT = Path(r"C:\Users\user\PycharmProjects\PythonProject\data")
    OUT = Path(r"C:\Users\user\PycharmProjects\PythonProject\data\output_aug")
    DATA_ROOT = OUT

    # Noise augmentation parameters (tuned to be more subtle)
    SNR_DB = 30.0
    JITTER_SIGMA = 0.012
    RT_SHIFT = (-1, 1)
    DT_SHIFT = (-1, 1)

    # Geometric parameters
    CROP_FRACTION = 0.04
    WARP_SIGMA = 0.12
    WARP_KNOTS = 4

    # Intensity parameters
    SCALE_RANGE = (0.95, 1.05)
    MIXUP_ALPHA = 0.3

    # Probabilities
    PROB_NOISE = 0.35
    PROB_JITTER = 0.35
    PROB_SHIFT = 0.25
    PROB_CROP = 0.25
    PROB_WARP = 0.15
    PROB_SCALE = 0.5

    # Evaluation parameters
    N_COMPARE = 20
    BINNING = 4
    MAX_SAMPLES_PER_FILE = 500
    MAX_SAMPLES_TOTAL = 20000
    TEST_SIZE = 0.25
    RANDOM_STATE = 42

class NoiseAugmentation:

    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)

    def add_gaussian_noise(self, arr: np.ndarray, snr_db: float = 28.0,
                           clip_zero: bool = True) -> np.ndarray:
        arr = np.clip(arr, 0, None) if clip_zero else arr
        power = float(np.mean(arr ** 2)) if arr.size else 0.0
        snr = 10 ** (snr_db / 10)
        sigma = np.sqrt(power / snr) if snr > 0 else 0.0
        out = arr + self.rng.normal(0.0, sigma, arr.shape)
        return np.clip(out, 0, None) if clip_zero else out

    def multiplicative_jitter(self, arr: np.ndarray, sigma: float = 0.015,
                              clip_zero: bool = True) -> np.ndarray:
        out = arr * self.rng.normal(1.0, sigma, size=arr.shape)
        return np.clip(out, 0, None) if clip_zero else out

    def small_axis_shift_fixed(self, arr: np.ndarray, rt_px: int = 0,
                               dt_px: int = 0) -> np.ndarray:
        if rt_px == 0 and dt_px == 0:
            return arr
        # Use scipy shift with proper edge handling
        return scipy_shift(arr, (rt_px, dt_px), mode='nearest', order=1)


class GeometricAugmentation:

    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)

    def random_crop_and_pad(self, arr: np.ndarray, crop_fraction: float = 0.05,
                            pad_mode: str = 'edge') -> np.ndarray:
        if crop_fraction <= 0 or crop_fraction >= 0.5:
            return arr

        h, w = arr.shape

        # Calculate crop amounts (random within range)
        max_crop_h = int(h * crop_fraction)
        max_crop_w = int(w * crop_fraction)

        crop_top = self.rng.integers(0, max_crop_h + 1)
        crop_bottom = self.rng.integers(0, max_crop_h + 1)
        crop_left = self.rng.integers(0, max_crop_w + 1)
        crop_right = self.rng.integers(0, max_crop_w + 1)

        # Crop
        cropped = arr[crop_top:h - crop_bottom, crop_left:w - crop_right]

        # Pad back to original size
        pad_width = ((crop_top, crop_bottom), (crop_left, crop_right))
        padded = np.pad(cropped, pad_width, mode=pad_mode)

        return padded

    def time_warping(self, arr: np.ndarray, sigma: float = 0.2,
                     knot_points: int = 4) -> np.ndarray:
        if sigma <= 0:
            return arr

        h, w = arr.shape

        # Create control point grid
        knot_h = np.linspace(0, h - 1, knot_points)
        knot_w = np.linspace(0, w - 1, knot_points)

        # Generate random displacements for control points
        # Scale by sigma and grid spacing
        disp_scale_h = sigma * (h / knot_points)
        disp_scale_w = sigma * (w / knot_points)

        displacements_h = self.rng.normal(0, disp_scale_h, knot_points)
        displacements_w = self.rng.normal(0, disp_scale_w, knot_points)

        # Interpolate displacements to full grid
        interp_h = interp1d(knot_h, displacements_h, kind='cubic',
                            fill_value='extrapolate')
        interp_w = interp1d(knot_w, displacements_w, kind='cubic',
                            fill_value='extrapolate')

        # Create dense coordinate grids
        coords_h = np.arange(h, dtype=np.float32)
        coords_w = np.arange(w, dtype=np.float32)

        # Apply warping
        warped_h = coords_h + interp_h(coords_h)
        warped_w = coords_w + interp_w(coords_w)

        # Clip to valid range
        warped_h = np.clip(warped_h, 0, h - 1)
        warped_w = np.clip(warped_w, 0, w - 1)

        # Create meshgrid for map_coordinates
        grid_h, grid_w = np.meshgrid(warped_h, warped_w, indexing='ij')

        # Apply coordinate mapping
        warped = map_coordinates(arr, [grid_h, grid_w], order=1, mode='nearest')

        return warped


class IntensityAugmentation:
    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)

    def amplitude_scaling(self, arr: np.ndarray,
                          scale_range: Tuple[float, float] = (0.95, 1.05),
                          clip_zero: bool = True) -> np.ndarray:
        scale = self.rng.uniform(scale_range[0], scale_range[1])
        out = arr * scale
        return np.clip(out, 0, None) if clip_zero else out

    def mixup(self, arr1: np.ndarray, arr2: np.ndarray,
              alpha: float = 0.3) -> np.ndarray:
        if arr1.shape != arr2.shape:
            raise ValueError(f"Arrays must have same shape: {arr1.shape} vs {arr2.shape}")

        # Sample mixing coefficient from Beta distribution
        lam = self.rng.beta(alpha, alpha)

        # Linear interpolation
        mixed = lam * arr1 + (1 - lam) * arr2

        return mixed


class AugmentationStrategy:

    @staticmethod
    def strategy_all_noise(arr, noise_aug, config):
        rt_px = noise_aug.rng.integers(config.RT_SHIFT[0], config.RT_SHIFT[1] + 1)
        dt_px = noise_aug.rng.integers(config.DT_SHIFT[0], config.DT_SHIFT[1] + 1)

        A = noise_aug.small_axis_shift_fixed(arr, rt_px=rt_px, dt_px=dt_px)
        A = noise_aug.multiplicative_jitter(A, sigma=config.JITTER_SIGMA)
        A = noise_aug.add_gaussian_noise(A, snr_db=config.SNR_DB)
        return A, {"rt_shift": rt_px, "dt_shift": dt_px}

    @staticmethod
    def strategy_selective_noise(arr, noise_aug, config):
        A = arr.copy()
        metadata = {}

        # Shift
        if noise_aug.rng.random() < config.PROB_SHIFT:
            rt_px = noise_aug.rng.integers(config.RT_SHIFT[0], config.RT_SHIFT[1] + 1)
            dt_px = noise_aug.rng.integers(config.DT_SHIFT[0], config.DT_SHIFT[1] + 1)
            A = noise_aug.small_axis_shift_fixed(A, rt_px=rt_px, dt_px=dt_px)
            metadata["shift"] = True
        else:
            metadata["shift"] = False

        # Jitter
        if noise_aug.rng.random() < config.PROB_JITTER:
            A = noise_aug.multiplicative_jitter(A, sigma=config.JITTER_SIGMA)
            metadata["jitter"] = True
        else:
            metadata["jitter"] = False

        # Noise
        if noise_aug.rng.random() < config.PROB_NOISE:
            A = noise_aug.add_gaussian_noise(A, snr_db=config.SNR_DB)
            metadata["noise"] = True
        else:
            metadata["noise"] = False

        return A, metadata

    @staticmethod
    def strategy_geometric_only(arr, geom_aug, config):
        A = arr.copy()
        metadata = {}

        if geom_aug.rng.random() < config.PROB_CROP:
            A = geom_aug.random_crop_and_pad(A, crop_fraction=config.CROP_FRACTION)
            metadata["crop"] = True
        else:
            metadata["crop"] = False

        if geom_aug.rng.random() < config.PROB_WARP:
            A = geom_aug.time_warping(A, sigma=config.WARP_SIGMA,
                                      knot_points=config.WARP_KNOTS)
            metadata["warp"] = True
        else:
            metadata["warp"] = False

        return A, metadata

    @staticmethod
    def strategy_intensity_only(arr, intensity_aug, config):
        A = arr.copy()
        metadata = {}

        if intensity_aug.rng.random() < config.PROB_SCALE:
            A = intensity_aug.amplitude_scaling(A, scale_range=config.SCALE_RANGE)
            metadata["scale"] = True
        else:
            metadata["scale"] = False

        return A, metadata

    @staticmethod
    def strategy_mixed(arr, noise_aug, geom_aug, intensity_aug, config):
        A = arr.copy()
        metadata = {}

        # Randomly select which augmentation categories to apply
        apply_noise = noise_aug.rng.random() < 0.5
        apply_geom = geom_aug.rng.random() < 0.4
        apply_intensity = intensity_aug.rng.random() < 0.6

        # Geometric augmentations (apply first as they change structure)
        if apply_geom:
            if geom_aug.rng.random() < config.PROB_CROP:
                A = geom_aug.random_crop_and_pad(A, crop_fraction=config.CROP_FRACTION)
                metadata["crop"] = True

            if geom_aug.rng.random() < config.PROB_WARP:
                A = geom_aug.time_warping(A, sigma=config.WARP_SIGMA,
                                          knot_points=config.WARP_KNOTS)
                metadata["warp"] = True

        # Intensity augmentations
        if apply_intensity:
            if intensity_aug.rng.random() < config.PROB_SCALE:
                A = intensity_aug.amplitude_scaling(A, scale_range=config.SCALE_RANGE)
                metadata["scale"] = True

        # Noise augmentations (apply last)
        if apply_noise:
            if noise_aug.rng.random() < config.PROB_SHIFT:
                rt_px = noise_aug.rng.integers(config.RT_SHIFT[0], config.RT_SHIFT[1] + 1)
                dt_px = noise_aug.rng.integers(config.DT_SHIFT[0], config.DT_SHIFT[1] + 1)
                A = noise_aug.small_axis_shift_fixed(A, rt_px=rt_px, dt_px=dt_px)
                metadata["shift"] = True

            if noise_aug.rng.random() < config.PROB_JITTER:
                A = noise_aug.multiplicative_jitter(A, sigma=config.JITTER_SIGMA)
                metadata["jitter"] = True

            if noise_aug.rng.random() < config.PROB_NOISE:
                A = noise_aug.add_gaussian_noise(A, snr_db=config.SNR_DB)
                metadata["noise"] = True

        return A, metadata


class AugmentationPipeline:

    def __init__(self, config: Config = None, strategy: str = "mixed"):
        self.config = config or Config()
        self.noise_aug = NoiseAugmentation()
        self.geom_aug = GeometricAugmentation()
        self.intensity_aug = IntensityAugmentation()
        self.strategy = strategy

        # Create output directory
        self.config.OUT.mkdir(parents=True, exist_ok=True)

    def apply_augmentation(self, arr: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if self.strategy == "all_noise":
            return AugmentationStrategy.strategy_all_noise(
                arr, self.noise_aug, self.config)

        elif self.strategy == "selective_noise":
            return AugmentationStrategy.strategy_selective_noise(
                arr, self.noise_aug, self.config)

        elif self.strategy == "geometric_only":
            return AugmentationStrategy.strategy_geometric_only(
                arr, self.geom_aug, self.config)

        elif self.strategy == "intensity_only":
            return AugmentationStrategy.strategy_intensity_only(
                arr, self.intensity_aug, self.config)

        elif self.strategy == "mixed":
            return AugmentationStrategy.strategy_mixed(
                arr, self.noise_aug, self.geom_aug, self.intensity_aug, self.config)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def process_mea(self, mea_path: Path) -> Dict:

        # Read MEA file
        spec = ims.Spectrum.read_mea(str(mea_path))
        M = np.clip(spec.values.astype(np.float32), 0, None)

        # Apply augmentation strategy
        A, metadata = self.apply_augmentation(M)

        # Save results
        base = mea_path.stem
        np.save(self.config.OUT / f"{base}_M.npy", M)
        np.save(self.config.OUT / f"{base}_M_aug.npy", A)
        np.save(self.config.OUT / f"{base}_rt.npy", spec.ret_time)
        np.save(self.config.OUT / f"{base}_dt.npy", spec.drift_time)

        return {
            "file": str(mea_path),
            "shape": f"{M.shape[0]}x{M.shape[1]}",
            "strategy": self.strategy,
            "clean_min": float(M.min()),
            "clean_max": float(M.max()),
            "aug_min": float(A.min()),
            "aug_max": float(A.max()),
            **metadata
        }

    def process_all_files(self) -> pd.DataFrame:
        files = sorted(self.config.ROOT.rglob("*.mea"))
        log = []

        for f in files:
            try:
                log.append(self.process_mea(f))
                print(f"Processed: {f.name}")
            except Exception as e:
                log.append({"file": str(f), "error": repr(e)})
                print(f"Error processing {f.name}: {e}")

        df = pd.DataFrame(log)
        df.to_csv(self.config.OUT / f"augmentation_log_{self.strategy}.csv", index=False)
        print(f"\nProcessed {len(files)} files → {self.config.OUT}")

        return df

    def create_mixup_dataset(self, n_pairs: int = None) -> pd.DataFrame:
        # Find all original files
        orig_files = sorted(self.config.DATA_ROOT.glob("*_M.npy"))

        if len(orig_files) < 2:
            print("Need at least 2 files for mixup")
            return pd.DataFrame()

        # Determine number of pairs
        if n_pairs is None or n_pairs > len(orig_files) // 2:
            n_pairs = len(orig_files) // 2

        log = []
        for i in range(n_pairs):
            # Randomly select two different files
            idx1, idx2 = self.intensity_aug.rng.choice(
                len(orig_files), size=2, replace=False)

            file1 = orig_files[idx1]
            file2 = orig_files[idx2]

            # Load arrays
            arr1 = np.load(file1)
            arr2 = np.load(file2)

            # Check if same shape
            if arr1.shape != arr2.shape:
                print(f"Shape mismatch, skipping: {file1.name} vs {file2.name}")
                continue

            # Apply mixup
            mixed = self.intensity_aug.mixup(arr1, arr2, alpha=self.config.MIXUP_ALPHA)

            # Save
            base1 = file1.stem.replace("_M", "")
            base2 = file2.stem.replace("_M", "")
            out_name = f"mixup_{base1}_{base2}_M_aug.npy"
            np.save(self.config.OUT / out_name, mixed)

            log.append({
                "file1": file1.name,
                "file2": file2.name,
                "output": out_name,
                "shape": f"{mixed.shape[0]}x{mixed.shape[1]}"
            })

            print(f"Created mixup {i + 1}/{n_pairs}: {out_name}")

        df = pd.DataFrame(log)
        df.to_csv(self.config.OUT / "mixup_log.csv", index=False)
        print(f"\nCreated {len(log)} mixup augmentations")

        return df


def calculate_psnr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mse = np.mean((x - y) ** 2)

    if mse == 0:
        return float("inf"), 0.0

    peak = np.max(x) - np.min(x)
    psnr_db = 20 * np.log10(peak / np.sqrt(mse + 1e-12))

    return psnr_db, mse


def bin_array(arr: np.ndarray, binning: int = 2) -> np.ndarray:
    if binning <= 1:
        return arr

    if arr.ndim == 2:
        r, c = arr.shape
        r2, c2 = r // binning, c // binning
        arr = arr[:r2 * binning, :c2 * binning]
        return arr.reshape(r2, binning, c2, binning).mean(axis=(1, 3))

    elif arr.ndim == 3:
        n, r, c = arr.shape
        r2, c2 = r // binning, c // binning
        arr = arr[:, :r2 * binning, :c2 * binning]
        return arr.reshape(n, r2, binning, c2, binning).mean(axis=(2, 4))

    return arr


def evaluate_augmentation_reliability(config: Config = None, strategy_name: str = "unknown"):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score)

    config = config or Config()
    DATA_ROOT = config.DATA_ROOT

    pairs = []
    for fname in os.listdir(DATA_ROOT):
        if fname.endswith("_M.npy") and not fname.startswith("mixup"):
            base = fname[:-6]
            aug_name = base + "_M_aug.npy"
            aug_path = os.path.join(DATA_ROOT, aug_name)
            orig_path = os.path.join(DATA_ROOT, fname)
            if os.path.exists(aug_path):
                pairs.append((orig_path, aug_path))

    if not pairs:
        raise RuntimeError("No matching _M.npy and _M_aug.npy pairs found")

    print(f"Found {len(pairs)} pairs of original/augmented files")

    # Load and prepare data
    X_parts, y_parts = [], []

    for orig_path, aug_path in pairs:
        Xo = np.load(orig_path)
        Xa = np.load(aug_path)

        if Xo.shape != Xa.shape:
            print(f"Shape mismatch, skipping: {orig_path}")
            continue

        Xo = bin_array(Xo, config.BINNING)
        Xa = bin_array(Xa, config.BINNING)

        n_take = min(len(Xo), config.MAX_SAMPLES_PER_FILE)
        idx_o = np.random.choice(len(Xo), n_take, replace=False)
        idx_a = np.random.choice(len(Xa), n_take, replace=False)

        X_parts.append(Xo[idx_o].reshape(n_take, -1).astype(np.float32))
        y_parts.append(np.zeros(n_take, dtype=np.int8))
        X_parts.append(Xa[idx_a].reshape(n_take, -1).astype(np.float32))
        y_parts.append(np.ones(n_take, dtype=np.int8))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    print(f"Loaded {len(X)} samples ({np.sum(y == 0)} orig / {np.sum(y == 1)} aug)")
    print(f"Feature dimension: {X.shape[1]}")

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # Downsample if needed
    if len(X) > config.MAX_SAMPLES_TOTAL:
        idx = np.random.choice(len(X), config.MAX_SAMPLES_TOTAL, replace=False)
        X, y = X[idx], y[idx]
        print(f"Downsampled to {len(X)} samples")

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE,
                                                max_depth=10),  # Limit depth to avoid overfitting
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=config.RANDOM_STATE)
    }

    print(f"RELIABILITY EVALUATION - Strategy: {strategy_name}")

    results = []
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_s, y_train)

        y_pred = clf.predict(X_test_s)
        y_proba = clf.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"\n{name}:")
        print(f"  Accuracy : {acc:.3f}")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall   : {rec:.3f}")
        print(f"  F1-score : {f1:.3f}")
        print(f"  ROC-AUC  : {auc:.3f}")

        results.append({
            "strategy": strategy_name,
            "classifier": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc
        })

    return pd.DataFrame(results)

def main():
    print("\nAvailable strategies:")
    print("1. all_noise - Original (all noise augs, always)")
    print("2. selective_noise - Noise augs probabilistically")
    print("3. geometric_only - Only crop/warp")
    print("4. intensity_only - Only amplitude scaling")
    print("5. mixed - Random mix of all")

    strategy = input("\nSelect strategy (1-5) or press Enter for 'mixed': ").strip()

    strategy_map = {
        "1": "all_noise",
        "2": "selective_noise",
        "3": "geometric_only",
        "4": "intensity_only",
        "5": "mixed",
        "": "mixed"
    }

    strategy_name = strategy_map.get(strategy, "mixed")
    print(f"\nUsing strategy: {strategy_name}")

    # Create pipeline
    config = Config()
    pipeline = AugmentationPipeline(config, strategy=strategy_name)

    # Process all files
    print("\nProcessing files")
    log_df = pipeline.process_all_files()

    # Optional: Create mixup augmentations
    if strategy_name in ["mixed", "intensity_only"]:
        create_mixup = input("\nCreate mixup augmentations? (y/n): ").lower().strip()
        if create_mixup == 'y':
            n_mixup = int(input("How many mixup pairs? (default=10): ") or "10")
            pipeline.create_mixup_dataset(n_pairs=n_mixup)

    # Evaluate
    print("\nEvaluating augmentation reliability")
    results_df = evaluate_augmentation_reliability(config, strategy_name=strategy_name)
    results_df.to_csv(config.OUT / f"evaluation_{strategy_name}.csv", index=False)


if __name__ == "__main__":
    main()
