"""
GC-IMS augmentation pipeline with sequential transformer autoencoders.

Data flow:
    X (M, N) -> [AE1] -> Z (D, N) -> [AE2] -> E (D, D)
    E (D, D) -> [AE2 decode] -> Z (D, N) -> [AE1 decode] -> X (M, N)
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from classifier import classify_spectra
from config import config
from data_loader import (
    build_preprocessor,
    load_spectral_data,
    match_synthetic_to_real_distribution,
)
from dataset import (
    TimeseriesDataset,
    extract_columns_as_timeseries,
    extract_rows_as_timeseries,
)
from gcims_debug_viz import plot_pipeline_visualization
from models import TransformerAutoencoder
from trainer import train_autoencoder
from utils import (
    diagnose_latent_quality,
    latent_stats_per_class,
    plot_matrix,
    plot_training_history,
    sample_latent_vectors,
)


def undersample_rows(rows, prob, rng=None):
    if rng is None:
        rng = np.random.default_rng(config.SEED)
    rows = np.asarray(rows)
    stds = np.std(rows, axis=1)
    median = np.median(stds)

    keep = stds >= median
    low_variance = ~keep
    keep[low_variance] = rng.random(np.sum(low_variance)) < prob
    return rows[keep]


def set_random_seeds():
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)


def standardize_latent(train_latent, val_latent):
    mean = float(train_latent.mean())
    std = float(train_latent.std())
    std = max(std, 1e-8)

    return (
        ((train_latent - mean) / std).astype(np.float32),
        ((val_latent - mean) / std).astype(np.float32),
        {"mean": mean, "std": std},
    )


def denormalize_latent(latent, params):
    return latent * params["std"] + params["mean"]


def build_loader(data, shuffle):
    return DataLoader(
        TimeseriesDataset(data),
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
    )


def build_autoencoder(input_dim):
    return TransformerAutoencoder(
        input_dim=input_dim,
        latent_dim=config.D,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
    )


def encode_matrix(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for mat in X:
            cols = torch.tensor(mat.T, dtype=torch.float32, device=config.DEVICE)
            latent = model.encode(cols).cpu().numpy().T
            out.append(latent)
    return np.asarray(out, dtype=np.float32)


def encode_latent_matrix(model, Z):
    model.eval()
    out = []
    with torch.no_grad():
        for mat in Z:
            rows = torch.tensor(mat, dtype=torch.float32, device=config.DEVICE)
            latent = model.encode(rows).cpu().numpy()
            out.append(latent)
    return np.asarray(out, dtype=np.float32)


def reconstruct_sample(AE1, AE2, X_sample, Z_sample, E_sample, z_norm_params):
    AE1.eval()
    AE2.eval()

    with torch.no_grad():
        Z_t = torch.tensor(Z_sample.T, dtype=torch.float32, device=config.DEVICE)
        X_ae1_recon = AE1.decode(Z_t).cpu().numpy().T

        E_t = torch.tensor(E_sample, dtype=torch.float32, device=config.DEVICE)
        Z_ae2_recon_norm = AE2.decode(E_t).cpu().numpy()
        Z_ae2_recon = denormalize_latent(Z_ae2_recon_norm, z_norm_params)

        Z_rec_t = torch.tensor(Z_ae2_recon.T, dtype=torch.float32, device=config.DEVICE)
        X_full_recon = AE1.decode(Z_rec_t).cpu().numpy().T

    return X_ae1_recon, Z_ae2_recon, X_full_recon


def decode_synthetic(AE2, AE1, latent_mats, z_norm_params, verbose=False):
    AE1.eval()
    AE2.eval()
    out = []

    with torch.no_grad():
        for idx, E in enumerate(latent_mats):
            try:
                E_t = torch.tensor(E, dtype=torch.float32, device=config.DEVICE)
                Z_rec_norm = AE2.decode(E_t).cpu().numpy()
                Z_rec = denormalize_latent(Z_rec_norm, z_norm_params)

                Z_rec_t = torch.tensor(Z_rec.T, dtype=torch.float32, device=config.DEVICE)
                X_rec = AE1.decode(Z_rec_t).cpu().numpy().T

                if X_rec.shape != (config.M, config.N):
                    fixed = np.zeros((config.M, config.N), dtype=np.float32)
                    rows = min(config.M, X_rec.shape[0])
                    cols = min(config.N, X_rec.shape[1])
                    fixed[:rows, :cols] = X_rec[:rows, :cols]
                    X_rec = fixed

                out.append(X_rec.astype(np.float32))
            except Exception as exc:
                if verbose:
                    print(f"Error decoding sample {idx}: {exc}")

    if verbose:
        print(f"Successfully decoded {len(out)} / {len(latent_mats)} samples")

    return np.asarray(out, dtype=np.float32)


def synthetic_count_per_class(y_train):
    classes, counts = np.unique(y_train, return_counts=True)
    return {
        cls: max(1, math.ceil(count * config.SYNTHETIC_MULTIPLIER))
        for cls, count in zip(classes, counts)
    }


def plot_orientation_check(sample):
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(sample, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("Retention Time (GC)")
    ax.set_ylabel("Drift Time (IMS)")
    ax.set_title("Dimension Verification")
    plt.colorbar(im, ax=ax, label="Intensity")
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/dimension_verification.png", dpi=300)
    plt.close(fig)


def main():
    set_random_seeds()

    print("=" * 60)
    print("GC-IMS Augmentation Pipeline")
    print("=" * 60)
    print(f"Device: {config.DEVICE}\n")
    rng = np.random.default_rng(config.SEED)

    X_raw, y = load_spectral_data(
        data_root=config.DATA_PATH,
        selected_classes=config.SELECTED_CLASSES,
        verbose=True,
    )

    print(f"\nLoaded {len(X_raw)} spectra")
    plot_orientation_check(X_raw[0])

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw,
        y,
        test_size=1 - config.TRAIN_SPLIT,
        stratify=y,
        random_state=config.SEED,
    )

    preprocessor = build_preprocessor(X_train_raw, method=config.NORMALIZATION_METHOD)
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    config.M, config.N = X_train.shape[1], X_train.shape[2]

    print("\nTrain-fitted preprocessing:")
    print(f"  Common raw shape: {preprocessor.common_shape}")
    print(f"  RIP target column: {preprocessor.rip_target_col}")
    print(f"  RIP cut half-width: {preprocessor.rip_cut_half_width}")
    print(f"  Row window: {preprocessor.row_slice}")
    print(f"  Column window: {preprocessor.col_slice}")
    print(f"  Model input shape: ({config.M}, {config.N})")
    plot_matrix(X_train[0], title="Preprocessed Training Spectrum")

    print("\nNormalized train statistics:")
    print(f"  Mean: {X_train.mean():.6f}")
    print(f"  Std:  {X_train.std():.6f}")
    print(f"  Min:  {X_train.min():.6f}")
    print(f"  Max:  {X_train.max():.6f}")

    print(f"\nTraining:   {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)

    print("\nTraining AE1 (column autoencoder)")
    train_cols = undersample_rows(
        extract_columns_as_timeseries(X_train),
        prob=config.UNDERSAMPLE_PROB,
        rng=rng,
    )
    val_cols = extract_columns_as_timeseries(X_val)
    if config.UNDERSAMPLE_VALIDATION:
        val_cols = undersample_rows(val_cols, prob=config.UNDERSAMPLE_PROB, rng=rng)

    AE1 = build_autoencoder(config.M)
    hist1 = train_autoencoder(
        AE1,
        build_loader(train_cols, shuffle=True),
        build_loader(val_cols, shuffle=False),
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        patience=config.PATIENCE,
        model_name="ae1",
    )
    plot_training_history(hist1, "AE1_Training_Curve")

    Z_train_raw = encode_matrix(AE1, X_train)
    Z_val_raw = encode_matrix(AE1, X_val)
    Z_train, Z_val, z_norm_params = standardize_latent(Z_train_raw, Z_val_raw)

    print("\nTraining AE2 (row autoencoder)")
    train_rows = undersample_rows(
        extract_rows_as_timeseries(Z_train),
        prob=config.UNDERSAMPLE_PROB,
        rng=rng,
    )
    val_rows = extract_rows_as_timeseries(Z_val)
    if config.UNDERSAMPLE_VALIDATION:
        val_rows = undersample_rows(val_rows, prob=config.UNDERSAMPLE_PROB, rng=rng)

    AE2 = build_autoencoder(config.N)
    hist2 = train_autoencoder(
        AE2,
        build_loader(train_rows, shuffle=True),
        build_loader(val_rows, shuffle=False),
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        patience=config.PATIENCE,
        model_name="ae2",
    )
    plot_training_history(hist2, "AE2_Training_Curve")

    E_train_raw = encode_latent_matrix(AE2, Z_train)
    E_val_raw = encode_latent_matrix(AE2, Z_val)
    E_train, E_val, e_norm_params = standardize_latent(E_train_raw, E_val_raw)

    plot_matrix(Z_train[0], title="AE1 Latent Matrix Z (Normalized)")
    plot_matrix(E_train[0], title="AE2 Latent Matrix E (Normalized)")

    X_ae1_recon, Z_ae2_recon, X_full_recon = reconstruct_sample(
        AE1,
        AE2,
        X_train[0],
        Z_train_raw[0],
        E_train_raw[0],
        z_norm_params,
    )
    plot_pipeline_visualization(
        X_original=X_train[0],
        Z_encoded=Z_train_raw[0],
        Z_decoded=Z_ae2_recon,
        X_reconstructed=X_full_recon,
        sample_idx=0,
    )
    plot_matrix(X_ae1_recon, title="AE1 Only Reconstruction")

    diff_full = np.abs(X_train[0] - X_full_recon)
    plot_matrix(diff_full, title="Full Pipeline Reconstruction Error")
    print(f"Full pipeline mean absolute error: {diff_full.mean():.6f}")

    print("\nFitting class-conditional latent models")
    stats = latent_stats_per_class(E_train, y_train)

    synth_counts = synthetic_count_per_class(y_train)
    synth_latent = []
    synth_labels = []

    for cls, (mu, cov) in stats.items():
        n_synth = synth_counts[cls]
        samples = sample_latent_vectors(
            mu,
            cov,
            n_synth,
            method=config.SAMPLING_METHOD,
            std_scale=config.SYNTHETIC_STD_SCALE,
        )
        synth_latent.append(samples.reshape(-1, config.D, config.D))
        synth_labels.extend([cls] * n_synth)
        print(f"Generated {n_synth} latent samples for class '{cls}'")

    synth_latent = np.concatenate(synth_latent, axis=0)
    synth_latent_denorm = denormalize_latent(synth_latent, e_norm_params)

    X_synth = decode_synthetic(
        AE2,
        AE1,
        synth_latent_denorm,
        z_norm_params=z_norm_params,
        verbose=True,
    )

    if config.MATCH_SYNTHETIC_DISTRIBUTION:
        X_synth = match_synthetic_to_real_distribution(X_synth, X_train, method="statistics")

    X_synth = np.clip(X_synth, 0.0, 1.0).astype(np.float32)
    plot_matrix(X_synth[0], title="Synthetic GC-IMS")

    diagnose_latent_quality(
        E_real=E_train,
        E_synth=synth_latent,
        y_real=np.asarray(y_train),
        y_synth=np.asarray(synth_labels),
    )

    synth_labels_encoded = le.transform(synth_labels)

    print("\nBaseline classifier (real only)")
    baseline = classify_spectra(X_train, y_train_encoded, X_val, y_val_encoded, le)

    print("\nAugmented classifier (real + synthetic)")
    augmented = classify_spectra(
        np.concatenate([X_train, X_synth], axis=0),
        np.concatenate([y_train_encoded, synth_labels_encoded], axis=0),
        X_val,
        y_val_encoded,
        le,
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Baseline Accuracy:   {baseline['accuracy']:.4f}")
    print(f"Augmented Accuracy:  {augmented['accuracy']:.4f}")
    print(f"Improvement:         {augmented['accuracy'] - baseline['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
