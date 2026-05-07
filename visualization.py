"""
Visualization utilities for GC-IMS pipeline.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from sklearn.preprocessing import LabelEncoder
from config import config
from sklearn.decomposition import PCA


def plot_training_history(history: Dict, title: str):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label="Train Loss")
    plt.plot(history['val_loss'], label="Val Loss")
    plt.yscale("log")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{config.RESULTS_PATH}/{title.replace(' ', '_')}.png", dpi=300)
    plt.close()


def plot_reconstruction_comparison(original: np.ndarray, reconstructed: np.ndarray, n_samples: int = 3):
    plt.figure(figsize=(12, 4 * n_samples))

    for i in range(n_samples):
        # Original
        plt.subplot(n_samples, 2, 2*i + 1)
        vmin = np.min(original)
        vmax = np.max(original)
        plt.imshow(original[i], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title(f"Original Sample {i+1}")
        plt.colorbar()

        # Reconstruction
        plt.subplot(n_samples, 2, 2*i + 2)
        plt.imshow(reconstructed[i], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title(f"Reconstructed Sample {i+1}")
        plt.colorbar()

    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/reconstruction_comparison.png", dpi=300)
    plt.close()


def plot_synthetic_samples(X_synth: np.ndarray, y_synth: np.ndarray, label_encoder: LabelEncoder, n_samples: int = 6):
    unique_labels = np.unique(y_synth)

    plt.figure(figsize=(12, 4 * len(unique_labels)))

    for i, label in enumerate(unique_labels):
        class_samples = X_synth[y_synth == label]
        name = label_encoder.inverse_transform([label])[0]

        for j in range(min(2, len(class_samples))):
            plt.subplot(len(unique_labels), 2, 2*i + j + 1)
            plt.imshow(class_samples[j], aspect='auto', cmap='viridis')
            plt.title(f"Synth {name} — Sample {j+1}")
            plt.colorbar()

    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/synthetic_samples.png", dpi=300)
    plt.close()

def plot_real_vs_synth(X_real, X_synth, y_real, y_synth, label_encoder, n_samples=3):

    unique_labels = np.unique(y_real)

    for label in unique_labels:
        name = label_encoder.inverse_transform([label])[0]

        real_samples = X_real[y_real == label][:n_samples]
        synth_samples = X_synth[y_synth == label][:n_samples]

        vmin = min(real_samples.min(), synth_samples.min())
        vmax = max(real_samples.max(), synth_samples.max())

        plt.figure(figsize=(12, 4 * n_samples))
        plt.suptitle(f"Class: {name} — Real vs Synthetic")

        for i in range(n_samples):
            plt.subplot(n_samples, 2, 2*i + 1)
            plt.imshow(real_samples[i], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            plt.title(f"Real {name} {i+1}")
            plt.colorbar()

            plt.subplot(n_samples, 2, 2*i + 2)
            plt.imshow(synth_samples[i], aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            plt.title(f"Synthetic {name} {i+1}")
            plt.colorbar()

        plt.tight_layout()
        plt.savefig(f"{config.RESULTS_PATH}/real_vs_synth_{name}.png", dpi=300)
        plt.close()

def plot_pca_real_synth(E_real, E_synth, y_real, y_synth, label_encoder):
    """
    PCA projection of real and synthetic latent matrices.
    Matches the validation figures in the paper.
    """
    flat_real = E_real.reshape(len(E_real), -1)
    flat_synth = E_synth.reshape(len(E_synth), -1)

    X = np.concatenate([flat_real, flat_synth], axis=0)
    y = np.concatenate([y_real, y_synth])
    domain = np.array(['Real'] * len(flat_real) + ['Synthetic'] * len(flat_synth))

    pca = PCA(n_components=2)
    proj = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))

    for label in np.unique(y):
        cls_name = label_encoder.inverse_transform([label])[0]

        mask_real = (y == label) & (domain == 'Real')
        mask_synth = (y == label) & (domain == 'Synthetic')

        plt.scatter(proj[mask_real, 0], proj[mask_real, 1], s=12, label=f"{cls_name} (real)")
        plt.scatter(proj[mask_synth, 0], proj[mask_synth, 1], s=12, marker='x', label=f"{cls_name} (synthetic)")

    plt.title("PCA: Real vs Synthetic Latent Distributions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/pca_real_vs_synth.png", dpi=300)
    plt.close()
