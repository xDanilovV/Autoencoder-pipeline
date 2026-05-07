"""
Enhanced visualization utilities for debugging the GC-IMS pipeline.
Add these functions to your utils.py or create a new file.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from config import config


def plot_encoding_decoding_steps(sample_idx=0):
    """
    Visualize a single sample through the entire encoding/decoding pipeline.
    Call this at each major step in your pipeline.
    """
    pass  # This is a placeholder - see the integration code below


def plot_spectra_grid(matrices, titles, suptitle="Spectra Comparison", n_cols=3, figsize=(15, 10)):
    """
    Plot multiple GC-IMS spectra in a grid.

    Args:
        matrices: list of 2D numpy arrays
        titles: list of titles for each subplot
        suptitle: overall figure title
        n_cols: number of columns in grid
        figsize: figure size
    """
    n_samples = len(matrices)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(suptitle, fontsize=16)

    # Flatten axes for easy iteration
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else [axes]

    for idx, (mat, title) in enumerate(zip(matrices, titles)):
        ax = axes[idx]

        # Plot the matrix
        im = ax.imshow(mat, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Retention Time')
        ax.set_ylabel('Drift Time')

        # Add colorbar
        plt.colorbar(im, ax=ax)

    # Hide empty subplots
    for idx in range(len(matrices), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/{suptitle.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_pipeline_visualization(X_original, Z_encoded, Z_decoded, X_reconstructed, sample_idx=0):
    """
    Show a single sample at each stage of the autoencoding pipeline.

    Args:
        X_original: original spectrum (M, N)
        Z_encoded: latent representation after AE1 (D, N)
        Z_decoded: reconstructed latent after AE2 decode (D, N)
        X_reconstructed: final reconstructed spectrum (M, N)
        sample_idx: which sample to visualize
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Pipeline Visualization - Sample {sample_idx}', fontsize=16)

    # Original
    im0 = axes[0, 0].imshow(X_original, aspect='auto', cmap='viridis', origin='lower')
    axes[0, 0].set_title(f'Original Spectrum\nShape: {X_original.shape}')
    axes[0, 0].set_xlabel('Retention Time (N)')
    axes[0, 0].set_ylabel('Drift Time (M)')
    plt.colorbar(im0, ax=axes[0, 0])

    # AE1 Encoded (latent Z)
    im1 = axes[0, 1].imshow(Z_encoded, aspect='auto', cmap='plasma', origin='lower')
    axes[0, 1].set_title(f'After AE1 Encoding (Z)\nShape: {Z_encoded.shape}')
    axes[0, 1].set_xlabel('Retention Time (N)')
    axes[0, 1].set_ylabel('Latent Dimension (D)')
    plt.colorbar(im1, ax=axes[0, 1])

    # AE2 Decoded (reconstructed Z)
    im2 = axes[1, 0].imshow(Z_decoded, aspect='auto', cmap='plasma', origin='lower')
    axes[1, 0].set_title(f'After AE2 Decoding (Z reconstructed)\nShape: {Z_decoded.shape}')
    axes[1, 0].set_xlabel('Retention Time (N)')
    axes[1, 0].set_ylabel('Latent Dimension (D)')
    plt.colorbar(im2, ax=axes[1, 0])

    # Final Reconstruction
    im3 = axes[1, 1].imshow(X_reconstructed, aspect='auto', cmap='viridis', origin='lower')
    axes[1, 1].set_title(f'Final Reconstruction\nShape: {X_reconstructed.shape}')
    axes[1, 1].set_xlabel('Retention Time (N)')
    axes[1, 1].set_ylabel('Drift Time (M)')
    plt.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/pipeline_viz_sample_{sample_idx}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_real_vs_synthetic_comparison(X_real, X_synth, y_real, y_synth, n_samples_per_class=3):
    """
    Compare real and synthetic spectra side by side for each class.

    Args:
        X_real: real spectra (n_real, M, N)
        X_synth: synthetic spectra (n_synth, M, N)
        y_real: real labels
        y_synth: synthetic labels
        n_samples_per_class: how many samples to show per class
    """
    unique_classes = np.unique(y_real)

    for cls in unique_classes:
        # Get samples for this class
        real_mask = (y_real == cls)
        synth_mask = (y_synth == cls)

        real_samples = X_real[real_mask][:n_samples_per_class]
        synth_samples = X_synth[synth_mask][:n_samples_per_class]

        n_real = len(real_samples)
        n_synth = len(synth_samples)
        n_rows = max(n_real, n_synth)

        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
        fig.suptitle(f'Class: {cls} - Real vs Synthetic', fontsize=16)

        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # Global min/max for consistent coloring
        vmin = min(real_samples.min(), synth_samples.min())
        vmax = max(real_samples.max(), synth_samples.max())

        for i in range(n_rows):
            # Real sample
            if i < n_real:
                im = axes[i, 0].imshow(real_samples[i], aspect='auto', cmap='viridis',
                                       origin='lower', vmin=vmin, vmax=vmax)
                axes[i, 0].set_title(f'Real Sample {i + 1}')
                axes[i, 0].set_ylabel('Drift Time')
                if i == n_rows - 1:
                    axes[i, 0].set_xlabel('Retention Time')
                plt.colorbar(im, ax=axes[i, 0])
            else:
                axes[i, 0].axis('off')

            # Synthetic sample
            if i < n_synth:
                im = axes[i, 1].imshow(synth_samples[i], aspect='auto', cmap='viridis',
                                       origin='lower', vmin=vmin, vmax=vmax)
                axes[i, 1].set_title(f'Synthetic Sample {i + 1}')
                if i == n_rows - 1:
                    axes[i, 1].set_xlabel('Retention Time')
                plt.colorbar(im, ax=axes[i, 1])
            else:
                axes[i, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f"{config.RESULTS_PATH}/real_vs_synth_class_{cls}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_intensity_distributions(X_real, X_synth, title="Intensity Distribution Comparison"):
    """
    Compare the intensity distributions of real vs synthetic spectra.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Flatten all values
    real_values = X_real.flatten()
    synth_values = X_synth.flatten()

    # Histograms
    axes[0].hist(real_values, bins=100, alpha=0.7, label='Real', color='blue', density=True)
    axes[0].hist(synth_values, bins=100, alpha=0.7, label='Synthetic', color='orange', density=True)
    axes[0].set_xlabel('Intensity')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Intensity Distribution')
    axes[0].legend()
    axes[0].set_yscale('log')

    # Statistics comparison
    stats_text = f"""
    Real Stats:
    Mean: {real_values.mean():.4f}
    Std: {real_values.std():.4f}
    Min: {real_values.min():.4f}
    Max: {real_values.max():.4f}

    Synthetic Stats:
    Mean: {synth_values.mean():.4f}
    Std: {synth_values.std():.4f}
    Min: {synth_values.min():.4f}
    Max: {synth_values.max():.4f}
    """

    axes[1].text(0.1, 0.5, stats_text, transform=axes[1].transAxes,
                 fontsize=10, verticalalignment='center', family='monospace')
    axes[1].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/intensity_distributions.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
