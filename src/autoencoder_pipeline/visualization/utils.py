"""
Utility functions for latent modeling and visualization.
Includes improved covariance handling and sampling.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from autoencoder_pipeline.config import config


def plot_matrix(matrix, title="Matrix Visualization", cmap='viridis'):
    """Plot a 2D matrix as a heatmap."""
    plt.figure(figsize=(12, 6))
    plt.imshow(matrix, aspect='auto', cmap=cmap, origin='lower')
    plt.colorbar(label='Intensity')
    plt.title(title)
    plt.xlabel('Retention Time')
    plt.ylabel('Drift Time')
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/{title.replace(' ', '_')}.png", dpi=300)
    plt.close()


def plot_training_history(history, title="Training History"):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/{title.replace(' ', '_')}.png", dpi=300)
    plt.close()


def latent_stats_per_class(E, y):
    """
    Compute mean and covariance for latent matrices per class.

    Args:
        E: Latent matrices (n_samples, D, D)
        y: Labels

    Returns:
        Dictionary: {class_label: (mean_vector, covariance_matrix)}
    """
    stats = {}
    unique_classes = np.unique(y)

    print("\n=== Computing Class-Conditional Statistics ===")

    for cls in unique_classes:
        mask = (y == cls)
        E_class = E[mask]  # (n_class, D, D)

        # Flatten each matrix to a vector
        E_flat = E_class.reshape(len(E_class), -1)  # (n_class, D*D)

        # Compute mean and covariance
        mu = np.mean(E_flat, axis=0)  # (D*D,)

        if len(E_flat) < 2:
            raise ValueError(f"Class '{cls}' needs at least 2 samples for latent statistics.")

        if config.SAMPLING_METHOD == "shrinkage":
            cov = LedoitWolf().fit(E_flat).covariance_
        elif config.SAMPLING_METHOD == "diagonal":
            variances = np.var(E_flat, axis=0, ddof=1)
            cov = np.diag(np.maximum(variances, 1e-6))
        else:
            cov = np.cov(E_flat, rowvar=False)
            cov = cov + 1e-6 * np.eye(cov.shape[0])

        # Check condition number
        try:
            cond_number = np.linalg.cond(cov)
            if cond_number > 1e10:
                print(f"  Warning: Class '{cls}' has ill-conditioned covariance (cond={cond_number:.2e})")
                cov = cov + 1e-4 * np.eye(cov.shape[0])
        except:
            print(f"  Warning: Class '{cls}' covariance condition check failed")

        stats[cls] = (mu, cov)

        print(f"  Class '{cls}': {len(E_class)} samples")
        print(f"    μ shape: {mu.shape}")
        print(f"    Σ shape: {cov.shape}")
        print(f"    μ range: [{mu.min():.4f}, {mu.max():.4f}]")
        print(f"    Σ diagonal range: [{np.diag(cov).min():.4e}, {np.diag(cov).max():.4e}]")

    return stats


def sample_latent_vectors(mu, cov, n_samples, method="shrinkage", std_scale=1.0):
    """
    Sample latent vectors from a multivariate Gaussian distribution.
    Improved with better numerical stability.

    Args:
        mu: Mean vector (D*D,)
        cov: Covariance matrix (D*D, D*D)
        n_samples: Number of samples to generate
        method: 'shrinkage', 'diagonal', 'cholesky', 'eigenvalue', or 'svd'

    Returns:
        Samples array (n_samples, D*D)
    """
    print(f"\n  Sampling {n_samples} latent vectors using {method} decomposition...")

    try:
        scaled_cov = cov * float(std_scale ** 2)

        if method in {"shrinkage", "diagonal"}:
            samples = np.random.multivariate_normal(mu, scaled_cov, size=n_samples)

        elif method == 'cholesky':
            # Standard Cholesky decomposition (fastest if cov is well-conditioned)
            L = np.linalg.cholesky(scaled_cov)
            # Sample from standard normal and transform
            z = np.random.randn(n_samples, len(mu))
            samples = mu + z @ L.T

        elif method == 'eigenvalue':
            # Eigenvalue decomposition (more stable for ill-conditioned matrices)
            eigenvalues, eigenvectors = np.linalg.eigh(scaled_cov)

            # Clip negative eigenvalues (shouldn't exist theoretically but can due to numerics)
            eigenvalues = np.maximum(eigenvalues, 1e-10)

            # Reconstruct with clipped eigenvalues
            sqrt_eigenvalues = np.sqrt(eigenvalues)
            transform = eigenvectors @ np.diag(sqrt_eigenvalues)

            # Sample and transform
            z = np.random.randn(n_samples, len(mu))
            samples = mu + z @ transform.T

        elif method == 'svd':
            # SVD decomposition (most stable but slowest)
            U, s, Vt = np.linalg.svd(scaled_cov)

            # Clip small singular values
            s = np.maximum(s, 1e-10)

            # Reconstruct
            sqrt_s = np.sqrt(s)
            transform = U @ np.diag(sqrt_s)

            # Sample and transform
            z = np.random.randn(n_samples, len(mu))
            samples = mu + z @ transform.T

        else:
            # Fallback to numpy's multivariate_normal with added regularization
            cov_reg = scaled_cov + 1e-5 * np.eye(len(scaled_cov))
            samples = np.random.multivariate_normal(mu, cov_reg, size=n_samples)

        print(f"  Successfully sampled {n_samples} vectors")
        print(f"    Sample mean: {samples.mean():.6f} (target: {mu.mean():.6f})")
        print(f"    Sample std: {samples.std():.6f} (target: {np.sqrt(np.diag(scaled_cov)).mean():.6f})")

        return samples.astype(np.float32)

    except np.linalg.LinAlgError as e:
        print(f"  ERROR: Sampling failed with {method} method: {e}")
        print(f"  Falling back to regularized numpy sampling...")

        # Heavy regularization fallback
        cov_reg = scaled_cov + 1e-3 * np.eye(len(scaled_cov))
        samples = np.random.multivariate_normal(mu, cov_reg, size=n_samples)
        return samples.astype(np.float32)


def pca_real_vs_synthetic(X_real, X_synth, title="PCA: Real vs Synthetic"):
    """
    Visualize real vs synthetic spectra using PCA.

    Args:
        X_real: Real spectra (n, M, N)
        X_synth: Synthetic spectra (n, M, N)
        title: Plot title
    """
    # Flatten spectra
    real_flat = X_real.reshape(len(X_real), -1)
    synth_flat = X_synth.reshape(len(X_synth), -1)

    # Combine for PCA
    combined = np.vstack([real_flat, synth_flat])

    # Apply PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(combined)

    # Split back
    real_proj = projected[:len(X_real)]
    synth_proj = projected[len(X_real):]

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(real_proj[:, 0], real_proj[:, 1],
                alpha=0.6, s=50, label='Real', marker='o', edgecolors='black', linewidths=0.5)
    plt.scatter(synth_proj[:, 0], synth_proj[:, 1],
                alpha=0.6, s=50, label='Synthetic', marker='x', linewidths=2)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/pca_comparison.png", dpi=300)
    plt.close()

    print(f"\nPCA Analysis:")
    print(f"  PC1 variance explained: {pca.explained_variance_ratio_[0] * 100:.2f}%")
    print(f"  PC2 variance explained: {pca.explained_variance_ratio_[1] * 100:.2f}%")
    print(f"  Total variance explained: {pca.explained_variance_ratio_[:2].sum() * 100:.2f}%")


def diagnose_latent_quality(E_real, E_synth, y_real, y_synth):
    """
    Diagnostic plots to assess quality of synthetic latent matrices.

    Args:
        E_real: Real latent matrices (n, D, D)
        E_synth: Synthetic latent matrices (n, D, D)
        y_real: Real labels
        y_synth: Synthetic labels
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Latent Space Quality Diagnostics', fontsize=16)

    # Flatten for analysis
    E_real_flat = E_real.reshape(len(E_real), -1)
    E_synth_flat = E_synth.reshape(len(E_synth), -1)

    # 1. Distribution comparison
    axes[0, 0].hist(E_real_flat.flatten(), bins=50, alpha=0.5, label='Real', density=True)
    axes[0, 0].hist(E_synth_flat.flatten(), bins=50, alpha=0.5, label='Synthetic', density=True)
    axes[0, 0].set_xlabel('Latent Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Value Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Mean comparison
    real_means = E_real_flat.mean(axis=1)
    synth_means = E_synth_flat.mean(axis=1)
    axes[0, 1].hist(real_means, bins=30, alpha=0.5, label='Real')
    axes[0, 1].hist(synth_means, bins=30, alpha=0.5, label='Synthetic')
    axes[0, 1].set_xlabel('Mean Latent Value')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Mean Value per Sample')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Std comparison
    real_stds = E_real_flat.std(axis=1)
    synth_stds = E_synth_flat.std(axis=1)
    axes[0, 2].hist(real_stds, bins=30, alpha=0.5, label='Real')
    axes[0, 2].hist(synth_stds, bins=30, alpha=0.5, label='Synthetic')
    axes[0, 2].set_xlabel('Std of Latent Values')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Std per Sample')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. PCA visualization
    combined = np.vstack([E_real_flat, E_synth_flat])
    pca = PCA(n_components=2)
    projected = pca.fit_transform(combined)

    real_proj = projected[:len(E_real)]
    synth_proj = projected[len(E_real):]

    axes[1, 0].scatter(real_proj[:, 0], real_proj[:, 1], alpha=0.6, s=30, label='Real')
    axes[1, 0].scatter(synth_proj[:, 0], synth_proj[:, 1], alpha=0.6, s=30, label='Synthetic', marker='x')
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    axes[1, 0].set_title('PCA Projection')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Per-class comparison
    unique_classes = np.unique(y_real)
    for cls in unique_classes:
        real_mask = (y_real == cls)
        synth_mask = (y_synth == cls)

        if np.any(real_mask) and np.any(synth_mask):
            real_class_mean = E_real_flat[real_mask].mean(axis=0)
            synth_class_mean = E_synth_flat[synth_mask].mean(axis=0)

            axes[1, 1].scatter(real_class_mean, synth_class_mean, alpha=0.6, s=10, label=cls)

    axes[1, 1].plot([E_real_flat.min(), E_real_flat.max()],
                    [E_real_flat.min(), E_real_flat.max()],
                    'r--', alpha=0.5, label='y=x')
    axes[1, 1].set_xlabel('Real Mean Latent Value')
    axes[1, 1].set_ylabel('Synthetic Mean Latent Value')
    axes[1, 1].set_title('Per-Class Mean Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Correlation matrix comparison
    real_corr = np.corrcoef(E_real_flat[:10].T)  # Use subset for speed
    synth_corr = np.corrcoef(E_synth_flat[:10].T)

    diff_corr = np.abs(real_corr - synth_corr)
    im = axes[1, 2].imshow(diff_corr, cmap='hot', vmin=0, vmax=1)
    axes[1, 2].set_title('|Real Corr - Synth Corr|')
    plt.colorbar(im, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/latent_diagnostics.png", dpi=300)
    plt.close(fig)
