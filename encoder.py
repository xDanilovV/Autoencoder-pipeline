"""
Encoding and decoding utilities with batching for performance.
"""
import torch
import numpy as np
from tqdm.auto import tqdm
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_dataset_first_autoencoder(model, X: np.ndarray, batch_size: int = None) -> np.ndarray:
    """
    Encodes each column of X using AE1 with BATCHING for speed.

    Args:
        model: First autoencoder
        X : (n_samples, M, N)
        batch_size: Number of columns to process at once
    Returns:
        Z : (n_samples, D, N)
    """
    if batch_size is None:
        batch_size = config.ENCODING_BATCH_SIZE

    model.eval()
    model = model.to(device)

    n_samples, M, N = X.shape
    Z = np.zeros((n_samples, config.D, N), dtype=np.float32)

    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Encoding samples (AE1)"):
            # Extract all columns for this sample: shape (N, M)
            columns = X[i, :, :].T  # Transpose to get (N, M)

            # Process in batches
            for j in range(0, N, batch_size):
                end_idx = min(j + batch_size, N)
                batch_cols = torch.FloatTensor(columns[j:end_idx]).to(device)

                # Encode batch
                latent_batch = model.encode(batch_cols).cpu().numpy()

                # Store in Z: shape (D, batch_size) transposed to (batch_size, D)
                Z[i, :, j:end_idx] = latent_batch.T

    return Z


def encode_dataset_second_autoencoder(model, Z: np.ndarray, batch_size: int = None) -> np.ndarray:
    """
    Encodes each latent row using AE2 with BATCHING for speed.

    Args:
        model: Second autoencoder
        Z : (n_samples, D, N)
        batch_size: Number of rows to process at once
    Returns:
        E : (n_samples, D, D)
    """
    if batch_size is None:
        batch_size = config.ENCODING_BATCH_SIZE

    model.eval()
    model = model.to(device)

    n_samples, D, N = Z.shape
    E = np.zeros((n_samples, D, D), dtype=np.float32)

    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Encoding samples (AE2)"):
            # Extract all rows for this sample: shape (D, N)
            rows = Z[i, :, :]

            # Process in batches
            for j in range(0, D, batch_size):
                end_idx = min(j + batch_size, D)
                batch_rows = torch.FloatTensor(rows[j:end_idx]).to(device)

                # Encode batch
                latent_batch = model.encode(batch_rows).cpu().numpy()

                # Store in E
                E[i, j:end_idx, :] = latent_batch

    return E


def decode_latent_matrices(model1, model2, E: np.ndarray, batch_size: int = None) -> np.ndarray:
    """
    Decode samples from the fully latent space E → Z → X with BATCHING.

    Args:
        model1 : First autoencoder  (columns / retention time)
        model2 : Second autoencoder (rows / drift evolution)
        E      : (n_samples, D, D)
        batch_size: Number of sequences to decode at once

    Returns:
        X_reconstructed : (n_samples, M, N)
    """
    if batch_size is None:
        batch_size = config.ENCODING_BATCH_SIZE

    model1.eval()
    model2.eval()
    model1 = model1.to(device)
    model2 = model2.to(device)

    n_samples, D, _ = E.shape
    M, N = config.M, config.N

    X_rec = np.zeros((n_samples, M, N), dtype=np.float32)

    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Decoding latent matrices"):

            # Step 1 — Decode each row via AE2 → reconstruct Z
            Z_rec = np.zeros((D, N), dtype=np.float32)

            for j in range(0, D, batch_size):
                end_idx = min(j + batch_size, D)
                batch_latent = torch.FloatTensor(E[i, j:end_idx, :]).to(device)
                batch_rec = model2.decode(batch_latent).cpu().numpy()
                Z_rec[j:end_idx, :] = batch_rec

            # Step 2 — Decode each column via AE1 → reconstruct X
            # Z_rec is (D, N), we need to decode N columns of size D each
            Z_rec_T = Z_rec.T  # Shape: (N, D)

            for j in range(0, N, batch_size):
                end_idx = min(j + batch_size, N)
                batch_latent = torch.FloatTensor(Z_rec_T[j:end_idx, :]).to(device)
                batch_rec = model1.decode(batch_latent).cpu().numpy()
                X_rec[i, :, j:end_idx] = batch_rec.T

    return X_rec