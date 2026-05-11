"""Spectral similarity metrics for real and synthetic GC-IMS spectra."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from autoencoder_pipeline.config import config


def evaluate_spectral_similarity(X_train, y_train, X_val, y_val, X_synth, y_synth):
    """Compare synthetic spectra against a real-real nearest-neighbor baseline."""
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_synth = np.asarray(X_synth, dtype=np.float32)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_synth = np.asarray(y_synth)

    train_emb, val_emb, synth_emb, pca = _fit_pca_embeddings(X_train, X_val, X_synth)

    real_pairs = _nearest_same_class_pairs(
        reference=X_train,
        reference_labels=y_train,
        reference_emb=train_emb,
        query=X_val,
        query_labels=y_val,
        query_emb=val_emb,
    )
    synth_pairs = _nearest_same_class_pairs(
        reference=X_train,
        reference_labels=y_train,
        reference_emb=train_emb,
        query=X_synth,
        query_labels=y_synth,
        query_emb=synth_emb,
    )

    rows = [
        _summarize_pairs("real validation -> nearest real train", real_pairs),
        _summarize_pairs("synthetic -> nearest real train", synth_pairs),
    ]
    rows[0].update(_distribution_metrics(train_emb, val_emb, "real_real", pca))
    rows[1].update(_distribution_metrics(train_emb, synth_emb, "real_synthetic", pca))

    table = pd.DataFrame(rows)
    class_table = _class_centroid_table(train_emb, y_train, synth_emb, y_synth)

    csv_path = config.RESULTS_PATH / "spectral_similarity_table.csv"
    md_path = config.RESULTS_PATH / "spectral_similarity_table.md"
    class_csv_path = config.RESULTS_PATH / "spectral_class_centroid_distances.csv"
    class_md_path = config.RESULTS_PATH / "spectral_class_centroid_distances.md"

    table.to_csv(csv_path, index=False)
    _write_markdown_table(table, md_path)
    class_table.to_csv(class_csv_path, index=False)
    _write_markdown_table(class_table, class_md_path)
    _plot_pair_examples(synth_pairs[:6], config.RESULTS_PATH / "synthetic_nearest_real_pairs.png")

    print("\nSaved spectral similarity diagnostics:")
    print(f"  CSV: {csv_path}")
    print(f"  Markdown: {md_path}")
    print(f"  Class centroids: {class_csv_path}")
    print(table.to_string(index=False))

    return table, class_table


def _fit_pca_embeddings(X_train, X_val, X_synth):
    n_components = min(
        int(config.SPECTRAL_METRIC_PCA_COMPONENTS),
        len(X_train) - 1,
        np.prod(X_train.shape[1:]),
    )
    pca = PCA(n_components=n_components, random_state=config.SEED)
    train_flat = _flatten(X_train)
    val_flat = _flatten(X_val)
    synth_flat = _flatten(X_synth)
    train_emb = pca.fit_transform(train_flat)
    val_emb = pca.transform(val_flat)
    synth_emb = pca.transform(synth_flat)
    return train_emb, val_emb, synth_emb, pca


def _nearest_same_class_pairs(reference, reference_labels, reference_emb, query, query_labels, query_emb):
    pairs = []
    for idx, label in enumerate(query_labels):
        candidate_indices = np.where(reference_labels == label)[0]
        if candidate_indices.size == 0:
            continue

        diffs = reference_emb[candidate_indices] - query_emb[idx]
        distances = np.linalg.norm(diffs, axis=1)
        best_local = int(np.argmin(distances))
        best_idx = int(candidate_indices[best_local])
        pairs.append({
            "label": str(label),
            "query": query[idx],
            "reference": reference[best_idx],
            "nn_pca_distance": float(distances[best_local]),
        })
    return pairs


def _summarize_pairs(name, pairs):
    metric_rows = []
    for pair in pairs:
        metric = _pair_metrics(pair["query"], pair["reference"])
        metric["nn_pca_distance"] = pair["nn_pca_distance"]
        metric_rows.append(metric)

    if not metric_rows:
        return {"comparison": name, "n_pairs": 0}

    keys = sorted(metric_rows[0])
    out = {"comparison": name, "n_pairs": len(metric_rows)}
    for key in keys:
        values = np.asarray([row[key] for row in metric_rows], dtype=np.float64)
        out[f"{key}_mean"] = float(np.nanmean(values))
        out[f"{key}_std"] = float(np.nanstd(values))
    return out


def _pair_metrics(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    diff = a - b
    a_flat = _sample_pixels(a)
    b_flat = _sample_pixels(b)

    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "pearson": _pearson(a_flat, b_flat),
        "cosine": _cosine(a_flat, b_flat),
        "global_ssim": _global_ssim(a_flat, b_flat),
        "peak_mask_iou": _peak_mask_iou(a, b),
        "peak_pixel_count_ratio": _peak_pixel_count_ratio(a, b),
        "intensity_quantile_distance": _quantile_distance(a_flat, b_flat),
    }


def _distribution_metrics(train_emb, query_emb, prefix, pca):
    return {
        "embedding_mmd_rbf": _mmd_rbf(train_emb, query_emb),
        "embedding_centroid_distance": float(np.linalg.norm(train_emb.mean(axis=0) - query_emb.mean(axis=0))),
        "metric_pca_components": int(pca.n_components_),
        "metric_pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    }


def _class_centroid_table(train_emb, y_train, synth_emb, y_synth):
    rows = []
    for label in sorted(set(y_train) & set(y_synth)):
        real = train_emb[y_train == label]
        synth = synth_emb[y_synth == label]
        if len(real) == 0 or len(synth) == 0:
            continue
        rows.append({
            "class": str(label),
            "real_samples": int(len(real)),
            "synthetic_samples": int(len(synth)),
            "centroid_distance": float(np.linalg.norm(real.mean(axis=0) - synth.mean(axis=0))),
            "real_within_class_radius": float(np.mean(np.linalg.norm(real - real.mean(axis=0), axis=1))),
            "synthetic_within_class_radius": float(np.mean(np.linalg.norm(synth - synth.mean(axis=0), axis=1))),
        })
    return pd.DataFrame(rows)


def _flatten(X):
    return X.reshape(len(X), -1)


def _sample_pixels(X):
    flat = np.asarray(X, dtype=np.float32).ravel()
    max_pixels = int(config.SPECTRAL_METRIC_MAX_PIXELS)
    if flat.size <= max_pixels:
        return flat
    indices = np.linspace(0, flat.size - 1, max_pixels, dtype=np.int64)
    return flat[indices]


def _pearson(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _cosine(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _global_ssim(a, b):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_a = float(np.mean(a))
    mu_b = float(np.mean(b))
    var_a = float(np.var(a))
    var_b = float(np.var(b))
    cov = float(np.mean((a - mu_a) * (b - mu_b)))
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(numerator / max(denominator, 1e-12))


def _peak_mask(X):
    threshold = max(
        float(np.percentile(X, config.SPECTRAL_METRIC_PEAK_PERCENTILE)),
        float(config.SPECTRAL_METRIC_PEAK_MIN_THRESHOLD),
    )
    return X >= threshold


def _peak_mask_iou(a, b):
    a_mask = _peak_mask(a)
    b_mask = _peak_mask(b)
    union = np.logical_or(a_mask, b_mask).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a_mask, b_mask).sum() / union)


def _peak_pixel_count_ratio(a, b):
    a_count = int(_peak_mask(a).sum())
    b_count = int(_peak_mask(b).sum())
    if b_count == 0:
        return float(a_count == 0)
    return float(a_count / b_count)


def _quantile_distance(a, b):
    qs = np.linspace(0.0, 1.0, 101)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def _mmd_rbf(a, b):
    combined = np.vstack([a, b])
    if len(combined) < 2:
        return 0.0

    sample = combined[: min(len(combined), 200)]
    diffs = sample[:, None, :] - sample[None, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    median_sq = float(np.median(sq_dists[sq_dists > 0])) if np.any(sq_dists > 0) else 1.0
    gamma = 1.0 / max(2.0 * median_sq, 1e-8)

    k_aa = _rbf_kernel_mean(a, a, gamma)
    k_bb = _rbf_kernel_mean(b, b, gamma)
    k_ab = _rbf_kernel_mean(a, b, gamma)
    return float(k_aa + k_bb - 2.0 * k_ab)


def _rbf_kernel_mean(a, b, gamma):
    diffs = a[:, None, :] - b[None, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    return float(np.mean(np.exp(-gamma * sq_dists)))


def _plot_pair_examples(pairs, path: Path):
    if not pairs:
        return

    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))
    if n == 1:
        axes = np.asarray([axes])

    for row, pair in enumerate(pairs):
        real = pair["reference"]
        synth = pair["query"]
        vmin = min(float(real.min()), float(synth.min()))
        vmax = max(float(real.max()), float(synth.max()))

        axes[row, 0].imshow(real, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"Nearest real: {pair['label']}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(synth, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title("Synthetic")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


def _write_markdown_table(table, path):
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
