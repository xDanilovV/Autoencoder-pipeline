"""Synthetic latent and spectrum generation helpers."""
import numpy as np
from sklearn.preprocessing import LabelEncoder

from autoencoder_pipeline.config import config
from autoencoder_pipeline.visualization.utils import latent_stats_per_class, sample_latent_vectors


def generate_synthetic_latent_matrices(
    E: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder | None = None,
    multiplier: float | None = None,
):
    stats = latent_stats_per_class(E, y)
    multiplier = config.SYNTHETIC_MULTIPLIER if multiplier is None else multiplier

    E_synth_list = []
    y_synth_list = []

    for label, (mu, cov) in stats.items():
        n_label = int(np.sum(y == label))
        n_synth = max(1, int(np.ceil(n_label * multiplier)))

        if label_encoder is not None:
            class_name = label_encoder.inverse_transform([label])[0]
        else:
            class_name = str(label)

        print(f"Generating {n_synth} synthetic samples for class '{class_name}'")

        flat_synth = sample_latent_vectors(
            mu,
            cov,
            n_synth,
            method=config.SAMPLING_METHOD,
            std_scale=config.SYNTHETIC_STD_SCALE,
        )
        E_synth = flat_synth.reshape(n_synth, config.D, config.D)

        E_synth_list.append(E_synth)
        y_synth_list.extend([label] * n_synth)

    return np.concatenate(E_synth_list, axis=0), np.array(y_synth_list)
