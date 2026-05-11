"""
Classification utilities for evaluating augmentation.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from autoencoder_pipeline.config import config

SEED = config.SEED


def classify_spectra(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    label_encoder: LabelEncoder,
    n_components: int | None = None,
    run_name: str = "classifier",
):
    """
    Baseline classifier to evaluate augmentation.
    Applies PCA → Random Forest.
    """

    # Flatten (M, N) → (M*N)
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat  = X_test.reshape(len(X_test),  -1)

    # PCA
    print("\nApplying PCA...")
    pca_full = PCA()
    pca_full.fit(X_train_flat)

    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    target_variance = config.PCA_COMPONENTS if n_components is None else n_components

    if isinstance(target_variance, float) and 0 < target_variance < 1:
        n_components = np.searchsorted(cumulative, target_variance) + 1
    else:
        n_components = int(target_variance)

    print(f"Using {n_components} PCA components")

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)

    print("Explained variance:", np.sum(pca.explained_variance_ratio_))

    # Random Forest
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(X_test_pca)

    # Metrics
    metrics = compute_classification_metrics(y_test, y_pred)
    print("\nAccuracy:", metrics["accuracy"])
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}/confusion_matrix_{safe_filename(run_name)}.png", dpi=300)
    plt.close()

    metrics.update({
        "pca_components": int(n_components),
        "pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    })

    return {
        **metrics,
        "predictions": y_pred,
        "confusion_matrix": cm,
        "pca": pca,
        "classifier": clf
    }


def compute_classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def safe_filename(name):
    return "".join(char if char.isalnum() else "_" for char in name).strip("_").lower()
