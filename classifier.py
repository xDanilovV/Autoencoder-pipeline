"""
Classification utilities for evaluating augmentation.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from config import config

SEED = 42


def classify_spectra(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    label_encoder: LabelEncoder,
    n_components: int | None = None
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
    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", acc)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

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
    plt.savefig(f"{config.RESULTS_PATH}/confusion_matrix.png", dpi=300)
    plt.show()

    return {
        "accuracy": acc,
        "predictions": y_pred,
        "confusion_matrix": cm,
        "pca": pca,
        "classifier": clf
    }
