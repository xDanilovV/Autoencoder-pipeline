"""Configuration for the GC-IMS augmentation pipeline."""
import torch
from pathlib import Path


class Config:
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    DATA_PATH = Path("data_fermentation")
    SELECTED_CLASSES = None
    RESULTS_PATH = Path("results")
    MODEL_PATH = Path("models")  # For saving trained models

    # Create directories if they don't exist
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)
    MODEL_PATH.mkdir(exist_ok=True, parents=True)

    # Data split
    SEED = 137
    TRAIN_SPLIT = 0.85
    M = None  # Drift time dimension (rows)
    N = None  # Retention time dimension (columns)
    D = 32  # Paper default; larger values make latent covariance unstable
    RIP_CROP_ROWS = 10
    ALIGN_RIP = True
    CUT_RIP = True
    RIP_SEARCH_FRACTION = (0.25, 0.55)
    RIP_CUT_HALF_WIDTH = 28
    ROI_INTENSITY_THRESHOLD = 0.04
    ROI_BACKGROUND_PERCENTILE = 60
    ROI_PROFILE_SMOOTH = 31
    ROI_MARGIN = 64
    ROI_MIN_SIZE = 128
    MAX_MODEL_ROWS = 1024
    MAX_MODEL_COLS = 512
    NORMALIZATION_METHOD = "log"

    NHEAD = 4  # Number of attention heads in transformer
    NUM_LAYERS = 2  # Number of transformer layers
    DIM_FEEDFORWARD = 128  # Feedforward dimension in transformer
    DROPOUT = 0.1  # Dropout rate

    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 150  # Increased to match paper (was 50)
    LEARNING_RATE = 0.0005
    PATIENCE = 20  # Increased patience for early stopping (was 5)
    PEAK_LOSS_WEIGHT = 3.0

    # Encoding batch size (for faster encoding in encoder.py)
    ENCODING_BATCH_SIZE = 64  # Process this many sequences at once during encoding

    # Undersampling
    UNDERSAMPLE_PROB = 0.25  # Keep 25% of low-variance timeseries
    UNDERSAMPLE_VALIDATION = True

    # Sampling method for synthetic generation
    # Options: 'shrinkage', 'diagonal', 'eigenvalue', 'svd'
    SAMPLING_METHOD = "shrinkage"
    PCA_COMPONENTS = 0.99
    SYNTHETIC_MULTIPLIER = 1.0  # 1.0 doubles each class, matching the paper
    SYNTHETIC_STD_SCALE = 0.75  # Temper latent variability to avoid noisy samples

    # Post-processing
    MATCH_SYNTHETIC_DISTRIBUTION = True  # Adjust synthetic to match real distribution


config = Config()


# Print configuration on import
def print_config():
    print("\n" + "="*60)
    print("Configuration Summary")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    print(f"Data path: {config.DATA_PATH}")
    print(f"Results path: {config.RESULTS_PATH}")
    print(f"\nArchitecture:")
    print(f"  Latent dimension (D): {config.D}")
    print(f"  Transformer heads: {config.NHEAD}")
    print(f"  Transformer layers: {config.NUM_LAYERS}")
    print(f"  Feedforward dim: {config.DIM_FEEDFORWARD}")
    print(f"\nTraining:")
    print(f"  Seed: {config.SEED}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Max epochs: {config.NUM_EPOCHS}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Early stopping patience: {config.PATIENCE}")
    print(f"  Peak loss weight: {config.PEAK_LOSS_WEIGHT}")
    print(f"\nPreprocessing:")
    print(f"  Normalization: {config.NORMALIZATION_METHOD}")
    print(f"  Align RIP: {config.ALIGN_RIP}")
    print(f"  Cut RIP: {config.CUT_RIP}")
    print(f"  ROI threshold: {config.ROI_INTENSITY_THRESHOLD}")
    print(f"  ROI margin: {config.ROI_MARGIN}")
    print(f"  Max model shape: ({config.MAX_MODEL_ROWS}, {config.MAX_MODEL_COLS})")
    print(f"\nSampling:")
    print(f"  Method: {config.SAMPLING_METHOD}")
    print(f"  Match distribution: {config.MATCH_SYNTHETIC_DISTRIBUTION}")
    print("="*60 + "\n")


if __name__ == "__main__":
    print_config()
