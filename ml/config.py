import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "ml"
MODEL_DIR = ML_DIR / "model"
EVAL_DIR = ML_DIR / "evaluation"
DATASET_DIR = ML_DIR / "dataset"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

MODEL_WEIGHTS_PATH = MODEL_DIR / "deepfake_audio_cnn.pt"
MODEL_METADATA_PATH = MODEL_DIR / "metadata.json"
PREPROCESSING_CONFIG_PATH = MODEL_DIR / "preprocessing.json"
METRICS_PATH = EVAL_DIR / "metrics.json"

# Audio Parameters
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
WINDOW_SECONDS = float(os.getenv("WINDOW_SECONDS", "3.0"))
HOP_SECONDS = float(os.getenv("HOP_SECONDS", "1.0"))
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)  # 48000 samples
HOP_SAMPLES = int(SAMPLE_RATE * HOP_SECONDS)        # 16000 samples

# Feature Extraction Parameters (Log-Mel Spectrogram)
N_MELS = int(os.getenv("MEL_BANDS", "128"))
N_FFT = int(os.getenv("FFT_SIZE", "1024"))
HOP_LENGTH = int(os.getenv("HOP_LENGTH", "256"))
WIN_LENGTH = int(os.getenv("WIN_LENGTH", "1024"))
F_MIN = float(os.getenv("F_MIN", "0.0"))
F_MAX = float(os.getenv("F_MAX", "8000.0"))

# Expected Spectrogram Dimensions
# Time frames = ceil(48000 / 256) ≈ 188
TIME_FRAMES = 1 + (WINDOW_SAMPLES // HOP_LENGTH)  # 188 frames

# Training Parameters
EPOCHS = int(os.getenv("EPOCHS", "50"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.0001"))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", "0.20"))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "7"))
LR_REDUCTION_PATIENCE = int(os.getenv("LR_REDUCTION_PATIENCE", "3"))
LR_REDUCTION_FACTOR = float(os.getenv("LR_REDUCTION_FACTOR", "0.5"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# Calibration & Decision
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
SUSPICIOUS_THRESHOLD = float(os.getenv("SUSPICIOUS_THRESHOLD", "0.40"))
SMOOTHING_WINDOW = int(os.getenv("SMOOTHING_WINDOW", "5"))
MIN_RMS_THRESHOLD = float(os.getenv("MIN_RMS_THRESHOLD", "0.005"))
