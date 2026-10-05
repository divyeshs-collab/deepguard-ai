import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

class Settings:
    # Model Metadata & Versioning
    MODEL_NAME: str = "DeepGuard CNN Audio Classifier"
    MODEL_VERSION: str = "1.0.0"
    MODEL_FRAMEWORK: str = "PyTorch / TorchScript (CNN Architecture)"
    
    # Audio Configuration
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))
    WINDOW_SECONDS: float = float(os.getenv("WINDOW_SECONDS", "3.0"))
    HOP_SECONDS: float = float(os.getenv("HOP_SECONDS", "1.0"))
    
    # Spectrogram Features
    MEL_BANDS: int = int(os.getenv("MEL_BANDS", "128"))
    FFT_SIZE: int = int(os.getenv("FFT_SIZE", "1024"))
    HOP_LENGTH: int = int(os.getenv("HOP_LENGTH", "256"))
    WIN_LENGTH: int = int(os.getenv("WIN_LENGTH", "1024"))
    
    # Detection & Smoothing
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
    SUSPICIOUS_THRESHOLD: float = float(os.getenv("SUSPICIOUS_THRESHOLD", "0.40"))
    SMOOTHING_WINDOW: int = int(os.getenv("SMOOTHING_WINDOW", "5"))
    MIN_RMS_THRESHOLD: float = float(os.getenv("MIN_RMS_THRESHOLD", "0.005"))
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    ML_DIR: Path = BASE_DIR / "ml"
    MODEL_DIR: Path = ML_DIR / "model"
    EVAL_DIR: Path = ML_DIR / "evaluation"
    MODEL_WEIGHTS_PATH: Path = MODEL_DIR / "deepfake_audio_cnn.pt"
    MODEL_METADATA_PATH: Path = MODEL_DIR / "metadata.json"
    PREPROCESSING_CONFIG_PATH: Path = MODEL_DIR / "preprocessing.json"
    METRICS_PATH: Path = EVAL_DIR / "metrics.json"

    # Server Configuration
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1")

settings = Settings()
