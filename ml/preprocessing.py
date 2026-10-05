import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ml.config import (
    SAMPLE_RATE, WINDOW_SECONDS, HOP_SECONDS, WINDOW_SAMPLES,
    N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH, TIME_FRAMES,
    PREPROCESSING_CONFIG_PATH
)
from ml.audio_processor import AudioProcessor
from ml.feature_extractor import MelSpectrogramExtractor

class PreprocessingPipeline:
    """
    End-to-End audio preprocessing pipeline:
    Raw Audio Buffer -> Mono -> Resample 16kHz -> Norm -> Quality Check -> Log-Mel Spectrogram -> CNN Tensor.
    """

    def __init__(self, config_path: Optional[Path] = PREPROCESSING_CONFIG_PATH):
        self.config_path = config_path
        self.audio_processor = AudioProcessor(target_sample_rate=SAMPLE_RATE)
        self.feature_extractor = MelSpectrogramExtractor(
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            expected_frames=TIME_FRAMES
        )
        self.save_config()

    def get_config(self) -> Dict[str, Any]:
        """Returns preprocessing configuration dictionary."""
        return {
            "sample_rate": SAMPLE_RATE,
            "window_seconds": WINDOW_SECONDS,
            "hop_seconds": HOP_SECONDS,
            "window_samples": WINDOW_SAMPLES,
            "feature_representation": "Log-Mel Spectrogram",
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "win_length": WIN_LENGTH,
            "expected_shape": [1, 1, N_MELS, TIME_FRAMES],
            "silence_rms_threshold": 0.005,
            "normalization": "peak_and_per_window_standardization"
        }

    def save_config(self):
        """Saves preprocessing configuration to ml/model/preprocessing.json."""
        if self.config_path:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.get_config(), f, indent=2)

    def preprocess_window(self, raw_audio_data: Any) -> Tuple[Optional[torch.Tensor], bool, str, Dict[str, float]]:
        """
        Processes a single audio segment (expected ~3 seconds):
        Returns:
            tensor: torch.Tensor of shape (1, 1, 128, 188) or None if unusable
            is_usable: bool
            quality_status: 'GOOD', 'INSUFFICIENT AUDIO', or 'LOW AUDIO QUALITY — DETECTION LIMITED'
            metrics: dict of acoustic indicators
        """
        # 1. Process audio stream
        clean_audio, is_usable, status, metrics = self.audio_processor.process_raw_audio(raw_audio_data)

        if not is_usable:
            return None, False, status, metrics

        # 2. Pad or truncate to exact window length (48000 samples)
        window_audio = self.audio_processor.pad_or_truncate(clean_audio, WINDOW_SAMPLES)

        # 3. Extract Log-Mel Spectrogram (128, 188)
        spec = self.feature_extractor.extract_features(window_audio)

        # 4. Convert to 4D Tensor for CNN (1, 1, 128, 188)
        tensor = self.feature_extractor.to_tensor(spec)

        return tensor, True, "GOOD", metrics
