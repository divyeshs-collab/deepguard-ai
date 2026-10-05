import io
import math
import numpy as np
import soundfile as sf
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ml.config import (
    SAMPLE_RATE, WINDOW_SAMPLES, HOP_SAMPLES, MIN_RMS_THRESHOLD
)

class AudioProcessor:
    """
    Robust audio preprocessing engine designed for live telephony call streams.
    Handles mono conversion, resampling to 16kHz, amplitude normalization,
    silence/low-energy gating, and window extraction.
    """

    def __init__(self, target_sample_rate: int = SAMPLE_RATE):
        self.target_sample_rate = target_sample_rate

    def load_audio(self, audio_source: Any) -> Tuple[np.ndarray, int]:
        """
        Loads audio from bytes, file-like object, path, or existing numpy array.
        Returns: (audio_array: np.ndarray, original_sample_rate: int)
        """
        if isinstance(audio_source, np.ndarray):
            return audio_source.astype(np.float32), self.target_sample_rate

        if isinstance(audio_source, (bytes, bytearray)):
            try:
                # Attempt decoding standard audio container (WAV/FLAC/OGG)
                with io.BytesIO(audio_source) as bio:
                    data, sr = sf.read(bio, dtype="float32")
                    return data, sr
            except Exception:
                # Fallback: Assume raw 16-bit PCM mono 16kHz stream from WebSocket
                pcm_data = np.frombuffer(audio_source, dtype=np.int16)
                float_data = pcm_data.astype(np.float32) / 32768.0
                return float_data, self.target_sample_rate

        if isinstance(audio_source, str):
            data, sr = sf.read(audio_source, dtype="float32")
            return data, sr

        raise ValueError(f"Unsupported audio source type: {type(audio_source)}")

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Converts multi-channel audio to single-channel mono."""
        if audio.ndim == 1:
            return audio
        elif audio.ndim == 2:
            # Average across channels
            return np.mean(audio, axis=1)
        else:
            raise ValueError(f"Invalid audio dimensions: {audio.shape}")

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """
        Resamples audio array to target sample rate using linear interpolation or sinc-based resampler.
        """
        if orig_sr == self.target_sample_rate or len(audio) == 0:
            return audio

        num_output_samples = int(round(len(audio) * float(self.target_sample_rate) / orig_sr))
        resampled = np.interp(
            np.linspace(0, len(audio), num_output_samples, endpoint=False),
            np.arange(len(audio)),
            audio
        ).astype(np.float32)
        return resampled

    def normalize_amplitude(self, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Normalizes audio peak amplitude while avoiding division by zero."""
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            return (audio / peak) * target_peak
        return audio

    def compute_audio_metrics(self, audio: np.ndarray) -> Dict[str, float]:
        """Calculates RMS energy, Peak amplitude, and Zero-Crossing Rate."""
        if len(audio) == 0:
            return {"rms": 0.0, "peak": 0.0, "zcr": 0.0, "snr_db": -99.0}

        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0)
        
        # Estimate coarse SNR (Signal vs estimated floor)
        sorted_energies = np.sort(np.abs(audio))
        noise_floor = np.mean(sorted_energies[:max(1, int(len(audio) * 0.10))]) + 1e-7
        snr_db = float(20 * np.log10((peak + 1e-7) / noise_floor))

        return {
            "rms": rms,
            "peak": peak,
            "zcr": zcr,
            "snr_db": snr_db
        }

    def check_audio_quality(self, audio: np.ndarray) -> Tuple[bool, str, Dict[str, float]]:
        """
        Evaluates whether the given audio slice contains sufficient usable speech
        or is too silent / degraded.
        Returns: (is_usable: bool, status_code: str, metrics: Dict[str, float])
        """
        metrics = self.compute_audio_metrics(audio)

        # 1. Check for silence or extremely low energy
        if metrics["rms"] < MIN_RMS_THRESHOLD:
            return False, "INSUFFICIENT AUDIO", metrics

        # 2. Check for severe clipping distortion
        clipped_ratio = np.mean(np.abs(audio) >= 0.999)
        if clipped_ratio > 0.15:
            return False, "LOW AUDIO QUALITY — DETECTION LIMITED", metrics

        # 3. Check for pure DC offset or flatline
        if np.std(audio) < 1e-4:
            return False, "INSUFFICIENT AUDIO", metrics

        return True, "GOOD", metrics

    def pad_or_truncate(self, audio: np.ndarray, target_length: int = WINDOW_SAMPLES) -> np.ndarray:
        """Pads with zeros or truncates to the exact target length."""
        if len(audio) == target_length:
            return audio
        elif len(audio) < target_length:
            pad_width = target_length - len(audio)
            return np.pad(audio, (0, pad_width), mode='constant', constant_values=0.0)
        else:
            return audio[:target_length]

    def extract_rolling_windows(self, audio: np.ndarray, window_size: int = WINDOW_SAMPLES, hop_size: int = HOP_SAMPLES) -> List[np.ndarray]:
        """
        Splits a continuous stream into overlapping windows.
        For 16kHz: 3.0s window = 48000 samples, 1.0s hop = 16000 samples.
        """
        windows = []
        if len(audio) < window_size:
            # If shorter than full window, return padded window if it has sufficient signal
            windows.append(self.pad_or_truncate(audio, window_size))
            return windows

        for start in range(0, len(audio) - window_size + 1, hop_size):
            end = start + window_size
            windows.append(audio[start:end])

        return windows

    def process_raw_audio(self, raw_audio: Any) -> Tuple[np.ndarray, bool, str, Dict[str, float]]:
        """
        Full preprocessing pipeline for incoming audio:
        Raw Audio -> Mono -> Resample 16kHz -> Amplitude Normalization -> Quality Check.
        """
        data, orig_sr = self.load_audio(raw_audio)
        mono_data = self.to_mono(data)
        resampled_data = self.resample(mono_data, orig_sr)
        norm_data = self.normalize_amplitude(resampled_data)
        is_usable, status, metrics = self.check_audio_quality(norm_data)

        return norm_data, is_usable, status, metrics
