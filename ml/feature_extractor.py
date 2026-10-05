import numpy as np
from typing import Optional, Tuple
import torch
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ml.config import (
    SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH, F_MIN, F_MAX, TIME_FRAMES
)

class MelSpectrogramExtractor:
    """
    Log-Mel Spectrogram feature extractor matching exact specification:
    - Sample Rate: 16 kHz
    - Number of Mel Bands: 128
    - FFT Size: 1024
    - Hop Length: 256
    - Window Length: 1024
    Outputs image-like tensor suitable for CNN ingestion.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
        expected_frames: int = TIME_FRAMES
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2.0
        self.expected_frames = expected_frames

        # Precompute Hann window
        self.window = np.hanning(self.win_length).astype(np.float32)

        # Precompute Mel Filterbank matrix (n_mels, 1 + n_fft // 2)
        self.mel_basis = self._create_mel_filterbank()

    def _hz_to_mel(self, hz: np.ndarray) -> np.ndarray:
        """Converts frequency in Hz to Mel scale."""
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: np.ndarray) -> np.ndarray:
        """Converts Mel scale value to frequency in Hz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _create_mel_filterbank(self) -> np.ndarray:
        """Constructs triangular Mel filterbank matrix."""
        num_fft_bins = 1 + self.n_fft // 2
        fft_freqs = np.linspace(0, self.sample_rate / 2.0, num_fft_bins)

        mel_min = self._hz_to_mel(np.array(self.f_min))
        mel_max = self._hz_to_mel(np.array(self.f_max))
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        weights = np.zeros((self.n_mels, num_fft_bins), dtype=np.float32)
        for m in range(self.n_mels):
            f_left = hz_points[m]
            f_center = hz_points[m + 1]
            f_right = hz_points[m + 2]

            # Up-slope
            up = (fft_freqs - f_left) / (f_center - f_left + 1e-8)
            # Down-slope
            down = (f_right - fft_freqs) / (f_right - f_center + 1e-8)

            triangular = np.maximum(0.0, np.minimum(up, down))
            
            # Slaney area normalization
            enorm = 2.0 / (f_right - f_left + 1e-8)
            weights[m, :] = triangular * enorm

        return weights

    def compute_stft(self, audio: np.ndarray) -> np.ndarray:
        """
        Computes Short-Time Fourier Transform power spectrum.
        audio: 1D numpy array of length N
        Returns: magnitude squared spectrogram (1 + n_fft // 2, time_frames)
        """
        if len(audio) < self.win_length:
            audio = np.pad(audio, (0, self.win_length - len(audio)), mode='constant')

        num_frames = 1 + (len(audio) - self.win_length) // self.hop_length
        num_bins = 1 + self.n_fft // 2
        stft_matrix = np.empty((num_bins, num_frames), dtype=np.complex64)

        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.win_length
            segment = audio[start:end] * self.window
            spectrum = np.fft.rfft(segment, n=self.n_fft)
            stft_matrix[:, i] = spectrum

        # Return power spectrogram (|X|^2)
        power_spec = np.abs(stft_matrix) ** 2
        return power_spec

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts Log-Mel Spectrogram from audio array.
        Returns: 2D numpy array of shape (n_mels, time_frames) e.g. (128, 188)
        """
        power_spec = self.compute_stft(audio)
        
        # Apply Mel Filterbank: (128, num_bins) @ (num_bins, frames) -> (128, frames)
        mel_spec = np.dot(self.mel_basis, power_spec)

        # Log Scaling (dB scale representation: 10 * log10(max(mel, 1e-10)))
        log_mel_spec = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))

        # Standardize / Normalize to zero mean, unit variance per window for consistent CNN input
        mean = np.mean(log_mel_spec)
        std = np.std(log_mel_spec) + 1e-6
        norm_mel_spec = (log_mel_spec - mean) / std

        # Ensure exact frame width
        curr_frames = norm_mel_spec.shape[1]
        if curr_frames < self.expected_frames:
            pad_width = self.expected_frames - curr_frames
            norm_mel_spec = np.pad(norm_mel_spec, ((0, 0), (0, pad_width)), mode='edge')
        elif curr_frames > self.expected_frames:
            norm_mel_spec = norm_mel_spec[:, :self.expected_frames]

        return norm_mel_spec.astype(np.float32)

    def to_tensor(self, log_mel_spec: np.ndarray) -> torch.Tensor:
        """
        Converts 2D (128, frames) numpy array to 4D CNN tensor:
        (batch_size=1, channels=1, height=128, width=188)
        """
        tensor = torch.from_numpy(log_mel_spec).float()
        return tensor.unsqueeze(0).unsqueeze(0)
