import numpy as np
from typing import Optional

class AudioAugmentation:
    """
    Training-time audio and spectrogram augmentations:
    - Additive background noise
    - Random gain
    - Time shifting
    - Time and frequency masking (SpecAugment)
    - Codec / compression simulation
    Applied ONLY during training, NEVER during validation or testing.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def add_background_noise(self, audio: np.ndarray, snr_range: tuple = (10, 30)) -> np.ndarray:
        """Adds synthetic Gaussian/pink noise within target SNR range."""
        snr = np.random.uniform(snr_range[0], snr_range[1])
        signal_power = np.mean(audio ** 2) + 1e-9
        noise_power = signal_power / (10.0 ** (snr / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), size=len(audio)).astype(np.float32)
        return audio + noise

    def random_gain(self, audio: np.ndarray, min_gain: float = 0.7, max_gain: float = 1.3) -> np.ndarray:
        """Scales amplitude by random gain factor."""
        gain = np.random.uniform(min_gain, max_gain)
        return audio * gain

    def time_shift(self, audio: np.ndarray, max_shift_ratio: float = 0.2) -> np.ndarray:
        """Circularly shifts audio along time axis."""
        shift_samples = int(np.random.uniform(-max_shift_ratio, max_shift_ratio) * len(audio))
        return np.roll(audio, shift_samples)

    def simulate_telephony_compression(self, audio: np.ndarray) -> np.ndarray:
        """
        Simulates 8-bit A-law / mu-law quantization effects common in telephony PSTN networks.
        """
        mu = 255.0
        # mu-law compression
        companded = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)
        # 8-bit quantization
        quantized = np.round(companded * 128.0) / 128.0
        # Inverse mu-law expansion
        expanded = np.sign(quantized) * ((1.0 + mu) ** np.abs(quantized) - 1.0) / mu
        return expanded.astype(np.float32)

    def spec_augment(
        self,
        spectrogram: np.ndarray,
        freq_mask_max: int = 16,
        time_mask_max: int = 24,
        num_freq_masks: int = 2,
        num_time_masks: int = 2
    ) -> np.ndarray:
        """
        SpecAugment: Frequency and Time masking applied directly to Log-Mel Spectrogram.
        spectrogram shape: (n_mels, time_frames) e.g. (128, 188)
        """
        augmented = spectrogram.copy()
        n_mels, time_frames = augmented.shape
        mean_val = np.mean(augmented)

        # Frequency masking
        for _ in range(num_freq_masks):
            f_len = np.random.randint(0, freq_mask_max)
            f_start = np.random.randint(0, max(1, n_mels - f_len))
            augmented[f_start:f_start + f_len, :] = mean_val

        # Time masking
        for _ in range(num_time_masks):
            t_len = np.random.randint(0, time_mask_max)
            t_start = np.random.randint(0, max(1, time_frames - t_len))
            augmented[:, t_start:t_start + t_len] = mean_val

        return augmented

    def apply_training_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """Combines time-domain augmentations stochastically."""
        aug_audio = audio.copy()
        if np.random.rand() > 0.4:
            aug_audio = self.random_gain(aug_audio)
        if np.random.rand() > 0.5:
            aug_audio = self.time_shift(aug_audio)
        if np.random.rand() > 0.4:
            aug_audio = self.add_background_noise(aug_audio)
        if np.random.rand() > 0.6:
            aug_audio = self.simulate_telephony_compression(aug_audio)
        return aug_audio
