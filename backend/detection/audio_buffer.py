import threading
import numpy as np
from typing import Optional, List, Tuple

class AudioStreamBuffer:
    """
    Thread-safe rolling audio buffer designed for live telephony streams.
    Accumulates incoming PCM audio chunks (e.g. 100ms-500ms).
    Emits 3.0s (48,000 samples @ 16kHz) windows with a 1.0s (16,000 samples) hop size.
    """

    def __init__(self, sample_rate: int = 16000, window_seconds: float = 3.0, hop_seconds: float = 1.0):
        self.sample_rate = sample_rate
        self.window_size = int(sample_rate * window_seconds) # 48,000
        self.hop_size = int(sample_rate * hop_seconds)       # 16,000
        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self.total_samples_received = 0
        self.windows_emitted = 0

    def append(self, audio_chunk: np.ndarray):
        """Appends new audio samples to the rolling buffer."""
        if audio_chunk is None or len(audio_chunk) == 0:
            return

        # Ensure float32 mono
        if audio_chunk.ndim > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
        audio_chunk = audio_chunk.astype(np.float32)

        with self._lock:
            self._buffer = np.concatenate([self._buffer, audio_chunk])
            self.total_samples_received += len(audio_chunk)

    def get_next_window(self) -> Optional[np.ndarray]:
        """
        Extracts the next 3.0s window if sufficient audio has accumulated.
        Advances buffer by hop_size (1.0s) to provide overlapping windows.
        """
        with self._lock:
            if len(self._buffer) < self.window_size:
                return None

            window = self._buffer[:self.window_size].copy()
            # Advance buffer by hop_size
            self._buffer = self._buffer[self.hop_size:]
            self.windows_emitted += 1
            return window

    def has_sufficient_audio(self) -> bool:
        """Checks if buffer has reached at least one full window."""
        with self._lock:
            return len(self._buffer) >= self.window_size

    def buffered_duration(self) -> float:
        """Returns currently buffered audio duration in seconds."""
        with self._lock:
            return len(self._buffer) / self.sample_rate

    def total_streamed_duration(self) -> float:
        """Returns total audio duration received in seconds."""
        return self.total_samples_received / self.sample_rate

    def clear(self):
        """Resets the buffer."""
        with self._lock:
            self._buffer = np.zeros(0, dtype=np.float32)
            self.total_samples_received = 0
            self.windows_emitted = 0
