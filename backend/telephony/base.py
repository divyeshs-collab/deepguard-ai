from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional
import numpy as np

class TelephonyProvider(ABC):
    """
    Abstract interface for telephony audio stream providers.
    Supports modular integration with Twilio Media Streams, SIP RTP trunking,
    Telnyx, AudioCodes, or Mock/Demo streams.
    """

    @abstractmethod
    def get_provider_name(self) -> str:
        """Returns the human-readable provider name."""
        pass

    @abstractmethod
    async def start_call_stream(self, call_id: str, options: Optional[Dict[str, Any]] = None) -> bool:
        """Initializes a media stream for the given call ID."""
        pass

    @abstractmethod
    async def stop_call_stream(self, call_id: str) -> bool:
        """Terminates active audio stream for the call."""
        pass

    @abstractmethod
    async def stream_audio_chunks(self, call_id: str, chunk_duration_sec: float = 0.5) -> AsyncGenerator[np.ndarray, None]:
        """
        Asynchronous generator emitting live audio chunks as 1D float32 numpy arrays @ 16kHz.
        """
        pass

class AudioStreamAdapter:
    """
    Standardizes disparate telephony audio formats (mu-law, 8kHz PCM, Opus, stereo)
    into unified 16kHz mono float32 chunks for the Detection Engine.
    """

    @staticmethod
    def decode_mulaw_to_float(mulaw_bytes: bytes) -> np.ndarray:
        """Decodes standard 8-bit G.711 mu-law telephony payload."""
        import audioop
        # 8-bit mu-law to 16-bit linear PCM
        linear_pcm = audioop.ulaw2lin(mulaw_bytes, 2)
        # Convert to float32
        data_int16 = np.frombuffer(linear_pcm, dtype=np.int16)
        return data_int16.astype(np.float32) / 32768.0

    @staticmethod
    def convert_pcm_bytes_to_float(pcm_bytes: bytes, dtype: np.dtype = np.int16) -> np.ndarray:
        """Decodes raw 16-bit linear PCM bytes."""
        data_int = np.frombuffer(pcm_bytes, dtype=dtype)
        return data_int.astype(np.float32) / 32768.0
