import asyncio
import io
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional
import numpy as np
import soundfile as sf

from backend.config import settings
from backend.telephony.base import TelephonyProvider
from ml.audio_processor import AudioProcessor

class DemoAudioStreamProvider(TelephonyProvider):
    """
    Simulated Telephony Provider for developer and investor demonstrations.
    Reads an authentic human conversation or synthetic deepfake voice call WAV file
    and streams it in real time (e.g. 500ms chunks) over the WebSocket pipeline,
    exactly mirroring a live carrier telephony stream.
    """

    def __init__(self, demo_dir: Optional[Path] = None):
        self.demo_dir = demo_dir or (settings.ML_DIR / "dataset" / "demo_calls")
        self.processor = AudioProcessor(target_sample_rate=settings.SAMPLE_RATE)
        self.active_streams: Dict[str, Dict[str, Any]] = {}

    def get_provider_name(self) -> str:
        return "DeepGuard Telephony Simulator (Carrier Stream)"

    def list_available_scenarios(self) -> list:
        """Returns catalog of pre-configured demo call scenarios."""
        return [
            {
                "id": "human_authentic_call",
                "title": "Scenario 1: Authentic Human Executive",
                "description": "Natural vocal dynamics, conversational micro-prosody, uncompressed vocal tract harmonics.",
                "expected_classification": "LIKELY HUMAN",
                "file": "human_authentic_call.wav"
            },
            {
                "id": "deepfake_cloned_call",
                "title": "Scenario 2: Cloned CEO Voice Attack",
                "description": "AI neural vocoder clone with high-frequency aliasing and sub-band phase discontinuities.",
                "expected_classification": "HIGH CONFIDENCE SYNTHETIC",
                "file": "deepfake_cloned_call.wav"
            },
            {
                "id": "phishing_synthetic_call",
                "title": "Scenario 3: Synthetic Phishing Bot",
                "description": "Autoregressive speech generator with rigid pitch and metallic comb-filtered harmonics.",
                "expected_classification": "HIGH CONFIDENCE SYNTHETIC",
                "file": "phishing_synthetic_call.wav"
            },
            {
                "id": "ambient_silence",
                "title": "Scenario 4: Silence / Low Energy Line",
                "description": "Silent line testing gating system behavior (triggers 'INSUFFICIENT AUDIO').",
                "expected_classification": "INSUFFICIENT AUDIO",
                "file": "ambient_silence.wav"
            }
        ]

    async def start_call_stream(self, call_id: str, options: Optional[Dict[str, Any]] = None) -> bool:
        options = options or {}
        scenario_id = options.get("scenario", "deepfake_cloned_call")
        custom_filepath = options.get("custom_filepath")

        if custom_filepath and Path(custom_filepath).exists():
            file_to_load = Path(custom_filepath)
        else:
            scenario_meta = next((s for s in self.list_available_scenarios() if s["id"] == scenario_id), None)
            target_name = scenario_meta["file"] if scenario_meta else "deepfake_cloned_call.wav"
            file_to_load = self.demo_dir / target_name

        if not file_to_load.exists():
            # If demo files don't exist yet, generate them dynamically
            from ml.dataset import DatasetGenerator
            generator = DatasetGenerator(settings.ML_DIR / "dataset", settings.SAMPLE_RATE)
            generator.build_dataset()

        # Load and resample audio
        data, sr = sf.read(str(file_to_load), dtype="float32")
        mono = self.processor.to_mono(data)
        resampled = self.processor.resample(mono, sr)

        self.active_streams[call_id] = {
            "audio": resampled,
            "sample_rate": settings.SAMPLE_RATE,
            "cursor": 0,
            "is_running": True,
            "scenario": scenario_id
        }
        return True

    async def stop_call_stream(self, call_id: str) -> bool:
        if call_id in self.active_streams:
            self.active_streams[call_id]["is_running"] = False
            del self.active_streams[call_id]
            return True
        return False

    async def stream_audio_chunks(self, call_id: str, chunk_duration_sec: float = 0.5) -> AsyncGenerator[np.ndarray, None]:
        """Streams audio chunks in synchronized real-time paces."""
        session = self.active_streams.get(call_id)
        if not session:
            return

        audio = session["audio"]
        sr = session["sample_rate"]
        chunk_size = int(chunk_duration_sec * sr)
        total_len = len(audio)

        cursor = 0
        while session.get("is_running", False):
            if cursor >= total_len:
                # Loop back or complete
                cursor = 0

            chunk = audio[cursor:cursor + chunk_size]
            cursor += chunk_size

            # If chunk is shorter than requested, pad
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')

            yield chunk
            # Yield at realistic real-time telephony pacing
            await asyncio.sleep(chunk_duration_sec)
