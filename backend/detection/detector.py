import time
from typing import Dict, Any, Optional, List
import numpy as np

from backend.config import settings
from backend.detection.audio_buffer import AudioStreamBuffer
from backend.detection.inference import AsyncInferenceEngine
from backend.detection.smoothing import TemporalSmoother

class LiveCallDetector:
    """
    Main Call Deepfake Detection Coordinator.
    Coordinates the real-time pipeline:
    Audio Chunks -> Buffer -> 3s Window -> Async CNN -> Calibrator -> Temporal Smoother -> Event Payload.
    """

    def __init__(self, call_id: str):
        self.call_id = call_id
        self.start_time = time.time()
        self.buffer = AudioStreamBuffer(
            sample_rate=settings.SAMPLE_RATE,
            window_seconds=settings.WINDOW_SECONDS,
            hop_seconds=settings.HOP_SECONDS
        )
        self.smoother = TemporalSmoother(
            window_size=settings.SMOOTHING_WINDOW,
            high_confidence_threshold=settings.CONFIDENCE_THRESHOLD,
            suspicious_threshold=settings.SUSPICIOUS_THRESHOLD
        )
        self.inference_engine = AsyncInferenceEngine()
        self.is_active = True
        self.window_counter = 0
        self.recent_predictions: List[Dict[str, Any]] = []

    def reset(self):
        """Resets detector state."""
        self.start_time = time.time()
        self.buffer.clear()
        self.smoother.reset()
        self.window_counter = 0
        self.recent_predictions.clear()
        self.is_active = True

    def get_elapsed_seconds(self) -> float:
        """Returns elapsed call duration in seconds."""
        return round(time.time() - self.start_time, 1)

    async def ingest_audio_chunk(self, audio_chunk: np.ndarray, request_explanation: bool = False) -> List[Dict[str, Any]]:
        """
        Ingests incoming streaming audio chunk.
        If a new 3.0-second window is ready, runs async inference and temporal smoothing.
        Returns a list of detection updates (usually 0 or 1 per hop).
        """
        self.buffer.append(audio_chunk)
        updates = []

        # Process any complete windows available
        while True:
            window = self.buffer.get_next_window()
            if window is None:
                break

            self.window_counter += 1
            # Request Grad-CAM explanation every 3rd window or when explicitly requested
            need_cam = request_explanation or (self.window_counter % 3 == 1)

            # Asynchronous CNN Inference
            infer_result = await self.inference_engine.infer_window_async(window, compute_cam=need_cam)

            # Temporal Smoothing
            is_usable = infer_result["is_usable"]
            quality_status = infer_result["quality_status"]
            calibrated_conf = infer_result["calibrated_confidence"]

            smooth_res = self.smoother.update(
                raw_confidence=calibrated_conf,
                is_usable=is_usable,
                quality_status=quality_status
            )

            # Build detection update event
            elapsed = self.get_elapsed_seconds()
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins:02d}:{secs:02d}"

            payload = {
                "timestamp": int(time.time()),
                "call_id": self.call_id,
                "elapsed_time": time_str,
                "window_index": self.window_counter,
                "classification": smooth_res["classification"],
                "risk_level": smooth_res["risk_level"],
                "model_confidence": round(float(calibrated_conf), 4),
                "smoothed_confidence": smooth_res["smoothed_confidence"],
                "window_duration": settings.WINDOW_SECONDS,
                "inference_latency_ms": infer_result.get("inference_latency_ms", 0.0),
                "total_latency_ms": infer_result.get("latency_ms", 0.0),
                "speech_detected": is_usable,
                "quality_status": quality_status,
                "state_description": smooth_res["state_description"],
                "explanation": infer_result.get("explanation"),
                "acoustic_metrics": infer_result.get("metrics")
            }

            self.recent_predictions.append(payload)
            if len(self.recent_predictions) > 50:
                self.recent_predictions.pop(0)

            updates.append(payload)

        return updates
