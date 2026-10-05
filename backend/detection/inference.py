import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch

from backend.config import settings
from ml.audio_processor import AudioProcessor
from ml.feature_extractor import MelSpectrogramExtractor
from ml.model import DeepfakeAudioCNN
from ml.calibration import ModelCalibrator
from ml.explainability import GradCAMExplainer

class AsyncInferenceEngine:
    """
    Non-blocking inference worker.
    Runs CNN forward pass and Grad-CAM in a background thread pool
    so that high-throughput WebSocket audio streams are never stalled.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AudioProcessor(target_sample_rate=settings.SAMPLE_RATE)
        self.extractor = MelSpectrogramExtractor(sample_rate=settings.SAMPLE_RATE)
        self.calibrator = ModelCalibrator.load(settings.MODEL_DIR / "calibration.json")
        self.model = DeepfakeAudioCNN(in_channels=1).to(self.device)

        if settings.MODEL_WEIGHTS_PATH.exists():
            try:
                self.model.load_state_dict(torch.load(str(settings.MODEL_WEIGHTS_PATH), map_location=self.device))
                print(f"[+] AsyncInferenceEngine: Weights loaded from {settings.MODEL_WEIGHTS_PATH}")
            except Exception as e:
                print(f"[!] Warning: Failed loading weights: {e}")
        self.model.eval()

        self.explainer = GradCAMExplainer(self.model)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cnn-inference")

    def _sync_infer(self, audio_window: np.ndarray, compute_cam: bool = False) -> Dict[str, Any]:
        """Synchronous CPU/GPU inference routine executed in worker thread."""
        t_start = time.perf_counter()

        # 1. Quality check
        clean_audio = self.processor.normalize_amplitude(audio_window)
        is_usable, quality_status, metrics = self.processor.check_audio_quality(clean_audio)

        if not is_usable:
            return {
                "is_usable": False,
                "quality_status": quality_status,
                "raw_confidence": 0.0,
                "calibrated_confidence": 0.0,
                "latency_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
                "metrics": metrics,
                "explanation": None
            }

        # 2. Extract Log-Mel Spectrogram
        t_pre = time.perf_counter()
        spec = self.extractor.extract_features(clean_audio)
        preproc_ms = (time.perf_counter() - t_pre) * 1000.0

        # 3. Model forward pass
        tensor = self.extractor.to_tensor(spec).to(self.device)
        t_infer = time.perf_counter()

        explanation = None
        if compute_cam:
            heatmap, raw_score = self.explainer.generate_heatmap(tensor)
            explanation = self.explainer.analyze_influencing_regions(heatmap, spec)
        else:
            with torch.no_grad():
                raw_score = self.model(tensor).item()

        infer_ms = (time.perf_counter() - t_infer) * 1000.0
        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        # 4. Calibrate confidence
        calibrated_score = self.calibrator.calibrate(raw_score)

        return {
            "is_usable": True,
            "quality_status": "GOOD",
            "raw_confidence": round(float(raw_score), 4),
            "calibrated_confidence": round(float(calibrated_score), 4),
            "latency_ms": round(total_latency_ms, 2),
            "inference_latency_ms": round(infer_ms, 2),
            "audio_preproc_latency_ms": round(preproc_ms, 2),
            "metrics": metrics,
            "explanation": explanation
        }

    async def infer_window_async(self, audio_window: np.ndarray, compute_cam: bool = False) -> Dict[str, Any]:
        """Asynchronously executes inference in thread pool without blocking event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._sync_infer, audio_window, compute_cam)
