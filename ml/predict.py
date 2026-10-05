import sys
import json
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import torch

# Ensure project root is importable
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ml.config import (
    MODEL_WEIGHTS_PATH, MODEL_DIR, CONFIDENCE_THRESHOLD, SUSPICIOUS_THRESHOLD
)
from ml.preprocessing import PreprocessingPipeline
from ml.model import DeepfakeAudioCNN
from ml.calibration import ModelCalibrator
from ml.explainability import GradCAMExplainer

class DeepfakeAudioPredictor:
    """Production Predictor for Deepfake Audio Detection with Grad-CAM Explainability."""

    def __init__(self, weights_path: Path = MODEL_WEIGHTS_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = PreprocessingPipeline()
        self.model = DeepfakeAudioCNN(in_channels=1).to(self.device)
        self.calibrator = ModelCalibrator.load(MODEL_DIR / "calibration.json")

        if weights_path.exists():
            self.model.load_state_dict(torch.load(str(weights_path), map_location=self.device))
        self.model.eval()

        self.explainer = GradCAMExplainer(self.model)

    def predict_window(self, raw_audio: Any, include_explainability: bool = True) -> Dict[str, Any]:
        """
        Runs inference on a single audio window.
        Returns:
            dict containing classification, model_confidence, quality_status, and acoustic analysis.
        """
        # 1. Preprocessing & Quality Gating
        tensor, is_usable, status, metrics = self.pipeline.preprocess_window(raw_audio)

        if not is_usable or tensor is None:
            return {
                "classification": status, # "INSUFFICIENT AUDIO" or "LOW AUDIO QUALITY — DETECTION LIMITED"
                "model_confidence": 0.0,
                "raw_score": 0.0,
                "is_usable": False,
                "status_code": status,
                "quality_metrics": metrics,
                "explanation": None
            }

        tensor = tensor.to(self.device)

        # 2. Grad-CAM & Inference
        if include_explainability:
            heatmap, raw_score = self.explainer.generate_heatmap(tensor)
            log_mel_np = tensor.squeeze().detach().cpu().numpy()
            explanation = self.explainer.analyze_influencing_regions(heatmap, log_mel_np)
        else:
            with torch.no_grad():
                raw_score = self.model(tensor).item()
            explanation = None

        # 3. Model Calibration
        calibrated_score = self.calibrator.calibrate(raw_score)

        # 4. State classification for single window
        if calibrated_score >= CONFIDENCE_THRESHOLD:
            classification = "AI-GENERATED"
        elif calibrated_score >= SUSPICIOUS_THRESHOLD:
            classification = "SUSPICIOUS"
        else:
            classification = "REAL"

        return {
            "classification": classification,
            "raw_score": round(float(raw_score), 4),
            "model_confidence": round(float(calibrated_score), 4),
            "is_usable": True,
            "status_code": "GOOD",
            "quality_metrics": metrics,
            "explanation": explanation
        }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        predictor = DeepfakeAudioPredictor()
        res = predictor.predict_window(audio_file)
        print(json.dumps(res, indent=2))
    else:
        print("Usage: python predict.py <path_to_audio_file>")
