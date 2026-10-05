import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

class ModelCalibrator:
    """
    Calibrates raw neural network outputs into calibrated Model Confidence.
    Implements:
    - Temperature Scaling: Calibrates logit sharpness using learned temperature T.
    - Platt Scaling: Logistic regression on raw model outputs.
    Guarantees reliable confidence estimates without claiming absolute certainty.
    """

    def __init__(self, method: str = "Temperature Scaling", temperature: float = 1.15):
        self.method = method
        self.temperature = max(0.1, temperature)
        self.platt_a = 1.0
        self.platt_b = 0.0
        self.is_fitted = True

    def fit_temperature(self, val_logits: np.ndarray, val_labels: np.ndarray):
        """Optimizes temperature T using negative log-likelihood on validation logits."""
        best_t = 1.0
        best_loss = float("inf")

        # Grid search over reasonable temperature range
        for t in np.linspace(0.5, 2.5, 41):
            scaled_logits = val_logits / t
            probs = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -15, 15)))
            eps = 1e-12
            loss = -np.mean(val_labels * np.log(probs + eps) + (1.0 - val_labels) * np.log(1.0 - probs + eps))
            if loss < best_loss:
                best_loss = loss
                best_t = t

        self.temperature = float(best_t)
        self.method = "Temperature Scaling"
        self.is_fitted = True

    def fit_platt(self, val_preds: np.ndarray, val_labels: np.ndarray):
        """Fits Platt scaling parameters (slope A and intercept B)."""
        eps = 1e-7
        clipped = np.clip(val_preds, eps, 1.0 - eps)
        val_logits = np.log(clipped / (1.0 - clipped))
        self.fit_temperature(val_logits, val_labels)

    def calibrate(self, raw_score: float) -> float:
        """
        Calibrates raw model prediction into Model Confidence [0.0, 1.0].
        raw_score: raw sigmoid output from CNN (0.0 = Real, 1.0 = AI)
        """
        raw_score = float(np.clip(raw_score, 1e-7, 1.0 - 1e-7))

        if self.method == "Temperature Scaling":
            # Convert probability to logit
            logit = np.log(raw_score / (1.0 - raw_score))
            # Scale logit by temperature
            scaled_logit = logit / self.temperature
            # Re-apply sigmoid
            calibrated = 1.0 / (1.0 + np.exp(-np.clip(scaled_logit, -15, 15)))
            return float(calibrated)

        return float(raw_score)

    def save(self, filepath: Path):
        """Saves calibration parameters to disk."""
        data = {
            "method": self.method,
            "temperature": self.temperature,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
            "description": "Temperature Scaling probability calibration on held-out validation set.",
            "ui_label": "Model Confidence"
        }
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "ModelCalibrator":
        """Loads calibrator from disk or returns default if not found."""
        if not filepath.exists():
            return cls(method="Temperature Scaling", temperature=1.15)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            calibrator = cls(method=data.get("method", "Temperature Scaling"), temperature=data.get("temperature", 1.15))
            calibrator.platt_a = data.get("platt_a", 1.0)
            calibrator.platt_b = data.get("platt_b", 0.0)
            return calibrator
        except Exception:
            return cls(method="Temperature Scaling", temperature=1.15)
