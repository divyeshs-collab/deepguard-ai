from collections import deque
from typing import List, Dict, Any, Tuple
import numpy as np

class TemporalSmoother:
    """
    Temporal aggregator and state engine for real-time call audio detection.
    Prevents single anomalous windows from triggering false alarms by computing
    Exponential Moving Averages (EMA) and requiring sustained evidence over
    a configurable smoothing window (default: 5 windows).
    """

    def __init__(
        self,
        window_size: int = 5,
        alpha: float = 0.40,
        high_confidence_threshold: float = 0.70,
        suspicious_threshold: float = 0.35,
        sustained_fake_count: int = 3
    ):
        self.window_size = window_size
        self.alpha = alpha  # Smoothing factor for EMA
        self.high_confidence_threshold = high_confidence_threshold
        self.suspicious_threshold = suspicious_threshold
        self.sustained_fake_count = sustained_fake_count

        self.history = deque(maxlen=window_size)
        self.smoothed_confidence: float = 0.0
        self.consecutive_synthetic_windows: int = 0
        self.total_processed_windows: int = 0

    def reset(self):
        """Resets smoother state for a new call."""
        self.history.clear()
        self.smoothed_confidence = 0.0
        self.consecutive_synthetic_windows = 0
        self.total_processed_windows = 0

    def update(self, raw_confidence: float, is_usable: bool = True, quality_status: str = "GOOD") -> Dict[str, Any]:
        """
        Updates the smoother with a new window prediction.
        Returns the smoothed classification state and risk assessment.
        """
        if not is_usable:
            return {
                "classification": quality_status,  # "INSUFFICIENT AUDIO" or "LOW AUDIO QUALITY — DETECTION LIMITED"
                "raw_confidence": 0.0,
                "smoothed_confidence": round(self.smoothed_confidence, 4),
                "risk_level": "UNKNOWN",
                "state_description": "Awaiting clear speech segment.",
                "history_length": len(self.history)
            }

        self.total_processed_windows += 1
        self.history.append(raw_confidence)

        # Compute Exponential Moving Average (EMA)
        if len(self.history) == 1:
            self.smoothed_confidence = raw_confidence
        else:
            self.smoothed_confidence = (self.alpha * raw_confidence) + ((1.0 - self.alpha) * self.smoothed_confidence)

        # Track consecutive synthetic indicators
        if raw_confidence >= self.high_confidence_threshold:
            self.consecutive_synthetic_windows += 1
        else:
            self.consecutive_synthetic_windows = max(0, self.consecutive_synthetic_windows - 1)

        # Determine classification state
        # 1. ANALYZING: Initial warmup period
        if len(self.history) < 2:
            classification = "ANALYZING"
            risk_level = "LOW"
            desc = "Gathering initial audio frames for baseline assessment."

        # 2. HIGH CONFIDENCE SYNTHETIC: Sustained high confidence over multiple windows
        elif self.smoothed_confidence >= self.high_confidence_threshold and self.consecutive_synthetic_windows >= self.sustained_fake_count:
            classification = "HIGH CONFIDENCE SYNTHETIC"
            risk_level = "CRITICAL"
            desc = f"Multiple consecutive windows ({self.consecutive_synthetic_windows}) exhibit strong synthetic voice characteristics."

        # 3. SUSPICIOUS: Elevated confidence or brief anomaly not yet sustained
        elif self.smoothed_confidence >= self.suspicious_threshold or raw_confidence >= self.high_confidence_threshold:
            classification = "SUSPICIOUS"
            risk_level = "HIGH" if self.smoothed_confidence >= 0.55 else "MODERATE"
            desc = "Acoustic anomalies detected; awaiting additional windows for confirmation."

        # 4. LIKELY HUMAN: Consistent human speech dynamics
        else:
            classification = "LIKELY HUMAN"
            risk_level = "LOW"
            desc = "Acoustic features and vocal tract resonances match natural human speech."

        return {
            "classification": classification,
            "raw_confidence": round(float(raw_confidence), 4),
            "smoothed_confidence": round(float(self.smoothed_confidence), 4),
            "risk_level": risk_level,
            "state_description": desc,
            "consecutive_synthetic_count": self.consecutive_synthetic_windows,
            "history_length": len(self.history),
            "total_windows": self.total_processed_windows
        }
