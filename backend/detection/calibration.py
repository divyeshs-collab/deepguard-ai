from pathlib import Path
from ml.calibration import ModelCalibrator
from backend.config import settings

def get_calibrator() -> ModelCalibrator:
    """Loads calibrator configuration for backend inference."""
    calib_path = settings.MODEL_DIR / "calibration.json"
    return ModelCalibrator.load(calib_path)
