import os
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.config import settings
from ml.model import DeepfakeAudioCNN
from ml.calibration import ModelCalibrator

router = APIRouter(prefix="/api/model", tags=["Model Information"])

@router.get("/info")
def get_model_info():
    """
    Returns full model metadata, architecture layer specs, input representation,
    and framework information.
    """
    metadata_path = settings.MODEL_METADATA_PATH
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Fallback to dynamic introspection
    cnn = DeepfakeAudioCNN(in_channels=1)
    arch = cnn.get_architecture_summary()
    calib = ModelCalibrator.load(settings.MODEL_DIR / "calibration.json")

    return {
        "model": {
            "name": "CNN Deepfake Audio Classifier",
            "version": settings.MODEL_VERSION,
            "framework": settings.MODEL_FRAMEWORK,
            "task": "Binary Audio Classification",
            "classes": ["REAL", "AI-GENERATED"],
            "input": "Log-Mel Spectrogram",
            "sample_rate": settings.SAMPLE_RATE,
            "window_seconds": settings.WINDOW_SECONDS,
            "hop_seconds": settings.HOP_SECONDS,
            "feature_shape": [1, 1, settings.MEL_BANDS, 188]
        },
        "architecture": arch,
        "calibration": {
            "method": calib.method,
            "temperature": calib.temperature,
            "ui_label": "Model Confidence"
        }
    }

@router.get("/metrics")
def get_model_metrics():
    """
    Dynamically loads evaluation metrics from ml/evaluation/metrics.json.
    Does NOT hard-code fake numbers; displays 'Not evaluated' if evaluation hasn't been executed.
    """
    metrics_path = settings.METRICS_PATH
    if not metrics_path.exists():
        return {
            "status": "Not evaluated",
            "message": "Model evaluation has not been executed yet. Run 'python ml/evaluate.py' to produce verified benchmarks.",
            "metrics": None
        }

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "status": "Evaluated",
            "metrics": data
        }
    except Exception as e:
        return {
            "status": "Not evaluated",
            "error": str(e),
            "metrics": None
        }

@router.get("/evaluation-asset/{asset_name}")
def get_evaluation_asset(asset_name: str):
    """Serves generated evaluation plots (loss, accuracy, ROC, PR, EER, CM, etc.)."""
    safe_name = Path(asset_name).name
    asset_path = settings.EVAL_DIR / safe_name
    if not asset_path.exists():
        raise HTTPException(status_code=404, detail=f"Evaluation asset '{asset_name}' not found.")
    return FileResponse(str(asset_path), media_type="image/png")
