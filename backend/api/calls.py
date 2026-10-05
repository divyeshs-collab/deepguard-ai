import os
import uuid
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import soundfile as sf
import numpy as np

from backend.config import settings
from backend.telephony.demo_stream import DemoAudioStreamProvider
from backend.detection.detector import LiveCallDetector

router = APIRouter(prefix="/api/call", tags=["Call Management"])

# Global session registries
ACTIVE_DETECTORS: Dict[str, LiveCallDetector] = {}
DEMO_PROVIDER = DemoAudioStreamProvider()

class StartCallRequest(BaseModel):
    call_id: Optional[str] = None
    mode: str = "demo"  # "demo" (pre-recorded carrier simulation) or "live" (mic/telephony stream)
    scenario: Optional[str] = "deepfake_cloned_call"

class CallResponse(BaseModel):
    call_id: str
    status: str
    mode: str
    message: str
    timestamp: int

@router.get("/scenarios")
def list_demo_scenarios():
    """Lists available pre-recorded call demonstration scenarios."""
    return {"scenarios": DEMO_PROVIDER.list_available_scenarios()}

@router.post("/start", response_model=CallResponse)
async def start_call(req: StartCallRequest):
    """Initializes a new real-time call monitoring session."""
    call_id = req.call_id or f"call-{uuid.uuid4().hex[:8]}"

    # Instantiate or reset detector
    detector = LiveCallDetector(call_id=call_id)
    ACTIVE_DETECTORS[call_id] = detector

    if req.mode == "demo":
        await DEMO_PROVIDER.start_call_stream(call_id, options={"scenario": req.scenario})

    return CallResponse(
        call_id=call_id,
        status="CONNECTED",
        mode=req.mode,
        message=f"Call analysis initialized for session {call_id}.",
        timestamp=int(time.time())
    )

@router.post("/stop")
async def stop_call(call_id: str = Form(...)):
    """Terminates an active call monitoring session."""
    if call_id in ACTIVE_DETECTORS:
        ACTIVE_DETECTORS[call_id].is_active = False
        del ACTIVE_DETECTORS[call_id]

    await DEMO_PROVIDER.stop_call_stream(call_id)

    return {
        "call_id": call_id,
        "status": "DISCONNECTED",
        "message": f"Call session {call_id} terminated.",
        "timestamp": int(time.time())
    }

@router.get("/status")
def get_call_status(call_id: Optional[str] = None):
    """Retrieves real-time status of active or primary call."""
    if call_id and call_id in ACTIVE_DETECTORS:
        det = ACTIVE_DETECTORS[call_id]
        recent = det.recent_predictions[-1] if det.recent_predictions else None
        return {
            "call_id": call_id,
            "status": "CONNECTED" if det.is_active else "IDLE",
            "elapsed_seconds": det.get_elapsed_seconds(),
            "windows_analyzed": det.window_counter,
            "latest_prediction": recent
        }

    # If no call_id provided, return most active session or IDLE
    if ACTIVE_DETECTORS:
        first_id, det = next(iter(ACTIVE_DETECTORS.items()))
        recent = det.recent_predictions[-1] if det.recent_predictions else None
        return {
            "call_id": first_id,
            "status": "CONNECTED" if det.is_active else "IDLE",
            "elapsed_seconds": det.get_elapsed_seconds(),
            "windows_analyzed": det.window_counter,
            "latest_prediction": recent
        }

    return {
        "call_id": None,
        "status": "IDLE",
        "elapsed_seconds": 0.0,
        "windows_analyzed": 0,
        "latest_prediction": None
    }

@router.post("/upload")
async def upload_test_audio(file: UploadFile = File(...)):
    """
    Offline Testing Mode: Upload WAV, MP3, or FLAC audio for immediate analysis
    or subsequent streaming playback.
    """
    allowed_exts = [".wav", ".mp3", ".flac", ".ogg"]
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {file_ext}. Supported: {allowed_exts}")

    upload_dir = settings.BASE_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / f"upload_{uuid.uuid4().hex[:8]}{file_ext}"

    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)

    # Validate audio readable
    try:
        data, sr = sf.read(str(temp_path), dtype="float32")
        duration = len(data) / sr
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=400, detail=f"Corrupted or invalid audio file: {e}")

    call_id = f"upload-{uuid.uuid4().hex[:6]}"
    detector = LiveCallDetector(call_id=call_id)
    ACTIVE_DETECTORS[call_id] = detector

    # Register as demo stream target
    await DEMO_PROVIDER.start_call_stream(call_id, options={"custom_filepath": str(temp_path)})

    return {
        "call_id": call_id,
        "filename": file.filename,
        "duration_seconds": round(duration, 2),
        "sample_rate": sr,
        "status": "READY_FOR_ANALYSIS",
        "message": "File successfully uploaded and queued for streaming analysis."
    }
