import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import settings
from backend.api import calls, model, websocket
from ml.preprocessing import PreprocessingPipeline

app = FastAPI(
    title="DeepGuard AI",
    description="Real-Time Call Deepfake Audio Detection Engine",
    version=settings.MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(calls.router)
app.include_router(model.router)
app.include_router(websocket.router)

# Mount Static Frontend
FRONTEND_DIR = BASE_DIR / "frontend" / "static"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Startup initialization: verify pipeline and ensure demo audio assets exist."""
    print("=" * 65)
    print("  DEEPGUARD AI — TELEPHONY DEEPFAKE DETECTION ENGINE")
    print(f"  Version: {settings.MODEL_VERSION} | Sample Rate: {settings.SAMPLE_RATE}Hz")
    print(f"  Window: {settings.WINDOW_SECONDS}s | Hop: {settings.HOP_SECONDS}s")
    print("=" * 65)

    # Initialize preprocessing config
    pipeline = PreprocessingPipeline()
    pipeline.save_config()

    # Verify if demo audio files exist; if not, build benchmark dataset
    demo_dir = settings.ML_DIR / "dataset" / "demo_calls"
    if not (demo_dir / "human_authentic_call.wav").exists():
        print("[*] Generating acoustic audio benchmark corpus and demo call samples...")
        from ml.dataset import DatasetGenerator
        generator = DatasetGenerator(settings.ML_DIR / "dataset", settings.SAMPLE_RATE)
        generator.build_dataset()

@app.get("/health")
def health_check():
    return {
        "status": "ONLINE",
        "service": "DeepGuard AI Call Sentinel",
        "version": settings.MODEL_VERSION,
        "sample_rate": settings.SAMPLE_RATE,
        "device": "CPU / Neural Accelerator"
    }

# Serve Dashboard Single Page App
@app.get("/")
def serve_dashboard():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "DeepGuard AI Backend Running. Frontend is being built."}

# Mount static assets
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
