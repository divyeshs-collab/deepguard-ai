#!/usr/bin/env bash
# DeepGuard AI - Remote Host Deployment Script (Linux / macOS / Cloud VM)
set -e

HOST_IP="${1:-0.0.0.0}"
HOST_PORT="${2:-8000}"

echo "===================================================================="
echo "  DEEPGUARD AI — REMOTE HOST DEPLOYMENT SCRIPT"
echo "  Target Host: ${HOST_IP} | Port: ${HOST_PORT}"
echo "===================================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 1. System packages check (Debian/Ubuntu)
if command -v apt-get &> /dev/null; then
    echo "[*] Ensuring required audio system packages (libsndfile1, ffmpeg)..."
    sudo apt-get update -qq && sudo apt-get install -y -qq libsndfile1 ffmpeg curl || true
fi

# 2. Virtual environment setup
if [ ! -d "venv" ]; then
    echo "[*] Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "[*] Activating virtual environment..."
source venv/bin/activate

# 3. Dependency installation
echo "[*] Installing Python dependencies from requirements.txt..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 4. Verify/generate dataset & trained model if missing
if [ ! -f "ml/model/deepfake_audio_cnn.pt" ]; then
    echo "[*] Pre-trained weights not found. Generating corpus and training model..."
    python ml/dataset.py
    python ml/train.py
    python ml/evaluate.py
fi

# 5. Launch FastAPI server
echo "[+] Starting DeepGuard AI Sentinel on http://${HOST_IP}:${HOST_PORT} ..."
exec python -m uvicorn backend.app:app --host "${HOST_IP}" --port "${HOST_PORT}"
