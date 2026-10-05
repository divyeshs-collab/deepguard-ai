"""
DeepGuard AI - Application Launcher
Initializes the ML CNN engine, boots the real-time carrier media stream server,
and launches the enterprise cybersecurity dashboard in your default browser.
"""

import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path

def main():
    print("=" * 68)
    print("  DEEPGUARD AI — REAL-TIME CALL DEEPFAKE DETECTION SENTINEL")
    print("  Enterprise Telephony Media Stream & CNN Analysis Platform")
    print("=" * 68)

    root_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(root_dir))

    # 1. Verify Dataset and Pre-generate Demo Calls if needed
    demo_dir = root_dir / "ml" / "dataset" / "demo_calls"
    if not (demo_dir / "human_authentic_call.wav").exists():
        print("[*] Generating benchmark voice corpus and demo telephony scenarios...")
        try:
            from ml.dataset import DatasetGenerator
            generator = DatasetGenerator(root_dir / "ml" / "dataset", 16000)
            generator.build_dataset()
        except Exception as e:
            print(f"[!] Warning: Could not pre-generate dataset: {e}")

    # 2. Check if model weights exist; if not, train or initialize weights
    weights_path = root_dir / "ml" / "model" / "deepfake_audio_cnn.pt"
    if not weights_path.exists():
        print("[*] Model weights not found. Running training and evaluation pipeline...")
        try:
            from ml.train import ModelTrainer
            from ml.evaluate import ModelEvaluator
            trainer = ModelTrainer()
            trainer.train()
            evaluator = ModelEvaluator()
            evaluator.evaluate()
        except Exception as e:
            print(f"[!] Warning during initial model training: {e}")

    # 3. Parse host and port configuration
    import argparse
    import socket

    parser = argparse.ArgumentParser(description="DeepGuard AI Application Launcher")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="Host address to bind (e.g. 0.0.0.0 for all interfaces, 127.0.0.1 for local)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "0")), help="Port number to bind (0 for automatic discovery)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    def is_port_available(p, h):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.4)
                test_host = '127.0.0.1' if h in ('0.0.0.0', '') else h
                return s.connect_ex((test_host, p)) != 0
        except Exception:
            return True

    selected_host = args.host
    selected_port = args.port

    # If port is 0 or already occupied, find a free available port
    if selected_port == 0 or not is_port_available(selected_port, selected_host):
        if selected_port != 0:
            print(f"[!] Notice: Port {selected_port} is already in use by another process.")
        found_port = None
        for p in [8050, 8080, 8000, 8001, 8888, 5050]:
            if is_port_available(p, selected_host):
                found_port = p
                break
        if found_port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                found_port = s.getsockname()[1]
        selected_port = found_port
        print(f"[+] Automatically switched to available port: {selected_port}")

    # Determine local network IP
    local_ip = "127.0.0.1"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
    except Exception:
        pass

    # 4. Schedule browser opening
    browser_url = f"http://localhost:{selected_port}"
    if not args.no_browser:
        def open_browser():
            time.sleep(1.8)
            print(f"[+] Opening DeepGuard AI Sentinel in browser: {browser_url}")
            webbrowser.open(browser_url)

        import threading
        threading.Thread(target=open_browser, daemon=True).start()

    # 5. Launch FastAPI Uvicorn Server
    print("=" * 68)
    print("  DEEPGUARD AI SENTINEL ONLINE")
    print(f"  Local Access:    http://localhost:{selected_port}")
    print(f"  Local Loopback:  http://127.0.0.1:{selected_port}")
    print(f"  Network Access:  http://{local_ip}:{selected_port}")
    print("=" * 68)

    cmd = [
        sys.executable, "-m", "uvicorn", "backend.app:app",
        "--host", selected_host,
        "--port", str(selected_port)
    ]
    subprocess.run(cmd, cwd=str(root_dir))

if __name__ == "__main__":
    main()
