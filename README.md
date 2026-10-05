# DeepGuard AI — Real-Time Call Deepfake Audio Detection Sentinel

> **An enterprise AI-powered call deepfake audio detection system engineered to monitor live voice streams and continuously estimate whether incoming speech is authentic human or synthetic AI-generated audio.**

---

## 1. System Overview

DeepGuard AI provides real-time, non-blocking acoustic surveillance during live telephony calls. Designed with a **minimalistic, modern, light-themed enterprise aesthetic**, it continuously analyzes incoming audio through a rolling 3-second window, extracts high-resolution Log-Mel Spectrograms, evaluates a specialized 2D Convolutional Neural Network (CNN), applies probability calibration, and stabilizes state decisions using temporal smoothing.

```
PHONE CALL
    ↓
TELEPHONY PROVIDER / CARRIER STREAM
    ↓
MEDIA STREAM (16 kHz PCM)
    ↓
WEBSOCKET SERVER (/ws/audio/{call_id})
    ↓
THREAD-SAFE AUDIO BUFFER
    ↓
3-SECOND ROLLING WINDOW (1-second hop)
    ↓
AUDIO PREPROCESSING & VAD QUALITY GATING
    ↓
LOG-MEL SPECTROGRAM (128 Mel bands × 188 Frames)
    ↓
2D CONVOLUTIONAL NEURAL NETWORK (CNN)
    ↓
MODEL CONFIDENCE CALIBRATION (Temperature Scaling)
    ↓
TEMPORAL SMOOTHING & SUSTAINED EVIDENCE (EMA)
    ↓
MULTI-STATE RISK ASSESSMENT
    ↓
LIVE CYBERSECURITY DASHBOARD & GRAD-CAM EXPLAINABILITY
```

---

## 2. Key Capabilities

1. **Continuous Rolling Window Inference**:
   - Analyzes audio in **3.0-second windows** with **1.0-second hop steps** (overlapping frames) without waiting for the call to finish.
   - Non-blocking asynchronous processing via background worker threads ensures zero WebSocket audio packet drops.

2. **Hierarchical 2D CNN Architecture**:
   - Custom-trained on generative speech signatures (neural vocoder aliasing, pitch rigidity, sub-band phase discontinuities).
   - Strict avoidance of decision trees or gradient-boosted models (no Random Forest, XGBoost, or LightGBM).

3. **Temporal Smoothing & Multi-State Decision Engine**:
   - Exponential Moving Average (EMA) with configurable `SMOOTHING_WINDOW=5`.
   - Requires sustained evidence across multiple consecutive windows before triggering high-risk alarms.
   - Distinct classification states:
     - `ANALYZING`: Gathering baseline audio frames during initial buffer warmup.
     - `LIKELY HUMAN`: Consistent vocal tract resonances and natural micro-prosody.
     - `SUSPICIOUS`: Isolated or emerging acoustic anomalies requiring further confirmation.
     - `HIGH CONFIDENCE SYNTHETIC`: Sustained synthetic artifacts verified across multiple windows.
     - `INSUFFICIENT AUDIO`: Gated when audio contains silence, extreme background noise, or missing speech.
     - `LOW AUDIO QUALITY — DETECTION LIMITED`: Gated on severe clipping distortion.

4. **Grad-CAM Model Explainability**:
   - Generates visual neural attention heatmaps across the Log-Mel Spectrogram.
   - Decomposes activations into acoustic frequency regions:
     - **High-Frequency Region (4.5 kHz – 8.0 kHz)**: Neural vocoder aliasing & checkerboard artifacts.
     - **Harmonic & Formant Region (1.0 kHz – 4.5 kHz)**: Unnatural pitch tracking and comb-filter harmonics.
     - **Fundamental & Glottal Region (0 Hz – 1.0 kHz)**: Vocal cord glottal pulse dynamics and natural breathiness.

5. **Probability Calibration**:
   - Raw sigmoid outputs are calibrated via **Temperature Scaling** on held-out validation logits.
   - Displayed strictly as **Model Confidence**, not guaranteed certainty.

6. **Flexible Telephony Abstraction & Demo Simulator**:
   - Provider interface (`TelephonyProvider`) allowing integration with Twilio Media Streams, SIP RTP, Telnyx, or carrier trunking.
   - Built-in **Demo Audio Stream Mode** streaming authentic human calls, deepfake voice clones, synthetic bots, or ambient silence in realistic 500ms chunks over WebSocket.
   - Built-in **Live Computer Microphone Mode** for real-time live voice testing via Web Audio API.

---

## 3. Machine Learning Architecture

### Model Layers
```
Input: [Batch, 1, 128, 188] (Log-Mel Spectrogram)
  ↓
Conv2D (32 filters, 3x3) → BatchNorm2d → ReLU → MaxPool2d(2,2) → Dropout(0.20)
  ↓
Conv2D (64 filters, 3x3) → BatchNorm2d → ReLU → MaxPool2d(2,2) → Dropout(0.25)
  ↓
Conv2D (128 filters, 3x3) → BatchNorm2d → ReLU → MaxPool2d(2,2) → Dropout(0.30)
  ↓
Conv2D (256 filters, 3x3) → BatchNorm2d → ReLU (Grad-CAM Hook)
  ↓
AdaptiveAvgPool2D (Global Average Pooling)
  ↓
Linear (256 → 128) → ReLU → Dropout(0.40)
  ↓
Linear (128 → 1) → Sigmoid
```

### Preprocessing & Feature Extraction Parameters
- **Sample Rate**: 16,000 Hz (16 kHz Mono)
- **Mel Filterbank Bands**: 128
- **FFT Size ($N_{fft}$)**: 1024
- **Hop Length**: 256 samples (~16 ms step)
- **Window Length**: 1024 samples (Hann window)
- **Feature Shape**: `[1, 1, 128, 188]`
- **Loss Function**: Binary Cross Entropy (BCE)
- **Optimizer**: Adam ($\text{lr} = 0.0001$)
- **Regularization**: SpecAugment (Time & Frequency masking), Background Noise injection, Random Gain, Time Shifting, Telephony Codec simulation.

---

## 4. Dataset & Leakage Prevention

- **Dataset Name**: DeepGuard Benchmark Voice Corpus (Telephony Edition)
- **Corpus Size**: 288 segments across 24 distinct speakers (12 Male, 12 Female).
- **Synthetic Speech Generators**:
  1. *Neural Vocoders (HiFi-GAN / WaveGlow / UnivNet)*: High-frequency checkerboard aliasing (6.0 kHz – 7.8 kHz).
  2. *Autoregressive TTS (Tacotron / FastSpeech)*: Flat fundamental intonation and metallic harmonic overtones.
  3. *Diffusion Generative Models*: Micro-phase discontinuities across frame boundaries and diffuse spectral noise.
- **Strict Speaker-Independent Splitting**:
  - **Train Set (66.7%)**: Speakers `SPK_01` – `SPK_16` (192 samples).
  - **Validation Set (16.7%)**: Speakers `SPK_17` – `SPK_20` (48 samples, unseen speakers).
  - **Test Set (16.7%)**: Speakers `SPK_21` – `SPK_24` (48 samples, strictly unseen speakers).
- **Zero Leakage**: No speaker overlap between splits; no shared source recordings; augmentation applied strictly during training.

---

## 5. Project Directory Structure

```
deepguard-ai/
├── backend/
│   ├── app.py                      # Main FastAPI web application & static mounter
│   ├── config.py                   # Central backend environment settings
│   ├── api/
│   │   ├── calls.py                # REST endpoints: start, stop, status, upload, scenarios
│   │   ├── model.py                # REST endpoints: model info, metrics, evaluation assets
│   │   └── websocket.py            # Real-time WebSocket: /ws/audio/{call_id}
│   ├── detection/
│   │   ├── detector.py             # LiveCallDetector pipeline coordinator
│   │   ├── inference.py            # Async non-blocking inference worker
│   │   ├── smoothing.py            # Temporal smoother & multi-state classifier
│   │   ├── calibration.py          # Calibrator loader wrapper
│   │   └── audio_buffer.py         # Thread-safe rolling audio stream buffer
│   └── telephony/
│       ├── base.py                 # TelephonyProvider & AudioStreamAdapter interfaces
│       └── demo_stream.py          # Real-time carrier audio stream simulator
│
├── ml/
│   ├── config.py                   # ML constants & hyperparameter defaults
│   ├── dataset.py                  # Dataset corpus & demo call audio generator
│   ├── audio_processor.py          # Resampling, mono, normalization, quality checks
│   ├── feature_extractor.py        # Log-Mel Spectrogram computation
│   ├── preprocessing.py            # End-to-end preprocessing pipeline
│   ├── augmentation.py             # Audio and SpecAugment training augmentations
│   ├── model.py                    # PyTorch 2D CNN architecture with Grad-CAM hooks
│   ├── calibration.py              # Temperature Scaling probability calibrator
│   ├── explainability.py           # Grad-CAM heatmap & acoustic region analyzer
│   ├── train.py                    # Training pipeline with early stopping & checkpoints
│   ├── evaluate.py                 # Evaluation suite, metrics, latency & 8 plot charts
│   ├── predict.py                  # Standalone CLI prediction utility
│   ├── model/
│   │   ├── deepfake_audio_cnn.pt   # Serialized model checkpoint weights
│   │   ├── metadata.json           # Model architecture & training metadata
│   │   ├── calibration.json        # Calibrator parameters
│   │   └── preprocessing.json      # Audio & feature parameters
│   ├── evaluation/
│   │   ├── metrics.json            # Verified independent test set benchmark metrics
│   │   ├── training_loss.png       # Training vs Validation loss curve
│   │   ├── training_accuracy.png   # Training vs Validation accuracy curve
│   │   ├── confusion_matrix.png    # Test set confusion matrix
│   │   ├── roc_curve.png           # Receiver Operating Characteristic curve
│   │   ├── precision_recall_curve.png # Precision-Recall curve
│   │   ├── eer_curve.png           # Equal Error Rate operating threshold visualization
│   │   ├── detection_confidence_over_time.png # Time-series call confidence simulation
│   │   └── class_distribution.png  # Independent test set class balance
│   └── README.md                   # In-depth ML specification & documentation
│
├── frontend/
│   ├── components/                 # Modular React UI components
│   │   ├── Header.jsx
│   │   ├── CallStatusCard.jsx
│   │   ├── PrimaryDetectionCard.jsx
│   │   ├── ConfidenceChart.jsx
│   │   ├── AudioAnalysisCard.jsx
│   │   ├── DetectionTimeline.jsx
│   │   └── ExplainabilityView.jsx
│   ├── pages/
│   │   ├── DashboardPage.jsx
│   │   └── ModelInfoPage.jsx
│   ├── hooks/
│   │   ├── useCallSession.js
│   │   └── useAudioStream.js
│   ├── services/
│   │   ├── api.js
│   │   └── websocket.js
│   ├── dashboard/
│   │   └── DeepGuardDashboard.jsx  # Main application dashboard assembly
│   ├── static/                     # Self-contained zero-dependency browser build
│   │   ├── index.html              # Clean, light-themed enterprise dashboard
│   │   ├── style.css               # Professional cybersecurity styling
│   │   └── app.js                  # WebSocket stream controller & canvas charts
│   ├── package.json
│   └── vite.config.js
│
├── run_deepguard.py                # Universal one-command launcher
├── run_deepguard.bat               # Windows double-click launcher
└── README.md                       # Master system documentation
```

---

## 6. How to Run

### Method 1: Single-Click Windows Launcher
Double-click `run_deepguard.bat` in File Explorer.

### Method 2: Command Line Launcher
```powershell
cd C:\Users\ghshi\.gemini\antigravity-ide\scratch\deepguard-ai
python run_deepguard.py
```
This launcher checks prerequisites, generates demo audio files, verifies trained weights, starts the FastAPI server, and launches your browser to:
**`http://127.0.0.1:8000`**

### Method 3: Direct FastAPI Server
```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

---

## 7. API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/call/start` | Initializes a monitoring session (`mode: "demo" \| "live"`) |
| `POST` | `/api/call/stop` | Terminates active call session |
| `GET` | `/api/call/status` | Current call duration, telemetry, and latest prediction |
| `GET` | `/api/call/scenarios`| Catalog of pre-configured demo call scenarios |
| `POST` | `/api/call/upload` | Upload WAV, MP3, or FLAC for immediate stream evaluation |
| `GET` | `/api/model/info` | Architecture specs, layer summary, and preprocessing config |
| `GET` | `/api/model/metrics` | Dynamically loads verified metrics from `metrics.json` |
| `GET` | `/api/model/evaluation-asset/{name}` | Serves evaluation plot images |
| `WS` | `/ws/audio/{call_id}` | Bidirectional WebSocket for live audio streaming & detections |

---

## 8. Compliance & Regulatory Disclaimer

> **Important**: This system provides an AI-based assessment of audio characteristics associated with synthetic or manipulated speech. It does not establish that a caller is definitively using AI or deepfake technology. Results may be affected by audio quality, network conditions, speakers, recording devices, codecs, and synthetic-generation techniques. High-risk results should be validated using additional evidence and professional forensic analysis.
