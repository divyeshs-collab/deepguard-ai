# DeepGuard AI — Machine Learning Engine Documentation

## 1. System Overview
DeepGuard AI is a real-time deepfake speech detection engine engineered for live telephony call monitoring. The core model employs a 2D Convolutional Neural Network (CNN) trained directly on Log-Mel Spectrograms to recognize acoustic and spectral artifacts left by neural vocoders, voice conversion models, and speech synthesis pipelines.

---

## 2. Dataset Specification

- **Dataset Name**: DeepGuard Benchmark Voice Corpus (Telephony Edition)
- **Dataset Source**: Standardized multi-speaker acoustic recordings simulating conversational telephony (G.711 / AMR-WB codecs) combined with state-of-the-art neural speech synthesis algorithms.
- **Sample Rate**: 16,000 Hz (16 kHz) Mono
- **Window Segment Length**: 3.0 seconds (48,000 PCM samples)
- **Total Samples**: 288 segments across 24 distinct speakers
  - **Real Human Samples**: 144 segments
  - **Synthetic / Deepfake Samples**: 144 segments
- **Speaker Demographics**:
  - Total Speakers: 24 (12 Female, 12 Male)
  - Varied Fundamental Pitch: 105 Hz – 240 Hz
  - Diverse resonant vocal tract formants (F1–F4)

---

## 3. Synthetic Generation Methods
The deepfake partition incorporates three distinct generative speech architectures:
1. **Neural Vocoders (HiFi-GAN / WaveGlow / UnivNet)**:
   - High-frequency periodic checkerboard artifacts (6.0 kHz – 7.8 kHz).
   - Micro-phase aliasing across sub-band filters.
2. **Autoregressive / Non-Autoregressive TTS (Tacotron / FastSpeech)**:
   - Unnaturally flat fundamental frequency ($F_0$) contours.
   - Robotic comb-filter harmonic overtones and rigid phoneme durations.
3. **Diffusion Generative Models**:
   - Discontinuous phase transitions across frame boundaries.
   - Spectral blur and subtle high-frequency diffuse noise.

---

## 4. Speaker-Independent Splitting & Leakage Prevention
To guarantee fair, generalizable evaluation and eliminate data leakage:
- **Speaker Independence**: Speakers in the training split **never** appear in the validation or test splits.
- **Train Split (66.7%)**: Speakers `SPK_01` through `SPK_16` (192 samples).
- **Validation Split (16.7%)**: Speakers `SPK_17` through `SPK_20` (48 samples, unseen speakers).
- **Test Split (16.7%)**: Speakers `SPK_21` through `SPK_24` (48 samples, completely unseen speakers).
- **No Source Duplication**: Audio utterances are generated from unique linguistic prompts with zero cross-split duplication.

---

## 5. Log-Mel Spectrogram Configuration
Audio is transformed into an image-like feature tensor:
- **Sample Rate**: 16 kHz
- **Mel Filterbank Bands**: 128
- **FFT Size ($N_{fft}$)**: 1024
- **Hop Length**: 256 samples (~16 ms step)
- **Window Length**: 1024 samples (Hann window)
- **Output Tensor Dimensions**: `[Batch, 1, 128, 188]` (128 Mel bands $\times$ 188 time frames)

---

## 6. CNN Architecture
The classifier adheres to the specified hierarchical convolutional design:
1. **Input**: `[1, 128, 188]`
2. **Block 1**: `Conv2D(32, 3x3)` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `MaxPool2D(2x2)` $\to$ `Dropout(0.20)`
3. **Block 2**: `Conv2D(64, 3x3)` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `MaxPool2D(2x2)` $\to$ `Dropout(0.25)`
4. **Block 3**: `Conv2D(128, 3x3)` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `MaxPool2D(2x2)` $\to$ `Dropout(0.30)`
5. **Block 4**: `Conv2D(256, 3x3)` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `AdaptiveAvgPool2D(1, 1)`
6. **Dense 1**: `Linear(256 -> 128)` $\to$ `ReLU` $\to$ `Dropout(0.40)`
7. **Output**: `Linear(128 -> 1)` $\to$ `Sigmoid`

---

## 7. Model Calibration & Explainability
- **Confidence Calibration**: Employs **Temperature Scaling** fitted on held-out validation logits to produce a reliable **Model Confidence** metric rather than uncalibrated certainty.
- **Explainability**: Integrated **Grad-CAM** generates gradient activation heatmaps over the Log-Mel Spectrogram to identify which acoustic regions (e.g. high-frequency vocoder aliasing, harmonic tracking, or glottal pulses) influenced the prediction.

---

## 8. Dataset Limitations
- Performance may degrade under extreme background noise ($SNR < 5\text{ dB}$) or severe packet loss.
- Telephony networks with severe aggressive speech compression (e.g., G.729 @ 8kbps) may obscure subtle high-frequency vocoder cues.
- The model serves as an AI-assisted detection tool and does not constitute definitive legal proof.
