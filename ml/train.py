import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure project root is importable
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ml.config import (
    SAMPLE_RATE, WINDOW_SAMPLES, TIME_FRAMES, EPOCHS, BATCH_SIZE,
    LEARNING_RATE, EARLY_STOPPING_PATIENCE, LR_REDUCTION_PATIENCE,
    LR_REDUCTION_FACTOR, RANDOM_SEED, MODEL_VERSION,
    MODEL_DIR, EVAL_DIR, DATASET_DIR, MODEL_WEIGHTS_PATH, MODEL_METADATA_PATH
)
from ml.audio_processor import AudioProcessor
from ml.feature_extractor import MelSpectrogramExtractor
from ml.augmentation import AudioAugmentation
from ml.model import DeepfakeAudioCNN
from ml.calibration import ModelCalibrator
from ml.dataset import DatasetGenerator

# Set deterministic seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class DeepfakeVoiceDataset(Dataset):
    """PyTorch Dataset for Log-Mel Spectrogram deepfake audio classification."""

    def __init__(self, samples: list, is_training: bool = False):
        self.samples = samples
        self.is_training = is_training
        self.processor = AudioProcessor(target_sample_rate=SAMPLE_RATE)
        self.extractor = MelSpectrogramExtractor(sample_rate=SAMPLE_RATE)
        self.augmenter = AudioAugmentation(sample_rate=SAMPLE_RATE)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        file_path = item["file_path"]
        label = float(item["label"])

        # Load raw audio
        audio, orig_sr = self.processor.load_audio(file_path)
        audio = self.processor.to_mono(audio)
        audio = self.processor.resample(audio, orig_sr)
        audio = self.processor.normalize_amplitude(audio)
        audio = self.processor.pad_or_truncate(audio, WINDOW_SAMPLES)

        # Apply augmentation ONLY during training
        if self.is_training:
            audio = self.augmenter.apply_training_augmentation(audio)

        # Extract Log-Mel Spectrogram
        spec = self.extractor.extract_features(audio)

        # SpecAugment during training
        if self.is_training and np.random.rand() > 0.4:
            spec = self.augmenter.spec_augment(spec)

        # Shape: (1, 128, 188)
        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return spec_tensor, label_tensor

class ModelTrainer:
    """End-to-end CNN Training, Checkpointing, and Calibration Pipeline."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Training device: {self.device}")
        self.model = DeepfakeAudioCNN(in_channels=1).to(self.device)
        self.calibrator = ModelCalibrator()

    def train(self) -> dict:
        print("=" * 65)
        print("  DEEPGUARD AI — CNN MODEL TRAINING PIPELINE")
        print(f"  Version: {MODEL_VERSION} | Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE}")
        print("=" * 65)

        # 1. Ensure dataset exists or generate it
        manifest_path = DATASET_DIR / "manifest.json"
        if not manifest_path.exists():
            print("[*] Generating benchmark dataset with speaker-independent split...")
            generator = DatasetGenerator(DATASET_DIR, SAMPLE_RATE)
            generator.build_dataset()

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        samples = manifest_data["samples"]
        train_samples = [s for s in samples if s["split"] == "train"]
        val_samples = [s for s in samples if s["split"] == "val"]
        test_samples = [s for s in samples if s["split"] == "test"]

        print(f"[+] Loaded Manifest: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

        train_ds = DeepfakeVoiceDataset(train_samples, is_training=True)
        val_ds = DeepfakeVoiceDataset(val_samples, is_training=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Binary Cross Entropy Loss & Adam Optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=LR_REDUCTION_PATIENCE, factor=LR_REDUCTION_FACTOR
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_weights = None

        start_time = time.time()

        for epoch in range(1, EPOCHS + 1):
            # Training Phase
            self.model.train()
            total_train_loss = 0.0
            correct_train = 0
            total_train = 0

            for specs, labels in train_loader:
                specs, labels = specs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                preds = self.model(specs)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * specs.size(0)
                predicted = (preds >= 0.5).float()
                correct_train += (predicted == labels).sum().item()
                total_train += specs.size(0)

            epoch_train_loss = total_train_loss / max(1, total_train)
            epoch_train_acc = correct_train / max(1, total_train)

            # Validation Phase
            self.model.eval()
            total_val_loss = 0.0
            correct_val = 0
            total_val = 0
            val_preds_list = []
            val_labels_list = []

            with torch.no_grad():
                for specs, labels in val_loader:
                    specs, labels = specs.to(self.device), labels.to(self.device)
                    preds = self.model(specs)
                    loss = criterion(preds, labels)

                    total_val_loss += loss.item() * specs.size(0)
                    predicted = (preds >= 0.5).float()
                    correct_val += (predicted == labels).sum().item()
                    total_val += specs.size(0)

                    val_preds_list.extend(preds.cpu().numpy().flatten())
                    val_labels_list.extend(labels.cpu().numpy().flatten())

            epoch_val_loss = total_val_loss / max(1, total_val)
            epoch_val_acc = correct_val / max(1, total_val)

            history["train_loss"].append(round(epoch_train_loss, 4))
            history["val_loss"].append(round(epoch_val_loss, 4))
            history["train_acc"].append(round(epoch_train_acc, 4))
            history["val_acc"].append(round(epoch_val_acc, 4))

            scheduler.step(epoch_val_loss)

            if epoch % 5 == 0 or epoch == 1 or epoch_val_loss < best_val_loss:
                print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
                      f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.3f} | "
                      f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.3f}")

            # Early Stopping and Checkpointing
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                best_model_weights = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"[*] Early stopping triggered at epoch {epoch}. (Patience: {EARLY_STOPPING_PATIENCE})")
                    break

        training_time_sec = round(time.time() - start_time, 2)
        print(f"[+] Training completed in {training_time_sec}s. Best Val Loss: {best_val_loss:.4f}")

        # Restore best model weights
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)

        # 3. Model Calibration (Temperature Scaling on Validation logits)
        val_preds_arr = np.array(val_preds_list)
        val_labels_arr = np.array(val_labels_list)
        self.calibrator.fit_platt(val_preds_arr, val_labels_arr)
        calib_path = MODEL_DIR / "calibration.json"
        self.calibrator.save(calib_path)
        print(f"[+] Calibrator fitted: Temperature = {self.calibrator.temperature:.3f}")

        # 4. Save model weights
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(MODEL_WEIGHTS_PATH))
        print(f"[+] Model checkpoint saved: {MODEL_WEIGHTS_PATH}")

        # 5. Save model metadata
        metadata = {
            "model": {
                "name": "CNN Deepfake Audio Classifier",
                "version": MODEL_VERSION,
                "framework": "PyTorch (TorchScript compatible)",
                "task": "Binary Audio Classification (REAL vs AI-GENERATED)",
                "input": "Log-Mel Spectrogram",
                "sample_rate": SAMPLE_RATE,
                "window_seconds": 3.0,
                "hop_seconds": 1.0,
                "feature_shape": [1, 128, TIME_FRAMES]
            },
            "training": {
                "date": datetime.now().isoformat(),
                "epochs_trained": len(history["train_loss"]),
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "random_seed": RANDOM_SEED,
                "duration_seconds": training_time_sec,
                "device": str(self.device),
                "best_val_loss": round(best_val_loss, 4),
                "history": history
            },
            "architecture": self.model.get_architecture_summary(),
            "calibration": {
                "method": self.calibrator.method,
                "temperature": self.calibrator.temperature,
                "ui_label": "Model Confidence"
            }
        }

        with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[+] Model metadata saved: {MODEL_METADATA_PATH}")
        return metadata

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
