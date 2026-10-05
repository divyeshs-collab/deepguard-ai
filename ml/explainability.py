import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List

class GradCAMExplainer:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN Spectrogram explainability.
    Identifies which frequency bands and time segments of the Log-Mel Spectrogram
    contributed most strongly to the model's prediction.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

    def generate_heatmap(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Computes Grad-CAM heatmap over the input spectrogram tensor.
        input_tensor: [1, 1, 128, 188]
        Returns:
            heatmap: [128, 188] normalized numpy array in range [0.0, 1.0]
            prediction_score: float
        """
        input_tensor = input_tensor.clone().detach()
        input_tensor.requires_grad = True

        # Forward pass
        score = self.model(input_tensor)
        prob = score.item()

        # Target class gradient (AI-Generated score)
        self.model.zero_grad()
        score.backward(retain_graph=True)

        gradients = self.model.gradients
        activations = self.model.activations

        if gradients is None or activations is None:
            # Fallback if hooks were not triggered: compute input gradient saliency map
            if input_tensor.grad is not None:
                saliency = torch.abs(input_tensor.grad).squeeze().cpu().numpy()
                norm_saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
                return norm_saliency, prob
            # Return uniform activation
            return np.ones((128, input_tensor.shape[-1]), dtype=np.float32) * 0.5, prob

        # Global average pooling of gradients: [1, 256, 1, 1]
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps: [1, 1, H', W']
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample CAM back to input spectrogram dimensions: [1, 1, 128, 188]
        cam_upsampled = F.interpolate(
            cam,
            size=(input_tensor.size(2), input_tensor.size(3)),
            mode='bilinear',
            align_corners=False
        )

        heatmap = cam_upsampled.squeeze().detach().cpu().numpy()

        # Min-max normalize
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap, prob

    def analyze_influencing_regions(self, heatmap: np.ndarray, log_mel_spec: np.ndarray) -> Dict[str, Any]:
        """
        Decomposes heatmap into acoustic domain regions:
        1. High-frequency spectral region (>4.5 kHz, bands 80-127) - vocoder phase/aliasing artifacts
        2. Harmonic & Formant structure region (1 kHz - 4.5 kHz, bands 30-79) - unnatural synthetic pitch tracking
        3. Low-frequency fundamental region (<1 kHz, bands 0-29) - glottal pulse & breath dynamics
        4. Temporal modulation - frame-to-frame discontinuities
        """
        total_energy = float(np.sum(heatmap) + 1e-8)

        # Region slices along Mel axis (128 bands total)
        low_band = heatmap[0:30, :]       # ~0 - 1000 Hz
        mid_band = heatmap[30:80, :]      # ~1000 - 4500 Hz
        high_band = heatmap[80:128, :]    # ~4500 - 8000 Hz

        low_pct = round(float(np.sum(low_band) / total_energy) * 100.0, 1)
        mid_pct = round(float(np.sum(mid_band) / total_energy) * 100.0, 1)
        high_pct = round(float(np.sum(high_band) / total_energy) * 100.0, 1)

        # Temporal variance in activation
        temporal_profile = np.mean(heatmap, axis=0)
        temporal_fluctuation = float(np.std(temporal_profile))

        # Determine prominent contributing characteristics
        top_regions = []
        if high_pct >= 35.0:
            top_regions.append({
                "region": "High-Frequency Spectral Region",
                "frequency_range": "4.5 kHz – 8.0 kHz",
                "contribution_pct": high_pct,
                "acoustic_significance": "Acoustic characteristics associated with neural vocoder aliasing and high-frequency spectral flattening."
            })
        if mid_pct >= 35.0:
            top_regions.append({
                "region": "Harmonic Structure & Formant Region",
                "frequency_range": "1.0 kHz – 4.5 kHz",
                "contribution_pct": mid_pct,
                "acoustic_significance": "Energy distribution across formant resonances and vocal tract transfer harmonics."
            })
        if low_pct >= 30.0:
            top_regions.append({
                "region": "Fundamental Frequency & Glottal Region",
                "frequency_range": "0 Hz – 1.0 kHz",
                "contribution_pct": low_pct,
                "acoustic_significance": "Vocal cord fundamental pitch dynamics, natural breathiness, and pitch micro-variations."
            })

        if not top_regions:
            top_regions.append({
                "region": "Harmonic Structure Region",
                "frequency_range": "1.0 kHz – 4.5 kHz",
                "contribution_pct": mid_pct,
                "acoustic_significance": "Harmonic alignment across conversational speech bands."
            })

        return {
            "title": "Audio Regions Influencing This Prediction",
            "disclaimer": "This is a model explanation of neural attention, not definitive proof of synthetic voice generation.",
            "frequency_breakdown": {
                "high_frequency_pct": high_pct,
                "mid_harmonic_pct": mid_pct,
                "low_fundamental_pct": low_pct
            },
            "temporal_modulation_score": round(temporal_fluctuation * 100.0, 2),
            "top_contributing_regions": top_regions,
            "heatmap_grid_summary": {
                "rows_mel_bands": heatmap.shape[0],
                "cols_time_frames": heatmap.shape[1],
                "peak_activation_mel_band": int(np.argmax(np.mean(heatmap, axis=1)))
            }
        }
