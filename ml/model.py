import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class DeepfakeAudioCNN(nn.Module):
    """
    Convolutional Neural Network for Deepfake & Synthetic Voice Classification.
    Architecture:
      Input: [Batch, 1, 128, TimeFrames]
      Block 1: Conv2D(32, 3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2,2) -> Dropout(0.20)
      Block 2: Conv2D(64, 3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2,2) -> Dropout(0.25)
      Block 3: Conv2D(128, 3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2,2) -> Dropout(0.30)
      Block 4: Conv2D(256, 3x3) -> BatchNorm2d -> ReLU -> AdaptiveAvgPool2d((1,1))
      Dense:   Linear(256 -> 128) -> ReLU -> Dropout(0.40)
      Output:  Linear(128 -> 1) -> Sigmoid
    """

    def __init__(self, in_channels: int = 1):
        super(DeepfakeAudioCNN, self).__init__()

        # Block 1: 32 filters
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.20)

        # Block 2: 64 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.25)

        # Block 3: 128 filters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.30)

        # Block 4: 256 filters
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Classification Head
        self.fc1 = nn.Linear(256, 128)
        self.relu_fc = nn.ReLU(inplace=True)
        self.drop_fc = nn.Dropout(0.40)
        self.out = nn.Linear(128, 1)

        # Gradients hook for Grad-CAM
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns: Sigmoid probability [Batch, 1] in range [0.0, 1.0].
        (0.0 = Real Human, 1.0 = AI-Generated / Synthetic)
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # Block 4 (Target for Grad-CAM)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        if x.requires_grad:
            self.activations = x
            h = x.register_hook(self.activations_hook)

        # Global Average Pooling: [Batch, 256, 1, 1] -> [Batch, 256]
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # Classification Head
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.drop_fc(x)
        logits = self.out(x)
        output = torch.sigmoid(logits)

        return output

    def get_architecture_summary(self) -> Dict[str, Any]:
        """Returns clean serializable summary of network layers."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "name": "DeepfakeAudioCNN",
            "type": "2D Convolutional Neural Network",
            "layers": [
                {"layer": "Input", "shape": [1, 128, 188], "description": "Log-Mel Spectrogram"},
                {"layer": "Conv2D_1", "filters": 32, "kernel": "3x3", "activation": "ReLU", "pooling": "MaxPool2d(2x2)", "dropout": 0.20},
                {"layer": "Conv2D_2", "filters": 64, "kernel": "3x3", "activation": "ReLU", "pooling": "MaxPool2d(2x2)", "dropout": 0.25},
                {"layer": "Conv2D_3", "filters": 128, "kernel": "3x3", "activation": "ReLU", "pooling": "MaxPool2d(2x2)", "dropout": 0.30},
                {"layer": "Conv2D_4", "filters": 256, "kernel": "3x3", "activation": "ReLU", "pooling": "GlobalAveragePooling2D"},
                {"layer": "Dense_1", "units": 128, "activation": "ReLU", "dropout": 0.40},
                {"layer": "Output", "units": 1, "activation": "Sigmoid", "classes": ["REAL", "AI-GENERATED"]}
            ],
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_size_kb": round(total_params * 4 / 1024, 2)
        }
