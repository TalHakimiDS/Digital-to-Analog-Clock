"""
Stage 1: Improved Digital Clock Reader (V2)

This version predicts the six digits of HH:MM:SS as
classification targets (0–9 for each digit) instead of
regressing normalized time directly. That gives much
better exact-match accuracy for discrete times.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialAttention(nn.Module):
    """Spatial attention to focus on clock digits."""

    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention


class DigitalClockReaderV2(nn.Module):
    """
    Improved CNN-based model to read time from digital clock images.

    Improvements:
    1. Spatial attention to focus on digits
    2. Separate prediction heads for hour/minute/second
    3. Deeper feature extraction
    4. Better regularization
    """

    def __init__(self, pretrained=True, dropout_rate=0.5, num_digit_classes: int = 10):
        super(DigitalClockReaderV2, self).__init__()

        # Load pretrained ResNet18 (torchvision >= 0.13 uses weights=)
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except AttributeError:
            # Fallback for older torchvision
            self.backbone = models.resnet18(pretrained=pretrained)

        # Extract intermediate features for attention
        self.layer1 = nn.Sequential(*list(self.backbone.children())[:5])
        self.layer2 = nn.Sequential(*list(self.backbone.children())[5:6])
        self.layer3 = nn.Sequential(*list(self.backbone.children())[6:7])
        self.layer4 = nn.Sequential(*list(self.backbone.children())[7:8])

        # Spatial attention after layer3
        self.attention = SpatialAttention(256)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Shared feature extraction
        self.shared_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Single classifier that predicts 6 digits × num_digit_classes logits
        self.num_digit_classes = num_digit_classes
        self.digit_head = nn.Linear(256, 6 * num_digit_classes)

    def forward(self, x):
        """
        Forward pass with attention.

        Args:
            x: Input tensor (batch_size, 3, H, W)

        Returns:
            logits: Tensor of shape (batch_size, 6, 10)
                    for digits [H1, H2, M1, M2, S1, S2]
        """
        # Extract features
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Apply spatial attention
        x = self.attention(x)

        x = self.layer4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # Shared features
        features = self.shared_fc(x)

        # Digit logits
        logits = self.digit_head(features)  # (batch, 6 * num_digit_classes)
        logits = logits.view(-1, 6, self.num_digit_classes)
        return logits


class DigitCrossEntropyLoss(nn.Module):
    """
    Cross-entropy over the 6 digit positions.

    pred_logits: (batch, 6, 10)
    target_digits: (batch, 6) with integer digit labels 0–9
    """

    def __init__(self):
        super(DigitCrossEntropyLoss, self).__init__()

    def forward(self, pred_logits: torch.Tensor, target_digits: torch.Tensor) -> torch.Tensor:
        # Flatten over digit positions so we can use a single CE call
        batch_size, num_digits, num_classes = pred_logits.shape
        pred_flat = pred_logits.view(batch_size * num_digits, num_classes)
        target_flat = target_digits.view(batch_size * num_digits)
        return F.cross_entropy(pred_flat, target_flat)


def create_digital_reader_v2(config_dict):
    """Factory function for improved digital reader."""
    model = DigitalClockReaderV2(
        pretrained=config_dict.get('pretrained', True),
        dropout_rate=config_dict.get('dropout_rate', 0.5)
    )
    return model
