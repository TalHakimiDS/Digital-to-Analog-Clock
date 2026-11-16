"""
Stage 1: Digital Clock Reader Model
Reads digital clock images and outputs (hour, minute, second)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class DigitalClockReader(nn.Module):
    """
    CNN-based model to read time from digital clock images.

    Architecture:
        ResNet18 (pretrained) → FC(512) → Dropout → FC(256) → FC(3)
        Output: (hour, minute, second) normalized to [0, 1]
    """

    def __init__(self, pretrained=True, dropout_rate=0.3):
        super(DigitalClockReader, self).__init__()

        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)

        # Remove the final FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Custom head for regression
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 3),  # (hour, minute, second)
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Tensor of shape (batch_size, 3) with normalized (h, m, s)
        """
        features = self.backbone(x)
        time_output = self.regressor(features)
        return time_output


def create_digital_reader(config_dict):
    """
    Factory function to create DigitalClockReader from config.

    Args:
        config_dict: Configuration dictionary

    Returns:
        DigitalClockReader model
    """
    model = DigitalClockReader(
        pretrained=config_dict.get('pretrained', True),
        dropout_rate=config_dict.get('dropout_rate', 0.3)
    )
    return model
