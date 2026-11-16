"""
Stage 2: Analog Clock Manipulator Model
Takes analog clock + target time, outputs hand angles as sin/cos
"""

import torch
import torch.nn as nn
import torchvision.models as models


class AnalogClockManipulator(nn.Module):
    """
    Model to predict hand angles for analog clocks given target time.

    Architecture:
        Image → ResNet18 → features
        Time → Embedding
        Concat → FC layers → 6 outputs (sin/cos for 3 hands)
    """

    def __init__(self, pretrained=True, time_embedding_dim=32, dropout_rate=0.3):
        super(AnalogClockManipulator, self).__init__()

        # Image encoder
        self.backbone = models.resnet18(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(3, time_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.ReLU(inplace=True)
        )

        # Combined predictor
        combined_features = num_features + time_embedding_dim
        self.predictor = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 6),  # [sin_h, cos_h, sin_m, cos_m, sin_s, cos_s]
            nn.Tanh()  # Output in [-1, 1] for sin/cos
        )

    def forward(self, image, time):
        """
        Forward pass.

        Args:
            image: Tensor of shape (batch_size, 3, H, W)
            time: Tensor of shape (batch_size, 3) with normalized (h, m, s)

        Returns:
            Tensor of shape (batch_size, 6) with sin/cos angles
        """
        # Extract image features
        img_features = self.backbone(image)

        # Embed time
        time_features = self.time_embedding(time)

        # Concatenate
        combined = torch.cat([img_features, time_features], dim=1)

        # Predict angles
        angles_sincos = self.predictor(combined)

        return angles_sincos


def create_analog_manipulator(config_dict):
    """
    Factory function to create AnalogClockManipulator from config.

    Args:
        config_dict: Configuration dictionary

    Returns:
        AnalogClockManipulator model
    """
    model = AnalogClockManipulator(
        pretrained=config_dict.get('pretrained', True),
        time_embedding_dim=config_dict.get('time_embedding_dim', 32),
        dropout_rate=config_dict.get('dropout_rate', 0.3)
    )
    return model
