"""
Stage 2: Improved Analog Clock Manipulator (V2)
Enhanced architecture for better hand angle prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


class CircularMSELoss(nn.Module):
    """
    Loss function for circular values (angles).
    Properly handles angle wraparound (0° = 360°).
    """

    def __init__(self):
        super(CircularMSELoss, self).__init__()

    def forward(self, pred_sincos, target_sincos):
        """
        Args:
            pred_sincos: (batch_size, 6) - [sin_h, cos_h, sin_m, cos_m, sin_s, cos_s]
            target_sincos: (batch_size, 6)
        """
        # MSE on sin/cos values
        mse = F.mse_loss(pred_sincos, target_sincos)

        # Additional penalty for angle difference
        batch_size = pred_sincos.size(0)
        angle_loss = 0.0

        for i in range(3):  # 3 hands
            sin_pred = pred_sincos[:, 2*i]
            cos_pred = pred_sincos[:, 2*i + 1]
            sin_target = target_sincos[:, 2*i]
            cos_target = target_sincos[:, 2*i + 1]

            # Angle difference using atan2
            pred_angle = torch.atan2(sin_pred, cos_pred)
            target_angle = torch.atan2(sin_target, cos_target)

            # Circular distance
            angle_diff = pred_angle - target_angle
            circular_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
            angle_loss += (circular_diff ** 2).mean()

        return mse + 0.5 * angle_loss / 3.0


class AnalogClockManipulatorV2(nn.Module):
    """
    Improved model to predict hand angles for analog clocks.

    Improvements:
    1. Better time embedding with positional encoding
    2. Cross-attention between image and time
    3. Separate heads for each hand
    4. Normalization of sin/cos outputs
    """

    def __init__(self, pretrained=True, time_embedding_dim=64, dropout_rate=0.4):
        super(AnalogClockManipulatorV2, self).__init__()

        # Image encoder
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except AttributeError:
            self.backbone = models.resnet18(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Enhanced time embedding with positional encoding
        self.time_embedding = nn.Sequential(
            nn.Linear(3, time_embedding_dim),
            nn.LayerNorm(time_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.LayerNorm(time_embedding_dim),
            nn.ReLU(inplace=True)
        )

        # Cross-attention between image features and time
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=time_embedding_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # Project image features to attention dim
        self.image_proj = nn.Linear(num_features, time_embedding_dim)

        # Combined predictor
        combined_dim = time_embedding_dim * 2
        self.shared_predictor = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Separate heads for each hand (better accuracy)
        self.hour_hand_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # sin, cos
            nn.Tanh()
        )

        self.minute_hand_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh()
        )

        self.second_hand_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh()
        )

    def normalize_sincos(self, sincos_pairs):
        """Normalize sin/cos pairs to unit circle."""
        sin_val = sincos_pairs[:, 0:1]
        cos_val = sincos_pairs[:, 1:2]
        magnitude = torch.sqrt(sin_val**2 + cos_val**2) + 1e-8
        return torch.cat([sin_val / magnitude, cos_val / magnitude], dim=1)

    def forward(self, image, time):
        """
        Forward pass with cross-attention.

        Args:
            image: Tensor (batch_size, 3, H, W)
            time: Tensor (batch_size, 3) normalized (h, m, s)

        Returns:
            Tensor (batch_size, 6) with sin/cos angles
        """
        batch_size = image.size(0)

        # Extract image features
        img_features = self.backbone(image)  # (batch, 512)
        img_features_proj = self.image_proj(img_features)  # (batch, 64)

        # Embed time
        time_features = self.time_embedding(time)  # (batch, 64)

        # Cross-attention (time queries image)
        # Add sequence dimension
        time_query = time_features.unsqueeze(1)  # (batch, 1, 64)
        img_key_value = img_features_proj.unsqueeze(1)  # (batch, 1, 64)

        attended_features, _ = self.cross_attention(
            time_query, img_key_value, img_key_value
        )
        attended_features = attended_features.squeeze(1)  # (batch, 64)

        # Combine features
        combined = torch.cat([attended_features, time_features], dim=1)  # (batch, 128)

        # Shared prediction
        shared_features = self.shared_predictor(combined)

        # Separate hand predictions
        hour_sincos = self.hour_hand_head(shared_features)
        minute_sincos = self.minute_hand_head(shared_features)
        second_sincos = self.second_hand_head(shared_features)

        # Normalize to unit circle
        hour_sincos = self.normalize_sincos(hour_sincos)
        minute_sincos = self.normalize_sincos(minute_sincos)
        second_sincos = self.normalize_sincos(second_sincos)

        # Concatenate
        angles_sincos = torch.cat([hour_sincos, minute_sincos, second_sincos], dim=1)

        return angles_sincos


def create_analog_manipulator_v2(config_dict):
    """Factory function for improved analog manipulator."""
    model = AnalogClockManipulatorV2(
        pretrained=config_dict.get('pretrained', True),
        time_embedding_dim=config_dict.get('time_embedding_dim', 64),
        dropout_rate=config_dict.get('dropout_rate', 0.4)
    )
    return model
