"""
Configuration file for Digital-to-Analog Clock Conversion System
Author: Advanced Topics in Data Science - Final Project
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DIGITAL_CLOCKS_DIR = DATA_DIR / "digital_clocks"
ANALOG_CLOCKS_DIR = DATA_DIR / "analog_clocks"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

# Create directories if they don't exist
for directory in [DATA_DIR, DIGITAL_CLOCKS_DIR, ANALOG_CLOCKS_DIR,
                  CHECKPOINTS_DIR, OUTPUTS_DIR, VISUALIZATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Maximum 400 unique time combinations as per project requirements
MAX_UNIQUE_TIMES = 400

# Number of image variations per unique time
DIGITAL_VARIATIONS_PER_TIME = 5  # Total: 400 * 5 = 2000 digital images
ANALOG_VARIATIONS_PER_TIME = 10  # Total: 400 * 10 = 4000 analog images

# Image dimensions
IMAGE_SIZE = 224  # Standard size for ResNet
DIGITAL_CLOCK_SIZE = (224, 224)
ANALOG_CLOCK_SIZE = (224, 224)

# Train/Val/Test split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# STAGE 1: DIGITAL CLOCK READER CONFIGURATION
# ============================================================================
STAGE1_CONFIG = {
    'model_name': 'digital_clock_reader',
    'architecture': 'resnet18',
    'pretrained': True,  # Use ImageNet pretrained weights
    'num_outputs': 3,  # (hour, minute, second)
    'output_mode': 'regression',  # 'regression' or 'classification'

    # Training hyperparameters
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'optimizer': 'adam',
    'scheduler': 'reduce_on_plateau',
    'scheduler_patience': 10,
    'scheduler_factor': 0.5,

    # Regularization
    'dropout_rate': 0.3,
    'early_stopping_patience': 20,

    # Data augmentation
    'augmentation': {
        'rotation_range': 5,  # degrees
        'brightness_range': (0.8, 1.2),
        'contrast_range': (0.8, 1.2),
        'hue_shift': 0.1,
        'noise_std': 0.05,
    },

    # Normalization (ImageNet stats)
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225],
}

# ============================================================================
# STAGE 2: ANALOG CLOCK MANIPULATOR CONFIGURATION
# ============================================================================
STAGE2_CONFIG = {
    'model_name': 'analog_clock_manipulator',
    'architecture': 'resnet18',
    'pretrained': True,

    # Output: 6 values (sin, cos for each of 3 hands)
    'num_outputs': 6,  # (sin_h, cos_h, sin_m, cos_m, sin_s, cos_s)
    'time_embedding_dim': 64,  # Embedding dimension for time input (updated for V2)

    # Training hyperparameters
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'optimizer': 'adam',
    'scheduler': 'reduce_on_plateau',
    'scheduler_patience': 10,
    'scheduler_factor': 0.5,

    # Regularization
    'dropout_rate': 0.3,
    'early_stopping_patience': 20,

    # Hand rendering parameters
    'hand_rendering': {
        'hour_hand_length': 0.50,   # Relative to clock radius
        'minute_hand_length': 0.70,
        'second_hand_length': 0.85, # Slightly longer so it nearly reaches tick marks
        'hour_hand_width': 6,       # pixels
        'minute_hand_width': 4,
        'second_hand_width': 2,
        'hour_hand_color': (0, 0, 0),    # RGB
        'minute_hand_color': (0, 0, 0),
        'second_hand_color': (220, 0, 0),  # Slightly darker red for second hand
    },

    # Normalization
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225],
}

# ============================================================================
# DIGITAL CLOCK GENERATION PARAMETERS
# ============================================================================
DIGITAL_CLOCK_STYLES = {
    # Keep training focused on readability to improve exact-match accuracy.
    # You can re-add more styles later once baseline accuracy is very strong.
    'fonts': [
        '/System/Library/Fonts/Helvetica.ttc',
        '/System/Library/Fonts/Monaco.dfont',
        '/System/Library/Fonts/Courier.dfont',
    ],
    'font_sizes': [60, 72, 84, 96],
    'colors': [
        '#000000',  # Black
    ],
    'backgrounds': [
        '#FFFFFF',  # White
        '#F0F0F0',  # Light gray
        '#E8E8E8',  # Very light gray
    ],
    'formats': [
        'HH:MM:SS',
    ],
}

# ============================================================================
# ANALOG CLOCK GENERATION PARAMETERS
# ============================================================================
ANALOG_CLOCK_STYLES = {
    'backgrounds': [
        '#FFFFFF',  # White
        '#F0F0F0',  # Light gray
        '#FFF8DC',  # Cornsilk
        '#FFFACD',  # Lemon chiffon
        '#F5F5DC',  # Beige
    ],
    'face_colors': [
        '#FFFFFF',
        '#F0F0F0',
        '#E8E8E8',
    ],
    'number_styles': ['roman', 'arabic', 'none', 'dots'],
    'tick_marks': [True, False],
    'border_widths': [2, 4, 6, 8],
    'border_colors': ['#000000', '#333333', '#666666'],
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================
EVALUATION_CONFIG = {
    'metrics': [
        'exact_match_accuracy',  # All three components correct
        'hour_accuracy',
        'minute_accuracy',
        'second_accuracy',
        'mean_absolute_error',
        'mean_angle_error',  # For Stage 2
    ],
    'angle_tolerance': 5.0,  # degrees - acceptable error for hand angles
}

# ============================================================================
# LOGGING & VISUALIZATION
# ============================================================================
LOGGING_CONFIG = {
    'log_interval': 10,  # batches
    'save_interval': 5,  # epochs
    'visualize_predictions': True,
    'num_visualize_samples': 16,
    'tensorboard': True,
}

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
import torch
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

# Dataloading defaults
NUM_WORKERS = 4 if DEVICE == 'cuda' else 0
PIN_MEMORY = True if DEVICE == 'cuda' else False

print(f"🚀 Configuration loaded. Using device: {DEVICE}")
