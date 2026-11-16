# Digital-to-Analog Clock Converter

A computer-vision pipeline that takes **a photo of a digital clock** and **a photo of any analog clock**, and produces a new image of that analog clock showing the digital clock's time. The output keeps the clock's face, and the new hands copy the colours and proportions of the old ones.

Team project for the *Advanced Topics in Data Science* course (B.Sc. Data Science). Every conversion step uses networks we trained ourselves in PyTorch.

## Results

| Stage | Test set | Metric | Result |
|---|---|---|---|
| Digital reader | 300 images | Exact HH:MM:SS match | **100%** |
| Angle network | 600 images | Mean hand-angle error | **1.52°** (hour 0.88°, minute 2.21°, second 1.46°) |
| Angle network | 600 images | Median hand-angle error | hour 0.52°, minute 1.58°, second 1.28° |

The course allowed **at most 400 unique times** in training. We used 400 times × (5 digital + 10 analog) variations, about 6,000 synthetic images in total.

> The test images come from the same synthetic generators as the training images, with times not used in training. Accuracy on real photos is lower and depends on the clock design.

## Pipeline

```
digital clock image ──► [Stage 1] Digital reader (CNN) ──► (h, m, s)
                                                              │
analog clock image ─────────────────────────────────────┐     ▼
                                                         ├─► [Stage 2] Angle network ──► hand angles
                                                         │
                                                         └─► [Stage 3] Hand editor ──► output image
```

### Stage 1: Digital reader (`src/models/digital_reader_v2.py`)
- ResNet-18 backbone with a **spatial-attention module** after `layer3`. A 1×1 conv with a sigmoid produces a mask that focuses the network on the digit area.
- **Six parallel digit heads** (10 classes each) instead of regressing the time. Regression mixed up neighbouring values, and classification fixed that.
- Optional test-time augmentation at inference (`--no-tta` turns it off).

### Stage 2: Angle network (`src/models/analog_manipulator_v2.py`)
- Image features from a ResNet-18 are combined with an embedding of the target time through **cross-attention**.
- Outputs **sin/cos pairs** for each hand, normalised to the unit circle. This avoids the jump between 359° and 0°.
- Trained with a **circular loss**: MSE on sin/cos plus an `atan2` wrapped angle-difference term.
- Cosine-annealing warm restarts.

### Stage 3: Hand editor (`src/utils/hand_editor.py`)
- Finds the existing hands as line candidates with `HoughLinesP`, filtered by length and position relative to the clock centre.
- Erases them with a mask fill and **TELEA inpainting** on the mask boundary to avoid smudges.
- Redraws the hands at the predicted angles (`src/utils/render_hands.py`), with colour (sampled in HSV) and proportions taken from the original hands.

### Demo (`app.py`)
A Gradio app that converts uploaded image pairs and shows a **live ticking clock**: the input clock animated to the current time, with the second hand moving every second.

## Project structure

```
app.py                     Gradio demo
config.py                  paths and hyper-parameters
src/models/                digital reader + angle network (v1 baselines and v2)
src/data/                  synthetic clock generators and PyTorch datasets
src/training/              training scripts for stage 1 and stage 2
src/utils/                 time/angle utilities, hand detection, rendering
```

## Running

```bash
pip install -r requirements.txt

# 1. generate the synthetic datasets (400 unique times)
python src/data/generate_digital_clocks.py
python src/data/generate_analog_clocks.py

# 2. train both networks (checkpoints are written to checkpoints/)
python src/training/train_stage1_v2.py
python src/training/train_stage2_v2.py

# 3. launch the demo
python app.py
```

Trained weights are not included in this repository, so run the training steps first.

## Tech

PyTorch · torchvision · OpenCV · scikit-image · Gradio · NumPy · Selenium (for clocks rendered from HTML/CSS)
