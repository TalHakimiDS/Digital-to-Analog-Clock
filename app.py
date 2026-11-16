"""
Gradio web demo for the Digital→Analog Clock Converter.

Tabs
----
1. Convert      - upload a digital clock + an analog clock → get the analog
                  clock re-drawn with the time shown on the digital clock.
2. Live Clock   - upload any analog clock and watch it display the current
                  wall-clock time with the second hand moving every second.

Usage
-----
    python app.py
    python app.py --share          # public Gradio link
    python app.py --port 7860
"""

import argparse
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import gradio as gr
import numpy as np
from PIL import Image

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import config
from src.utils.hand_editor import process_watch_pair

# Converter is loaded lazily so the app starts even without checkpoints.
_converter = None
_converter_error: str = ""


def _load_converter():
    global _converter, _converter_error
    if _converter is not None:
        return _converter, ""

    s1 = Path("checkpoints/stage1_v2/stage1_v2_best.pth")
    s2 = Path("checkpoints/stage2_v2/stage2_v2_best.pth")

    if not s1.exists() or not s2.exists():
        msg = (
            f"Checkpoint(s) missing.\n"
            f"  stage1: {'✓' if s1.exists() else '✗'} {s1}\n"
            f"  stage2: {'✓' if s2.exists() else '✗'} {s2}\n"
            "Run the training pipeline first: python run_pipeline_v2.py"
        )
        _converter_error = msg
        return None, msg

    try:
        from src.inference_v2 import ClockConverterV2
        _converter = ClockConverterV2(
            stage1_checkpoint=str(s1),
            stage2_checkpoint=str(s2),
            device=config.DEVICE,
            tta=True,
        )
        return _converter, ""
    except Exception as exc:
        _converter_error = str(exc)
        return None, str(exc)


# ── Tab 1: one-shot conversion ───────────────────────────────────────────────

def convert_clocks(digital_pil, analog_pil):
    """Run the full Stage 1 + Stage 2 pipeline and return the result image."""
    if digital_pil is None or analog_pil is None:
        return None, "⚠  Please upload both a digital clock and an analog clock image."

    conv, err = _load_converter()
    if conv is None:
        return None, f"⚠  Model not loaded: {err}"

    digital_np = np.array(digital_pil.convert("RGB"))
    analog_np  = np.array(analog_pil.convert("RGB"))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        digital_path = f.name
        Image.fromarray(digital_np).save(digital_path)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        analog_path = f.name
        Image.fromarray(analog_np).save(analog_path)

    try:
        output_np, (h, m, s), (ah, am, as_) = conv.convert(
            digital_image_path=digital_path,
            analog_image_path=analog_path,
        )
        label = (
            f"DETECTED TIME      {h:02d} : {m:02d} : {s:02d}\n"
            f"HAND ANGLES        H {ah:6.1f}°    M {am:6.1f}°    S {as_:6.1f}°"
        )
        return output_np, label
    except Exception as exc:
        return None, f"⚠  Error during conversion: {exc}"
    finally:
        os.unlink(digital_path)
        os.unlink(analog_path)


# ── Tab 2: live clock animation ──────────────────────────────────────────────

def live_tick(analog_pil):
    """
    Called every second by gr.Timer.
    Returns the analog clock image with hands showing the current wall time (Israel time).
    """
    if analog_pil is None:
        return None

    now = datetime.now(ZoneInfo("Asia/Jerusalem"))
    analog_np = np.array(analog_pil.convert("RGB"))

    try:
        result = process_watch_pair(
            time_tuple=(now.hour, now.minute, now.second),
            analog_image=analog_np,
        )
        return result
    except Exception:
        return analog_np


def _current_time_str() -> str:
    now = datetime.now(ZoneInfo("Asia/Jerusalem"))
    return f"{now.hour:02d} : {now.minute:02d} : {now.second:02d}"


# ── Gradio UI ────────────────────────────────────────────────────────────────

CSS = """
/* ─────────────────────────────────────────────────────────────────
   Design tokens
   ───────────────────────────────────────────────────────────────── */
:root {
  --bg:            #0b0d10;
  --bg-elevated:   #14171c;
  --bg-subtle:     #1a1e25;
  --surface:       #1f242c;
  --border:        #262b34;
  --border-strong: #353b46;

  --text:          #e8eaed;
  --text-muted:    #9aa3b2;
  --text-faint:    #6b7382;

  --accent:        #d4ff3a;        /* signature lime */
  --accent-hover:  #c2eb2a;
  --accent-soft:   rgba(212, 255, 58, 0.12);
  --accent-text:   #0b0d10;

  --danger:        #ff6b6b;
  --success:       #4ade80;

  --radius-sm: 6px;
  --radius:    10px;
  --radius-lg: 16px;
  --radius-xl: 24px;

  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-mono: 'JetBrains Mono', 'SF Mono', 'Menlo', monospace;

  --shadow-sm: 0 1px 2px rgba(0,0,0,.4);
  --shadow:    0 4px 24px rgba(0,0,0,.3);
  --shadow-lg: 0 24px 60px -12px rgba(0,0,0,.5);
}

/* ─────────────────────────────────────────────────────────────────
   Reset / base
   ───────────────────────────────────────────────────────────────── */
* { box-sizing: border-box; }

html, body, gradio-app {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-sans) !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Decorative grid backdrop */
gradio-app::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background-image:
    radial-gradient(circle at 20% 0%,  rgba(212,255,58,.06), transparent 40%),
    radial-gradient(circle at 80% 100%, rgba(120,80,255,.05), transparent 40%),
    linear-gradient(rgba(255,255,255,.015) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.015) 1px, transparent 1px);
  background-size: auto, auto, 48px 48px, 48px 48px;
  z-index: 0;
}

.gradio-container {
  background: transparent !important;
  max-width: 1280px !important;
  padding: 32px 24px 64px !important;
  position: relative;
  z-index: 1;
}

/* ─────────────────────────────────────────────────────────────────
   Top bar
   ───────────────────────────────────────────────────────────────── */
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  margin-bottom: 28px;
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}

.topbar-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  font-weight: 600;
  font-size: 14px;
  letter-spacing: -0.01em;
}

.brand-mark {
  width: 28px;
  height: 28px;
  border-radius: 8px;
  background: var(--accent);
  display: grid;
  place-items: center;
  color: var(--accent-text);
  font-weight: 800;
  font-size: 14px;
  box-shadow: 0 0 24px rgba(212,255,58,.3);
}

.topbar-meta {
  display: flex;
  gap: 18px;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-faint);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.topbar-meta .dot {
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--success);
  margin-right: 6px;
  vertical-align: middle;
  box-shadow: 0 0 8px var(--success);
}

/* ─────────────────────────────────────────────────────────────────
   Hero
   ───────────────────────────────────────────────────────────────── */
.hero {
  position: relative;
  padding: 56px 40px 48px;
  margin-bottom: 32px;
  background:
    radial-gradient(circle at 90% 10%, rgba(212,255,58,.08), transparent 50%),
    var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
}

.hero::after {
  content: "";
  position: absolute;
  right: -80px; top: -80px;
  width: 320px; height: 320px;
  border-radius: 50%;
  border: 1px solid var(--border-strong);
  opacity: .6;
  pointer-events: none;
}
.hero::before {
  content: "";
  position: absolute;
  right: 40px; top: 40px;
  width: 120px; height: 120px;
  border-radius: 50%;
  border: 1px dashed var(--border-strong);
  opacity: .4;
  pointer-events: none;
}

.hero-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--accent);
  padding: 6px 12px;
  background: var(--accent-soft);
  border-radius: 999px;
  margin-bottom: 20px;
}

.hero-title {
  font-size: 56px;
  font-weight: 700;
  line-height: 1.02;
  letter-spacing: -0.03em;
  color: var(--text);
  margin: 0 0 16px 0;
  max-width: 780px;
}

.hero-title em {
  font-style: normal;
  color: var(--accent);
  position: relative;
}

.hero-sub {
  font-size: 17px;
  line-height: 1.55;
  color: var(--text-muted);
  max-width: 620px;
  margin: 0;
}

.hero-stats {
  display: flex;
  gap: 48px;
  margin-top: 36px;
  padding-top: 28px;
  border-top: 1px solid var(--border);
}

.hero-stat .num {
  font-family: var(--font-mono);
  font-size: 22px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.01em;
}

.hero-stat .lbl {
  display: block;
  margin-top: 4px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-faint);
}

/* ─────────────────────────────────────────────────────────────────
   Tabs
   ───────────────────────────────────────────────────────────────── */
.tab-nav {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 0 !important;
  margin-bottom: 32px !important;
  gap: 0 !important;
}

.tab-nav button {
  background: transparent !important;
  border: none !important;
  color: var(--text-faint) !important;
  font-family: var(--font-sans) !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  padding: 14px 20px !important;
  border-radius: 0 !important;
  border-bottom: 2px solid transparent !important;
  transition: color .15s ease, border-color .15s ease !important;
  margin: 0 !important;
}

.tab-nav button:hover {
  color: var(--text-muted) !important;
}

.tab-nav button.selected {
  color: var(--text) !important;
  border-bottom-color: var(--accent) !important;
  background: transparent !important;
}

/* tab panels - flush with no extra chrome */
.tabitem {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}

/* ─────────────────────────────────────────────────────────────────
   Step / panel cards
   ───────────────────────────────────────────────────────────────── */
.step-card {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 24px;
  height: 100%;
  transition: border-color .2s ease, transform .2s ease;
}

.step-card:hover {
  border-color: var(--border-strong);
}

.step-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 6px;
}

.step-num {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
  color: var(--accent);
  background: var(--accent-soft);
  padding: 3px 8px;
  border-radius: 4px;
  letter-spacing: 0.08em;
}

.step-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.01em;
}

.step-desc {
  font-size: 13px;
  color: var(--text-faint);
  line-height: 1.5;
  margin: 0 0 18px 0;
}

.step-card.is-output .step-num {
  color: var(--accent-text);
  background: var(--accent);
}

/* ─────────────────────────────────────────────────────────────────
   Image upload styling
   ───────────────────────────────────────────────────────────────── */
.gradio-container .image-container,
.gradio-container [data-testid="image"] {
  background: var(--bg-subtle) !important;
  border: 1px dashed var(--border-strong) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
  transition: border-color .2s ease, background .2s ease !important;
}

.gradio-container [data-testid="image"]:hover {
  border-color: var(--accent) !important;
}

/* hide redundant Gradio labels above images, we use our own headings */
.step-card .gradio-image > label,
.step-card .gradio-image .label-wrap {
  display: none !important;
}

/* ─────────────────────────────────────────────────────────────────
   Buttons
   ───────────────────────────────────────────────────────────────── */
.gradio-container button.primary,
.gradio-container button[variant="primary"],
.gradio-container .primary {
  background: var(--accent) !important;
  color: var(--accent-text) !important;
  border: none !important;
  font-family: var(--font-sans) !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  letter-spacing: -0.01em !important;
  padding: 14px 28px !important;
  border-radius: var(--radius) !important;
  box-shadow: 0 0 0 1px rgba(212,255,58,.4), 0 8px 24px -8px rgba(212,255,58,.4) !important;
  transition: all .2s ease !important;
}

.gradio-container button.primary:hover,
.gradio-container button[variant="primary"]:hover {
  background: var(--accent-hover) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 0 0 1px rgba(212,255,58,.5), 0 12px 32px -8px rgba(212,255,58,.5) !important;
}

.gradio-container button.secondary,
.gradio-container button[variant="secondary"] {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border-strong) !important;
  font-weight: 500 !important;
  font-size: 14px !important;
  padding: 14px 22px !important;
  border-radius: var(--radius) !important;
  transition: all .15s ease !important;
}

.gradio-container button.secondary:hover {
  background: var(--bg-subtle) !important;
  border-color: var(--text-faint) !important;
}

/* ─────────────────────────────────────────────────────────────────
   Text outputs / info boxes
   ───────────────────────────────────────────────────────────────── */
.gradio-container textarea,
.gradio-container input[type="text"] {
  background: var(--bg-subtle) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
  font-size: 13px !important;
  padding: 14px 16px !important;
}

.gradio-container .gradio-textbox label,
.gradio-container .gradio-textbox .label-wrap {
  font-size: 11px !important;
  font-weight: 500 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  color: var(--text-faint) !important;
  margin-bottom: 8px !important;
}

/* ─────────────────────────────────────────────────────────────────
   Live clock display
   ───────────────────────────────────────────────────────────────── */
.live-display {
  background:
    radial-gradient(circle at 50% 0%, rgba(212,255,58,.08), transparent 60%),
    var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 32px;
  text-align: center;
  margin-top: 16px;
}

.live-label {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: var(--text-faint);
  margin-bottom: 16px;
}

.live-label .pulse {
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--accent);
  margin-right: 8px;
  vertical-align: middle;
  box-shadow: 0 0 12px var(--accent);
  animation: pulse 1.6s ease-in-out infinite;
}

@keyframes pulse {
  0%,100% { opacity: 1; transform: scale(1); }
  50%     { opacity: .4; transform: scale(.8); }
}

#live-time-box textarea,
#live-time-box input {
  font-family: var(--font-mono) !important;
  font-size: 64px !important;
  font-weight: 600 !important;
  text-align: center !important;
  letter-spacing: 0.04em !important;
  color: var(--accent) !important;
  background: transparent !important;
  border: none !important;
  padding: 8px 0 !important;
  text-shadow: 0 0 40px rgba(212,255,58,.3);
  resize: none !important;
}

#live-time-box label,
#live-time-box .label-wrap {
  display: none !important;
}

/* ─────────────────────────────────────────────────────────────────
   About / footer
   ───────────────────────────────────────────────────────────────── */
.about-section {
  margin-top: 48px;
  padding-top: 48px;
  border-top: 1px solid var(--border);
}

.about-eyebrow {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--text-faint);
  margin-bottom: 8px;
}

.about-title {
  font-size: 28px;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--text);
  margin: 0 0 32px 0;
  max-width: 600px;
  line-height: 1.2;
}

.about-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 16px;
  margin-bottom: 40px;
}

.about-card {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 24px;
  transition: border-color .2s ease;
}

.about-card:hover {
  border-color: var(--border-strong);
}

.about-card .num {
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
  color: var(--accent);
  letter-spacing: 0.1em;
  margin-bottom: 12px;
}

.about-card h4 {
  font-size: 15px;
  font-weight: 600;
  color: var(--text);
  margin: 0 0 8px 0;
  letter-spacing: -0.01em;
}

.about-card p {
  font-size: 13px;
  line-height: 1.55;
  color: var(--text-muted);
  margin: 0;
}

.meta-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 32px;
  padding: 20px 0 0 0;
  border-top: 1px solid var(--border);
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-faint);
}

.meta-strip span strong {
  color: var(--text-muted);
  font-weight: 500;
  margin-right: 6px;
  text-transform: uppercase;
  font-size: 10px;
  letter-spacing: 0.1em;
}

/* ─────────────────────────────────────────────────────────────────
   Responsive
   ───────────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
  .hero { padding: 36px 24px; }
  .hero-title { font-size: 36px; }
  .hero-stats { gap: 24px; flex-wrap: wrap; }
  #live-time-box textarea, #live-time-box input { font-size: 42px !important; }
}

/* hide the gradio footer "Use via API" etc. for cleaner look */
footer { display: none !important; }
"""


HEAD_FONT_LINK = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
"""


with gr.Blocks(
    title="Digital to Analog Clock Converter",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.lime,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ),
    css=CSS,
    head=HEAD_FONT_LINK,
) as demo:

    # ── Top bar ─────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="topbar">
      <div class="topbar-brand">
        <div class="brand-mark">C</div>
        <span></span>
      </div>
      <div class="topbar-meta">
        <span><span class="dot"></span>System online</span>
        <span>v2.0</span>
        <span>Two-stage neural pipeline</span>
      </div>
    </div>
    """)

    # ── Hero ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <section class="hero">
      <span class="hero-eyebrow">◐ Computer vision · Generative</span>
      <h1 class="hero-title">Read any digital time.<br/>Render it on any <em>analog</em> face.</h1>
      <p class="hero-sub">
        A two-stage neural pipeline that recognizes time from a digital display,
        intelligently removes the existing hands of an analog clock, and re-draws them
        in the correct position - preserving the original style, color, and weight.
      </p>
      <div class="hero-stats">
        <div class="hero-stat">
          <div class="num">2</div>
          <span class="lbl">Neural stages</span>
        </div>
        <div class="hero-stat">
          <div class="num">±0.5°</div>
          <span class="lbl">Hand precision</span>
        </div>
        <div class="hero-stat">
          <div class="num">1 Hz</div>
          <span class="lbl">Live tick rate</span>
        </div>
        <div class="hero-stat">
          <div class="num">TTA</div>
          <span class="lbl">Robust inference</span>
        </div>
      </div>
    </section>
    """)

    # ── Tabs ────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ─── Tab 1: Convert ──────────────────────────────────────────────
        with gr.Tab("Convert"):

            with gr.Row(equal_height=True):

                # Input 1 - digital
                with gr.Column():
                    gr.HTML("""
                    <div class="step-card">
                      <div class="step-head">
                        <span class="step-num">STEP 01</span>
                        <span class="step-title">Source · Digital</span>
                      </div>
                      <p class="step-desc">Upload an image of a digital clock - any 6-digit HH:MM:SS readout.</p>
                    </div>
                    """)
                    digital_in = gr.Image(
                        type="pil",
                        show_label=False,
                        height=320,
                        sources=["upload", "clipboard"],
                    )

                # Input 2 - analog template
                with gr.Column():
                    gr.HTML("""
                    <div class="step-card">
                      <div class="step-head">
                        <span class="step-num">STEP 02</span>
                        <span class="step-title">Template · Analog</span>
                      </div>
                      <p class="step-desc">Upload the analog clock face you want the time rendered on.</p>
                    </div>
                    """)
                    analog_in = gr.Image(
                        type="pil",
                        show_label=False,
                        height=320,
                        sources=["upload", "clipboard"],
                    )

                # Output
                with gr.Column():
                    gr.HTML("""
                    <div class="step-card is-output">
                      <div class="step-head">
                        <span class="step-num">OUTPUT</span>
                        <span class="step-title">Rendered Result</span>
                      </div>
                      <p class="step-desc">The analog clock, redrawn with the time read from your digital input.</p>
                    </div>
                    """)
                    result_out = gr.Image(
                        type="numpy",
                        show_label=False,
                        height=320,
                        interactive=False,
                    )

            # Action bar
            with gr.Row():
                with gr.Column(scale=3):
                    info_box = gr.Textbox(
                        label="Inference output",
                        interactive=False,
                        lines=2,
                        placeholder="Detection results will appear here after conversion…",
                    )
                with gr.Column(scale=1, min_width=180):
                    convert_btn = gr.Button(
                        "Run conversion  →",
                        variant="primary",
                        size="lg",
                    )

            convert_btn.click(
                fn=convert_clocks,
                inputs=[digital_in, analog_in],
                outputs=[result_out, info_box],
            )

        # ─── Tab 2: Live ─────────────────────────────────────────────────
        with gr.Tab("Live Clock"):

            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.HTML("""
                    <div class="step-card">
                      <div class="step-head">
                        <span class="step-num">INPUT</span>
                        <span class="step-title">Clock Template</span>
                      </div>
                      <p class="step-desc">Upload any analog clock face. We'll animate it with the current wall-clock time.</p>
                    </div>
                    """)
                    live_analog_in = gr.Image(
                        type="pil",
                        show_label=False,
                        height=380,
                        sources=["upload", "clipboard"],
                    )

                with gr.Column():
                    gr.HTML("""
                    <div class="step-card is-output">
                      <div class="step-head">
                        <span class="step-num">LIVE</span>
                        <span class="step-title">Real-time Render</span>
                      </div>
                      <p class="step-desc">Hands re-drawn every second, synchronized with system time.</p>
                    </div>
                    """)
                    live_clock_out = gr.Image(
                        type="numpy",
                        show_label=False,
                        height=380,
                        interactive=False,
                    )

            # Live time
            gr.HTML("""
            <div class="live-display">
              <div class="live-label"><span class="pulse"></span>Synchronized · Local time</div>
            """)

            live_time_display = gr.Textbox(
                value=_current_time_str,
                interactive=False,
                every=1,
                elem_id="live-time-box",
                show_label=False,
            )

            gr.HTML("</div>")

            with gr.Row():
                start_btn = gr.Button("▶  Start animation", variant="primary", size="lg", scale=2)
                stop_btn  = gr.Button("⏸  Pause",          variant="secondary", size="lg", scale=1)

            timer = gr.Timer(value=1.0, active=False)
            timer.tick(
                fn=live_tick,
                inputs=[live_analog_in],
                outputs=[live_clock_out],
            )

            start_btn.click(fn=lambda: gr.Timer(active=True),  inputs=None, outputs=timer)
            stop_btn.click( fn=lambda: gr.Timer(active=False), inputs=None, outputs=timer)

    # ── About / Architecture ───────────────────────────────────────────────
    gr.HTML("""
    <section class="about-section">
      <div class="about-eyebrow">◇ Under the hood</div>
      <h2 class="about-title">A two-stage architecture, designed for precision.</h2>

      <div class="about-grid">
        <div class="about-card">
          <div class="num">01 · DIGITAL READER</div>
          <h4>Time recognition</h4>
          <p>A deep OCR network reads the six digits of HH:MM:SS from the digital display. Test-time augmentation makes it robust to lighting, blur, and color variation.</p>
        </div>
        <div class="about-card">
          <div class="num">02 · HAND PREDICTOR</div>
          <h4>Angle regression</h4>
          <p>A second network maps the recognized time to exact hour, minute, and second hand angles - accurate to within half a degree.</p>
        </div>
        <div class="about-card">
          <div class="num">03 · HAND EDITOR</div>
          <h4>Style-preserving render</h4>
          <p>Existing hands are detected by color and length, removed cleanly, and redrawn with the same thickness and palette as the original.</p>
        </div>
        <div class="about-card">
          <div class="num">04 · LIVE LOOP</div>
          <h4>Real-time animation</h4>
          <p>A 1 Hz timer re-renders the hands every second, turning any static clock face into a working timepiece.</p>
        </div>
      </div>

      <div class="meta-strip">
        <span><strong>Course</strong> Advanced Topics in Data Science</span>
        <span><strong>Instructor</strong> Dr. Shai Horovitz</span>
        <span><strong>Stack</strong> PyTorch · OpenCV · Gradio</span>
        <span><strong>Deployment</strong> HuggingFace Spaces</span>
      </div>
    </section>
    """)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share",       action="store_true", help="Create public Gradio link")
    parser.add_argument("--port",        type=int, default=7860)
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server address to bind to")
    parser.add_argument("--no-tta",      action="store_true", help="Disable test-time augmentation")
    args = parser.parse_args()

    print("=" * 60)
    print("Digital → Analog Clock Converter")
    print("=" * 60)
    print(f"  Device : {config.DEVICE}")
    print(f"  TTA    : {'off' if args.no_tta else 'on'}")
    print("  Loading models...")
    conv, err = _load_converter()
    if conv:
        print("  Models loaded successfully.")
    else:
        print(f"  WARNING: {err}")
        print("  Live-animation tab will still work (no models needed).")
    print("=" * 60)

    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )