"""
Generate synthetic analog clock images for training Stage 2
Uses PIL to draw analog clock faces with hands at various positions
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import json
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List
import config
from src.utils.time_utils import generate_unique_times, time_to_angles, angles_to_sincos
import math


EXTENDED_STYLES = {
    # Light faces (original set)
    'light_faces': [
        ('#FFFFFF', '#000000'),   # white face, black border/hands
        ('#F0F0F0', '#222222'),
        ('#FFF8DC', '#333333'),   # cornsilk
        ('#FFFACD', '#444444'),   # lemon chiffon
        ('#F5F5DC', '#333333'),   # beige
        ('#E8F4F8', '#1a3a5c'),   # light blue face, navy accents
    ],
    # Dark / coloured faces - hands are drawn in light colours
    'dark_faces': [
        ('#1a1a2e', '#c8d0e0'),   # navy, silver hands
        ('#2d2d2d', '#e8e8e8'),   # charcoal, white hands
        ('#3c2415', '#d4a96a'),   # dark brown, gold hands
        ('#0d3320', '#a8d4b0'),   # dark green, pale-green hands
        ('#1c1c3a', '#f0e0c0'),   # midnight blue, cream hands
    ],
    # Gold / cream tones
    'warm_faces': [
        ('#f5e6c8', '#6b4b1a'),   # cream face, brown accents
        ('#e8d5a0', '#5a3e10'),
        ('#d4c080', '#3d2a00'),
    ],
}

# Hand style presets: (hour_w, min_w, sec_w, hour_len, min_len, sec_len, style)
HAND_PRESETS = [
    (6,  4,  2,  0.50, 0.70, 0.85, 'classic'),
    (8,  5,  2,  0.48, 0.68, 0.82, 'baton'),      # thick baton hands
    (4,  3,  1,  0.52, 0.72, 0.87, 'slim'),       # thin dress-watch hands
    (7,  5,  2,  0.45, 0.65, 0.80, 'sport'),      # short, thick sport hands
    (5,  3,  2,  0.55, 0.75, 0.90, 'skeleton'),   # long slim hands
]


class AnalogClockGenerator:
    """Generate synthetic analog clock images with various styles."""

    def __init__(self, output_dir: str, image_size: Tuple[int, int] = (224, 224)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.center = (image_size[0] // 2, image_size[1] // 2)
        self.radius = min(image_size) // 2 - 20
        self.metadata = []

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _draw_clock_face(
        self,
        draw: ImageDraw.Draw,
        bg_color: str,
        face_color: str,
        border_width: int,
        border_color: str,
        number_style: str,
        tick_marks: bool
    ):
        """Draw the clock face (background, border, numbers, ticks)."""
        # Draw outer circle (border)
        border_rgb = self._hex_to_rgb(border_color)
        draw.ellipse(
            [self.center[0] - self.radius, self.center[1] - self.radius,
             self.center[0] + self.radius, self.center[1] + self.radius],
            fill=self._hex_to_rgb(face_color),
            outline=border_rgb,
            width=border_width
        )

        # Draw center dot
        center_dot_radius = 5
        draw.ellipse(
            [self.center[0] - center_dot_radius, self.center[1] - center_dot_radius,
             self.center[0] + center_dot_radius, self.center[1] + center_dot_radius],
            fill=border_rgb
        )

        # Draw tick marks
        if tick_marks:
            for i in range(60):
                angle = math.radians(i * 6 - 90)  # -90 to start from 12 o'clock
                if i % 5 == 0:  # Hour marks
                    outer_len = self.radius - 5
                    inner_len = self.radius - 15
                    width = 3
                else:  # Minute marks
                    outer_len = self.radius - 5
                    inner_len = self.radius - 10
                    width = 1

                x1 = self.center[0] + inner_len * math.cos(angle)
                y1 = self.center[1] + inner_len * math.sin(angle)
                x2 = self.center[0] + outer_len * math.cos(angle)
                y2 = self.center[1] + outer_len * math.sin(angle)

                draw.line([(x1, y1), (x2, y2)], fill=border_rgb, width=width)

        # Draw numbers
        if number_style == 'arabic':
            try:
                font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 16)
            except:
                font = ImageFont.load_default()

            for i in range(1, 13):
                angle = math.radians(i * 30 - 90)
                num_radius = self.radius - 30
                x = self.center[0] + num_radius * math.cos(angle)
                y = self.center[1] + num_radius * math.sin(angle)

                text = str(i)
                bbox = draw.textbbox((x, y), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                draw.text((x - text_width/2, y - text_height/2), text, fill=border_rgb, font=font)

        elif number_style == 'roman':
            try:
                font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 14)
            except:
                font = ImageFont.load_default()

            roman = ['XII', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI']
            for i in range(12):
                angle = math.radians(i * 30 - 90)
                num_radius = self.radius - 30
                x = self.center[0] + num_radius * math.cos(angle)
                y = self.center[1] + num_radius * math.sin(angle)

                text = roman[i]
                bbox = draw.textbbox((x, y), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                draw.text((x - text_width/2, y - text_height/2), text, fill=border_rgb, font=font)

        elif number_style == 'dots':
            for i in range(12):
                angle = math.radians(i * 30 - 90)
                num_radius = self.radius - 25
                x = self.center[0] + num_radius * math.cos(angle)
                y = self.center[1] + num_radius * math.sin(angle)
                dot_radius = 3
                draw.ellipse(
                    [x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius],
                    fill=border_rgb
                )

    def _draw_hand(
        self,
        draw: ImageDraw.Draw,
        angle_deg: float,
        length_ratio: float,
        width: int,
        color: Tuple[int, int, int],
        hand_style: str = 'classic',
    ):
        """Draw a clock hand with optional style variation."""
        angle_rad = math.radians(angle_deg - 90)
        length = self.radius * length_ratio

        end_x = self.center[0] + length * math.cos(angle_rad)
        end_y = self.center[1] + length * math.sin(angle_rad)

        draw.line([self.center, (end_x, end_y)], fill=color, width=width)

        if hand_style == 'baton':
            # Baton: short counterweight tail in opposite direction
            tail_len = length * 0.18
            tx = self.center[0] - tail_len * math.cos(angle_rad)
            ty = self.center[1] - tail_len * math.sin(angle_rad)
            draw.line([self.center, (tx, ty)], fill=color, width=max(2, width - 1))

    def generate_clock_image(
        self,
        hour: int,
        minute: int,
        second: int,
        bg_color: str,
        face_color: str,
        border_width: int,
        border_color: str,
        number_style: str,
        tick_marks: bool,
        hand_color: Tuple[int, int, int] = (30, 30, 30),
        second_color: Tuple[int, int, int] = (200, 10, 10),
        hand_preset: tuple = None,
        hand_style: str = 'classic',
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Generate a single analog clock image.

        Returns:
            Tuple of (PIL Image, hand angles as sin/cos array)
        """
        bg_rgb = self._hex_to_rgb(bg_color)
        img = Image.new('RGB', self.image_size, bg_rgb)
        draw = ImageDraw.Draw(img)

        self._draw_clock_face(
            draw, bg_color, face_color, border_width,
            border_color, number_style, tick_marks
        )

        hour_angle, minute_angle, second_angle = time_to_angles(hour, minute, second)

        if hand_preset:
            h_w, m_w, s_w, h_len, m_len, s_len, _ = hand_preset
        else:
            h_w   = config.STAGE2_CONFIG['hand_rendering']['hour_hand_width']
            m_w   = config.STAGE2_CONFIG['hand_rendering']['minute_hand_width']
            s_w   = config.STAGE2_CONFIG['hand_rendering']['second_hand_width']
            h_len = config.STAGE2_CONFIG['hand_rendering']['hour_hand_length']
            m_len = config.STAGE2_CONFIG['hand_rendering']['minute_hand_length']
            s_len = config.STAGE2_CONFIG['hand_rendering']['second_hand_length']

        self._draw_hand(draw, hour_angle,   h_len, h_w, hand_color,   hand_style)
        self._draw_hand(draw, minute_angle, m_len, m_w, hand_color,   hand_style)
        self._draw_hand(draw, second_angle, s_len, s_w, second_color, hand_style)

        sincos_angles = angles_to_sincos(hour_angle, minute_angle, second_angle)
        return img, sincos_angles

    def generate_dataset(
        self,
        unique_times: List[Tuple[int, int, int]],
        variations_per_time: int = 10
    ):
        """Generate complete dataset of analog clock images."""
        print(f"\n🕐 Generating analog clock dataset...")
        print(f"   Unique times: {len(unique_times)}")
        print(f"   Variations per time: {variations_per_time}")
        print(f"   Total images: {len(unique_times) * variations_per_time}")

        image_counter = 0

        # Flatten all face/hand colour combos for balanced sampling
        all_face_combos = (
            EXTENDED_STYLES['light_faces'] +
            EXTENDED_STYLES['dark_faces'] +
            EXTENDED_STYLES['warm_faces']
        )

        for time_idx, (hour, minute, second) in enumerate(tqdm(unique_times, desc="Generating clocks")):
            for _ in range(variations_per_time):
                # Pick a face + border/accent colour pair
                face_hex, accent_hex = random.choice(all_face_combos)
                accent_rgb = self._hex_to_rgb(accent_hex)

                # Background = slightly different shade from face
                bg_color   = face_hex
                face_color = face_hex
                border_color = accent_hex

                # Determine if this is a dark face → light hands
                face_rgb = self._hex_to_rgb(face_hex)
                luminance = 0.299 * face_rgb[0] + 0.587 * face_rgb[1] + 0.114 * face_rgb[2]
                if luminance < 100:
                    hand_color   = accent_rgb                   # light hands on dark face
                    second_color = (220, 80, 80)                # readable red
                else:
                    hand_color   = (30, 30, 30)                 # dark hands on light face
                    second_color = (200, 10, 10)

                border_width = random.choice(config.ANALOG_CLOCK_STYLES['border_widths'])
                number_style = random.choice(config.ANALOG_CLOCK_STYLES['number_styles'])
                tick_marks   = random.choice(config.ANALOG_CLOCK_STYLES['tick_marks'])
                hand_preset  = random.choice(HAND_PRESETS)
                hand_style   = hand_preset[6]

                img, sincos_angles = self.generate_clock_image(
                    hour, minute, second,
                    bg_color, face_color, border_width, border_color,
                    number_style, tick_marks,
                    hand_color=hand_color,
                    second_color=second_color,
                    hand_preset=hand_preset,
                    hand_style=hand_style,
                )

                filename = f"analog_{image_counter:05d}.png"
                filepath = self.output_dir / filename
                img.save(filepath)

                self.metadata.append({
                    'filename': filename,
                    'time_index': time_idx,
                    'hour': hour,
                    'minute': minute,
                    'second': second,
                    'sincos_angles': sincos_angles.tolist(),
                    'bg_color': bg_color,
                    'face_color': face_color,
                    'number_style': number_style,
                    'hand_style': hand_style,
                    'dark_face': luminance < 100,
                })

                image_counter += 1

        # Save metadata
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"✓ Generated {image_counter} images")
        print(f"✓ Metadata saved to {metadata_path}")


def main():
    """Main function to generate analog clock dataset."""
    print("=" * 60)
    print("ANALOG CLOCK DATASET GENERATION")
    print("=" * 60)

    # Generate unique times
    unique_times = generate_unique_times(
        num_times=config.MAX_UNIQUE_TIMES,
        seed=config.RANDOM_SEED
    )

    # Create generator
    generator = AnalogClockGenerator(
        output_dir=config.ANALOG_CLOCKS_DIR,
        image_size=config.ANALOG_CLOCK_SIZE
    )

    # Generate dataset
    generator.generate_dataset(
        unique_times=unique_times,
        variations_per_time=config.ANALOG_VARIATIONS_PER_TIME
    )

    print("\n✅ Analog clock dataset generation complete!")
    print(f"   Output directory: {config.ANALOG_CLOCKS_DIR}")


if __name__ == '__main__':
    main()
