"""
Generate synthetic digital clock images for training Stage 1
Uses PIL to create digital clock displays with various styles
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
from typing import Tuple, List, Dict
import config
from src.utils.time_utils import generate_unique_times


class DigitalClockGenerator:
    """Generate synthetic digital clock images with various styles."""

    def __init__(self, output_dir: str, image_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            output_dir: Directory to save generated images
            image_size: Size of generated images (width, height)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.metadata = []

        # Get available system fonts
        self.fonts = self._get_available_fonts()

    def _get_available_fonts(self) -> List[str]:
        """
        Get list of available system fonts.
        Fallback to default fonts if specified fonts not available.
        """
        available_fonts = []

        # Try to use specified fonts from config
        for font_name in config.DIGITAL_CLOCK_STYLES['fonts']:
            try:
                # Try to create a test font
                ImageFont.truetype(font_name, 20)
                available_fonts.append(font_name)
            except:
                pass

        # Fallback fonts
        fallback_fonts = [
            '/System/Library/Fonts/Helvetica.ttc',
            '/System/Library/Fonts/Courier.dfont',
            '/System/Library/Fonts/Monaco.dfont',
            '/Library/Fonts/Arial.ttf',
            '/Library/Fonts/Courier New.ttf',
        ]

        for font_path in fallback_fonts:
            if os.path.exists(font_path) and font_path not in available_fonts:
                available_fonts.append(font_path)

        # If still no fonts, use default
        if not available_fonts:
            available_fonts = [None]  # Will use default PIL font

        print(f"✓ Found {len(available_fonts)} available fonts")
        return available_fonts

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color string to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _get_font(self, font_name: str, font_size: int) -> ImageFont.ImageFont:
        """Get a PIL font object."""
        try:
            if font_name is None:
                return ImageFont.load_default()
            return ImageFont.truetype(font_name, font_size)
        except:
            return ImageFont.load_default()

    def _format_time(self, hour: int, minute: int, second: int, format_type: str) -> str:
        """
        Format time string according to format type.

        Args:
            hour, minute, second: Time components
            format_type: One of 'HH:MM:SS', 'HH MM SS', 'HH-MM-SS'

        Returns:
            Formatted time string
        """
        h_str = f"{hour:02d}"
        m_str = f"{minute:02d}"
        s_str = f"{second:02d}"

        if format_type == 'HH:MM:SS':
            return f"{h_str}:{m_str}:{s_str}"
        elif format_type == 'HH MM SS':
            return f"{h_str} {m_str} {s_str}"
        elif format_type == 'HH-MM-SS':
            return f"{h_str}-{m_str}-{s_str}"
        else:
            return f"{h_str}:{m_str}:{s_str}"

    def generate_clock_image(
        self,
        hour: int,
        minute: int,
        second: int,
        font_name: str,
        font_size: int,
        text_color: str,
        bg_color: str,
        format_type: str,
        add_noise: bool = True
    ) -> Image.Image:
        """
        Generate a single digital clock image with better contrast and readability.

        Args:
            hour, minute, second: Time to display
            font_name: Font to use
            font_size: Font size
            text_color: Text color (hex)
            bg_color: Background color (hex)
            format_type: Time format
            add_noise: Whether to add noise for augmentation

        Returns:
            PIL Image of the digital clock
        """
        # Create blank image
        bg_rgb = self._hex_to_rgb(bg_color)
        img = Image.new('RGB', self.image_size, bg_rgb)
        draw = ImageDraw.Draw(img)

        # Format time string
        time_str = self._format_time(hour, minute, second, format_type)

        # Get font (larger for better readability) but ensure it fits in the image
        base_size = min(font_size + 12, 96)
        font = self._get_font(font_name, base_size)

        # Iteratively shrink font until the full text fits inside the image width
        padding = 8
        max_text_width = self.image_size[0] - 2 * padding
        for _ in range(3):
            try:
                bbox = draw.textbbox((0, 0), time_str, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except Exception:
                text_width, text_height = draw.textsize(time_str, font=font)

            if text_width <= max_text_width:
                break

            # Reduce font size proportionally
            if base_size <= 24:
                break
            scale = max_text_width / max(text_width, 1)
            base_size = max(24, int(base_size * scale * 0.95))
            font = self._get_font(font_name, base_size)

        # Center text
        x = (self.image_size[0] - text_width) // 2
        y = (self.image_size[1] - text_height) // 2

        # Draw text with shadow for better readability
        text_rgb = self._hex_to_rgb(text_color)

        # Add subtle shadow
        shadow_offset = 2
        shadow_color = tuple(max(0, c - 40) if sum(bg_rgb) > 384 else min(255, c + 40) for c in text_rgb)
        draw.text((x + shadow_offset, y + shadow_offset), time_str, fill=shadow_color, font=font)

        # Draw main text
        draw.text((x, y), time_str, fill=text_rgb, font=font)

        # Add very light noise/augmentation (reduced for clarity)
        if add_noise:
            img_array = np.array(img).astype(np.float32)

            # Minimal Gaussian noise
            noise = np.random.normal(0, 1, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)

            # Very slight brightness variation
            brightness_factor = random.uniform(0.98, 1.02)
            img_array = np.clip(img_array * brightness_factor, 0, 255)

            img = Image.fromarray(img_array.astype(np.uint8))

        return img

    def generate_dataset(
        self,
        unique_times: List[Tuple[int, int, int]],
        variations_per_time: int = 5
    ):
        """
        Generate complete dataset of digital clock images.

        Args:
            unique_times: List of (hour, minute, second) tuples
            variations_per_time: Number of style variations per time
        """
        print(f"\n🎨 Generating digital clock dataset...")
        print(f"   Unique times: {len(unique_times)}")
        print(f"   Variations per time: {variations_per_time}")
        print(f"   Total images: {len(unique_times) * variations_per_time}")

        image_counter = 0

        for time_idx, (hour, minute, second) in enumerate(tqdm(unique_times, desc="Generating clocks")):
            for var_idx in range(variations_per_time):
                # Randomly select style parameters
                font_name = random.choice(self.fonts)
                font_size = random.choice(config.DIGITAL_CLOCK_STYLES['font_sizes'])
                text_color = random.choice(config.DIGITAL_CLOCK_STYLES['colors'])
                bg_color = random.choice(config.DIGITAL_CLOCK_STYLES['backgrounds'])
                format_type = random.choice(config.DIGITAL_CLOCK_STYLES['formats'])

                # Ensure sufficient contrast
                text_rgb = self._hex_to_rgb(text_color)
                bg_rgb = self._hex_to_rgb(bg_color)

                # If colors are too similar, swap one
                if abs(sum(text_rgb) - sum(bg_rgb)) < 150:
                    if sum(bg_rgb) < 384:  # Dark background
                        text_color = '#FFFFFF'
                    else:  # Light background
                        text_color = '#000000'

                # Generate image
                img = self.generate_clock_image(
                    hour, minute, second,
                    font_name, font_size, text_color, bg_color, format_type
                )

                # Save image
                filename = f"digital_{image_counter:05d}.png"
                filepath = self.output_dir / filename
                img.save(filepath)

                # Save metadata
                self.metadata.append({
                    'filename': filename,
                    'time_index': time_idx,
                    'hour': hour,
                    'minute': minute,
                    'second': second,
                    'font_size': font_size,
                    'text_color': text_color,
                    'bg_color': bg_color,
                    'format': format_type
                })

                image_counter += 1

        # Save metadata to JSON
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"✓ Generated {image_counter} images")
        print(f"✓ Metadata saved to {metadata_path}")


def main():
    """Main function to generate digital clock dataset."""
    print("=" * 60)
    print("DIGITAL CLOCK DATASET GENERATION")
    print("=" * 60)

    # Generate unique times
    unique_times = generate_unique_times(
        num_times=config.MAX_UNIQUE_TIMES,
        seed=config.RANDOM_SEED
    )

    # Create generator
    generator = DigitalClockGenerator(
        output_dir=config.DIGITAL_CLOCKS_DIR,
        image_size=config.DIGITAL_CLOCK_SIZE
    )

    # Generate dataset
    generator.generate_dataset(
        unique_times=unique_times,
        variations_per_time=config.DIGITAL_VARIATIONS_PER_TIME
    )

    print("\n✅ Digital clock dataset generation complete!")
    print(f"   Output directory: {config.DIGITAL_CLOCKS_DIR}")
    print(f"   Total images: {len(unique_times) * config.DIGITAL_VARIATIONS_PER_TIME}")


if __name__ == '__main__':
    main()
