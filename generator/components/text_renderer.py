# Copyright (C) 2021-2025, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import random

from PIL import Image, ImageDraw, ImageFont

from ..augmentations import AugmentationPipeline, RandomBlur, RandomPerspective, RandomRotate
from .config import GenerationConfig

__all__ = ["TextRenderer"]


class TextRenderer:
    """Handles text rendering and applying augmentations

    Args:
        config (GenerationConfig): Configuration for text rendering
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

        self.augmentations = AugmentationPipeline([
            RandomRotate(angle_range=config.rotation_range, prob=config.rotation_prob),
            RandomBlur(radius_range=config.blur_radius_range, prob=config.blur_prob),
            RandomPerspective(margin=config.perspective_margin, prob=config.perspective_prob),
        ])

    def render_text_to_image(self, text: str, font_path: str) -> Image.Image:
        """Render text to image with random augmentations"""
        font_size = random.randint(*self.config.font_size_range)
        font = ImageFont.truetype(font_path, font_size)

        # Get text dimensions
        left, top, right, bottom = font.getbbox(text)
        width = int((right - left) + 2 * self.config.padding)
        height = int((bottom - top) + 2 * self.config.padding)

        # Create image
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Apply bold effect
        offsets = [(0, 0)]
        if random.random() < self.config.bold_prob:
            offsets += [(1, 0), (0, 1), (1, 1)]

        # Draw text with bold effect
        for dx, dy in offsets:
            draw.text(
                (self.config.padding - left + dx, self.config.padding - top + dy), text, font=font, fill=(0, 0, 0, 255)
            )

        # Apply augmentations
        image = self.augmentations(image)
        return image
