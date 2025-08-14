import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from PIL.Image import Resampling, Transform

from .config import GenerationConfig

__all__ = ["TextRenderer"]


class TextRenderer:
    """Handles text rendering and applying augmentations

    Args:
        config (GenerationConfig): Configuration for text rendering
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

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
        # TODO: This should be a pipeline instead of a class method
        image = self._apply_augmentations(image)
        return image

    def _apply_augmentations(self, image: Image.Image) -> Image.Image:
        """Apply random augmentations to the image"""
        # Rotation
        if random.random() < self.config.rotation_prob:
            angle = random.uniform(*self.config.rotation_range)
            image = image.rotate(angle, expand=True, resample=Resampling.BICUBIC)

        # Blur
        if random.random() < self.config.blur_prob:
            radius = random.uniform(*self.config.blur_radius_range)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        # Perspective distortion
        if random.random() < self.config.perspective_prob:
            image = self._random_perspective_distort(image)

        return image

    def _random_perspective_distort(self, image: Image.Image) -> Image.Image:
        """Apply random perspective distortion"""
        width, height = image.size
        margin = self.config.perspective_margin

        def rand_offset():
            return random.randint(-margin, margin)

        # Source points (distorted)
        src_points = [
            (rand_offset(), rand_offset()),
            (width + rand_offset(), rand_offset()),
            (width + rand_offset(), height + rand_offset()),
            (rand_offset(), height + rand_offset()),
        ]
        # Destination points (regular rectangle)
        dst_points = [(0, 0), (width, 0), (width, height), (0, height)]

        coeffs = self._find_coeffs(src_points, dst_points)
        return image.transform(
            size=(width, height), method=Transform.PERSPECTIVE, data=coeffs, resample=Resampling.BICUBIC
        )

    def _find_coeffs(self, pa, pb):
        """Calculate perspective transform coefficients"""
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
        mat_a = np.array(matrix, dtype=np.float64)
        mat_b = np.array(pb).reshape(8)
        coeffs, _, _, _ = np.linalg.lstsq(mat_a, mat_b, rcond=None)
        return coeffs
