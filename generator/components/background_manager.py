# Copyright (C) 2021-2025, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
import random
import warnings

from PIL import Image

__all__ = ["BackgroundManager"]


class BackgroundManager:
    """Handles background image loading and cropping"""

    def __init__(self, bg_image_dir: str | None = None):
        self.bg_image_dir = bg_image_dir
        self.bg_images = self._load_background_images()
        if not self.bg_images:
            warnings.warn(
                f"No background images found in {bg_image_dir}. Using a blank background instead.", UserWarning
            )

    def _load_background_images(self) -> list[str]:
        """Load all background image paths"""
        return (
            [
                os.path.join(self.bg_image_dir, f)
                for f in os.listdir(self.bg_image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if self.bg_image_dir
            else []
        )

    def get_background_crop(self, size: tuple[int, int]) -> Image.Image:
        """Get a random background crop of the specified size, retrying if more than 3% of pixels are black.

        Args:
            size (tuple[int, int]): Desired crop size (width, height)

        Returns:
            Image.Image: Cropped background image
        """
        if not self.bg_images:
            return Image.new("RGB", size, (255, 255, 255))

        crop_width, crop_height = size
        attempts = 10

        for _ in range(attempts):
            bg_path = random.choice(self.bg_images)
            try:
                with Image.open(bg_path) as bg:
                    bg = bg.convert("RGB")
                    bg_width, bg_height = bg.size
                    if crop_width <= bg_width and crop_height <= bg_height:
                        x = random.randint(0, bg_width - crop_width)
                        y = random.randint(0, bg_height - crop_height)
                        return bg.crop((x, y, x + crop_width, y + crop_height))
            except Exception as e:
                print(f"Error loading background {bg_path}: {e}")
                continue

        return Image.new("RGB", size, (255, 255, 255))
