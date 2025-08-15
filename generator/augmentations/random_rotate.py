# Copyright (C) 2021-2025, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import random

from PIL import Image

__all__ = ["RandomRotate"]


class RandomRotate:
    """
    Randomly rotates the image by a specified angle.

    Args:
        angle_range (tuple): Range of angles to rotate the image, e.g., (-30, 30).
        prob (float): Probability of applying the rotation. Default is 1.0 (always apply).
    """

    def __init__(self, angle_range: tuple = (-30, 30), prob: float = 1.0):
        self.angle_range = angle_range
        self.prob = prob

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply the rotation to the given image.

        Args:
            image (Image.Image): The input image to rotate.

        Returns:
            Image.Image: The rotated image.
        """
        if random.random() < self.prob:
            angle = random.uniform(*self.angle_range)
            return image.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        return image

    def __repr__(self):
        return f"RandomRotate(angle_range={self.angle_range}, prob={self.prob})"
