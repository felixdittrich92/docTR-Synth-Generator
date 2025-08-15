# Copyright (C) 2021-2025, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import random

from PIL import Image, ImageFilter

__all__ = ["RandomBlur"]


class RandomBlur:
    """
    Randomly applies a Gaussian blur to the image.

    Args:
        radius_range (tuple): Range of blur radii, e.g., (0.1, 2.0).
        prob (float): Probability of applying the blur. Default is 1.0 (always apply).
    """

    def __init__(self, radius_range: tuple = (0.1, 2.0), prob: float = 1.0):
        self.radius_range = radius_range
        self.prob = prob

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply the blur to the given image.

        Args:
            image (Image.Image): The input image to blur.

        Returns:
            Image.Image: The blurred image.
        """
        if random.random() < self.prob:
            radius = random.uniform(*self.radius_range)
            return image.filter(ImageFilter.GaussianBlur(radius))
        return image

    def __repr__(self):
        return f"RandomBlur(radius_range={self.radius_range}, prob={self.prob})"
