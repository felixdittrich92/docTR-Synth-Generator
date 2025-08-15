# Copyright (C) 2021-2025, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import random

import numpy as np
from PIL import Image
from PIL.Image import Resampling, Transform

__all__ = ["RandomPerspective"]


class RandomPerspective:
    """
    Randomly applies perspective distortion to the image.

    Args:
        margin (int): Margin for the perspective distortion.
        prob (float): Probability of applying the perspective distortion. Default is 1.0 (always apply).
    """

    def __init__(self, margin: int = 10, prob: float = 1.0):
        self.margin = margin
        self.prob = prob

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply the perspective distortion to the given image.

        Args:
            image (Image.Image): The input image to distort.

        Returns:
            Image.Image: The distorted image.
        """
        if random.random() < self.prob:
            return self._random_perspective_distort(image)
        return image

    def __repr__(self):
        return f"RandomPerspective(margin={self.margin}, prob={self.prob})"

    def _random_perspective_distort(self, image: Image.Image) -> Image.Image:
        """Apply random perspective distortion"""
        width, height = image.size
        margin = self.margin

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
