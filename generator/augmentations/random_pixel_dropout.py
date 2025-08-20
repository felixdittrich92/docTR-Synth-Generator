# Copyright (C) 2021-2025, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import random

import numpy as np
from PIL import Image

__all__ = ["RandomPixelDropout"]


class RandomPixelDropout:
    """
    Randomly drops out pixels in the image by setting them to transparent.

    Args:
        pixel_dropout_range (tuple): Range of dropout probabilities, e.g., (0.2, 0.4).
        prob (float): Probability of applying the dropout. Default is 1.0 (always apply).
    """

    def __init__(self, pixel_dropout_range: tuple[float, float] = (0.2, 0.4), prob: float = 1.0):
        self.pixel_dropout_range = pixel_dropout_range
        self.prob = prob

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply the dropout to the given image.

        Args:
            image (Image.Image): The input image (RGBA recommended).

        Returns:
            Image.Image: The image with randomly dropped pixels.
        """
        if random.random() < self.prob:
            img_array = np.array(image, dtype=np.uint8)
            alpha_channel = img_array[:, :, 3]

            nonzero_idxes = np.argwhere(alpha_channel != 0)
            nonzero_count = nonzero_idxes.shape[0]
            if nonzero_count == 0:
                return image  # nothing to drop

            low = max(0, int(nonzero_count * self.pixel_dropout_range[0]))
            high = max(low, int(nonzero_count * self.pixel_dropout_range[1]))
            if high == 0:
                return image

            random_dropout_count = random.randint(low, high)

            # Vectorized pixel dropout
            coords = nonzero_idxes[np.random.choice(nonzero_count, size=random_dropout_count, replace=False)]
            img_array[coords[:, 0], coords[:, 1]] = (0, 0, 0, 0)

            return Image.fromarray(img_array, mode="RGBA")

        return image

    def __repr__(self):
        return f"RandomPixelDropout(pixel_dropout_range={self.pixel_dropout_range}, prob={self.prob})"
