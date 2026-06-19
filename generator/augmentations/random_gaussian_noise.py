# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import random

import numpy as np
from PIL import Image

__all__ = ["RandomGaussianNoise"]


class RandomGaussianNoise:
    """Adds zero-mean Gaussian sensor noise to the (composited) image.

    Real captures from phone cameras and scanners always carry sensor/grain
    noise, especially in low light. Adding it makes the recognizer robust to
    grainy inputs instead of overfitting to noise-free synthetic glyphs.

    Args:
        std_range (tuple[float, float]): Range of the noise standard deviation in
            8-bit intensity units (0-255). A value drawn from this range is used
            per image.
        prob (float): Probability of applying the noise.
        grayscale_prob (float): Probability that the noise is monochromatic
            (same value across channels), as produced by many real sensors,
            rather than independent per channel.
    """

    def __init__(
        self,
        std_range: tuple[float, float] = (2.0, 12.0),
        prob: float = 1.0,
        grayscale_prob: float = 0.5,
    ):
        self.std_range = std_range
        self.prob = prob
        self.grayscale_prob = grayscale_prob

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.prob:
            return image

        had_alpha = image.mode == "RGBA"
        alpha = np.array(image.getchannel("A")) if had_alpha else None

        rgb = np.array(image.convert("RGB"), dtype=np.float32)
        std = random.uniform(*self.std_range)

        if random.random() < self.grayscale_prob:
            noise = np.random.normal(0.0, std, size=rgb.shape[:2])[..., None]
        else:
            noise = np.random.normal(0.0, std, size=rgb.shape)

        noisy = np.clip(rgb + noise, 0, 255).astype(np.uint8)
        out = Image.fromarray(noisy, mode="RGB")

        if had_alpha and alpha is not None:
            out = out.convert("RGBA")
            out.putalpha(Image.fromarray(alpha))
        return out

    def __repr__(self):
        return f"RandomGaussianNoise(std_range={self.std_range}, prob={self.prob})"
