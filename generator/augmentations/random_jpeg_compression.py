# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import io
import random

from PIL import Image

__all__ = ["RandomJpegCompression"]


class RandomJpegCompression:
    """Re-encodes the image as JPEG at a random quality to introduce blocking artifacts.

    Almost every real-world document image the model will see at inference time
    has been through at least one lossy JPEG step (camera capture, scanner
    export, messaging apps, ...). Training on perfectly clean PNG glyphs leaves
    a domain gap; this augmentation closes it by injecting the characteristic
    8x8 ringing/blocking around text edges.

    Args:
        quality_range (tuple[int, int]): Range of JPEG quality factors (1-100).
            Lower values mean stronger artifacts.
        prob (float): Probability of applying the compression.
    """

    def __init__(self, quality_range: tuple[int, int] = (35, 90), prob: float = 1.0):
        self.quality_range = quality_range
        self.prob = prob

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.prob:
            return image

        had_alpha = image.mode == "RGBA"
        alpha = image.getchannel("A") if had_alpha else None

        rgb = image.convert("RGB")
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")

        if had_alpha and alpha is not None:
            compressed = compressed.convert("RGBA")
            compressed.putalpha(alpha)
        return compressed

    def __repr__(self):
        return f"RandomJpegCompression(quality_range={self.quality_range}, prob={self.prob})"
