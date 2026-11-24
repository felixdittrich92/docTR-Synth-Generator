# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Grayscale and black-and-white output, as scanners, copiers and fax machines deliver it.

Most documents in the wild are not colour images. Office scanners default to
grayscale, copiers and fax machines produce bitonal pages, and scanner apps on a
phone offer a "B&W" mode that thresholds the photo. A model trained only on colour
renders sees none of that.

Black-and-white uses adaptive (Bradley) thresholding - what those devices do - and
is only kept when every labelled word still shows ink afterwards. A faint word or a
light-on-dark band that thresholding would erase sends the page to grayscale
instead, so a label never points at blank paper.
"""

import random

import numpy as np
from PIL import Image, ImageFilter

from .config import GenerationConfig

__all__ = ["apply_scan_tone", "to_bitonal"]

Polygon = list[list[float]]


def to_bitonal(image: Image.Image, sensitivity: float | None = None, speckle: float | None = None) -> Image.Image:
    """Adaptive black-and-white thresholding, with a little toner speckle.

    A pixel turns black when it is darker than its neighbourhood mean by more than
    ``sensitivity``; a single global threshold would black out every shadow and
    erase every faint line.
    """
    gray = image.convert("L")
    radius = max(6, min(gray.size) // 40)
    local = np.asarray(gray.filter(ImageFilter.BoxBlur(radius)), dtype=np.float32)
    lum = np.asarray(gray, dtype=np.float32)
    k = random.uniform(0.08, 0.18) if sensitivity is None else sensitivity
    black = lum < local * (1.0 - k)
    density = random.uniform(0.0, 0.0015) if speckle is None else speckle
    if density > 0:
        noise = np.random.random(black.shape)
        black ^= noise < density  # toner dust on white, drop-outs in black
    return Image.fromarray(np.where(black, 0, 255).astype(np.uint8)).convert("RGB")


def _words_survive(bitonal: Image.Image, polygons: list[Polygon]) -> bool:
    """Every word box still holds both ink and paper (not erased, not flooded)."""
    arr = np.asarray(bitonal.convert("L")) < 128
    height, width = arr.shape
    for poly in polygons:
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        x0, x1 = max(0, int(min(xs))), min(width, int(max(xs)) + 1)
        y0, y1 = max(0, int(min(ys))), min(height, int(max(ys)) + 1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        ink = float(arr[y0:y1, x0:x1].mean())
        if ink < 0.02 or ink > 0.97:
            return False
    return True


# Thresholding needs pixels to work with: below these text heights glyphs break into
# blobs that no reader - human or model - can tell apart. A crop is measured by its
# word, a page by its *smallest* glyph (footnotes), hence the lower page floor.
MIN_BITONAL_TEXT_PX = 24.0  # crop height, which includes marks above and below the letters
MIN_BITONAL_PAGE_TEXT_PX = 10.0


def apply_scan_tone(
    config: GenerationConfig,
    image: Image.Image,
    polygons: list[Polygon] | None = None,
    text_height: float | None = None,
    min_text_px: float = MIN_BITONAL_TEXT_PX,
) -> Image.Image:
    """Deliver ``image`` in colour, grayscale or black-and-white (``RealismConfig``).

    With ``polygons`` given, black-and-white is only kept if every word survives it;
    otherwise the image is delivered in grayscale. Without them (a recognition crop)
    the caller's legibility check decides. Text smaller than ``min_text_px``
    (``text_height``, the smallest glyph height) is never thresholded.
    """
    realism = config.realism
    r = random.random()
    if r < realism.bitonal_prob:
        if text_height is not None and text_height < min_text_px:
            return image.convert("L").convert("RGB")
        bitonal = to_bitonal(image)
        if polygons is None or _words_survive(bitonal, polygons):
            return bitonal
        return image.convert("L").convert("RGB")
    if r < realism.bitonal_prob + realism.grayscale_prob:
        return image.convert("L").convert("RGB")
    return image
