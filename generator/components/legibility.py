# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Keep degradations proportional to the smallest text on the page.

Blur, smear and compression are destructive relative to *stroke width*, not in
absolute pixels. A 1.2px Gaussian is invisible on 30px type and turns 10px type
into a grey smudge, so a single global radius range cannot be right for both. A
page whose smallest glyph is 10px tall therefore gets a tighter budget than one
set in 24px, and the same degradation config produces readable output at either
size.

The numbers below come from stroke geometry: for a regular weight the stem is
roughly one eighth of the glyph box height, and counters (the holes in a, e, o)
close once ink is displaced by about half a stem. That puts the destructive
threshold near ``h/16`` and makes ``h/12`` a safe working cap for blur sigma,
with a matching cap on motion smear length.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

__all__ = ["DegradationBudget", "local_ink_contrast"]


def local_ink_contrast(image: Image.Image, alpha: Image.Image, window: int | None = None) -> float:
    """Worst ink-vs-paper luminance gap along a word, in a sliding window.

    A whole-crop average hides the failure that actually makes a crop unreadable:
    *part* of the word sitting on a dark band, a fold or a photographic patch of
    the background, where ink and paper meet at the same grey. The crop is cut
    into windows about one glyph height wide; each window compares its glyph core
    (``alpha >= 192``, or 75% of the peak for hairlines) with its paper (``alpha == 0``), discounted by how busy that
    paper is. The weakest window is the crop's legibility.

    Args:
        image: the finished (degraded) crop.
        alpha: glyph coverage aligned with ``image`` (same size).
        window: window width in pixels (default: the crop height).

    Returns:
        The smallest ``|ink - paper| - std(paper)`` over all windows, in 0-255
        luminance units; ``inf`` when nothing could be measured.
    """
    if image.size != alpha.size:
        return float("inf")
    lum = np.asarray(image.convert("L"), dtype=np.float32)
    cov = np.asarray(alpha, dtype=np.uint8)
    # Thin strokes at small sizes never reach full coverage after anti-aliasing, so
    # the core is relative to the glyph's own peak.
    core, paper = cov >= min(192, int(cov.max() * 0.75)), cov == 0
    width = lum.shape[1]
    win = max(8, window or lum.shape[0])
    worst = float("inf")
    for x0 in range(0, max(1, width - win // 2), max(4, win // 2)):
        sl = slice(x0, min(width, x0 + win))
        c, p = core[:, sl], paper[:, sl]
        if c.sum() < 4 or p.sum() < 8:
            continue
        ink, bg = lum[:, sl][c], lum[:, sl][p]
        worst = min(worst, abs(float(ink.mean()) - float(bg.mean())) - float(bg.std()))
    return worst


@dataclass
class DegradationBudget:
    """Per-page caps on destructive degradations.

    Attributes:
        max_blur_radius: Gaussian sigma ceiling, in final-image pixels.
        max_motion_length: camera-shake smear ceiling, in pixels.
        min_jpeg_quality: quality floor - 8x8 blocking is far more destructive
            on small type, so the floor rises as text shrinks.
        max_noise_std: sensor-noise ceiling.
    """

    max_blur_radius: float
    max_motion_length: int
    min_jpeg_quality: int
    max_noise_std: float

    @classmethod
    def for_text_height(cls, height_px: float, already_softened: bool = False) -> "DegradationBudget":
        """Derive the budget from the smallest glyph box on the page.

        ``already_softened`` halves the blur allowance for a page that has been
        through a downscale: the resample is itself a low-pass filter, so adding
        a full blur budget on top of it double-counts and is what turns dense
        small text to mush.
        """
        height = max(4.0, float(height_px))
        # h/16 is where counters begin to close (stem ~ h/8, ink displaced by
        # half a stem); the previous h/12 sat right on the edge for dense text.
        blur = height / 16.0
        if already_softened:
            blur *= 0.5
        return cls(
            max_blur_radius=blur,
            max_motion_length=max(0, int(height / 6.0)),
            # 35 for comfortable type, rising toward 65 as glyphs approach 8px.
            min_jpeg_quality=int(min(65, 35 + max(0.0, 18.0 - height) * 3.0)),
            max_noise_std=3.0 + height * 0.6,
        )
