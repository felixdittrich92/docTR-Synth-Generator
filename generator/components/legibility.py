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

__all__ = ["DegradationBudget"]


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
