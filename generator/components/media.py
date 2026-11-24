# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Give pages a physical size, a scan resolution and a delivery resample.

Sizing type in pixels makes small text impossible. At the pixel page sizes this
generator used (700-1100px wide, roughly 43-148 DPI for A4) a 6pt footnote is
eight pixels tall and no amount of careful augmentation keeps it readable, so
fine print simply could not be generated - under 2% of boxes were below 10px.

Modelling the physical page removes the conflict. A sheet has a size in inches
and is scanned at some DPI, type is specified in points, and a 6pt footnote at
300 DPI is 25px: genuinely small relative to the page, and perfectly sharp.

The second half matters just as much. Real images are then *delivered* at a
lower resolution than they were captured - resized for storage, upload or a
messaging app - and that downscale is where real data gets its characteristic
softening and aliasing. Rendering straight to the delivery size skips it, so
pages are rendered at scan resolution and resampled down afterwards, with the
word polygons scaled by the same factor.
"""

import random
from dataclasses import dataclass

from PIL import Image

from .config import GenerationConfig

__all__ = ["PageMedia", "sample_media", "media_for_pinned_size", "apply_delivery_resample"]

Polygon = list[list[float]]

# name -> (width_in, height_in, weight). Sizes are the real trim sizes.
_DOCUMENT_FORMATS: list[tuple[str, float, float, float]] = [
    ("a4", 8.27, 11.69, 5.0),
    ("letter", 8.5, 11.0, 4.0),
    ("legal", 8.5, 14.0, 1.0),
    ("a5", 5.83, 8.27, 1.5),
    ("tabloid", 11.0, 17.0, 0.8),
]
_RECEIPT_WIDTHS = (2.28, 3.15)  # 58mm and 80mm thermal rolls
_CARD = ("id_card", 3.37, 2.13)  # ID-1, landscape


@dataclass
class PageMedia:
    """A physical sheet, the resolution it was captured at, and its delivery size.

    Attributes:
        name: format name, for debugging.
        dpi: capture resolution the page is rendered at.
        width_px / height_px: render dimensions.
        delivery_long_edge: long edge of the final image, after resampling.
    """

    name: str
    dpi: float
    width_px: int
    height_px: int
    delivery_long_edge: int

    @property
    def delivery_scale(self) -> float:
        """Factor the rendered page is resized by (<=1 means a downscale)."""
        return self.delivery_long_edge / max(1, max(self.width_px, self.height_px))

    def points_to_px(self, points: float) -> int:
        """Convert a type size in points to render pixels at this resolution."""
        return max(1, int(round(points * self.dpi / 72.0)))

    def min_render_px_for_delivery(self, min_delivery_px: float) -> int:
        """Smallest render size that still lands above ``min_delivery_px`` after resampling.

        This is what replaces an absolute pixel floor: what has to stay legible
        is the *delivered* glyph, so a page that will be downscaled by half must
        render its smallest type twice as large.
        """
        return max(1, int(round(min_delivery_px / max(0.05, min(1.0, self.delivery_scale)))))


def media_for_pinned_size(config: GenerationConfig, width_px: int, height_px: int) -> PageMedia:
    """Media for a page whose pixel size the caller pinned.

    The sheet size is inferred from the pixels by assuming a portrait A4-ish
    aspect, which gives an effective DPI - so type is still specified in points
    and fine print still exists, while the caller's pixel contract is kept.
    """
    long_edge_in = 11.69 if height_px >= width_px else 8.27
    dpi = max(30.0, max(width_px, height_px) / long_edge_in)
    return PageMedia("pinned", dpi, width_px, height_px, max(width_px, height_px))


def sample_media(config: GenerationConfig, layout: str) -> PageMedia:
    """Pick a sheet, a scan resolution and a delivery size for one page."""
    cfg = config.media
    dpi = random.uniform(*cfg.dpi_range)

    if layout == "receipt":
        width_in = random.uniform(*_RECEIPT_WIDTHS)
        height_in = random.uniform(*cfg.receipt_length_range)
        name = "receipt"
    elif layout == "id_card":
        name, width_in, height_in = _CARD
        # The card is photographed on a surface, so the frame is larger than it.
        width_in *= random.uniform(1.35, 1.9)
        height_in *= random.uniform(1.4, 2.1)
    else:
        names = [f[0] for f in _DOCUMENT_FORMATS]
        weights = [f[3] for f in _DOCUMENT_FORMATS]
        name = random.choices(names, weights=weights, k=1)[0]
        _, width_in, height_in, _ = next(f for f in _DOCUMENT_FORMATS if f[0] == name)
        if random.random() < cfg.landscape_prob:
            width_in, height_in = height_in, width_in
            name += "_landscape"

    # Keep the render bounded: A4 at 400 DPI is 13 megapixels, which costs far
    # more than it adds. Drop the resolution rather than the page size.
    megapixels = (width_in * dpi) * (height_in * dpi) / 1e6
    if megapixels > cfg.max_render_megapixels:
        dpi *= (cfg.max_render_megapixels / megapixels) ** 0.5

    width_px = max(64, int(width_in * dpi))
    height_px = max(64, int(height_in * dpi))
    long_edge = max(width_px, height_px)
    delivery = int(min(long_edge, random.uniform(*cfg.delivery_long_edge_range)))
    return PageMedia(name, dpi, width_px, height_px, delivery)


def apply_delivery_resample(
    config: GenerationConfig,
    image: Image.Image,
    polygons: list[Polygon],
    media: PageMedia,
    min_text_px: float | None = None,
) -> tuple[Image.Image, list[Polygon], float]:
    """Resize the page to its delivery resolution, scaling the polygons with it.

    Returns the image, the moved polygons and the scale that was applied, so the
    caller can budget its degradations against *delivered* glyph sizes.

    ``min_text_px`` is the smallest glyph on the page, in render pixels. The
    downscale is limited so that glyph still arrives above
    ``media.min_delivery_text_px``: the page's planned scale is computed before
    layout, but a captured page is composited into a *larger* frame, so the
    real factor is harsher than planned and would otherwise push fine print
    below the floor the model promised.
    """
    if random.random() >= config.media.resample_prob:
        return image, polygons, 1.0

    long_edge = max(image.width, image.height)
    scale = media.delivery_long_edge / max(1, long_edge)
    if min_text_px is not None and min_text_px > 0:
        scale = max(scale, config.media.min_delivery_text_px / min_text_px)
    scale = min(scale, 1.0)
    if scale >= 0.995:
        return image, polygons, 1.0

    target = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    # A box filter is what an image pipeline actually uses for a downscale;
    # bicubic shows up when something resizes carelessly. Both are worth seeing.
    resample = random.choice([Image.Resampling.LANCZOS, Image.Resampling.BILINEAR, Image.Resampling.BICUBIC])
    out = image.resize(target, resample)

    if random.random() < config.media.upscale_after_prob:
        # A second generation: something downscaled the image, something else
        # blew it back up. Very common on anything that has been through chat.
        factor = random.uniform(1.1, 1.6)
        blown = (int(out.width * factor), int(out.height * factor))
        out = out.resize(blown, Image.Resampling.BICUBIC)
        scale *= factor

    moved = [[[x * scale, y * scale] for x, y in poly] for poly in polygons]
    return out, moved, scale
