# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import random
from collections import OrderedDict
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from ..augmentations import AugmentationPipeline, RandomBlur, RandomPerspective, RandomPixelDropout, RandomRotate
from .config import GenerationConfig

__all__ = ["TextRenderer", "TextStyle"]


@dataclass
class TextStyle:
    """Colour / weight styling for a single rendered word.

    Attributes:
        fill_color (tuple[int, int, int]): RGB ink colour.
        opacity (int): Glyph alpha (0-255); < 255 yields faded ink.
        bold_width (int): Faux-bold stroke width in *supersampled* pixels (0 = off).
        outline_color (tuple[int, int, int] | None): Outline colour, or None.
        outline_width (int): Outline stroke width in supersampled pixels.
    """

    fill_color: tuple[int, ...] = (0, 0, 0)
    opacity: int = 255
    bold_width: int = 0
    outline_color: tuple[int, int, int] | None = None
    outline_width: int = 0


class TextRenderer:
    """Handles text rendering and applying glyph-space augmentations.

    Performance notes:
      * ``ImageFont`` objects are cached per (path, size). Re-creating a FreeType
        face for a large variable font on every word dominated the runtime; the
        cache is tiny in memory (a handful of MB) because the loaded file buffer
        is shared across sizes.
      * The common path renders a single black *coverage* glyph that the
        generator recolours in-place, instead of rendering once to measure and
        again to draw. This removes a second expensive ``getbbox`` per image.

    Args:
        config (GenerationConfig): Configuration for text rendering.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.supersample = max(1, int(config.supersample))

        ss = self.supersample
        self.geometry_augs = AugmentationPipeline([
            RandomRotate(angle_range=config.rotation_range, prob=config.rotation_prob),
            RandomPerspective(margin=config.perspective_margin * ss, prob=config.perspective_prob),
        ])
        self.detail_augs = AugmentationPipeline([
            RandomPixelDropout(pixel_dropout_range=config.pixel_dropout_range, prob=config.pixel_dropout_prob),
            RandomBlur(radius_range=config.blur_radius_range, prob=config.blur_prob),
        ])

        self._font_cache: OrderedDict[tuple[str, int], ImageFont.FreeTypeFont] = OrderedDict()
        self._font_cache_size = 128

    def _get_font(self, font_path: str, size: int) -> ImageFont.FreeTypeFont:
        key = (font_path, size)
        font = self._font_cache.get(key)
        if font is None:
            font = ImageFont.truetype(font_path, size)
            self._font_cache[key] = font
            if len(self._font_cache) > self._font_cache_size:
                self._font_cache.popitem(last=False)
        else:
            self._font_cache.move_to_end(key)
        return font

    def sample_style(self) -> tuple[int, int, int]:
        """Sample the per-word font size and stroke widths (colour set later)."""
        font_size = random.randint(*self.config.font_size_range)
        bold_width = 0
        if random.random() < self.config.bold_prob:
            bold_width = random.randint(*self.config.bold_width_range) * self.supersample
        outline_width = 0
        if random.random() < self.config.outline_prob:
            outline_width = random.randint(*self.config.outline_width_range) * self.supersample
        return font_size, bold_width, outline_width

    def measure_size(self, text: str, font_path: str, font_size: int) -> tuple[int, int]:
        """Approximate the rendered (target-resolution) size (kept for compatibility)."""
        font = self._get_font(font_path, font_size)
        left, top, right, bottom = font.getbbox(text)
        width = int((right - left) + 2 * self.config.padding)
        height = int((bottom - top) + 2 * self.config.padding)
        return max(1, width), max(1, height)

    def _draw(self, text: str, font_path: str, font_size: int, style: TextStyle) -> Image.Image:
        """Draw the styled glyph at the supersampled scale (no augmentation yet)."""
        ss = self.supersample
        pad = self.config.padding * ss
        font = self._get_font(font_path, font_size * ss)

        stroke = max(style.bold_width, style.outline_width)
        left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
        width = int((right - left) + 2 * pad)
        height = int((bottom - top) + 2 * pad)

        image = Image.new("RGBA", (max(1, width), max(1, height)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        fill = (*style.fill_color, style.opacity)
        pos = (pad - left, pad - top)

        if style.outline_color is not None and style.outline_width > 0:
            draw.text(
                pos,
                text,
                font=font,
                fill=fill,
                stroke_width=style.outline_width,
                stroke_fill=(*style.outline_color, style.opacity),
            )
        elif style.bold_width > 0:
            draw.text(pos, text, font=font, fill=fill, stroke_width=style.bold_width, stroke_fill=fill)
        else:
            draw.text(pos, text, font=font, fill=fill)
        return image

    def _augment(self, image: Image.Image) -> Image.Image:
        """Geometry distortions (at supersample) -> downsample -> ink detail."""
        image = self.geometry_augs(image)
        if self.supersample > 1:
            new_size = (max(1, image.width // self.supersample), max(1, image.height // self.supersample))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        return self.detail_augs(image)

    def render_coverage(self, text: str, font_path: str, font_size: int, bold_width: int = 0) -> Image.Image:
        """Render a black glyph coverage map (RGBA) for the generator to recolour.

        This is the fast common path: a single draw + augment, recoloured later
        from the alpha channel. Anti-aliased edges are preserved by the alpha.
        """
        style = TextStyle(fill_color=(0, 0, 0), opacity=255, bold_width=bold_width)
        return self._augment(self._draw(text, font_path, font_size, style))

    def render_text_to_image(
        self,
        text: str,
        font_path: str,
        style: TextStyle,
        font_size: int,
    ) -> Image.Image:
        """Render fully-styled text (used for the rarer two-tone outline path)."""
        return self._augment(self._draw(text, font_path, font_size, style))
