# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import random
from collections import OrderedDict
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from ..augmentations import AugmentationPipeline, RandomBlur, RandomPerspective, RandomPixelDropout, RandomRotate
from .config import GenerationConfig
from .font_styles import Decorations, font_traits, set_weight

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
        word_augmentations (bool): per-word rotation, perspective, ink dropout and
            blur. Right for recognition crops, each its own capture; wrong on a page,
            where one word blurred or tilted on its own next to crisp neighbours is
            not something a scanner or camera produces.
    """

    def __init__(self, config: GenerationConfig, word_augmentations: bool = True):
        self.config = config
        self.supersample = max(1, int(config.realism.supersample))

        ss = self.supersample
        self.geometry_augs = AugmentationPipeline([
            RandomRotate(angle_range=config.realism.rotation_range, prob=config.realism.rotation_prob),
            RandomPerspective(margin=config.realism.perspective_margin * ss, prob=config.realism.perspective_prob),
        ])
        self.detail_augs = AugmentationPipeline([
            RandomPixelDropout(
                pixel_dropout_range=config.realism.pixel_dropout_range,
                prob=config.realism.pixel_dropout_prob,
            ),
            RandomBlur(radius_range=config.realism.blur_radius_range, prob=config.realism.blur_prob),
        ])
        if not word_augmentations:
            self.geometry_augs = AugmentationPipeline([])
            self.detail_augs = AugmentationPipeline([])

        self._font_cache: OrderedDict[tuple[str, int, int | None], ImageFont.FreeTypeFont] = OrderedDict()
        self._font_cache_size = 128

    def _get_font(self, font_path: str, size: int, weight: int | None = None) -> ImageFont.FreeTypeFont:
        """A cached face; ``weight`` selects an instance of a variable font (its ``wght`` axis)."""
        key = (font_path, size, weight)
        font = self._font_cache.get(key)
        if font is None:
            font = ImageFont.truetype(font_path, size)
            if weight is not None and font_traits(font_path).weight_range is not None:
                set_weight(font, weight)
            self._font_cache[key] = font
            if len(self._font_cache) > self._font_cache_size:
                self._font_cache.popitem(last=False)
        else:
            self._font_cache.move_to_end(key)
        return font

    def sample_style(self) -> tuple[int, int, int]:
        """Sample the per-word font size and stroke widths (colour set later).

        Bold and outline stroke widths are a *fraction of the font size* rather
        than an absolute pixel count. An absolute stroke that looks like normal
        bold at a large size fills the counters and merges glyphs at a small
        size, turning words into unreadable blobs; scaling with the font keeps
        the weight legible across the whole ``font_size_range``.
        """
        font_size = random.randint(*self.config.recognition.font_size_range)
        fs_ss = font_size * self.supersample

        bold_width = 0
        if random.random() < self.config.realism.bold_prob:
            frac = random.uniform(*self.config.realism.bold_width_frac_range)
            bold_width = max(1, round(fs_ss * frac))

        outline_width = 0
        if random.random() < self.config.realism.outline_prob:
            frac = random.uniform(*self.config.realism.outline_width_frac_range)
            outline_width = max(1, round(fs_ss * frac))
        return font_size, bold_width, outline_width

    def measure_size(self, text: str, font_path: str, font_size: int) -> tuple[int, int]:
        """Approximate the rendered (target-resolution) size (kept for compatibility)."""
        font = self._get_font(font_path, font_size)
        left, top, right, bottom = font.getbbox(text)
        width = int((right - left) + 2 * self.config.recognition.padding)
        height = int((bottom - top) + 2 * self.config.recognition.padding)
        return max(1, width), max(1, height)

    def _draw(
        self,
        text: str,
        font_path: str,
        font_size: int,
        style: TextStyle,
        weight: int | None = None,
        decorations: Decorations | None = None,
    ) -> Image.Image:
        """Draw the styled glyph at the supersampled scale (no augmentation yet).

        ``decorations`` add underline / strike-out lines from the font's own metrics,
        drawn in the same ink so they recolour, blur and warp with the word.
        """
        ss = self.supersample
        pad = self.config.recognition.padding * ss
        font = self._get_font(font_path, font_size * ss, weight)

        stroke = max(style.bold_width, style.outline_width)
        left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
        lines = self._decoration_lines(text, font, font_path, decorations) if decorations and decorations.any else []
        for lx0, ly0, lx1, ly1 in lines:  # grow the canvas so a low underline is not cut off
            left, top, right, bottom = min(left, lx0), min(top, ly0), max(right, lx1), max(bottom, ly1)
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
        for lx0, ly0, lx1, ly1 in lines:
            draw.rectangle((pos[0] + lx0, pos[1] + ly0, pos[0] + lx1, pos[1] + ly1), fill=fill)
        return image

    @staticmethod
    def _decoration_lines(
        text: str, font: ImageFont.FreeTypeFont, font_path: str, decorations: Decorations
    ) -> list[tuple[float, float, float, float]]:
        """Underline / double underline / strike-out rectangles, relative to the text origin.

        Positions and thicknesses come from the face's ``post`` and ``OS/2`` tables,
        so a light serif gets a hairline and a heavy grotesque a thick rule, as in a
        word processor. A double underline is two such rules, one thickness apart.
        """
        traits = font_traits(font_path)
        scale = font.size / traits.units_per_em
        baseline = font.getmetrics()[0]
        width = font.getlength(text)
        lines = []
        u_pos, u_thick = traits.underline
        t = max(1.0, u_thick * scale)
        y = baseline - u_pos * scale - t / 2
        for k in range(decorations.underline):
            lines.append((0.0, y + k * 2 * t, width, y + k * 2 * t + t))
        if decorations.strikethrough:
            s_pos, s_thick = traits.strikeout
            ts = max(1.0, s_thick * scale)
            ys = baseline - s_pos * scale - ts / 2
            lines.append((0.0, ys, width, ys + ts))
        return lines

    def _augment(self, image: Image.Image) -> Image.Image:
        """Geometry distortions (at supersample) -> downsample -> ink detail."""
        image = self.geometry_augs(image)
        if self.supersample > 1:
            new_size = (max(1, image.width // self.supersample), max(1, image.height // self.supersample))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        return self.detail_augs(image)

    def render_coverage(
        self,
        text: str,
        font_path: str,
        font_size: int,
        bold_width: int = 0,
        weight: int | None = None,
        decorations: Decorations | None = None,
    ) -> Image.Image:
        """Render a black glyph coverage map (RGBA) for the generator to recolour.

        This is the fast common path: a single draw + augment, recoloured later
        from the alpha channel. Anti-aliased edges are preserved by the alpha.
        ``weight`` picks an instance of a variable face, ``decorations`` adds lines.
        """
        style = TextStyle(fill_color=(0, 0, 0), opacity=255, bold_width=bold_width)
        return self._augment(self._draw(text, font_path, font_size, style, weight, decorations))

    def render_text_to_image(
        self,
        text: str,
        font_path: str,
        style: TextStyle,
        font_size: int,
    ) -> Image.Image:
        """Render fully-styled text (used for the rarer two-tone outline path)."""
        return self._augment(self._draw(text, font_path, font_size, style))
