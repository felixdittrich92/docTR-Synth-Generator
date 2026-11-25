# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Font style ground truth: what a face *is* (family, weight, italic) and how a word is marked.

Labels come from the font files, never from guesses about the rendered pixels:
weight from the variable ``wght`` axis or ``OS/2.usWeightClass``, italic from the
italic bit / angle of the face, underline and strike-out geometry from the font's
own ``post`` / ``OS/2`` metrics (as word processors draw them). Only the ink colour
is measured on the finished crop, because scanning or a colour cast can change it.
"""

import colorsys
import functools
import random
import re
from dataclasses import dataclass

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageFont

__all__ = [
    "StyleSpec",
    "Decorations",
    "FontTraits",
    "font_traits",
    "set_weight",
    "color_name",
    "COLOR_NAMES",
    "HIGHLIGHT_COLORS",
    "paint_highlight",
    "weight_name",
]

COLOR_NAMES = ("black", "gray", "white", "red", "orange", "brown", "yellow", "green", "blue", "purple")
# Marker pens: yellow, green, pink, blue, orange.
HIGHLIGHT_COLORS = {
    "yellow": (255, 240, 60),
    "green": (140, 245, 120),
    "pink": (255, 150, 210),
    "blue": (130, 210, 255),
    "orange": (255, 190, 90),
}
_WEIGHT_NAMES = {
    100: "Thin",
    200: "ExtraLight",
    300: "Light",
    400: "Regular",
    500: "Medium",
    600: "SemiBold",
    700: "Bold",
    800: "ExtraBold",
    900: "Black",
}
# Style words that end a family name in the name table of static fonts ("Fira Sans Bold").
_STYLE_WORDS = re.compile(
    r"\s+(Thin|Hairline|Extra ?Light|Ultra ?Light|Light|Book|Regular|Normal|Medium|Semi ?Bold|Demi ?Bold|Bold|"
    r"Extra ?Bold|Ultra ?Bold|Black|Heavy|Italic|Oblique)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FontTraits:
    """What a font file is, read from its tables."""

    family: str
    weight: int  # usWeightClass, or the default instance of a variable face
    italic: bool
    weight_range: tuple[int, int] | None  # the wght axis of a variable face
    units_per_em: int
    underline: tuple[int, int]  # (position, thickness) in font units; position < 0 is below the baseline
    strikeout: tuple[int, int]  # (position, thickness) in font units, above the baseline


@dataclass(frozen=True)
class Decorations:
    """Lines drawn with the word, in its ink: an underline (0, 1 or 2 lines) and a strike-out."""

    underline: int = 0
    strikethrough: bool = False

    @property
    def any(self) -> bool:
        return self.underline > 0 or self.strikethrough


@dataclass(frozen=True)
class StyleSpec:
    """The style one crop is drawn in (see :meth:`TextImageGenerator.generate_styled`).

    ``weight`` selects an instance of a variable face (``None``: the face as it is);
    ``highlight`` is a marker colour behind the word; ``ink_hue`` asks for a coloured
    ink, ``neutral`` for black/gray/white - the measured colour is what gets labelled.
    """

    weight: int | None = None
    decorations: Decorations = Decorations()
    highlight: tuple[int, int, int] | None = None
    ink_hue: tuple[int, int, int] | None = None
    neutral: bool = False


def _name(font: TTFont, *ids: int) -> str | None:
    table = font["name"] if "name" in font else None
    if table is None:
        return None
    for name_id in ids:
        rec = table.getDebugName(name_id)
        if rec:
            return str(rec).strip()
    return None


_NAMED_WEIGHTS = [  # longest first: "ExtraBold" before "Bold"
    ("extralight", 200), ("ultralight", 200), ("extrabold", 800), ("ultrabold", 800), ("semibold", 600),
    ("demibold", 600), ("hairline", 100), ("thin", 100), ("light", 300), ("medium", 500), ("black", 900),
    ("heavy", 900), ("bold", 700), ("regular", 400), ("normal", 400), ("book", 400),
]  # fmt: skip


def _named_weight(font: TTFont) -> int | None:
    """The weight a static face's style name states ("SemiBold Italic" -> 600)."""
    style = (_name(font, 17, 2) or "").lower().replace(" ", "").replace("-", "")
    return next((w for word, w in _NAMED_WEIGHTS if word in style), None)


def _standard_weight(value: int) -> int:
    """``usWeightClass`` to 100-900. 250/275 are the Windows stand-ins for ExtraLight."""
    if value < 10:  # a few old fonts store 1-9
        value *= 100
    if value in (250, 275):
        return 200
    return max(100, min(900, int(round(value / 100.0)) * 100))


@functools.lru_cache(maxsize=4096)
def font_traits(path: str) -> FontTraits:
    """Family, weight, italic and line metrics of a font file (cached per path)."""
    font = TTFont(path, fontNumber=0, lazy=True)
    family = _name(font, 16, 1) or path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    while (stripped := _STYLE_WORDS.sub("", family)) != family:
        family = stripped  # "Noto Sans Medium Italic" -> "Noto Sans"
    os2 = font["OS/2"] if "OS/2" in font else None
    post = font["post"] if "post" in font else None
    upem = int(font["head"].unitsPerEm) if "head" in font else 1000
    italic = bool(
        (os2 is not None and os2.fsSelection & 1)
        or (post is not None and abs(float(post.italicAngle)) > 0.5)
        or re.search(r"italic|oblique", path.rsplit("/", 1)[-1], re.IGNORECASE)
    )
    weight = _named_weight(font) or _standard_weight(int(os2.usWeightClass) if os2 is not None else 400)
    weight_range = None
    if "fvar" in font:
        for axis in font["fvar"].axes:
            if axis.axisTag == "wght":
                weight_range = (int(axis.minValue), int(axis.maxValue))
                weight = int(axis.defaultValue)
    underline = (int(post.underlinePosition), int(post.underlineThickness)) if post is not None else (0, 0)
    if underline[1] <= 0:  # missing metrics: what word processors fall back to
        underline = (-upem // 10, max(1, upem // 20))
    strikeout = (int(os2.yStrikeoutPosition), int(os2.yStrikeoutSize)) if os2 is not None else (0, 0)
    if strikeout[1] <= 0 or strikeout[0] <= 0:
        strikeout = (int(upem * 0.25), underline[1])
    font.close()
    return FontTraits(family, max(100, min(900, weight)), italic, weight_range, upem, underline, strikeout)


def set_weight(font: ImageFont.FreeTypeFont, weight: int) -> None:
    """Set the ``wght`` axis of a variable face (other axes stay at their defaults)."""
    axes = font.get_variation_axes()
    values: list[float] = [float(a.get("default") or 0) for a in axes]
    for i, axis in enumerate(axes):
        name = axis.get("name") or ""
        if (name.decode() if isinstance(name, bytes) else str(name)).lower() == "weight":
            lo, hi = float(axis.get("minimum") or weight), float(axis.get("maximum") or weight)
            values[i] = max(lo, min(hi, float(weight)))
    font.set_variation_by_axes(values)


def weight_name(weight: int) -> str:
    """CSS / OpenType weight name of the nearest standard weight ('Bold' for 700)."""
    return _WEIGHT_NAMES[min(_WEIGHT_NAMES, key=lambda w: abs(w - weight))]


def color_name(ink: np.ndarray, paper: np.ndarray) -> str:
    """Name the ink colour of a crop as a reader would: ``COLOR_NAMES``.

    Achromatic ink is black, gray, or white (lighter than its paper - text on a dark
    band); chromatic ink is named by hue, dark orange/red-brown being ``brown``.
    """
    r, g, b = (float(v) / 255.0 for v in ink[:3])
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    chroma = max(r, g, b) - min(r, g, b)
    ink_lum = 0.299 * r + 0.587 * g + 0.114 * b
    paper_lum = float(0.299 * paper[0] + 0.587 * paper[1] + 0.114 * paper[2]) / 255.0
    # Hue only counts with real colourfulness: near-black ink of any tint reads as black,
    # and a tint too faint to name reads as black / gray / white.
    if chroma < 0.12 or (v < 0.25 and chroma < 0.2):
        if ink_lum > paper_lum + 0.12:
            return "white"
        return "black" if ink_lum < 0.33 else "gray"
    hue = h * 360.0
    if hue < 15 or hue >= 335:
        return "red"  # dark red is maroon - still red, not brown
    if hue < 45:
        return "brown" if v < 0.55 else "orange"
    if hue < 70:
        return "brown" if v < 0.4 else "yellow"
    if hue < 165:
        return "green"
    if hue < 260:
        return "blue"
    return "purple"


def paint_highlight(
    background: Image.Image, box: tuple[int, int, int, int], colour: tuple[int, int, int]
) -> Image.Image:
    """A marker stroke over ``box`` on ``background``: multiplied, with ragged ends and a little slant."""
    bg = np.asarray(background.convert("RGB"), dtype=np.float32).copy()
    h, w = bg.shape[:2]
    x0, y0, x1, y1 = box
    height = max(2, y1 - y0)
    pad_x, pad_y = int(height * random.uniform(0.1, 0.5)), int(height * random.uniform(0.0, 0.2))
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
    if x1 <= x0 or y1 <= y0:
        return background
    mask = np.zeros((h, w), np.float32)
    slant = random.uniform(-0.04, 0.04) * height
    for x in range(x0, x1):
        t = (x - x0) / max(1, x1 - x0)
        dy = int(round(slant * (t - 0.5) * 2))
        lo, hi = max(0, y0 + dy + random.randint(-1, 1)), min(h, y1 + dy + random.randint(-1, 1))
        mask[lo:hi, x] = 1.0
    strength = random.uniform(0.55, 0.9)
    edge = np.clip(np.minimum(np.arange(w) - x0 + 1, x1 - np.arange(w)) / max(2.0, height * 0.3), 0, 1)
    mask *= edge[None, :] * strength
    tint = np.asarray(colour, np.float32) / 255.0
    out = bg * (1 - mask[..., None]) + bg * tint[None, None, :] * mask[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
