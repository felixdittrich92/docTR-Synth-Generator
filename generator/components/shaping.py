# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Shaping-level check that a font really draws a piece of text.

A font's cmap only says a codepoint *maps* to some glyph. It does not say that
the glyph has ink, nor that the font carries the shaping rules a complex script
needs. When a mark lands where its script does not allow one (a matra after a
matra, a virama at the start of a word), HarfBuzz - the engine behind Pillow's
raqm layout - inserts a dotted circle (U+25CC). The crop then shows a character
its label does not contain, which is exactly the kind of label noise a
recognition model cannot unlearn.

:class:`ShapingChecker` shapes the text with the same engine and rejects:

* ``.notdef`` glyphs (gid 0, the "tofu" box),
* dotted circles HarfBuzz inserted for broken clusters,
* clusters that should be visible but come out without any ink.

``uharfbuzz`` is optional. Without it every check passes and behaviour falls
back to plain cmap coverage.
"""

import unicodedata
from collections import OrderedDict

try:  # optional dependency: pip install "doctr-synth-generator[shaping]"
    import uharfbuzz as hb
except ImportError:  # pragma: no cover - exercised only without uharfbuzz
    hb = None

__all__ = ["ShapingChecker", "HAS_HARFBUZZ"]

HAS_HARFBUZZ = hb is not None
_DOTTED_CIRCLE = 0x25CC
# Characters that legitimately render without ink.
_INKLESS_CATEGORIES = {"Zs", "Zl", "Zp", "Cc", "Cf"}


class ShapingChecker:
    """Shape ``text`` with a font and report whether it renders cleanly.

    Args:
        enabled (bool): turn the check off without uninstalling ``uharfbuzz``.
        cache_size (int): number of HarfBuzz font objects kept open. A face
            holds its file in memory, so this bounds RAM for large font sets.
    """

    def __init__(self, enabled: bool = True, cache_size: int = 64):
        self.enabled = bool(enabled and HAS_HARFBUZZ)
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self._cache_size = max(1, cache_size)

    def _font(self, font_path: str):
        entry = self._cache.get(font_path)
        if entry is None:
            face = hb.Face(hb.Blob.from_file_path(font_path))
            font = hb.Font(face)
            entry = (font, font.get_nominal_glyph(_DOTTED_CIRCLE))
            self._cache[font_path] = entry
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(font_path)
        return entry

    def problem(self, text: str, font_path: str) -> str | None:
        """Return why ``text`` does not render cleanly with the font, or ``None``.

        Possible reasons: ``"notdef"``, ``"dotted_circle"``, ``"invisible"`` or
        ``"unreadable_font"``.
        """
        if not self.enabled or not text:
            return None
        try:
            font, dotted = self._font(font_path)
        except Exception:
            return "unreadable_font"
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(font, buf, {})
        infos = buf.glyph_infos
        if any(info.codepoint == 0 for info in infos):
            return "notdef"
        if dotted and "◌" not in text and any(info.codepoint == dotted for info in infos):
            return "dotted_circle"

        # Ink per cluster: a cluster made of visible characters must draw something.
        ink: dict[int, bool] = {}
        for info in infos:
            ext = font.get_glyph_extents(info.codepoint)
            has_ink = bool(ext and ext.width and ext.height)
            ink[info.cluster] = ink.get(info.cluster, False) or has_ink
        starts = sorted(ink)
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(text)
            chars = text[min(start, end) : max(start, end)] or text[start : start + 1]
            visible = any(unicodedata.category(c) not in _INKLESS_CATEGORIES for c in chars)
            if visible and not ink[start]:
                return "invisible"
        return None

    def ok(self, text: str, font_path: str) -> bool:
        """``True`` when ``text`` shapes without tofu, dotted circles or missing ink."""
        return self.problem(text, font_path) is None
