# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
import random

from fontTools.ttLib import TTFont

from .font_downloader import _HANDWRITING_FONTS, FontDownloader
from .shaping import ShapingChecker

__all__ = ["FontSelector"]


class FontSelector:
    """Handles font loading and selection based on character support.

    When ``auto_download`` is enabled and none of the locally available fonts
    covers every character of a piece of text, a matching open-source font is
    fetched on demand (see :class:`FontDownloader`), cached and added to the
    in-memory support table. This prevents words from being silently dropped
    just because the bundled fonts lack the required glyphs - the main cause of
    biased, latin-only synthetic datasets.

    Args:
        font_dir (str | None): Directory containing TTF/OTF font files. May be
            ``None``/empty when relying purely on ``auto_download``.
        auto_download (bool): Enable on-demand font downloading for uncovered text.
        font_cache_dir (str | None): Where downloaded fonts are cached. Defaults
            to ``<font_dir>/_downloaded`` (or ``./.font_cache`` when no font_dir).
        download_timeout (int): Per-request timeout for downloads in seconds.
        verify_shaping (bool): Also shape the text with HarfBuzz and reject a font
            that would draw a tofu box, a dotted circle or an empty glyph for it
            (see :class:`ShapingChecker`). A no-op when ``uharfbuzz`` is missing.
    """

    MAX_SHAPING_TRIES = 12

    def __init__(
        self,
        font_dir: str | None,
        auto_download: bool = False,
        font_cache_dir: str | None = None,
        download_timeout: int = 30,
        verify_shaping: bool = True,
    ):
        self.font_dir = font_dir
        self.shaping = ShapingChecker(enabled=verify_shaping)
        self.font_support_table: dict[str, set[str]] = {}
        # Handwriting faces live in the same support table (so they can be
        # rendered) but are excluded from ordinary printed-text resolution -
        # otherwise a downloaded script face leaks into body copy.
        self.handwriting_fonts: set[str] = set()
        self._handwriting_scripts: set[str] = set()  # scripts whose faces are fetched

        if font_cache_dir is None:
            font_cache_dir = os.path.join(font_dir, "_downloaded") if font_dir else ".font_cache"
        self.downloader = FontDownloader(cache_dir=font_cache_dir, timeout=download_timeout, enabled=auto_download)
        self.auto_download = auto_download

        self._load_fonts()
        # cache: frozenset(required_chars) -> font_path, to avoid re-resolving
        self._text_font_cache: dict = {}

        if not self.font_support_table and not auto_download:
            raise ValueError(
                f"No usable fonts found in '{font_dir}' and auto_download is disabled. "
                "Provide a font directory or set auto_download_fonts=True."
            )

    def _load_fonts(self):
        """Load all local fonts and build the character support table."""
        if not self.font_dir or not os.path.isdir(self.font_dir):
            return

        font_files = [f for f in os.listdir(self.font_dir) if f.lower().endswith((".ttf", ".otf"))]
        if not font_files:
            return

        print(f"Loading {len(font_files)} fonts...")
        for i, file in enumerate(font_files):
            font_path = os.path.join(self.font_dir, file)
            self._register_font(font_path, verbose=False)
            if (i + 1) % 10 == 0:
                print(f"Loaded {i + 1}/{len(font_files)} fonts")

    def _register_font(self, font_path: str, verbose: bool = True) -> bool:
        """Read a font file and add its supported characters to the table."""
        if font_path in self.font_support_table:
            return True
        try:
            font = TTFont(font_path, fontNumber=0, lazy=True)
            supported_chars = self._extract_supported_chars(font)
            self.font_support_table[font_path] = supported_chars
            font.close()
            return True
        except Exception as e:
            if verbose:
                print(f"Error reading font {font_path}: {e}")
            return False

    def _extract_supported_chars(self, font: TTFont) -> set[str]:
        """Extract supported characters from font."""
        supported_chars: set[str] = set()
        for cmap in font["cmap"].tables:
            if cmap.isUnicode():
                supported_chars.update(chr(cp) for cp in cmap.cmap.keys())
        return supported_chars

    def supports(self, text: str, font_path: str) -> bool:
        """Whether ``font_path`` covers every character of ``text`` and shapes it cleanly."""
        chars = self.font_support_table.get(font_path)
        if chars is None or not all(c in chars for c in text if not c.isspace()):
            return False
        return self.shaping.ok(text, font_path)

    def _first_clean(self, text: str, candidates: list[str]) -> str | None:
        """A random candidate that shapes ``text`` cleanly (cmap coverage is already given)."""
        if not self.shaping.enabled:
            return random.choice(candidates)
        # A broken cluster fails in every face, a face-specific gap (an empty
        # glyph, missing shaping rules) in a few - a dozen tries tells them apart
        # without shaping a malformed token against hundreds of fonts.
        order = random.sample(candidates, min(len(candidates), self.MAX_SHAPING_TRIES))
        return next((path for path in order if self.shaping.ok(text, path)), None)

    def get_handwriting_font(self, text: str) -> str | None:
        """Get a random handwriting face covering ``text``, or ``None`` if there is none.

        With downloads enabled every listed face of the text's scripts is fetched
        on first use (once, then cached), so crops vary across all of them.
        Returning ``None`` is expected for scripts with no handwriting face -
        callers fall back to a printed font.
        """
        required_chars = frozenset(c for c in text if not c.isspace())
        if self.auto_download:
            for script in (self.downloader.required_scripts(text) or set()) | {"latin"}:
                if script in self._handwriting_scripts:
                    continue
                self._handwriting_scripts.add(script)
                for path in self.downloader.handwriting_faces(script):
                    if self._register_font(path):
                        self.handwriting_fonts.add(path)
        local = [
            path
            for path, chars in self.font_support_table.items()
            if required_chars.issubset(chars) and (path in self.handwriting_fonts or self._looks_handwritten(path))
        ]
        return self._first_clean(text, local) if local else None

    @staticmethod
    def _looks_handwritten(font_path: str) -> bool:
        name = os.path.basename(font_path).lower()
        return any(family in name for faces in _HANDWRITING_FONTS.values() for family, _ in faces)

    def get_font_for_text(self, text: str) -> str | None:
        """Get a font that supports all characters in ``text``.

        A local font is preferred (chosen at random among all that match). If
        none matches and ``auto_download`` is enabled, a suitable font is
        downloaded, cached and reused.

        Args:
            text (str): Text to render

        Returns:
            str | None: Path to a suitable font file, or ``None`` if no font
            (local or downloadable) supports every character.
        """
        required_chars = frozenset(c for c in text if not c.isspace())

        matching_fonts = [
            path
            for path, chars in self.font_support_table.items()
            if required_chars.issubset(chars) and path not in self.handwriting_fonts
        ]
        if matching_fonts:
            chosen = self._first_clean(text, matching_fonts)
            if chosen or not self.auto_download:
                # ``None`` here means every local face that maps these characters
                # would draw a broken cluster - the text itself is malformed, and
                # a downloaded font would not shape it any better.
                return chosen

        if not self.auto_download:
            return None

        # Negative/positive resolution cache keyed by the exact required chars.
        if required_chars in self._text_font_cache:
            return self._text_font_cache[required_chars]

        downloaded = self.downloader.resolve(text)
        if downloaded:
            self._register_font(downloaded)
            if not self.supports(text, downloaded):
                downloaded = None
        self._text_font_cache[required_chars] = downloaded
        return downloaded
