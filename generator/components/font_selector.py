# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
import random

from fontTools.ttLib import TTFont

from .font_downloader import FontDownloader

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
    """

    def __init__(
        self,
        font_dir: str | None,
        auto_download: bool = False,
        font_cache_dir: str | None = None,
        download_timeout: int = 30,
    ):
        self.font_dir = font_dir
        self.font_support_table: dict[str, set[str]] = {}

        if font_cache_dir is None:
            font_cache_dir = os.path.join(font_dir, "_downloaded") if font_dir else ".font_cache"
        self.downloader = FontDownloader(cache_dir=font_cache_dir, timeout=download_timeout, enabled=auto_download)
        self.auto_download = auto_download

        self._load_fonts()
        # cache: frozenset(required_chars) -> font_path, to avoid re-resolving
        self._text_font_cache: dict[frozenset, str | None] = {}

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

        matching_fonts = [path for path, chars in self.font_support_table.items() if required_chars.issubset(chars)]
        if matching_fonts:
            return random.choice(matching_fonts)

        if not self.auto_download:
            return None

        # Negative/positive resolution cache keyed by the exact required chars.
        if required_chars in self._text_font_cache:
            return self._text_font_cache[required_chars]

        downloaded = self.downloader.resolve(text)
        if downloaded:
            self._register_font(downloaded)
        self._text_font_cache[required_chars] = downloaded
        return downloaded
