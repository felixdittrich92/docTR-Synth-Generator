# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
import random
import tempfile
import threading
import urllib.parse
import urllib.request

from fontTools.ttLib import TTFont

__all__ = ["FontDownloader"]


# Mirror of the official Google Fonts repository. ``raw.githubusercontent.com``
# serves the raw font binaries without authentication or rate limiting.
_GF_RAW = "https://raw.githubusercontent.com/google/fonts/main"

# Inclusive Unicode codepoint ranges -> a logical "script" key. Ordered roughly
# by frequency. Only the start/end of each block is needed; the lookup picks the
# first range that contains the codepoint.
_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0000, 0x024F, "latin"),  # Basic Latin + Latin-1 + Latin Extended A/B
    (0x0250, 0x02AF, "latin"),  # IPA extensions
    (0x1E00, 0x1EFF, "latin"),  # Latin Extended Additional
    (0x0370, 0x03FF, "greek"),
    (0x1F00, 0x1FFF, "greek"),  # Greek Extended
    (0x0400, 0x052F, "cyrillic"),
    (0x0531, 0x058F, "armenian"),
    (0x0590, 0x05FF, "hebrew"),
    (0x0600, 0x06FF, "arabic"),
    (0x0750, 0x077F, "arabic"),
    (0x0700, 0x074F, "syriac"),
    (0x0900, 0x097F, "devanagari"),
    (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "oriya"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
    (0x0D80, 0x0DFF, "sinhala"),
    (0x0E00, 0x0E7F, "thai"),
    (0x0E80, 0x0EFF, "lao"),
    (0x0F00, 0x0FFF, "tibetan"),
    (0x1000, 0x109F, "myanmar"),
    (0x10A0, 0x10FF, "georgian"),
    (0x1100, 0x11FF, "korean"),  # Hangul Jamo
    (0x3040, 0x309F, "japanese"),  # Hiragana
    (0x30A0, 0x30FF, "japanese"),  # Katakana
    (0xAC00, 0xD7AF, "korean"),  # Hangul syllables
    (0x4E00, 0x9FFF, "cjk"),  # CJK unified ideographs
    (0x3400, 0x4DBF, "cjk"),  # CJK extension A
    (0xF900, 0xFAFF, "cjk"),  # CJK compatibility ideographs
]

# script -> ordered list of (family_dir, candidate_filenames) on Google Fonts.
# Several filename candidates are tried because variable-font axis names
# (``[wght]`` vs ``[wdth,wght]``) differ between families. The first URL that
# returns HTTP 200 *and* passes coverage verification wins.
_SCRIPT_FONTS: dict[str, list[tuple[str, list[str]]]] = {
    "latin": [
        ("notosans", ["NotoSans[wdth,wght].ttf"]),
        ("notoserif", ["NotoSerif[wdth,wght].ttf"]),
    ],
    "greek": [("notosans", ["NotoSans[wdth,wght].ttf"])],
    "cyrillic": [("notosans", ["NotoSans[wdth,wght].ttf"])],
    "armenian": [("notosansarmenian", ["NotoSansArmenian[wdth,wght].ttf"])],
    "hebrew": [("notosanshebrew", ["NotoSansHebrew[wdth,wght].ttf"])],
    "arabic": [("notosansarabic", ["NotoSansArabic[wdth,wght].ttf"])],
    "syriac": [("notosanssyriac", ["NotoSansSyriac[wght].ttf"])],
    "devanagari": [("notosansdevanagari", ["NotoSansDevanagari[wdth,wght].ttf"])],
    "bengali": [("notosansbengali", ["NotoSansBengali[wdth,wght].ttf"])],
    "gurmukhi": [("notosansgurmukhi", ["NotoSansGurmukhi[wdth,wght].ttf"])],
    "gujarati": [("notosansgujarati", ["NotoSansGujarati[wdth,wght].ttf"])],
    "oriya": [("notosansoriya", ["NotoSansOriya[wdth,wght].ttf"])],
    "tamil": [("notosanstamil", ["NotoSansTamil[wdth,wght].ttf"])],
    "telugu": [("notosanstelugu", ["NotoSansTelugu[wdth,wght].ttf"])],
    "kannada": [("notosanskannada", ["NotoSansKannada[wdth,wght].ttf"])],
    "malayalam": [("notosansmalayalam", ["NotoSansMalayalam[wdth,wght].ttf"])],
    "sinhala": [("notosanssinhala", ["NotoSansSinhala[wdth,wght].ttf"])],
    "thai": [("notosansthai", ["NotoSansThai[wdth,wght].ttf"])],
    "lao": [("notosanslao", ["NotoSansLao[wdth,wght].ttf"])],
    "tibetan": [("notoseriftibetan", ["NotoSerifTibetan[wght].ttf"])],
    "myanmar": [("notosansmyanmar", ["NotoSansMyanmar[wdth,wght].ttf"])],
    "georgian": [("notosansgeorgian", ["NotoSansGeorgian[wdth,wght].ttf"])],
    "japanese": [("notosansjp", ["NotoSansJP[wght].ttf"])],
    "korean": [("notosanskr", ["NotoSansKR[wght].ttf"])],
    "cjk": [("notosanssc", ["NotoSansSC[wght].ttf"])],
}

# Handwriting faces, for form values and annotations that a person filled in by
# hand. Latin and Devanagari only - Google Fonts has no reliable handwriting
# coverage for most other scripts, and the caller falls back to a printed face.
_HANDWRITING_FONTS: dict[str, list[tuple[str, list[str]]]] = {
    "latin": [
        ("caveat", ["Caveat[wght].ttf", "Caveat-Regular.ttf"]),
        ("indieflower", ["IndieFlower-Regular.ttf", "IndieFlower.ttf"]),
        ("shadowsintolight", ["ShadowsIntoLight.ttf"]),
        ("architectsdaughter", ["ArchitectsDaughter-Regular.ttf", "ArchitectsDaughter.ttf"]),
        ("patrickhand", ["PatrickHand-Regular.ttf", "PatrickHand.ttf"]),
    ],
    "devanagari": [("kalam", ["Kalam-Regular.ttf", "Kalam.ttf"])],
}

# A small, ordered fallback chain. Noto Sans (latin/greek/cyrillic) is tried
# first because it covers the bulk of European text; CJK is the broadest net for
# ideographic scripts.
_FALLBACK_FONTS: list[tuple[str, list[str]]] = [
    ("notosans", ["NotoSans[wdth,wght].ttf"]),
    ("notosanssc", ["NotoSansSC[wght].ttf"]),
]

_DOWNLOAD_LOCK = threading.Lock()


def _codepoint_script(cp: int) -> str:
    for start, end, script in _SCRIPT_RANGES:
        if start <= cp <= end:
            return script
    return "unknown"


class FontDownloader:
    """Resolve and download open-source fonts that cover arbitrary Unicode text.

    Args:
        cache_dir (str): Directory used to store downloaded fonts. Created if missing.
        source_base_url (str): Base URL of the Google Fonts raw mirror.
        timeout (int): Per-request download timeout in seconds.
        enabled (bool): If ``False`` the downloader never hits the network and
            :meth:`resolve` always returns ``None`` (graceful no-op).
    """

    def __init__(
        self,
        cache_dir: str,
        source_base_url: str = _GF_RAW,
        timeout: int = 30,
        enabled: bool = True,
    ):
        self.cache_dir = cache_dir
        self.source_base_url = source_base_url.rstrip("/")
        self.timeout = timeout
        self.enabled = enabled
        # script -> resolved local path | None (negative results are cached too)
        self._resolved: dict[str, str | None] = {}
        # family__filename of downloads known to fail (404 etc.), to avoid
        # re-requesting - and re-logging - the same missing file for every word.
        self._failed_downloads: set[str] = set()
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def required_scripts(text: str) -> set[str]:
        """Return the set of Unicode scripts needed to render ``text``.

        Whitespace, ASCII digits and common punctuation are ignored because they
        are covered by virtually every font.
        """
        scripts: set[str] = set()
        for ch in text:
            if ch.isspace() or ch in "0123456789.,;:!?-_'\"()[]{}/\\@#%&*+=<>|~`$":
                continue
            scripts.add(_codepoint_script(ord(ch)))
        scripts.discard("unknown")
        return scripts

    def _download(self, family: str, filename: str) -> str | None:
        """Download ``family/filename`` into the cache, returning its local path."""
        local_name = f"{family}__{filename}".replace("[", "_").replace("]", "_").replace(",", "_")
        local_path = os.path.join(self.cache_dir, local_name)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path
        # Already known to be missing/unavailable: skip the network and the log.
        if local_name in self._failed_downloads:
            return None

        url = f"{self.source_base_url}/ofl/{family}/{urllib.parse.quote(filename)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "docTR-Synth-Generator"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
            if not data:
                self._failed_downloads.add(local_name)
                return None
            # Atomic write so concurrent workers never observe a partial file.
            fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".part")
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, local_path)
            return local_path
        except Exception as e:  # pragma: no cover - network dependent
            # Remember the failure so it is attempted (and logged) only once.
            self._failed_downloads.add(local_name)
            print(f"FontDownloader: {family}/{filename} unavailable ({e}); will not retry.")
            return None

    @staticmethod
    def _covers(font_path: str, required_chars: set[str]) -> bool:
        try:
            font = TTFont(font_path, fontNumber=0, lazy=True)
            supported: set[int] = set()
            for cmap in font["cmap"].tables:
                if cmap.isUnicode():
                    supported.update(cmap.cmap.keys())
            font.close()
            return all(ord(c) in supported for c in required_chars)
        except Exception:
            return False

    def _resolve_script(self, script: str, required_chars: set[str]) -> str | None:
        """Download (or reuse) a font covering ``required_chars`` for ``script``."""
        candidates = _SCRIPT_FONTS.get(script, []) + _FALLBACK_FONTS
        for family, filenames in candidates:
            for filename in filenames:
                with _DOWNLOAD_LOCK:
                    path = self._download(family, filename)
                if path and self._covers(path, required_chars):
                    return path
        return None

    def resolve_handwriting(self, text: str) -> str | None:
        """Return a handwriting face covering ``text``, or ``None``.

        ``None`` is a normal outcome, not an error: most scripts have no
        handwriting face here, and the caller falls back to a printed one.
        """
        if not self.enabled:
            return None
        required_chars = {c for c in text if not c.isspace()}
        if not required_chars:
            return None
        scripts = self.required_scripts(text) or {"latin"}
        if len(scripts) != 1:
            return None
        candidates = _HANDWRITING_FONTS.get(next(iter(scripts)), [])
        random.shuffle(candidates := list(candidates))
        for family, filenames in candidates:
            for filename in filenames:
                with _DOWNLOAD_LOCK:
                    path = self._download(family, filename)
                if path and self._covers(path, required_chars):
                    return path
        return None

    def resolve(self, text: str) -> str | None:
        """Return a local font path that fully covers ``text``, downloading if needed.

        Returns ``None`` if downloading is disabled or no known font covers the
        text (in which case the caller should fall back to skipping the sample).
        """
        if not self.enabled:
            return None

        required_chars = {c for c in text if not c.isspace()}
        if not required_chars:
            return None

        scripts = self.required_scripts(text) or {"latin"}

        # Single-script fast path: a cached resolution can be reused directly,
        # but only if it still covers the specific characters of this text.
        if len(scripts) == 1:
            script = next(iter(scripts))
            cached = self._resolved.get(script)
            if cached and self._covers(cached, required_chars):
                return cached
            resolved = self._resolve_script(script, required_chars)
            self._resolved[script] = resolved
            return resolved

        # Multi-script text: try to find a single font covering everything,
        # preferring the broad Noto Sans family before giving up.
        for family, filenames in _FALLBACK_FONTS:
            for filename in filenames:
                with _DOWNLOAD_LOCK:
                    path = self._download(family, filename)
                if path and self._covers(path, required_chars):
                    return path
        return None
