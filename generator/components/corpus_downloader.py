# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
import random
import tempfile
import threading
import urllib.request

from .font_downloader import FontDownloader

__all__ = ["CorpusDownloader"]

_FW_BASE = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018"

# Secondary source: ~5k most-frequent words for 109 languages, one word per line.
# Fills the languages hermitdave lacks (Odia, Khmer, Burmese, most Indic/SE-Asian
# scripts, ...) so far fewer vocabs fall back to pure synthesis.
_FW2_BASE = "https://raw.githubusercontent.com/frekwencja/most-common-words-multilingual/main/data/wordfrequency.info"
# A few codes differ from ISO 639-1 in the secondary source.
_FW2_ALIASES = {"zh": "zh-CN"}

# Some languages live under a different code or only ship a ``_full`` list.
_LANG_ALIASES: dict[str, list[str]] = {
    "zh": ["zh_cn", "zh_tw"],
    "ja": ["ja"],  # only ja_full.txt exists - handled by the filename candidates
    "pt": ["pt", "pt_br"],
}

# Expected Unicode script(s) per language, used to drop foreign-script
# contamination (e.g. stray English tokens inside a German list). Languages not
# listed here are accepted without script filtering.
_LANG_SCRIPTS: dict[str, set[str]] = {
    "en": {"latin"},
    "de": {"latin"},
    "fr": {"latin"},
    "es": {"latin"},
    "it": {"latin"},
    "pt": {"latin"},
    "nl": {"latin"},
    "pl": {"latin"},
    "cs": {"latin"},
    "sk": {"latin"},
    "tr": {"latin"},
    "vi": {"latin"},
    "id": {"latin"},
    "ro": {"latin"},
    "hu": {"latin"},
    "sv": {"latin"},
    "da": {"latin"},
    "no": {"latin"},
    "fi": {"latin"},
    "et": {"latin"},
    "lt": {"latin"},
    "lv": {"latin"},
    "hr": {"latin"},
    "sl": {"latin"},
    "sq": {"latin"},
    "ru": {"cyrillic"},
    "uk": {"cyrillic"},
    "bg": {"cyrillic"},
    "sr": {"cyrillic"},
    "mk": {"cyrillic"},
    "el": {"greek"},
    "ar": {"arabic"},
    "fa": {"arabic"},
    "ur": {"arabic"},
    "he": {"hebrew"},
    "hi": {"devanagari"},
    "mr": {"devanagari"},
    "th": {"thai"},
    "ja": {"japanese", "cjk"},
    "ko": {"korean"},
    "zh": {"cjk"},
    # Languages served mainly by the secondary source - listed so the script
    # filter drops foreign (usually Latin) strays from those frequency lists.
    "bn": {"bengali"},
    "as": {"bengali"},
    "or": {"oriya"},
    "my": {"myanmar"},
    "km": {"khmer"},
    "lo": {"lao"},
    "si": {"sinhala"},
    "ta": {"tamil"},
    "te": {"telugu"},
    "kn": {"kannada"},
    "ml": {"malayalam"},
    "pa": {"gurmukhi"},
    "gu": {"gujarati"},
    "ne": {"devanagari"},
    "ka": {"georgian"},
    "hy": {"armenian"},
    "am": {"ethiopic"},
}

_DOWNLOAD_LOCK = threading.Lock()


class CorpusDownloader:
    """Resolve and download real word lists for requested languages.

    Args:
        cache_dir (str): Directory used to cache downloaded corpora.
        source_base_url (str): Base URL of the frequency-list mirror.
        timeout (int): Per-request download timeout in seconds.
        min_word_length (int): Minimum word length (characters) to keep.
        max_word_length (int): Maximum word length (characters) to keep.
        filter_by_script (bool): Drop words whose script does not match the
            requested language (removes foreign-language contamination).
    """

    def __init__(
        self,
        cache_dir: str,
        source_base_url: str = _FW_BASE,
        secondary_base_url: str = _FW2_BASE,
        timeout: int = 30,
        min_word_length: int = 1,
        max_word_length: int = 24,
        filter_by_script: bool = True,
    ):
        self.cache_dir = cache_dir
        self.source_base_url = source_base_url.rstrip("/")
        self.secondary_base_url = secondary_base_url.rstrip("/") if secondary_base_url else None
        self.timeout = timeout
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.filter_by_script = filter_by_script
        os.makedirs(self.cache_dir, exist_ok=True)
        self._cache: dict[str, list[str]] = {}

    def _candidate_paths(self, language: str) -> list[str]:
        codes = [language] + _LANG_ALIASES.get(language, [])
        # de-dupe while keeping order
        seen: set[str] = set()
        codes = [c for c in codes if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]
        paths: list[str] = []
        for code in codes:
            paths.append(f"{code}/{code}_50k.txt")
            paths.append(f"{code}/{code}_full.txt")
        return paths

    def _fetch_url(self, url: str) -> bytes | None:
        """GET a URL, returning its bytes (or None on any failure/empty)."""
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "docTR-Synth-Generator"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
            return data or None
        except Exception:
            return None

    def _download_raw(self, language: str) -> str | None:
        """Download the raw frequency list text for a language (cached on disk).

        Tries the primary mirror (hermitdave) first, then the secondary mirror
        (frekwencja) for the many languages hermitdave does not cover.
        """
        local_path = os.path.join(self.cache_dir, f"{language}_freq.txt")
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            with open(local_path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()

        # Primary: hermitdave (``word count`` per line).
        for rel in self._candidate_paths(language):
            data = self._fetch_url(f"{self.source_base_url}/{rel}")
            if data:
                self._cache_to_disk(local_path, data)
                return data.decode("utf-8", "replace")

        # Secondary: frekwencja (one word per line, with a leading code line).
        if self.secondary_base_url:
            code = _FW2_ALIASES.get(language, language)
            data = self._fetch_url(f"{self.secondary_base_url}/{code}.txt")
            if data:
                text = data.decode("utf-8", "replace")
                lines = text.splitlines()
                if lines and lines[0].strip() in {code, language}:  # drop the header line
                    text = "\n".join(lines[1:])
                self._cache_to_disk(local_path, text.encode("utf-8"))
                return text

        print(f"CorpusDownloader: no frequency list found for language '{language}'")
        return None

    def _cache_to_disk(self, local_path: str, data: bytes) -> None:
        """Atomically write downloaded bytes to the on-disk cache."""
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".part")
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, local_path)

    def _clean(self, raw: str, language: str) -> list[str]:
        """Parse and filter the raw ``word count`` lines into a clean word list."""
        expected = _LANG_SCRIPTS.get(language) if self.filter_by_script else None
        words: list[str] = []
        seen: set[str] = set()

        for line in raw.splitlines():
            parts = line.split()
            if not parts:
                continue
            word = parts[0].strip()
            if not word or word in seen:
                continue
            if not (self.min_word_length <= len(word) <= self.max_word_length):
                continue
            # Must contain at least one letter/ideograph (drop pure punctuation).
            if not any(ch.isalpha() for ch in word):
                continue
            if expected is not None:
                scripts = FontDownloader.required_scripts(word)
                if scripts and not scripts.issubset(expected):
                    continue
            seen.add(word)
            words.append(word)
        return words

    def fetch(self, language: str) -> list[str]:
        """Return a cleaned, frequency-ordered word list for a single language."""
        if language in self._cache:
            return self._cache[language]
        with _DOWNLOAD_LOCK:
            raw = self._download_raw(language)
        words = self._clean(raw, language) if raw else []
        self._cache[language] = words
        return words

    def build_vocabulary(self, languages: list[str], words_per_language: int = 50000) -> list[str]:
        """Build a combined vocabulary across ``languages``.

        Args:
            languages (list[str]): ISO 639-1 codes, e.g. ``["en", "de"]``.
            words_per_language (int): Cap on words taken from each language
                (the most frequent ones are kept).

        Returns:
            list[str]: De-duplicated vocabulary (order: language-major,
            frequency within language).
        """
        vocab: list[str] = []
        seen: set[str] = set()
        for lang in languages:
            for word in self.fetch(lang)[:words_per_language]:
                if word not in seen:
                    seen.add(word)
                    vocab.append(word)
        return vocab


def apply_casing_variants(words: list[str], prob: float = 0.3, seed: int | None = None) -> list[str]:
    """Add Title-case / UPPERCASE variants so capital glyphs are represented.

    Frequency lists are almost entirely lowercase; a recognizer trained only on
    them never learns capital letters. With probability ``prob`` each word is
    additionally emitted in a capitalised form (cased scripts only).

    Args:
        words (list[str]): Input words.
        prob (float): Probability of emitting an extra cased variant per word.
        seed (int | None): Optional RNG seed for reproducibility.

    Returns:
        list[str]: Original words plus the generated cased variants.
    """
    rng = random.Random(seed)
    out = list(words)
    for w in words:
        if not w or not w[0].isalpha():
            continue
        # Only meaningful for scripts that have upper/lower case.
        if w.lower() == w.upper():
            continue
        if rng.random() < prob:
            out.append(w.title() if rng.random() < 0.7 else w.upper())
    return out


def generate_numeric_tokens(n: int, seed: int | None = None) -> list[str]:
    """Generate realistic numeric / date / price / code tokens.

    Real document crops (invoices, receipts, forms, IDs) are full of these, yet
    dictionary word lists contain none. Mixing a small fraction in makes the
    dataset markedly closer to real documents.

    Args:
        n (int): Number of tokens to generate.
        seed (int | None): Optional RNG seed.

    Returns:
        list[str]: Generated tokens (labels match exactly what will be rendered).
    """
    rng = random.Random(seed)
    tokens: list[str] = []
    for _ in range(n):
        kind = rng.choice(["int", "decimal", "price", "date", "code", "percent", "phone"])
        if kind == "int":
            tokens.append(str(rng.randint(0, 999999)))
        elif kind == "decimal":
            dec = rng.choice([",", "."])
            tokens.append(f"{rng.randint(0, 9999)}{dec}{rng.randint(0, 99):02d}")
        elif kind == "price":
            sym = rng.choice(["€", "$", "£", ""])
            dec = rng.choice([",", "."])
            amt = f"{rng.randint(0, 9999)}{dec}{rng.randint(0, 99):02d}"
            tokens.append(f"{amt}{sym}" if sym in ("€", "") else f"{sym}{amt}")
        elif kind == "date":
            sep = rng.choice([".", "/", "-"])
            d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1990, 2030)
            tokens.append(rng.choice([f"{d:02d}{sep}{m:02d}{sep}{y}", f"{y}{sep}{m:02d}{sep}{d:02d}"]))
        elif kind == "code":
            letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(rng.randint(2, 4)))
            tokens.append(f"{letters}-{rng.randint(100, 99999)}")
        elif kind == "percent":
            tokens.append(f"{rng.randint(0, 100)}%")
        else:  # phone
            tokens.append(f"+{rng.randint(1, 99)}{rng.randint(100, 999)}{rng.randint(100000, 9999999)}")
    return tokens
