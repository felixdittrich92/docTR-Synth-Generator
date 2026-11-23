# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Raw word sources per language: frequency lists, repairs and derived languages."""

import random
import re
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

from ..components.vocabs import _BASE_VOCABS, VOCABS
from .languages import LANGUAGES, letter_scripts

__all__ = [
    "Corpus",
    "http_get",
    "fix_prebase_matras",
    "syllable_words",
    "sr_latn_to_cyrl",
    "uz_latn_to_cyrl",
    "ku_latn_to_sorani",
]

_HERMIT = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018"
_FREKWENCJA = "https://raw.githubusercontent.com/frekwencja/most-common-words-multilingual/main/data/wordfrequency.info"

# Which list to download for a language code.
SOURCE = {"sr-Latn": "sr", "sr-Cyrl": "sr", "zh": "zh_cn"}
# hermitdave's lists for these are mis-decoded (e.g. Cyrillic read as Georgian).
SECONDARY_ONLY = {"ka", "si"}
# Beyond the first 50k entries of a "full" list, keep only words seen this often: the
# long tail is mostly names, typos and foreign words.
MIN_FREQ = 8

# Mojibake in the OpenSubtitles-derived lists (legacy code pages decoded as Latin-1),
# plus Romanian cedilla -> comma-below (the orthography docTR's vocab uses).
FIXES = {
    "is": str.maketrans("đĐūŪķĶũŨōõ", "ðÐþÞóÓýÝðð"),  # 'ūađ' -> 'það'
    "et": str.maketrans("ðÐþÞ", "šŠžŽ"),
    "ro": str.maketrans("şŞţŢãÃþÞºª", "șȘțȚăĂțȚșȘ"),
    "hu": str.maketrans("õÕûÛ", "őŐűŰ"),
}
THAI_MOJIBAKE = re.compile(r"(เ[ธน].){2,}")  # UTF-8 Thai read as TIS-620


def http_get(url: str, timeout: int = 120) -> bytes | None:
    """GET ``url``; ``None`` on any failure (callers fall back to another source)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "doctr-synth-generator"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read() or None
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Script conversions for languages without a public word list
# -----------------------------------------------------------------------------

_SR_DI = [("dž", "џ"), ("Dž", "Џ"), ("DŽ", "Џ"), ("lj", "љ"), ("Lj", "Љ"), ("LJ", "Љ"),
          ("nj", "њ"), ("Nj", "Њ"), ("NJ", "Њ")]  # fmt: skip
_SR_MAP = str.maketrans(
    "abvgdđežzijklmnoprstćufhcčšABVGDĐEŽZIJKLMNOPRSTĆUFHCČŠ",
    "абвгдђежзијклмнопрстћуфхцчшАБВГДЂЕЖЗИЈКЛМНОПРСТЋУФХЦЧШ",
)


def sr_latn_to_cyrl(word: str) -> str:
    """Serbian Latin -> Serbian Cyrillic (the two orthographies map 1:1)."""
    for a, b in _SR_DI:
        word = word.replace(a, b)
    return word.translate(_SR_MAP)


_UZ_MULTI = [("o'", "ў"), ("g'", "ғ"), ("oʻ", "ў"), ("gʻ", "ғ"), ("sh", "ш"), ("ch", "ч"),
             ("yo", "ё"), ("yu", "ю"), ("ya", "я"), ("ts", "ц")]  # fmt: skip
_UZ_MAP = str.maketrans("abdefghijklmnopqrstuvxyz'ʼ", "абдефгҳижклмнопқрстувхйзъъ")


def uz_latn_to_cyrl(word: str) -> str:
    """Uzbek Latin -> Uzbek Cyrillic."""
    word = word.lower()
    for a, b in _UZ_MULTI:
        word = word.replace(a, b)
    word = ("э" + word[1:]) if word.startswith("e") else word
    return word.translate(_UZ_MAP)


_KU_MAP = {"a": "ا", "b": "ب", "c": "ج", "ç": "چ", "d": "د", "e": "ە", "ê": "ێ", "f": "ف", "g": "گ",
           "h": "ه", "i": "", "î": "ی", "j": "ژ", "k": "ک", "l": "ل", "m": "م", "n": "ن", "o": "ۆ",
           "p": "پ", "q": "ق", "r": "ر", "s": "س", "ş": "ش", "t": "ت", "u": "و", "û": "وو", "v": "ڤ",
           "w": "و", "x": "خ", "y": "ی", "z": "ز"}  # fmt: skip


def ku_latn_to_sorani(word: str, rng: random.Random) -> str:
    """Kurmanji (Latin) -> Central Kurdish (Arabic script); ``""`` if not mappable."""
    word = word.lower()
    if not word or any(c not in _KU_MAP for c in word):
        return ""
    out = "".join(_KU_MAP[c] for c in word)
    out = "".join(
        "ڕ" if c == "ر" and rng.random() < 0.25 else "ڵ" if c == "ل" and rng.random() < 0.25 else c for c in out
    )
    return ("ئ" + out) if word[0] in "aeêiîouû" else out


def _to_polytonic(word: str, rng: random.Random, variants: dict[str, list[str]]) -> str:
    word = unicodedata.normalize("NFC", unicodedata.normalize("NFD", word).replace("\u0301", ""))
    return "".join(
        rng.choice(variants[c]) if variants.get(c) and rng.random() < (0.8 if i == 0 else 0.35) else c
        for i, c in enumerate(word)
    )


def _inject(word: str, table: dict[str, str], p: float, rng: random.Random) -> str:
    return "".join(table[c] if c in table and rng.random() < p else c for c in word)


def syllable_words(vocab_key: str, n: int, rng: random.Random, syllables: tuple[int, int] = (1, 4)) -> list[str]:
    """Pseudo-words for languages with no corpus - valid letter/mark order, no dotted circles.

    Fine for glyph coverage; they are not real words, so such languages get a low
    sampling weight.
    """
    letters_all = VOCABS[vocab_key]
    stem = {"javanese": "javanese", "sudanese": "sudanese"}.get(vocab_key)
    if stem:  # Brahmic: consonant (+ vowel sign) (+ virama + consonant)
        cons = list(_BASE_VOCABS[f"{stem}_consonants"])
        extra = _BASE_VOCABS.get(f"{stem}_vowels", "") + _BASE_VOCABS.get(f"{stem}_diacritics", "")
        marks = [c for c in extra if unicodedata.category(c).startswith("M") and unicodedata.combining(c) != 9]
        independent = [c for c in extra if unicodedata.category(c).startswith("L")]
        virama = next((c for c in letters_all if unicodedata.combining(c) == 9), "")

        def syllable() -> str:
            if independent and rng.random() < 0.1:
                return rng.choice(independent)
            s = rng.choice(cons)
            if virama and rng.random() < 0.12:
                s += virama + rng.choice(cons)
            return s + (rng.choice(marks) if marks and rng.random() < 0.55 else "")

    else:  # alphabetic: CV / V / CVC
        letters = [c for c in letters_all if c.isalpha() and c.islower()]
        vowels = [c for c in letters if unicodedata.normalize("NFD", c)[0] in "aeiou"]
        cons = [c for c in letters if c not in vowels]
        if not vowels or not cons:
            return []

        def syllable() -> str:
            s = rng.choice(cons) if rng.random() < 0.85 else ""
            return s + rng.choice(vowels) + (rng.choice(cons) if rng.random() < 0.15 else "")

    return ["".join(syllable() for _ in range(rng.randint(*syllables))) for _ in range(n)]


# -----------------------------------------------------------------------------
# Orthography repairs of the source lists
# -----------------------------------------------------------------------------

_YO_VOWELS = "aeiouẹọAEIOUẸỌ"


def yoruba_tone_variants(words: list[str], rng: random.Random) -> list[str]:
    """Toned variants (à, é, ọ̀ ...): the Yoruba list is almost entirely untoned."""
    out = []
    for w in words:
        if any(c in _YO_VOWELS for c in w):
            toned = "".join(c + rng.choice("\u0300\u0301") if c in _YO_VOWELS and rng.random() < 0.5 else c for c in w)
            if toned != w:
                out.append(unicodedata.normalize("NFC", toned))
    return out


def uzbek_apostrophes(words: list[str]) -> list[str]:
    """oʻ / gʻ / ʼ variants: the source list types all three as the ASCII apostrophe."""
    out = []
    for w in words:
        if "'" in w:
            v = re.sub(r"([oOgG])'", lambda m: m.group(1) + "\u02bb", w).replace("'", "\u02bc")
            if v != w:
                out.append(v)
    return out


def catalan_interpunct(words: list[str]) -> list[str]:
    """l·l spelled with the middle dot U+00B7, as the orthography prescribes."""
    out = [
        w.replace("\u0140l", "l\u00b7l").replace("\u013fL", "L\u00b7L") for w in words if "\u0140" in w or "\u013f" in w
    ]
    return [w for w in out if "\u0140" not in w and "\u013f" not in w]


# Indic vowel signs drawn *left* of their consonant but stored *after* it. Lists built
# from legacy encodings store them in visual order instead ("िकया" for "किया"), which a
# shaper can only draw as a dotted circle.
_PREBASE_MATRAS = set("\u093f\u09bf\u09c7\u09c8\u0a3f\u0abf\u0b47\u0bc6\u0bc7\u0bc8\u0d46\u0d47\u0d48\u0dd9")


def fix_prebase_matras(word: str) -> str:
    """Move a pre-base vowel sign that precedes its consonant back behind it."""
    if not _PREBASE_MATRAS.intersection(word):
        return word
    chars = list(word)
    i = 0
    while i < len(chars) - 1:
        c, nxt = chars[i], chars[i + 1]
        prev = chars[i - 1] if i else ""
        orphan = not prev or not unicodedata.category(prev).startswith("L") or unicodedata.combining(prev) == 9
        if c in _PREBASE_MATRAS and orphan and unicodedata.category(nxt) == "Lo":
            chars[i], chars[i + 1] = nxt, c
            i += 2
        else:
            i += 1
    return unicodedata.normalize("NFC", "".join(chars))


# -----------------------------------------------------------------------------
# Corpus
# -----------------------------------------------------------------------------


class Corpus:
    """Downloads (once) and caches word lists, and builds a raw word list per language.

    Args:
        cache_dir: where the downloaded lists are stored.
        full_lists: prefer hermitdave's ``*_full`` lists (2-3x more unique words
            than ``*_50k``; entries past rank 50k need ``MIN_FREQ`` occurrences).
        offline: never download, use only what is cached.
    """

    def __init__(self, cache_dir: str | Path, full_lists: bool = True, offline: bool = False):
        self.cache_dir = Path(cache_dir)
        self.full_lists = full_lists
        self.offline = offline
        self._raw: dict[str, list[str]] = {}

    def _cached(self, name: str, url: str) -> Path | None:
        path = self.cache_dir / name
        if path.exists():
            return path
        if self.offline:
            return None
        data = http_get(url)
        if not data:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def frequency_list(self, code: str) -> list[str]:
        """Words of a source list, most frequent first. hermitdave first, frekwencja fallback."""
        if code not in SECONDARY_ONLY:
            for variant in ("full", "50k") if self.full_lists else ("50k", "full"):
                path = self._cached(f"hermit_{code}_{variant}.txt", f"{_HERMIT}/{code}/{code}_{variant}.txt")
                if path is None:
                    continue
                words: list[str] = []
                for line in path.read_text("utf-8", errors="replace").splitlines():
                    parts = line.split()
                    if len(parts) >= 2 and parts[-1].isdigit():
                        if int(parts[-1]) < MIN_FREQ and len(words) >= 50_000:
                            break
                        words.append(parts[0])
                if words:
                    return words
        path = self._cached(f"frek_{code}.txt", f"{_FREKWENCJA}/{code}.txt")
        if path is None:
            return []
        lines = [ln.strip() for ln in path.read_text("utf-8", errors="replace").splitlines() if ln.strip()]
        if lines and lines[0] == code:
            lines = lines[1:]
        return [ln.split()[0] for ln in lines]

    def _source(self, lang: str) -> list[str]:
        """NFC-normalised, mojibake-repaired words of the list behind ``lang``."""
        if lang not in self._raw:
            code = SOURCE.get(lang, lang)
            fix = FIXES.get(code)
            words = [unicodedata.normalize("NFC", w.strip()) for w in self.frequency_list(code)]
            self._raw[lang] = [w.translate(fix) for w in words] if fix else words
        return self._raw[lang]

    def raw_words(self, lang: str, rng: random.Random) -> list[str]:
        """Uncleaned candidate words for ``lang``, most frequent first where known."""
        if lang == "sr-Cyrl":
            return [sr_latn_to_cyrl(w) if letter_scripts(w) == {"LATIN"} else w for w in self._source("sr-Cyrl")]
        if lang == "cnr":  # Montenegrin = Serbian Latin + ś/ź
            return [_inject(w, {"s": "ś", "z": "ź"}, 0.12, rng) for w in self._source("sr-Latn")]
        if lang == "uz-Cyrl":
            return [uz_latn_to_cyrl(w) for w in self._source("uz")]
        if lang == "sah":  # Turkic Cyrillic words + the Yakut-only letters
            donors = self._source("ky") + self._source("kk") + self._source("tt")
            return [_inject(w, {"г": "ҕ", "н": "ҥ", "о": "ө", "у": "ү", "х": "һ"}, 0.3, rng) for w in donors]
        if lang == "grc":
            extended = set(VOCABS["greek_extended"]) - set(VOCABS["greek"])
            variants: dict[str, list[str]] = defaultdict(list)
            for c in sorted(extended):
                if c.isalpha():
                    variants[unicodedata.normalize("NFD", c)[0]].append(c)
            return [_to_polytonic(w, rng, variants) for w in self._source("el")]
        if lang == "ckb":
            return [t for t in (ku_latn_to_sorani(w, rng) for w in self._source("ku")) if t]
        if lang in ("qu", "mi"):
            return syllable_words(LANGUAGES[lang][0], 8000, rng, syllables=(1, 4))
        if lang in ("jv", "su"):  # scriptio continua: phrase-length tokens
            return syllable_words(LANGUAGES[lang][0], 8000, rng, syllables=(2, 7))

        words = self._source(lang)
        if any(c in _PREBASE_MATRAS for c in VOCABS[LANGUAGES[lang][0]]):
            words = [fix_prebase_matras(w) for w in words]
        if lang == "yo":
            words = words + yoruba_tone_variants(words, rng)
        elif lang == "uz":
            words = words + uzbek_apostrophes(words)
        elif lang == "ca":
            words = words + catalan_interpunct(words)
        return words


SYNTHETIC_LANGUAGES = {"qu", "mi", "jv", "su"}  # no word list: syllable pseudo-words
