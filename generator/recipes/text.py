# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Token-level text helpers: casing, punctuation, homoglyphs, numbers, fillers."""

import random
import unicodedata
from collections import defaultdict

from ..components.corpus_downloader import generate_numeric_tokens
from ..components.vocabs import VOCABS
from .languages import LANGUAGES, PUNCT, primary_script, punct_style

__all__ = [
    "upper",
    "capitalize",
    "random_case",
    "attach_punct",
    "is_ambiguous",
    "digit_system",
    "mixes_digit_systems",
    "native_digit_systems",
    "native_number_tokens",
    "numeric_tokens",
    "digit_filler",
    "combining_filler",
    "cycle_sample",
]

# -----------------------------------------------------------------------------
# Casing
# -----------------------------------------------------------------------------


def upper(w: str, lang: str) -> str:
    """Language-aware upper case (Turkish dotted İ, Greek all-caps without tonos)."""
    if lang in ("tr", "az"):
        return w.replace("i", "İ").upper()
    if lang in ("el", "grc"):  # all-caps Greek drops the tonos, keeps the dialytika
        return unicodedata.normalize("NFC", unicodedata.normalize("NFD", w).replace("\u0301", "").upper())
    return w.upper()


def capitalize(w: str, lang: str) -> str:
    """First letter upper case. Not ``str.title()``: ``"it's".title() == "It'S"``."""
    if lang == "nl" and w.startswith("ij"):
        return "IJ" + w[2:]
    head = "İ" if (lang in ("tr", "az") and w[0] == "i") else w[0].upper()
    return head + w[1:]


def random_case(
    w: str,
    lang: str,
    charset: set[str],
    rng: random.Random,
    upper_prob: float = 0.08,
    title_prob: float = 0.25,
    max_len: int = 24,
    drop_ambiguous: bool = True,
) -> str:
    """Return ``w`` unchanged, title-cased or upper-cased; never leaves ``charset``."""
    if not w or w.upper() == w.lower() or primary_script(lang) == "GEORGIAN":  # caseless
        return w
    r = rng.random()
    if r < upper_prob:
        v = upper(w, lang)
    elif r < upper_prob + title_prob:
        v = capitalize(w, lang)
    else:
        return w
    if not set(v) <= charset or len(v) > max_len:
        return w
    return w if drop_ambiguous and is_ambiguous(v, primary_script(lang)) else v


def attach_punct(w: str, lang: str, charset: set[str], rng: random.Random, max_len: int = 24) -> str:
    """Attach script-appropriate punctuation, as word crops from real pages carry it."""
    trail, lead, wraps = PUNCT[punct_style(lang)]
    r = rng.random()
    if lang == "es" and r < 0.15:
        v = rng.choice([f"¿{w}?", f"¡{w}!"])
    elif lang in ("el", "grc") and r < 0.15:
        v = w + ";"  # the Greek question mark
    elif wraps and r < 0.2:
        v = rng.choice(wraps).format(w)
    elif lead and r < 0.32:
        v = rng.choice(lead) + w
    elif trail:
        v = w + rng.choice(trail)
    else:
        return w
    return v if set(v) <= charset and len(v) <= max_len else w


# -----------------------------------------------------------------------------
# Homoglyphs
# -----------------------------------------------------------------------------

# Letters that render identically to a Latin (or, for Greek, Cyrillic) letter in
# practically every font. A token made only of them (Russian 'се', Greek 'ΤΟ') is
# pixel-identical to a different string in another script, so whichever label it
# gets is noise. Such tokens are only ever labelled in the higher-priority script
# (Latin > Cyrillic > Greek), which makes the label a function of the image.
_CYR_LIKE_LAT = set("аеорсухіјѕһёїә" + "АВЕКМНОРСТХІЈЅҮЁЇӘ")
_GRK_LIKE_LAT = set("οινϊ" + "ΑΒΕΖΗΙΚΜΝΟΡΤΥΧΪ")
_GRK_LIKE_CYR = set("οϊ" + "ΑΒΓΕΗΙΚΜΟΠΡΤΥΦΧΪ")


def is_ambiguous(word: str, script: str) -> bool:
    """True if the word renders identically to a different string in another script."""
    letters = {c for c in word if c.isalpha()}
    if not letters or script not in ("CYRILLIC", "GREEK"):
        return False
    if script == "CYRILLIC":
        return letters <= _CYR_LIKE_LAT
    return letters <= _GRK_LIKE_LAT or letters <= _GRK_LIKE_CYR


# -----------------------------------------------------------------------------
# Numbers
# -----------------------------------------------------------------------------


def digit_system(c: str) -> str | None:
    """The numbering system of a character: ``"ASCII"``, a decimal run (identified by
    its zero: ٠ ۰ ० ๐ ...) or a script's own numerals (Ethiopic ፩፪፫); ``None`` otherwise."""
    if not unicodedata.category(c).startswith("N"):
        return None
    if c.isascii():
        return "ASCII"
    if unicodedata.category(c) == "Nd":
        return chr(ord(c) - unicodedata.digit(c))
    return unicodedata.name(c, "?").split(" ", 1)[0]


def mixes_digit_systems(word: str) -> bool:
    """'۰42', '፪543' - number strings no real document contains."""
    return len({d for c in word if (d := digit_system(c))}) > 1


def native_digit_systems(lang: str, charset: set[str]) -> list[dict[str, str]]:
    """Every complete non-ASCII digit run in the language's vocab, as ASCII->native maps.

    A list: Persian has both ٠-٩ and ۰-۹, Burmese both the Myanmar and the Shan digits.
    """
    runs: dict[str, dict[int, str]] = defaultdict(dict)
    for c in VOCABS[LANGUAGES[lang][0]]:
        if unicodedata.category(c) == "Nd" and not c.isascii() and c in charset:
            runs[chr(ord(c) - unicodedata.digit(c))][unicodedata.digit(c)] = c
    return [{str(d): run[d] for d in range(10)} for run in runs.values() if len(run) == 10]


def native_number_tokens(native: dict[str, str], n: int, rng: random.Random) -> list[str]:
    """Numbers written entirely in one script's own digits, every digit used evenly."""
    digits = list(native.values())
    bag: list[str] = []
    out = []
    for _ in range(n):
        length = rng.randint(1, 6)
        if len(bag) < length + 2:
            extra = digits * 4
            rng.shuffle(extra)
            bag += extra
        body = "".join(bag.pop() for _ in range(length))
        r = rng.random()
        if r < 0.25 and length >= 4:  # date-like
            body = body[:2] + rng.choice(".-/") + body[2:4] + rng.choice(".-/") + body[4:]
        elif r < 0.45 and length >= 3:  # decimal
            body = body[:-2] + rng.choice(",.") + body[-2:]
        elif r < 0.55:
            body += "%"
        out.append(body)
    return out


def numeric_tokens(
    lang: str, n: int, charset: set[str], rng: random.Random, native_prob: float = 0.4, allow_spaces: bool = False
) -> list[str]:
    """Prices, dates, codes, percentages ... - in ASCII or the language's own digits."""
    if n <= 0:
        return []
    systems = native_digit_systems(lang, charset)
    currency = sorted(c for c in charset if unicodedata.category(c) == "Sc") or ["$"]
    out: list[str] = []
    for t in generate_numeric_tokens(int(n * 2) + 200, seed=rng.randint(0, 2**31)):
        if " " in t and not allow_spaces:
            continue
        if rng.random() < 0.35:
            for s in "€$£¥":
                t = t.replace(s, rng.choice(currency))
        if systems and rng.random() < native_prob:
            t = t.translate({ord(k): v for k, v in rng.choice(systems).items()})
        if t and set(t) - {" "} <= charset and not mixes_digit_systems(t):
            out.append(t)
    rng.shuffle(out)
    return cycle_sample(out, n, rng) if out else []


def digit_filler(charset: set[str], missing: list[str], rng: random.Random, n_each: int) -> list[str]:
    """Same-system numbers for digits no language block produced (e.g. Shan digits)."""
    out = []
    for char in missing:
        system = digit_system(char)
        if system is None or system == "ASCII":
            continue
        siblings = sorted(c for c in charset if digit_system(c) == system) or [char]
        out += [char + "".join(rng.choice(siblings) for _ in range(rng.randint(0, 4))) for _ in range(n_each)]
    return out


def combining_filler(charset: set[str], missing: list[str], rng: random.Random, n_each: int) -> list[str]:
    """Base letter + combining mark tokens, for marks no real word supplied."""
    out = []
    for mark in missing:
        if not unicodedata.combining(mark):
            continue
        bases = sorted(
            c for c in charset if c.isalpha() and unicodedata.normalize("NFC", c + mark) != c + mark
        ) or sorted(c for c in charset if c.isalpha() and c.isascii())
        out += ["".join(rng.choice(bases) + mark for _ in range(rng.randint(1, 3))) for _ in range(n_each)]
    return out


def cycle_sample(words: list[str], k: int, rng: random.Random) -> list[str]:
    """``k`` samples where every word repeats as evenly as possible (no Zipf pile-up)."""
    if not words:
        return []
    out: list[str] = []
    while len(out) < k:
        batch = list(words)
        rng.shuffle(batch)
        out.extend(batch[: k - len(out)])
    return out
