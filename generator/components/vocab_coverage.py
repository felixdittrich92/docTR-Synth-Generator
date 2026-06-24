# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import random
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache

from .vocabs import VOCABS

__all__ = [
    "LANGUAGE_TO_VOCAB",
    "VOCAB_TO_LANGUAGE",
    "resolve_target_vocab",
    "resolve_vocab_charset",
    "filter_in_vocab",
    "augment_words_for_coverage",
]

# ISO 639-1 code -> VOCABS key (only mappings whose vocab actually exists).
LANGUAGE_TO_VOCAB: dict[str, str] = {
    "af": "afrikaans",
    "ar": "arabic",
    "az": "azerbaijani",
    "be": "belarusian",
    "bg": "bulgarian",
    "bn": "bengali",
    "ca": "catalan",
    "cs": "czech",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "et": "estonian",
    "eu": "basque",
    "fa": "persian",
    "fi": "finnish",
    "fr": "french",
    "ga": "irish",
    "gu": "gujarati",
    "he": "hebrew",
    "hi": "hindi",
    "hr": "croatian",
    "hu": "hungarian",
    "hy": "armenian",
    "id": "indonesian",
    "is": "icelandic",
    "it": "italian",
    "ja": "japanese",
    "ka": "georgian",
    "kn": "kannada",
    "ko": "korean",
    "lt": "lithuanian",
    "lv": "latvian",
    "mk": "macedonian",
    "ml": "malayalam",
    "mr": "marathi",
    "mt": "maltese",
    "my": "burmese",
    "nb": "norwegian",
    "nl": "dutch",
    "no": "norwegian",
    "or": "odia",
    "pl": "polish",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "si": "sinhala",
    "sk": "slovak",
    "sl": "slovene",
    "sq": "albanian",
    "sv": "swedish",
    "ta": "tamil",
    "te": "telugu",
    "th": "thai",
    "tr": "turkish",
    "uk": "ukrainian",
    "ur": "urdu",
    "vi": "vietnamese",
}


VOCAB_TO_LANGUAGE: dict[str, str] = {}
for _iso, _key in LANGUAGE_TO_VOCAB.items():
    VOCAB_TO_LANGUAGE.setdefault(_key, _iso)  # first ISO code wins


def resolve_vocab_charset(vocab: str | list[str] | tuple[str, ...] | set[str] | None) -> set[str] | None:
    """Resolve a vocab spec into the set of allowed characters.

    Args:
        vocab: a single :data:`VOCABS` key (e.g. ``"german"``), a literal string
            of characters, or an iterable of any of those. The result is the
            union over all of them - so ``["german", "urdu", "odia"]`` yields the
            combined character set those three vocabs cover.

    Returns:
        The union character set, or ``None`` when nothing resolves.
    """
    if not vocab:
        return None
    items = [vocab] if isinstance(vocab, str) else list(vocab)
    chars: set[str] = set()
    for item in items:
        if not item:
            continue
        chars |= set(VOCABS[item]) if item in VOCABS else set(item)
    return chars or None


def resolve_target_vocab(language: str | None, target_vocab: str | list[str] | None) -> set[str] | None:
    """Resolve the set of characters that must be covered.

    Args:
        language (str | None): ISO 639-1 code used to look up a vocab.
        target_vocab: Explicit override - a :data:`VOCABS` key, a literal string
            of characters, or a list of those (their union). Takes precedence
            over ``language`` when set.

    Returns:
        set[str] | None: The character set to cover, or ``None`` when nothing
        sensible can be resolved (e.g. CJK, which has no fixed small vocab).
    """
    if target_vocab:
        return resolve_vocab_charset(target_vocab)
    if language:
        key = LANGUAGE_TO_VOCAB.get(language.lower())
        if key and key in VOCABS:
            return set(VOCABS[key])
    return None


def filter_in_vocab(words: list[str], charset: set[str] | None) -> list[str]:
    """Keep only words whose every character lies within ``charset``.

    A no-op when ``charset`` is ``None``. This is what guarantees a recognition
    model trained on a fixed vocab never sees a label it cannot encode.
    """
    if not charset:
        return list(words)
    return [w for w in words if w and set(w) <= charset]


_LARGE_SCRIPT_THRESHOLD = 400  # scripts bigger than this (CJK, Hangul) are left to the corpus


@lru_cache(maxsize=4096)
def _script_of(char: str) -> str:
    """Coarse Unicode script bucket for a character, e.g. ``"LATIN"``/``"HEBREW"``.

    Digits, punctuation, symbols and separators are script-neutral and return
    ``"COMMON"``. Combining marks inherit the script named in their Unicode name
    so they stay attached to letters of the same script.
    """
    category = unicodedata.category(char)
    if category[0] in ("N", "P", "S", "Z", "C"):
        return "COMMON"
    try:
        name = unicodedata.name(char)
    except ValueError:
        return "COMMON"
    return name.split(" ", 1)[0]  # the first word of the name is the script/block


def _is_mark(char: str) -> bool:
    """True for combining marks (nonspacing, spacing-combining or enclosing).

    Uses the general category rather than the combining class, because spacing
    vowel signs (e.g. Devanagari matras) have a combining class of 0 yet still
    must follow a base letter.
    """
    return unicodedata.category(char).startswith("M")


def _is_virama(char: str) -> bool:
    """True for viramas/halants - stackers that must be followed by a consonant.

    Canonical combining class 9 marks this family across Brahmic scripts
    (Devanagari U+094D, Bengali U+09CD, Tamil U+0BCD, Myanmar U+1039, Khmer
    U+17D2, ...). A dangling virama renders a dotted circle, so synthesised
    tokens must keep it between two base letters.
    """
    return unicodedata.combining(char) == 9


def _make_token(
    char: str,
    script_words: dict[str, list[str]],
    script_base_letters: dict[str, list[str]],
    rng: random.Random,
) -> str:
    """Build a short, **single-script**, valid, word-like token containing ``char``.

    Two rules keep the result renderable: the token never mixes scripts (a Hebrew
    letter is only ever placed in a Hebrew token), and a combining mark is always
    attached to a base letter of the same script rather than left to render alone
    on a dotted circle.
    """
    script = _script_of(char)
    if script == "COMMON":
        # Script-neutral symbol/digit/punctuation: emit standalone (optionally
        # with a few digits) so the renderer is free to pick whatever font covers
        # it, without dragging in letters of an unrelated script.
        digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(0, 3)))
        return char + digits

    base_letters = script_base_letters.get(script, [])
    bases = script_words.get(script, [])

    if _is_mark(char):
        if _is_virama(char):
            # A virama/halant (canonical combining class 9) is a stacker: it must
            # sit BETWEEN two base letters (consonant + virama + consonant) to form
            # a valid conjunct, never dangling at the end (which renders a dotted
            # circle). This matters for Myanmar U+1039, Khmer U+17D2, Indic halants.
            for _ in range(6):  # prefer a real word where a letter already follows
                if not bases:
                    break
                word = rng.choice(bases)
                positions = [
                    i
                    for i in range(len(word) - 1)
                    if unicodedata.category(word[i]).startswith("L")
                    and unicodedata.category(word[i + 1]).startswith("L")
                ]
                if positions:
                    i = rng.choice(positions)
                    return (word[: i + 1] + char + word[i + 1 :])[:16]
            if base_letters:  # scaffold: always a consonant on both sides of the virama
                lead = "".join(rng.choice(base_letters) for _ in range(rng.randint(1, 2)))
                tail = "".join(rng.choice(base_letters) for _ in range(rng.randint(1, 2)))
                return (lead + char + tail)[:16]
            return ""
        # A non-virama combining mark must sit on a base letter. Prefer inserting
        # it right after a letter of a real same-script word.
        for _ in range(6):
            if not bases:
                break
            word = rng.choice(bases)
            positions = [i for i, c in enumerate(word) if unicodedata.category(c).startswith("L")]
            if positions:
                i = rng.choice(positions)
                return (word[: i + 1] + char + word[i + 1 :])[:16]
        # Fall back to a couple of same-script base letters.
        if base_letters:
            lead = "".join(rng.choice(base_letters) for _ in range(rng.randint(1, 2)))
            tail = "".join(rng.choice(base_letters) for _ in range(rng.randint(0, 2)))
            return (lead + char + tail)[:16]
        return ""  # no base letter available for this script -> cannot place validly

    # Non-mark character (letter or standalone modifier).
    if bases and rng.random() < 0.6:
        word = rng.choice(bases)
        pos = rng.randint(0, len(word))
        return (word[:pos] + char + word[pos:])[:16]
    if base_letters:
        left = "".join(rng.choice(base_letters) for _ in range(rng.randint(0, 3)))
        right = "".join(rng.choice(base_letters) for _ in range(rng.randint(0, 3)))
        return (left + char + right) or char
    return char


def augment_words_for_coverage(
    words: list[str],
    target_chars: set[str],
    min_count: int = 3,
    seed: int | None = None,
) -> tuple[list[str], int, int]:
    """Append synthetic tokens so every target character is well represented.

    Tokens are kept within a single script so they always render with one font.
    Characters belonging to very large scripts (CJK ideographs, Hangul) are left
    to the real corpus rather than synthesised, since exhaustive coverage there
    is neither meaningful nor practical.

    Args:
        words (list[str]): The real word pool.
        target_chars (set[str]): Characters that should each appear ``>= min_count``
            times across the returned list.
        min_count (int): Minimum occurrences required per target character.
        seed (int | None): Seed for reproducible token synthesis.

    Returns:
        tuple[list[str], int, int]: ``(augmented_words, n_tokens_added,
        n_chars_synthesised_for)``.
    """
    if not target_chars or min_count <= 0:
        return words, 0, 0

    rng = random.Random(seed)
    counts: Counter[str] = Counter()
    for word in words:
        counts.update(word)

    # Group target characters by script so synthesis never crosses scripts. Keep
    # base letters (category L) separate from marks: only base letters are used as
    # scaffolding, so a token never starts with - or consists only of - marks.
    script_letters: dict[str, list[str]] = defaultdict(list)
    script_base_letters: dict[str, list[str]] = defaultdict(list)
    for char in target_chars:
        script = _script_of(char)
        if script != "COMMON":
            script_letters[script].append(char)
            if unicodedata.category(char).startswith("L"):
                script_base_letters[script].append(char)
    for table in (script_letters, script_base_letters):
        for script in table:
            table[script].sort()
    script_words: dict[str, list[str]] = defaultdict(list)
    for word in words:
        if not word:
            continue
        scripts = {s for s in (_script_of(ch) for ch in word) if s != "COMMON"}
        if len(scripts) == 1:  # only pure single-script words make safe scaffolds
            script_words[next(iter(scripts))].append(word)

    # Huge scripts (e.g. thousands of CJK ideographs) can't be covered by
    # synthesis; rely on the corpus for those instead of bloating the dataset.
    oversized = {s for s, chars in script_letters.items() if len(chars) > _LARGE_SCRIPT_THRESHOLD}

    # Real words that already contain an under-represented character give
    # linguistically-attested clusters (correct diacritic combinations, real
    # orthography). Prefer repeating those over synthesising; synthesis is then
    # only a fallback for characters genuinely absent from the corpus (rare
    # punctuation, currency, capitals in a lower-cased corpus, ...).
    under_present = {c for c in target_chars if 0 < counts.get(c, 0) < min_count and _script_of(c) not in oversized}
    real_with: dict[str, list[str]] = defaultdict(list)
    if under_present:
        for word in words:
            if not word:
                continue
            for char in set(word) & under_present:
                real_with[char].append(word)

    extra: list[str] = []
    deficient = 0
    for char in sorted(target_chars):
        if counts.get(char, 0) >= min_count or _script_of(char) in oversized:
            continue
        deficient += 1
        attested = real_with.get(char)
        attempts = 0
        # Cap attempts per char so a stubborn case can never loop forever.
        while counts.get(char, 0) < min_count and attempts < min_count * 4 + 4:
            attempts += 1
            if attested:
                token = rng.choice(attested)  # real, attested word - correct combinations
            else:
                token = _make_token(char, script_words, script_base_letters, rng)
            if not token:
                break  # cannot place this character validly (no same-script base letter)
            extra.append(token)
            counts.update(token)  # a token may satisfy several characters at once

    return words + extra, len(extra), deficient
