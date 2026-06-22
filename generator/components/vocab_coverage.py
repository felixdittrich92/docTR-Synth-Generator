# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import random
from collections import Counter

from .vocabs import VOCABS

__all__ = ["LANGUAGE_TO_VOCAB", "resolve_target_vocab", "augment_words_for_coverage"]

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


def resolve_target_vocab(language: str | None, target_vocab: str | None) -> set[str] | None:
    """Resolve the set of characters that must be covered.

    Args:
        language (str | None): ISO 639-1 code used to look up a vocab.
        target_vocab (str | None): Explicit override - either a key in
            ``VOCABS`` (e.g. ``"german"``) or a literal string of characters.
            When set it takes precedence over ``language``.

    Returns:
        set[str] | None: The character set to cover, or ``None`` when nothing
        sensible can be resolved (e.g. CJK, which has no fixed small vocab).
    """
    if target_vocab:
        chars = VOCABS[target_vocab] if target_vocab in VOCABS else target_vocab
        return set(chars)
    if language:
        key = LANGUAGE_TO_VOCAB.get(language.lower())
        if key and key in VOCABS:
            return set(VOCABS[key])
    return None


def _make_token(char: str, base_words: list[str], letters: list[str], rng: random.Random) -> str:
    """Build a short, word-like token that contains ``char``."""
    # Symbols/currency from a different script than the main letters (e.g. ฿ in a
    # Latin vocab) cannot share a font with Latin glyphs, so a mixed token gets
    # skipped at render time. Emit them standalone (optionally with digits) so a
    # script-specific font can cover them.
    if not char.isalpha() and rng.random() < 0.4:
        digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(0, 3)))
        return char + digits
    if base_words and rng.random() < 0.6:
        # Inject the character into a real word - keeps a natural shape.
        word = rng.choice(base_words)
        if not word:
            word = "".join(rng.choice(letters) for _ in range(rng.randint(2, 5))) if letters else char
        pos = rng.randint(0, len(word))
        return (word[:pos] + char + word[pos:])[:16]
    # Otherwise surround the character with a few vocab letters.
    if letters:
        left = "".join(rng.choice(letters) for _ in range(rng.randint(0, 3)))
        right = "".join(rng.choice(letters) for _ in range(rng.randint(0, 3)))
        token = left + char + right
        return token or char
    return char


def augment_words_for_coverage(
    words: list[str],
    target_chars: set[str],
    min_count: int = 3,
    seed: int | None = None,
) -> tuple[list[str], int, int]:
    """Append synthetic tokens so every target character is well represented.

    Args:
        words (list[str]): The real word pool.
        target_chars (set[str]): Characters that must each appear ``>= min_count``
            times across the returned list.
        min_count (int): Minimum occurrences required per target character.
        seed (int | None): Seed for reproducible token synthesis.

    Returns:
        tuple[list[str], int, int]: ``(augmented_words, n_tokens_added,
        n_chars_that_were_missing_or_rare)``.
    """
    if not target_chars or min_count <= 0:
        return words, 0, 0

    rng = random.Random(seed)
    counts: Counter[str] = Counter()
    for word in words:
        counts.update(word)

    letters = sorted(c for c in target_chars if c.isalpha())
    base_words = [w for w in words if w]
    extra: list[str] = []
    deficient = sum(1 for c in target_chars if counts.get(c, 0) < min_count)

    # Deterministic order so runs are reproducible.
    for char in sorted(target_chars):
        attempts = 0
        # Cap attempts per char so a stubborn case can never loop forever.
        while counts.get(char, 0) < min_count and attempts < min_count * 4 + 4:
            token = _make_token(char, base_words, letters, rng)
            extra.append(token)
            counts.update(token)  # a token may satisfy several characters at once
            attempts += 1

    return words + extra, len(extra), deficient
