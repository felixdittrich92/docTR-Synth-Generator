# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Language table shared by the recipes: code -> docTR vocab, script family, presets."""

import unicodedata
from collections import Counter
from functools import cache

from ..components.vocabs import VOCABS

__all__ = [
    "LANGUAGES",
    "PRESETS",
    "resolve_languages",
    "script_of",
    "letter_scripts",
    "primary_script",
    "allowed_scripts",
    "punct_style",
    "PUNCT",
    "COMPLEX_SCRIPT_TAGS",
]

_LATIN = {
    "en": "english", "sq": "albanian", "af": "afrikaans", "az": "azerbaijani", "eu": "basque",
    "bs": "bosnian", "ca": "catalan", "hr": "croatian", "cs": "czech", "da": "danish",
    "nl": "dutch", "et": "estonian", "eo": "esperanto", "fr": "french", "fi": "finnish",
    "fy": "frisian", "gl": "galician", "de": "german", "ha": "hausa", "hu": "hungarian",
    "is": "icelandic", "id": "indonesian", "ga": "irish", "it": "italian", "lv": "latvian",
    "lt": "lithuanian", "lb": "luxembourgish", "mg": "malagasy", "ms": "malay", "mt": "maltese",
    "mi": "maori", "cnr": "montenegrin", "no": "norwegian", "pl": "polish", "pt": "portuguese",
    "qu": "quechua", "ro": "romanian", "gd": "scottish_gaelic", "sr-Latn": "serbian_latin",
    "sk": "slovak", "sl": "slovene", "so": "somali", "es": "spanish", "sw": "swahili",
    "sv": "swedish", "tl": "tagalog", "tr": "turkish", "uz": "uzbek_latin", "vi": "vietnamese",
    "cy": "welsh", "yo": "yoruba", "zu": "zulu",
}  # fmt: skip
_CYRILLIC = {
    "ru": "russian", "be": "belarusian", "uk": "ukrainian", "tt": "tatar", "tg": "tajik",
    "kk": "kazakh", "ky": "kyrgyz", "bg": "bulgarian", "mk": "macedonian", "mn": "mongolian",
    "sah": "yakut", "sr-Cyrl": "serbian_cyrillic", "uz-Cyrl": "uzbek_cyrillic",
}  # fmt: skip
_ARABIC = {
    "ar": "arabic",
    "fa": "persian",
    "ur": "urdu",
    "ps": "pashto",
    "ckb": "kurdish",
    "ug": "uyghur",
    "sd": "sindhi",
}
_INDIC = {
    "gu": "gujarati", "bn": "bengali", "ta": "tamil", "te": "telugu", "kn": "kannada",
    "si": "sinhala", "ml": "malayalam", "pa": "punjabi", "or": "odia",
}  # fmt: skip
_SEA = {"km": "khmer", "th": "thai", "lo": "lao", "my": "burmese", "jv": "javanese", "su": "sudanese"}

# code -> (docTR vocab key, script family). Every language vocab in vocabs.py is here
# except "sanskrit"/"devanagari" (same letters as "hindi") and the combined keys.
LANGUAGES: dict[str, tuple[str, str]] = {
    **{c: (k, "latin") for c, k in _LATIN.items()},
    **{c: (k, "cyrillic") for c, k in _CYRILLIC.items()},
    "el": ("greek", "greek"),
    "grc": ("greek_extended", "greek"),
    "he": ("hebrew", "hebrew"),
    **{c: (k, "arabic") for c, k in _ARABIC.items()},
    "hi": ("hindi", "devanagari"),
    "mr": ("marathi", "devanagari"),
    "ne": ("nepali", "devanagari"),
    **{c: (k, "indic") for c, k in _INDIC.items()},
    **{c: (k, "sea") for c, k in _SEA.items()},
    "hy": ("armenian", "hy_ka_am"),
    "ka": ("georgian", "hy_ka_am"),
    "am": ("ethiopic", "hy_ka_am"),
    "ja": ("japanese", "cjk"),
    "ko": ("korean", "cjk"),
    "zh": ("simplified_chinese", "cjk"),
}

# Named language sets. ``european`` is every Latin and Cyrillic vocab plus monotonic
# Greek - one model for the alphabets of Europe and the languages that share them.
# ``multilingual`` leaves out Chinese/Japanese/Korean: tens of thousands of classes
# need their own model and their own sampling strategy.
PRESETS: dict[str, list[str]] = {
    "european": [c for c, (_, g) in LANGUAGES.items() if g in ("latin", "cyrillic")] + ["el"],
    "multilingual": [c for c in LANGUAGES if c not in ("ja", "ko", "zh")],
    "all": list(LANGUAGES),
}


def resolve_languages(spec: str | list[str]) -> list[str]:
    """Turn a preset name or a comma separated list of codes into language codes.

    Raises:
        ValueError: for an unknown preset or code.
    """
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    out: list[str] = []
    for item in (i.strip() for i in items):
        if not item:
            continue
        if item in PRESETS:
            out.extend(PRESETS[item])
        elif item in LANGUAGES:
            out.append(item)
        else:
            raise ValueError(f"unknown language or preset {item!r}; presets: {sorted(PRESETS)}")
    return list(dict.fromkeys(out))


# -----------------------------------------------------------------------------
# Scripts
# -----------------------------------------------------------------------------


@cache
def script_of(c: str) -> str:
    """Unicode script bucket of a character (``"LATIN"``, ``"CJK"``, ``"DEVANAGARI"``...)."""
    name = unicodedata.name(c, "?")
    if name.startswith("CJK"):
        return "CJK"
    if name.startswith("MODIFIER LETTER"):  # Uzbek oʻ, Georgian ჼ: part of the host script
        return "GEORGIAN" if "GEORGIAN" in name else "LATIN"
    return name.split(" ")[0]


def letter_scripts(word: str) -> set[str]:
    """Scripts of the word's letters (digits/punctuation are script-neutral)."""
    return {script_of(c) for c in word if c.isalpha()}


@cache
def primary_script(lang: str) -> str:
    """The script most of the language's vocab letters belong to."""
    counts = Counter(script_of(c) for c in VOCABS[LANGUAGES[lang][0]] if c.isalpha())
    return counts.most_common(1)[0][0]


def allowed_scripts(lang: str) -> set[str]:
    """Scripts a word of this language may contain."""
    if lang == "ja":
        return {"CJK", "HIRAGANA", "KATAKANA", "KATAKANA-HIRAGANA"}
    return {primary_script(lang)}


# Script-appropriate punctuation: (trailing, leading, wrappers). Repetition = weight.
PUNCT: dict[str, tuple[str, str, list[str]]] = {
    "LATIN": (",,,,....;::!?)\"'", "(\"'", ["({})", '"{}"']),
    "CYRILLIC": (",,,,....;::!?)", "(", ["«{}»", "({})"]),
    "GREEK": (",,,,....;;·!)", "(", ["«{}»", "({})"]),
    "ARMENIAN": (",,,,։։։՝՞)", "(", ["«{}»"]),
    "HEBREW": (",,,,....:;?!", "(", ['"{}"', "({})"]),
    "ARABIC": ("،،،،...؛:؟!", "(", ["«{}»", "({})"]),
    "INDIC_DANDA": (",,,।।।।?!", "(", ["({})"]),
    "INDIC": (",,,,....?!", "(", ["({})"]),
    "KHMER": ("។។។៕?", "", []),
    "MYANMAR": ("။။။၊၊", "", []),
    "ETHIOPIC": ("።።።፣፣፤", "", []),
    "CJK": ("、、、。。。！？」』", "「『", ["「{}」", "（{}）"]),
    "JAVANESE": ("꧈꧈꧉", "", []),
    "NONE": ("", "", []),
}
_PUNCT_STYLE = {
    "DEVANAGARI": "INDIC_DANDA", "BENGALI": "INDIC_DANDA", "GURMUKHI": "INDIC_DANDA",
    "GUJARATI": "INDIC_DANDA", "ORIYA": "INDIC_DANDA", "TAMIL": "INDIC", "TELUGU": "INDIC",
    "KANNADA": "INDIC", "MALAYALAM": "INDIC", "SINHALA": "INDIC", "THAI": "NONE", "LAO": "NONE",
    "SUNDANESE": "LATIN", "GEORGIAN": "LATIN", "HANGUL": "LATIN", "HIRAGANA": "CJK",
}  # fmt: skip


def punct_style(lang: str) -> str:
    """Key into :data:`PUNCT` for a language."""
    if lang in ("ja", "zh"):
        return "CJK"
    s = primary_script(lang)
    return _PUNCT_STYLE.get(s, s if s in PUNCT else "LATIN")


# OpenType script tags a font must carry in GSUB before it may set a complex script.
# Without them HarfBuzz still produces glyphs - Arabic letters unjoined, Indic
# conjuncts as explicit halants - which no real document looks like.
COMPLEX_SCRIPT_TAGS: dict[str, set[str]] = {
    "ARABIC": {"arab"},
    "DEVANAGARI": {"deva", "dev2"},
    "BENGALI": {"beng", "bng2"},
    "GURMUKHI": {"guru", "gur2"},
    "GUJARATI": {"gujr", "gjr2"},
    "ORIYA": {"orya", "ory2"},
    "TAMIL": {"taml", "tml2"},
    "TELUGU": {"telu", "tel2"},
    "KANNADA": {"knda", "knd2"},
    "MALAYALAM": {"mlym", "mlm2"},
    "SINHALA": {"sinh"},
    "MYANMAR": {"mymr", "mym2"},
    "KHMER": {"khmr"},
    "JAVANESE": {"java"},
    "SUNDANESE": {"sund"},
}
