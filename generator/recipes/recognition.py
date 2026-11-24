# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Large-scale text-recognition dataset for docTR, with guaranteed vocab coverage.

    python -m generator.recipes.recognition prepare --languages european --out european_rec
    python -m generator.recipes.recognition render  --out european_rec      # resumable
    python -m generator.recipes.recognition verify  --out european_rec      # every vocab char rendered?
    python -m generator.recipes.recognition topup   --out european_rec      # render only what is missing
    python -m generator.recipes.recognition preview --out european_rec      # contact sheet per script

``--languages`` takes a preset (``european``, ``multilingual``, ``all``) or comma separated
codes (``de,fr,ru``). Output is what docTR's ``references/recognition/train.py`` expects:

    OUT/train/shard_0000/{images/, labels.json}   ->  --train_path OUT/train
    OUT/val/{images/, labels.json}                 ->  --val_path   OUT/val
    OUT/meta/vocab.txt                             ->  the model vocab
    OUT/meta/vocab_dropped.json                    ->  vocab characters left out, and why
    OUT/meta/stats.json                            ->  per-language numbers

How the labels are guaranteed to match the images:

* the model vocab is the union of the languages' docTR vocabs, restricted to what the
  vetted fonts can actually draw (characters no face inks are reported, not kept);
* every vocab character is placed in at least ``--min-char-count`` train labels (and
  ``--min-char-count-val`` val labels) - by real words where possible, else by
  script-valid synthetic tokens that HarfBuzz confirms render without a dotted circle;
* each crop is rendered with a face of its own language that HarfBuzz confirms draws
  it without tofu, dotted circles or empty glyphs; a crop no face draws is dropped;
* ``verify`` re-counts the characters of the labels that were actually written.

Why not ``generate_dataset(num_images=10_000_000)``? Word selection there repeats the
surplus with 1/rank weights (the top word of a 3k-word language gets thousands of
copies), val words reappear in train once words repeat, and there is no resume.
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from ..components.config import GenerationConfig
from ..components.generator import TextImageGenerator
from ..components.vocab_coverage import augment_words_for_coverage, is_large_script_char
from ..components.vocabs import VOCABS
from ..dataset_generator import SyntheticDatasetGenerator
from .corpus import SYNTHETIC_LANGUAGES, THAI_MOJIBAKE, Corpus
from .fonts import ManifestFontSelector, assign_fonts, collect_fonts, require_harfbuzz, scan_fonts, write_font_dir
from .languages import LANGUAGES, allowed_scripts, primary_script, resolve_languages, script_of
from .render import Progress, Worker, atomic_write_json, make_pool, render_part
from .text import (
    attach_punct,
    combining_filler,
    cycle_sample,
    digit_filler,
    is_ambiguous,
    mixes_digit_systems,
    native_digit_systems,
    native_number_tokens,
    numeric_tokens,
    random_case,
    symbol_token,
)

__all__ = ["prepare", "render", "verify", "topup", "main"]

# =============================================================================
# Defaults (all overridable on the command line)
# =============================================================================

NUM_TRAIN = 10_000_000
NUM_VAL = 100_000
SHARD_SIZE = 500_000  # images per train shard
CHUNK = 256  # crops per worker round-trip
SEED = 1234
WORDS_PER_LANG = 150_000
MIN_LEN, MAX_LEN = 1, 24
MIN_CHAR_COUNT_TRAIN = 1000
MIN_CHAR_COUNT_VAL = 10
# CJK ideographs and Hangul syllables: thousands per script, most of them rare in
# real text. Each still gets this many train labels (val: its own floor, if lower).
MIN_CHAR_COUNT_LARGE = 50
JPEG_QUALITY = 92

# Share of the dataset per script family; inside a family weight ~ pool_size ** T
# (T=1 proportional, T=0 equal). Without this the 52 Latin vocabs would swamp the
# 9 Indic ones. Families absent from a build are simply renormalised away.
GROUP_SHARE = {
    "latin": 0.34, "cyrillic": 0.12, "greek": 0.04, "hebrew": 0.04, "arabic": 0.12,
    "devanagari": 0.07, "indic": 0.13, "sea": 0.07, "hy_ka_am": 0.07, "cjk": 0.15,
}  # fmt: skip
TEMPERATURE = 0.7
# Fewer samples for languages whose words are synthetic or derived, more for English.
LANG_MULT = {"en": 2.5, "grc": 0.4, "sah": 0.5, "cnr": 0.5, "uz-Cyrl": 0.7,
             "qu": 0.4, "mi": 0.4, "jv": 0.4, "su": 0.4}  # fmt: skip
VAL_WORD_FRACTION = 0.03  # unique words held out per language: val shares no word with train
MIN_VAL_WORDS = 150
NUMERIC_RATIO = 0.12  # prices, dates, codes ... (share of all samples)
# Share of each language's samples drawn from its short words (<= 3 characters).
# Real crops are full of them - 'of', 'to', 'TO:', 'x', 'No' are ~18% of FUNSD and CORD
# words - but they are a few hundred of ~150k unique words, so even sampling over the
# vocabulary would almost never show one.
SHORT_SHARE = 0.3
SHORT_MAX_LEN = 3
# Short words are taken from the head of the frequency list and drawn Zipf-weighted:
# the short *tail* of a list is mostly junk ('s01', 'qin'), its head is 'of', 'to', 'a'.
SHORT_TOP_RANKS = 5000
# Machine-readable zones (passports, ID cards): 30 characters of A-Z, 0-9 and '<',
# read verbatim. Real word lists contain none of them.
MRZ_SHARE = 0.005
HANDWRITING_SHARE = 0.05
NATIVE_DIGIT_PROB = 0.4
# Case variants, drawn independently per word and added next to it: forms, receipts and
# headings are set in capitals far more than running text, so a pool of mostly lower-case
# corpus words gets ~15% ALL CAPS and ~25% Capitalised words (FUNSD: 28% / 31%, CORD:
# 30% / 20%; running text has fewer).
UPPER_PROB, TITLE_PROB = 0.25, 0.42
TITLE_PROB_OVERRIDE = {"de": 0.7, "lb": 0.7}  # nouns are capitalised; the lists are lower-cased
SHORT_UPPER_PROB, SHORT_TITLE_PROB = 0.12, 0.25  # 'of', 'и', 'la' are mostly lower-case
# Punctuation standing alone, as forms and tables have it (':' after a label, '-' for
# nil, '/' between fields). Glyphs that are ambiguous alone are left out: '|' (l, I, 1),
# a lone en/em dash (hyphen), '_' and the like (see text._NEVER_ALONE).
PUNCT_ALONE_SHARE = 0.02
_PUNCT_ALONE = set(":-/&%#*+=.,;!?•") | set("।。、،؟؛։።")
PUNCT_PROB = 0.12  # word crops from real pages carry attached punctuation
# Word-length tokens for scripts written without spaces: what a detector hands the recogniser.
JOIN_MAX = {"th": 3, "lo": 3, "km": 3, "my": 3, "ja": 3, "zh": 3}

# Mongolian *script* digits: Cyrillic Mongolian writes 0-9 and no Cyrillic face has them.
DROP_CHARS = "᠐᠑᠒᠓᠔᠕᠖᠗᠘᠙"


# =============================================================================
# prepare
# =============================================================================


def clean_words(
    lang: str, words: list[str], charset: set[str], limit: int, rng: random.Random
) -> tuple[list[str], Counter]:
    """Script, vocab, length and homoglyph filtering of one language's raw words."""
    allowed, script = allowed_scripts(lang), primary_script(lang)
    stats: Counter = Counter()
    out: dict[str, None] = {}
    for raw in words:
        w = unicodedata.normalize("NFC", raw.strip())
        if not (MIN_LEN <= len(w) <= MAX_LEN) or not any(c.isalpha() for c in w):
            stats["length_or_no_letters"] += 1
        elif not (sc := {script_of(c) for c in w if c.isalpha()}) or not sc <= allowed:
            stats["wrong_or_mixed_script"] += 1  # e.g. a Latin 'a' inside a Kazakh word
        elif not set(w) <= charset:
            stats["outside_vocab"] += 1  # e.g. Russian words in the Ukrainian list
        elif lang == "th" and THAI_MOJIBAKE.search(w):
            stats["mojibake"] += 1
        elif is_ambiguous(w, script):
            stats["ambiguous_homoglyph"] += 1  # 'се' is pixel-identical to Latin 'ce'
        else:
            out.setdefault(w, None)
            if len(out) >= limit:
                break
    kept = list(out)
    if lang in JOIN_MAX and kept:
        joined = [
            "".join(rng.choice(kept) for _ in range(rng.randint(2, JOIN_MAX[lang]))) for _ in range(len(kept) // 2)
        ]
        kept += [w for w in joined if len(w) <= MAX_LEN]
    stats["kept"] = len(kept)
    return kept, stats


def group_weights(pool_sizes: dict[str, int]) -> dict[str, float]:
    """Sampling weight per language: family share, then pool_size ** T inside a family."""
    groups: dict[str, list[str]] = defaultdict(list)
    for lang in pool_sizes:
        groups[LANGUAGES[lang][1]].append(lang)
    weights = {}
    for group, members in groups.items():
        raw = {m: (pool_sizes[m] ** TEMPERATURE) * LANG_MULT.get(m, 1.0) for m in members}
        total = sum(raw.values()) or 1.0
        for m in members:
            weights[m] = GROUP_SHARE[group] * raw[m] / total
    s = sum(weights.values())
    return {k: v / s for k, v in weights.items()}


def quotas(weights: dict[str, float], budget: int) -> dict[str, int]:
    """Integer split of ``budget`` by ``weights`` (largest weights take the remainder)."""
    q = {lang: int(budget * w) for lang, w in weights.items()}
    for lang in sorted(weights, key=weights.get, reverse=True)[: budget - sum(q.values())]:  # type: ignore[arg-type]
        q[lang] += 1
    return q


def _vocab_and_charsets(
    langs: list[str], selector: ManifestFontSelector, extra_chars: str
) -> tuple[str, dict[str, set[str]], dict[str, str]]:
    """Model vocab, per-language usable charsets, and the characters left out (and why)."""
    charsets: dict[str, set[str]] = {}
    dropped: dict[str, str] = {}
    for lang in langs:
        key = LANGUAGES[lang][0]
        usable = set()
        for c in VOCABS[key]:
            if c.isspace() or c in DROP_CHARS:
                continue
            if selector.candidates(c, lang):  # some face inks it (and can shape its script)
                usable.add(c)
        charsets[lang] = usable | set(extra_chars)
        missing = "".join(c for c in VOCABS[key] if not c.isspace() and c not in DROP_CHARS and c not in usable)
        if missing:
            dropped[key] = missing
    order = "".join(VOCABS[LANGUAGES[lang][0]] for lang in langs) + extra_chars
    vocab = "".join(c for c in dict.fromkeys(order) if any(c in cs for cs in charsets.values()))
    return vocab, charsets, dropped


def _fonts(args, langs: list[str], pools: dict[str, list[str]], meta: Path) -> dict[str, list[str]]:
    cache = Path(args.cache)
    paths = collect_fonts(Path(args.font_src), cache / "google_fonts", download_extra=not args.offline)
    print(f"Scanning {len(paths)} fonts (cached after the first run) ...")
    fonts = scan_fonts(paths, cache / "font_scan.json")
    if not any(LANGUAGES[lang][1] == "cjk" for lang in langs):
        # CJK faces cover Latin too, and a full CJK charset is ~20 MB per face in every
        # worker's support table. No language of this build needs them.
        fonts = {p: f for p, f in fonts.items() if sum(1 for c in f["chars"] if script_of(c) == "CJK") < 5000}
    freq = {lang: Counter(c for w in ws for c in w) for lang, ws in pools.items()}
    lang_fonts, impossible = assign_fonts(fonts, freq, langs)
    if args.max_fonts_per_lang:
        rng = random.Random(SEED)
        lang_fonts = {k: sorted(rng.sample(v, min(len(v), args.max_fonts_per_lang))) for k, v in lang_fonts.items()}
    by_lang = write_font_dir(fonts, lang_fonts, meta / "fonts")
    counts = sorted((len(v), k) for k, v in by_lang.items())
    print(f"  {len({f for v in by_lang.values() for f in v})} faces kept; fewest per language: {counts[:8]}")
    for lang, chars in impossible.items():
        if chars:
            print(f"  {lang}: no face draws {chars!r}")
    return by_lang


def prepare(args) -> None:
    """Corpora, fonts, vocab and the train/val label pools (written to ``OUT/meta``)."""
    require_harfbuzz()
    meta = Path(args.out, "meta")
    meta.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    langs = resolve_languages(args.languages)
    print(f"{len(langs)} languages: {' '.join(langs)}")

    print("Word pools ...")
    corpus = Corpus(Path(args.cache) / "corpora", full_lists=args.full_lists, offline=args.offline)
    pools: dict[str, list[str]] = {}
    clean_stats: dict[str, dict] = {}
    for lang in langs:
        own = set(VOCABS[LANGUAGES[lang][0]]) - set(DROP_CHARS)
        words, st = clean_words(lang, corpus.raw_words(lang, rng), own, args.words_per_lang, rng)
        if words:
            pools[lang], clean_stats[lang] = words, dict(st)
        else:
            print(f"  {lang:8s} no usable words - skipped")
    langs = [lang for lang in langs if lang in pools]

    print("Fonts ...")
    lang_fonts = _fonts(args, langs, pools, meta)
    selector = ManifestFontSelector(meta / "fonts", handwriting=False)
    langs = [lang for lang in langs if lang_fonts.get(lang)]
    vocab, charsets, dropped = _vocab_and_charsets(langs, selector, args.extra_chars)
    charset = set(vocab)
    print(
        f"  vocab: {len(vocab)} characters ({sum(map(len, dropped.values()))} no face draws - meta/vocab_dropped.json)"
    )

    print("Variants and train/val word split ...")
    train_words, val_words = {}, {}
    for lang in langs:
        cs = charsets[lang]
        words = [w for w in pools[lang] if set(w) <= cs and selector.candidates(w, lang)]
        if any(unicodedata.category(c).startswith("M") for c in cs):
            # Scripts with combining marks: a corpus word in the wrong logical order
            # shapes to a dotted circle in every face, so it never gets a slot.
            before = len(words)
            words = [w for w in words if selector.renders(w, lang, rng)]
            clean_stats[lang]["broken_cluster"] = before - len(words)
        title_p = TITLE_PROB_OVERRIDE.get(lang, TITLE_PROB)
        extra = [v for w in words if (v := random_case(w, lang, cs, rng, UPPER_PROB, 0.0)) != w]
        extra += [v for w in words if (v := random_case(w, lang, cs, rng, 0.0, title_p)) != w]
        words = list(dict.fromkeys(words + extra))
        extra = [v for w in words if rng.random() < PUNCT_PROB and (v := attach_punct(w, lang, cs, rng)) != w]
        words = list(dict.fromkeys(words + extra))
        rng.shuffle(words)
        n_val = min(max(MIN_VAL_WORDS, int(len(words) * VAL_WORD_FRACTION)), len(words) // 5)
        val_words[lang], train_words[lang] = words[:n_val], words[n_val:]
    # A held-out word must not come back through another language ("no" is en/es/it/...)
    # or through casing ("Mann" vs "mann").
    held_out = {w.casefold() for ws in val_words.values() for w in ws}
    for lang in train_words:
        train_words[lang] = [w for w in train_words[lang] if w.casefold() not in held_out]
    # Frequent short words per split, in frequency order (the pools keep rank order).
    short_words: dict[str, dict[str, list[str]]] = {"train": {}, "val": {}}
    for lang in langs:
        head = [w for w in pools[lang][:SHORT_TOP_RANKS] if len(w) <= SHORT_MAX_LEN]
        val_set = {w.casefold() for w in val_words[lang]}
        short_words["val"][lang] = [w for w in head if w.casefold() in val_set]
        short_words["train"][lang] = [w for w in head if w.casefold() not in held_out]

    weights = group_weights({k: len(v) for k, v in train_words.items() if v})
    n_num_train, n_num_val = round(args.num_train * NUMERIC_RATIO), round(args.num_val * NUMERIC_RATIO)
    q_train = quotas(weights, args.num_train - n_num_train)
    q_val = quotas(weights, args.num_val - n_num_val)

    def short_tokens(lang: str, ranked: list[str], n: int) -> list[str]:
        """``n`` short tokens: frequent short words (Zipf-weighted) and small numbers."""
        if n <= 0:
            return []
        cs = charsets[lang]
        n_num = n // 4 if ranked else n
        out = []
        if ranked:
            weights = [1.0 / (rank + 1) for rank in range(len(ranked))]
            title_p = SHORT_TITLE_PROB * TITLE_PROB_OVERRIDE.get(lang, TITLE_PROB) / TITLE_PROB
            for w in rng.choices(ranked, weights=weights, k=n - n_num):
                w = random_case(w, lang, cs, rng, SHORT_UPPER_PROB, title_p)
                out.append(attach_punct(w, lang, cs, rng) if rng.random() < PUNCT_PROB * 2 else w)
        # Quantities, item and page numbers: '1', '12', '3.', '(2)'.
        systems = native_digit_systems(lang, cs)
        for _ in range(n_num):
            t = str(rng.choice([rng.randint(0, 9), rng.randint(10, 99)])) + rng.choice(["", "", "", ".", ")", "x"])
            if systems and rng.random() < NATIVE_DIGIT_PROB:
                t = t.translate({ord(k): v for k, v in rng.choice(systems).items()})
            out.append(t if set(t) <= cs else t.rstrip(".)x"))
        return [t for t in out if t and set(t) <= cs]

    def build(
        words_by_lang: dict[str, list[str]], q: dict[str, int], n_num: int, floor: int, split: str
    ) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        n_total = sum(q.values()) + n_num
        out += [("*", t) for t in mrz_tokens(round(n_total * MRZ_SHARE), rng) if set(t) <= charset]
        for lang, ws in words_by_lang.items():
            n = q.get(lang, 0)
            n_short = round(n * SHORT_SHARE)
            alone = sorted(_PUNCT_ALONE & charsets[lang])
            n_alone = round(n * PUNCT_ALONE_SHARE) if alone else 0
            out += [(lang, t) for t in short_tokens(lang, short_words[split].get(lang, []), n_short)]
            out += [(lang, rng.choice(alone)) for _ in range(n_alone)]
            out += [(lang, w) for w in cycle_sample(ws, n - n_short - n_alone, rng)]
        num_w = {lang: weights[lang] for lang in words_by_lang}
        for lang, n in quotas(num_w, n_num).items():
            out += [(lang, t) for t in numeric_tokens(lang, n, charsets[lang], rng, NATIVE_DIGIT_PROB)]
            # A guaranteed block, so a script's own digits never depend on the coverage
            # helper (which pads digits with ASCII ones). ~3 tokens per required
            # occurrence: a number is ~3 digits long and the ten digits share the slots.
            for native in native_digit_systems(lang, charsets[lang]):
                out += [(lang, t) for t in native_number_tokens(native, max(n // 4, 3 * floor), rng)]
        return out

    def accept(token: str) -> bool:
        return selector.renders(token, rng=rng)

    def symbols_in_context(pool: list[tuple[str, str]], words_by_lang: dict[str, list[str]], floor: int):
        """Tokens that carry under-represented punctuation and symbols the way documents do."""
        counts = Counter(c for _, t in pool for c in t)
        out: list[tuple[str, str]] = []
        for c in vocab:
            if unicodedata.category(c)[0] not in "PS" or counts[c] >= floor:
                continue
            hosts = [lang for lang, ws in words_by_lang.items() if ws and c in charsets[lang]]
            need, tries = floor - counts[c], 0
            while hosts and need > 0 and tries < floor * 6:
                tries += 1
                lang = rng.choice(hosts)
                token = symbol_token(c, rng.choice(words_by_lang[lang]), rng)
                if (
                    token
                    and len(token) <= MAX_LEN
                    and set(token) <= charsets[lang]
                    and selector.candidates(token, lang)
                ):
                    out.append((lang, token))
                    need -= token.count(c)
        return out

    def complete(pool: list[tuple[str, str]], floor: int, seed: int) -> tuple[list[tuple[str, str]], int]:
        """Top every vocab character up to ``floor`` occurrences, with renderable tokens only."""
        texts = [t for _, t in pool]
        texts, n_added, _ = augment_words_for_coverage(
            texts, charset, min_count=floor, seed=seed, accept=accept, large_min_count=_large_floor(args)
        )
        pool = pool + [("*", t) for t in texts[len(pool) :]]
        counts = Counter(c for _, t in pool for c in t)
        # Marks the coverage helper cannot synthesise (Latin tone marks) and digits of
        # systems no language block produced (e.g. the Shan digits next to the Myanmar ones).
        short = [c for c in vocab if unicodedata.combining(c) and counts[c] < floor]
        thin = [c for c in vocab if unicodedata.category(c) == "Nd" and not c.isascii() and counts[c] < floor]
        fill = combining_filler(charset, short, rng, max(1, floor // 2)) + digit_filler(charset, thin, rng, floor)
        pool += [("*", t) for t in fill if accept(t)]
        pool = [(lang, t) for lang, t in pool if t and set(t) <= charset and not mixes_digit_systems(t)]
        return pool, n_added

    print("Sampling pools and completing vocab coverage ...")
    train = build(train_words, q_train, n_num_train, args.min_char_count, "train")
    train += symbols_in_context(train, train_words, args.min_char_count)
    train, n_tr = complete(train, args.min_char_count, SEED)
    train_texts = {t for _, t in train}
    val = [
        (lang, t)
        for lang, t in build(val_words, q_val, n_num_val, args.min_char_count_val, "val")
        if t not in train_texts
    ]
    val += symbols_in_context(val, val_words, args.min_char_count_val)
    val, n_va = complete(val, args.min_char_count_val, SEED + 1)
    rng.shuffle(train)
    rng.shuffle(val)

    counts = Counter(c for _, t in train for c in t)
    missing = [c for c in vocab if counts[c] == 0]
    stats = {
        "languages_arg": args.languages,
        "num_train": len(train),
        "num_val": len(val),
        "vocab_size": len(vocab),
        "coverage_tokens_added": {"train": n_tr, "val": n_va},
        "unique_train_texts": len(train_texts),
        "train_val_exact_overlap": len({t for _, t in val} & train_texts),
        "vocab_chars_missing_in_train": "".join(missing),
        "rarest_chars_train": [(c, counts[c]) for c in sorted(vocab, key=lambda c: counts[c])[:15]],
        "languages": {
            lang: {
                "vocab": LANGUAGES[lang][0],
                "group": LANGUAGES[lang][1],
                "script": primary_script(lang),
                "fonts": len(lang_fonts[lang]),
                "unique_train_words": len(train_words[lang]),
                "unique_val_words": len(val_words[lang]),
                "train_samples": q_train.get(lang, 0),
                "repeats_per_word": round(q_train.get(lang, 0) / max(1, len(train_words[lang])), 2),
                "synthetic_words": lang in SYNTHETIC_LANGUAGES,
                "cleaning": clean_stats[lang],
            }
            for lang in langs
        },
    }
    (meta / "vocab.txt").write_text(vocab, encoding="utf-8")
    dropped_info = {"mongolian_script_digits": DROP_CHARS, "no_font_draws_them": dropped}
    atomic_write_json(meta / "vocab_dropped.json", dropped_info, indent=1)
    _write_tsv(meta / "train_labels.tsv", train)
    _write_tsv(meta / "val_labels.tsv", val)
    atomic_write_json(meta / "stats.json", stats, indent=1)
    print(
        f"coverage top-up: {n_tr:,} train labels ({n_tr / max(1, len(train)):.0%}) - mostly real words repeated for "
        "their rare characters; the share falls as --num-train grows relative to --min-char-count"
    )
    print(
        f"train {len(train):,} / val {len(val):,} labels | vocab {len(vocab)} chars | "
        f"train/val overlap {stats['train_val_exact_overlap']} | missing chars: {''.join(missing) or 'none'}"
    )


def mrz_tokens(n: int, rng: random.Random) -> list[str]:
    """Lines of machine-readable zones (ICAO 9303 TD1 / TD3 style), at most 30 characters."""
    up = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits = "0123456789"
    out = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.35:  # document line: type, issuing state, number, check digit, filler
            line = rng.choice(["ID", "IP", "AC", "P<", "I<"]) + "".join(rng.choice(up) for _ in range(3))
            line += "".join(rng.choice(up + digits) for _ in range(9)) + rng.choice(digits)
        elif kind < 0.7:  # dates, sex, expiry, nationality
            line = "".join(rng.choice(digits) for _ in range(7)) + rng.choice("MF<")
            line += "".join(rng.choice(digits) for _ in range(7)) + "".join(rng.choice(up) for _ in range(3))
        else:  # names
            line = "".join(rng.choice(up) for _ in range(rng.randint(4, 10))) + "<<"
            line += "<".join(
                "".join(rng.choice(up) for _ in range(rng.randint(3, 8))) for _ in range(rng.randint(1, 2))
            )
        out.append((line + "<" * 30)[:30])
    return out


def _write_tsv(path: Path, rows: list[tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for lang, text in rows:
            if "\t" not in text and "\n" not in text:
                fh.write(f"{lang}\t{text}\n")


def _read_tsv(path: Path) -> list[tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            lang, _, text = line.rstrip("\n").partition("\t")
            if text:
                rows.append((lang, text))
    return rows


# =============================================================================
# render
# =============================================================================


class CropWorker(Worker):
    """Renders one recognition crop with a face of the sample's language.

    ``handwriting_share`` of the crops are written in a handwriting face (filled-in
    forms, passports, notes on a waybill) when one covers and cleanly shapes the
    text; otherwise they stay printed.
    """

    def __init__(self, cfg: GenerationConfig, font_dir: str, handwriting_share: float = 0.0):
        self.selector = ManifestFontSelector(
            font_dir, handwriting=handwriting_share > 0, cache_dir=cfg.resources.font_cache_dir
        )
        self.gen = TextImageGenerator(cfg, font_selector=self.selector)
        self.handwriting_share = handwriting_share
        self.save = ("JPEG", {"quality": cfg.core.output_jpeg_quality}) if cfg.core.output_jpeg else ("PNG", {})

    def render(self, payload, path):
        """Render ``(lang, text)``; the label record is the text itself."""
        lang, text = payload
        self.selector.language = None if lang == "*" else lang
        hand = self.selector.get_handwriting_font(text) if random.random() < self.handwriting_share else None
        img = self.gen.generate_image(text, font_path=hand)
        if img is None:
            return None
        img.save(path, self.save[0], **self.save[1])
        return text


def _handwriting_share(args) -> float:
    """Handwriting faces are downloaded on first use, so ``--offline`` turns them off."""
    return 0.0 if getattr(args, "offline", False) else getattr(args, "handwriting_share", HANDWRITING_SHARE)


def _config(args) -> GenerationConfig:
    return GenerationConfig.flat(
        task="recognition",
        output_dir=str(args.out),
        num_workers=args.workers,
        bg_image_dir=args.bg_dir,
        output_jpeg=not args.png,
        output_jpeg_quality=JPEG_QUALITY,
    )


def _finish_part(part: Path, final: Path, labels: dict[str, str]) -> None:
    atomic_write_json(part / "labels.json", dict(sorted(labels.items())))
    (part / "progress.jsonl").unlink(missing_ok=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(part, final)  # atomic: docTR never sees a half-written shard


def render(args) -> None:
    """Render val and every unfinished train shard (resumable at chunk granularity)."""
    require_harfbuzz()
    out, meta = Path(args.out), Path(args.out, "meta")
    if not (meta / "train_labels.tsv").exists():
        sys.exit("Run `prepare` first.")
    train, val = _read_tsv(meta / "train_labels.tsv"), _read_tsv(meta / "val_labels.tsv")
    cfg = _config(args)
    SyntheticDatasetGenerator(cfg).resolve_backgrounds()  # download once, before forking
    ext = "png" if args.png else "jpg"

    jobs = []
    if not (out / "val" / "labels.json").exists():
        jobs.append((val, out / "val", out / "_partial" / "val"))
    for s, start in enumerate(range(0, len(train), args.shard_size)):
        final = out / "train" / f"shard_{s:04d}"
        if not (final / "labels.json").exists():
            jobs.append((train[start : start + args.shard_size], final, out / "_partial" / final.name))
    if not jobs:
        print("Nothing to do - all shards are complete.")
        return
    progress = Progress(sum(len(j[0]) for j in jobs))
    print(f"Rendering {progress.total:,} crops in {len(jobs)} part(s) with {args.workers} workers ...")
    with make_pool(args.workers, CropWorker, (cfg, str(meta / "fonts"), _handwriting_share(args))) as pool:
        for rows, final, part in jobs:
            width = max(6, len(str(len(rows))))
            items = [(f"{i:0{width}d}.{ext}", row) for i, row in enumerate(rows)]
            labels = render_part(pool, items, part, CHUNK, progress, final.name)
            _finish_part(part, final, labels)
            print(f"\n  -> {final}: {len(labels):,} images, {len(rows) - len(labels)} dropped", flush=True)
    shutil.rmtree(out / "_partial", ignore_errors=True)
    print(f"Done in {(time.time() - progress.t0) / 3600:.2f} h.")


# =============================================================================
# verify / topup
# =============================================================================


def _dataset_labels(out: Path) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {"train": [], "val": []}
    for split, folders in (("val", [out / "val"]), ("train", sorted((out / "train").glob("shard_*")))):
        for folder in folders:
            f = folder / "labels.json"
            if f.exists():
                labels[split].extend(json.loads(f.read_text(encoding="utf-8")).values())
    return labels


def _coverage(out: Path) -> tuple[str, dict[str, Counter], dict[str, list[str]], list[str]]:
    vocab = (out / "meta" / "vocab.txt").read_text(encoding="utf-8")
    labels = _dataset_labels(out)
    counts = {split: Counter(c for w in ws for c in w) for split, ws in labels.items()}
    missing = {split: [c for c in vocab if counts[split][c] == 0] for split in counts}
    unexpected = sorted({c for cnt in counts.values() for c in cnt} - set(vocab))
    return vocab, counts, missing, unexpected


def _large_floor(args) -> int:
    return getattr(args, "min_char_count_large", MIN_CHAR_COUNT_LARGE)


def _floor_for(char: str, floor: int, large: int) -> int:
    """The floor a character is held to: CJK ideographs and Hangul have their own."""
    return min(floor, large) if is_large_script_char(char) else floor


def verify(args) -> bool:
    """Every vocab character present in the *rendered* labels of both splits?"""
    out = Path(args.out)
    vocab, counts, missing, unexpected = _coverage(out)
    n_train = sum(counts["train"].values())
    print(f"vocab {len(vocab)} chars | train: {n_train:,} chars, {len(counts['train'])} distinct | "
          f"val: {len(counts['val'])} distinct")  # fmt: skip
    ok = True
    for split in ("train", "val"):
        if missing[split]:
            ok = False
            print(f"\n{split}: {len(missing[split])} vocab characters never appear: {''.join(missing[split])}")
    if unexpected:
        ok = False
        print(f"\nlabels contain {len(unexpected)} characters NOT in vocab.txt: {''.join(unexpected)}")
    large = _large_floor(args)
    for split, floor in (("train", args.min_char_count), ("val", args.min_char_count_val)):
        thin = [(c, counts[split][c]) for c in vocab if 0 < counts[split][c] < _floor_for(c, floor, large)]
        if thin:
            print(f"\n{split}: {len(thin)} characters below the floor of {floor}: {thin[:20]}")
    if ok:
        print("\nOK - every vocab character appears in the rendered labels of both splits.")
    else:
        print("\nRun `topup` to render samples for exactly the missing characters.")
    return ok


def topup(args) -> None:
    """Render extra samples for characters the finished dataset never produced."""
    require_harfbuzz()
    out, meta = Path(args.out), Path(args.out, "meta")
    vocab, counts, missing, _ = _coverage(out)
    if not any(missing.values()):
        print("Nothing missing - the dataset covers the whole vocab.")
        return
    charset = set(vocab)
    rng = random.Random(SEED + 7)
    selector = ManifestFontSelector(meta / "fonts", handwriting=False)
    cfg = _config(args)
    SyntheticDatasetGenerator(cfg).resolve_backgrounds()
    ext = "png" if args.png else "jpg"
    with make_pool(args.workers, CropWorker, (cfg, str(meta / "fonts"), _handwriting_share(args))) as pool:
        for split in ("train", "val"):
            if not missing[split]:
                continue
            target = set(missing[split])
            floor = args.min_char_count if split == "train" else args.min_char_count_val
            seed_words = [w for w in _dataset_labels(out)[split][:200_000] if set(w) & target]
            words, _, _ = augment_words_for_coverage(
                seed_words,
                target,
                min_count=floor,
                seed=SEED + 7,
                accept=lambda t: selector.renders(t, rng=rng),
                large_min_count=_large_floor(args),
            )
            words = words[len(seed_words) :]
            words += combining_filler(charset, [c for c in target if unicodedata.combining(c)], rng, floor)
            words += digit_filler(charset, [c for c in target if unicodedata.category(c) == "Nd"], rng, floor)
            words = [w for w in words if w and set(w) <= charset and selector.renders(w, rng=rng)]
            if not words:
                print(f"{split}: no renderable token for {''.join(sorted(target))!r} - add a language that uses them")
                continue
            print(f"{split}: {len(target)} missing characters -> {len(words):,} new samples")
            items = [(f"topup_{i:06d}.{ext}", ("*", w)) for i, w in enumerate(words)]
            if split == "train":
                final = out / "train" / f"shard_{len(list((out / 'train').glob('shard_*'))):04d}_topup"
                labels = render_part(pool, items, out / "_partial" / final.name, CHUNK)
                _finish_part(out / "_partial" / final.name, final, labels)
            else:
                part = out / "_partial" / "val_topup"
                labels = render_part(pool, items, part, CHUNK)
                for name in labels:
                    os.replace(part / "images" / name, out / "val" / "images" / name)
                merged = json.loads((out / "val" / "labels.json").read_text(encoding="utf-8")) | labels
                atomic_write_json(out / "val" / "labels.json", dict(sorted(merged.items())))
    shutil.rmtree(out / "_partial", ignore_errors=True)


# =============================================================================
# CLI
# =============================================================================


def main(argv: list[str] | None = None) -> None:
    """Command line entry point (``python -m generator.recipes.recognition``)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["prepare", "render", "all", "verify", "topup", "preview"])
    p.add_argument("--out", required=True, help="dataset directory")
    p.add_argument("--languages", default="european", help="preset (european, multilingual, all) or codes: de,fr,ru")
    p.add_argument("--num-train", type=int, default=NUM_TRAIN)
    p.add_argument("--num-val", type=int, default=NUM_VAL)
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--words-per-lang", type=int, default=WORDS_PER_LANG)
    p.add_argument("--min-char-count", type=int, default=MIN_CHAR_COUNT_TRAIN, help="per-character floor in train")
    p.add_argument("--min-char-count-val", type=int, default=MIN_CHAR_COUNT_VAL, help="per-character floor in val")
    p.add_argument(
        "--min-char-count-large",
        type=int,
        default=MIN_CHAR_COUNT_LARGE,
        help="per-character floor for CJK ideographs and Hangul syllables (thousands each; most are rare)",
    )
    p.add_argument("--extra-chars", default="", help="characters added to the vocab on top of the docTR vocabs")
    p.add_argument("--font-src", default="fonts_v1", help="unzipped fonts_v1 release (downloaded if missing)")
    p.add_argument("--cache", default=".recipe_cache", help="corpora, extra fonts and the font scan")
    p.add_argument("--bg-dir", default=None, help="background images (default: the curated download)")
    p.add_argument("--max-fonts-per-lang", type=int, default=0, help="cap faces per language (0 = all)")
    p.add_argument("--no-full-lists", dest="full_lists", action="store_false", help="use the 50k word lists")
    p.add_argument("--offline", action="store_true", help="never download; use cached corpora and fonts only")
    p.add_argument("--png", action="store_true", help="PNG instead of JPEG")
    p.add_argument(
        "--handwriting-share",
        type=float,
        default=HANDWRITING_SHARE,
        help="share of crops written in a handwriting face (downloaded once and cached; printed if none fits; "
        "off with --offline)",
    )
    p.add_argument("--samples", type=int, default=6, help="preview: crops per script")
    args = p.parse_args(argv)

    if args.step in ("prepare", "all"):
        prepare(args)
    if args.step in ("render", "all"):
        render(args)
    if args.step in ("verify", "all"):
        if not verify(args):
            sys.exit(1)
    if args.step == "topup":
        topup(args)
        if not verify(args):
            sys.exit(1)
    if args.step == "preview":
        from .preview import recognition_sheet

        print(recognition_sheet(Path(args.out), args.samples, bg_dir=args.bg_dir))


if __name__ == "__main__":
    main()
