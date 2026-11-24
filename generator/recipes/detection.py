# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Large-scale multilingual text-detection dataset for docTR.

    python -m generator.recipes.detection prepare --languages all --out multilingual_det
    python -m generator.recipes.detection render  --out multilingual_det     # resumable
    python -m generator.recipes.detection preview --out multilingual_det     # pages + boxes

Output is a plain docTR detection dataset, loadable with ``DetectionDataset`` as is:

    OUT/train/images/*.jpg   OUT/train/labels.json
    OUT/val/images/*.jpg     OUT/val/labels.json
    OUT/meta/                pools, fonts, languages.json

    references/detection/train.py ... --train_path OUT/train --val_path OUT/val

Every page is set in faces of its own language (vetted like the recognition recipe:
inked glyphs, shaping rules for complex scripts, HarfBuzz-confirmed per word), so a
Polish page never falls back to a random face for every word with 'ł'. Rendering
happens in resumable parts under ``OUT/_parts``; the last step moves the images into
one folder and streams the parts into a single ``labels.json``.

Page density drives both realism and size: docTR resizes every page to
``--input_size`` (1024 by default), so a page carrying 1,200 words ends up with boxes
a couple of pixels tall. ``--density`` picks the typography preset.
"""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..components.config import GenerationConfig
from ..components.page_generator import PageGenerator
from ..components.vocab_coverage import augment_words_for_coverage
from ..components.vocabs import VOCABS
from ..dataset_generator import SyntheticDatasetGenerator
from .corpus import THAI_MOJIBAKE, Corpus
from .fonts import ManifestFontSelector, assign_fonts, collect_fonts, require_harfbuzz, scan_fonts, write_font_dir
from .languages import LANGUAGES, allowed_scripts, letter_scripts, resolve_languages
from .render import Progress, Worker, make_pool, render_part
from .text import attach_punct, numeric_tokens, random_case, symbol_token

__all__ = ["prepare", "render", "main"]

NUM_TRAIN = 25_000
NUM_VAL = 2_000
PART_SIZE = 5_000  # pages per render part (merged at the end)
CHUNK = 2  # pages per worker round-trip (pages take seconds each)
SEED = 4321
JPEG_QUALITY = 85
# docTR's detection trainer resizes every page to --input_size (1024), so pixels beyond
# ~1600 on the long edge only cost disk.
DELIVERY_LONG_EDGE = (1024.0, 1600.0)
MAX_LONG_EDGE = 1600
# docTR trains detection at --input_size 1024 (long edge). Body text on a real A4 scan
# is ~10-12 px tall there and footnotes ~6 px; below ~5 px print is no longer legible.
# A page is re-rolled when more than MAX_TINY_SHARE of its words fall below MIN_TEXT_PX,
# or when it carries more than MAX_BOXES_PER_PAGE words (denser than real forms, letters
# and reports - FUNSD's densest pages hold ~300 - and a wall of fine print, which
# also costs ~45 bytes per box in labels.json).
TRAIN_INPUT_SIZE = 1024
MIN_TEXT_PX = 5.0
MAX_TINY_SHARE = 0.03
MAX_BOXES_PER_PAGE = 800
PAGE_ATTEMPTS = 4
# Pages with no labelled text at all - blank paper, empty forms, charts, photos, the
# show-through of a page printed on the back. Hard negatives: they teach a detector
# what is *not* text, the main source of false positives on real documents.
NEGATIVE_SHARE = 0.05
# Typography presets: (body points, fine-print prob, newspaper px, max blocks, table rows,
# newspaper columns). docTR's DetectionDataset holds every box in RAM, so this is the main
# lever on labels.json size as well as on legibility after the resize to 1024 px.
PAGE_DENSITY = {
    "low": ((15.0, 21.0), 0.0, (18, 26), 10, (4, 9), (2, 3)),
    "medium": ((12.0, 16.0), 0.05, (14, 20), 18, (5, 14), (2, 4)),
    "high": ((8.5, 12.5), 0.22, (9, 15), 48, (6, 22), (3, 6)),  # the generator's own defaults
}
# Per-preset limits: (max boxes, delivered long edge, min text px at the training input
# size, max tiny share). Dense pages - a broadsheet carries 2,000+ words - keep many boxes
# and a larger delivered image, but their words must stay learnable at --input-size: a
# fine-tune on 4 px words (at 1024) learned that specks of fabric and grain are text and
# lost a quarter of its precision on real receipt photos (CORD 41% -> 32%). Training at
# docTR --input_size 1536/2048? Pass the same --input-size and the pages get denser.
DENSITY_LIMITS = {
    "low": (800, 1600, 5.0, 0.03),
    "medium": (1500, 1600, 5.5, 0.03),
    "high": (2500, 2048, 6.0, 0.03),
}
# --density mixed (the default) draws the preset per page. Real collections are mostly
# letters, forms and reports; broadsheets are the minority.
DENSITY_MIX = {"low": 0.55, "medium": 0.3, "high": 0.15}

WORDS_PER_LANG = 30_000
CANDIDATES_PER_PAGE = 600  # the page recycles these, so this bounds per-page vocabulary
CASE_VARIANT_PROB = 0.25
PUNCT_VARIANT_PROB = 0.18
NUMERIC_SHARE = 0.08  # share of page candidates that are numbers/dates/prices
MIX_PROB, MIX_SHARE = 0.3, 0.15  # non-Latin pages: 30% carry ~15% English tokens
COVERAGE_MIN = 3  # every drawable vocab char appears >= N times in its language pool
NO_SPACE = {"th": (2, 4), "lo": (2, 4), "km": (2, 4), "my": (2, 4), "ja": (2, 5), "zh": (2, 5)}

# Page share per writing system; split inside the family by LANG_MULT (default 1).
GROUP_SHARE = {
    "latin": 0.40, "cyrillic": 0.09, "greek": 0.025, "hebrew": 0.025, "arabic": 0.08,
    "devanagari": 0.04, "indic": 0.09, "sea": 0.05, "hy_ka_am": 0.03, "cjk": 0.17,
}  # fmt: skip
LANG_MULT = {"en": 3.0, "zh": 1.2, "ja": 1.2, "ko": 0.8, "grc": 0.4, "sah": 0.5, "cnr": 0.5,
             "uz-Cyrl": 0.7, "qu": 0.5, "mi": 0.5, "jv": 0.5, "su": 0.5, "ckb": 0.7}  # fmt: skip

# Layout mix, roughly as documents occur in the wild: mostly running text, forms and
# tables; ID cards and receipts are a small share. Vertical typesetting is normal for
# Chinese and Japanese, rare elsewhere.
LAYOUT_DEFAULT = {"paragraph": 0.34, "newspaper": 0.12, "form": 0.18, "table": 0.16,
                  "receipt": 0.08, "id_card": 0.07, "vertical": 0.03}  # fmt: skip
LAYOUT_CJK_VERTICAL = {"paragraph": 0.24, "newspaper": 0.18, "form": 0.12, "table": 0.11,
                       "receipt": 0.06, "id_card": 0.04, "vertical": 0.25}  # fmt: skip
CJK_VERTICAL = {"ja", "zh"}
# Dense typesetting belongs to newspapers, long reports and tables - a dense receipt or
# ID card does not exist.
LAYOUT_DENSE = {"newspaper": 0.5, "paragraph": 0.3, "table": 0.2}


# =============================================================================
# prepare
# =============================================================================


def _is_gb2312(w: str) -> bool:
    try:
        w.encode("gb2312")
        return True
    except UnicodeEncodeError:
        return False


def clean_pool(lang: str, words: list[str], rng: random.Random) -> tuple[list[str], Counter]:
    """Length, script and mojibake filtering; scripts without spaces get clause-length tokens."""
    allowed = allowed_scripts(lang)
    stats: Counter = Counter()
    out: dict[str, None] = {}
    for w in words:
        if not (1 <= len(w) <= 24) or not any(c.isalpha() for c in w):
            stats["length_or_no_letters"] += 1
        elif not (sc := letter_scripts(w)) or not sc <= allowed:
            stats["wrong_or_mixed_script"] += 1
        elif lang == "th" and THAI_MOJIBAKE.search(w):
            stats["mojibake"] += 1
        elif lang == "zh" and not _is_gb2312(w):
            stats["traditional_or_rare_hanzi"] += 1  # the zh_cn list mixes in ~9% traditional text
        else:
            out.setdefault(w, None)
            if len(out) >= WORDS_PER_LANG:
                break
    kept = list(out)
    if lang in NO_SPACE and kept:  # real Thai/CJK text has no inter-word spaces
        lo, hi = NO_SPACE[lang]
        kept = ["".join(rng.choice(kept) for _ in range(rng.randint(lo, hi))) for _ in range(min(len(kept), 20_000))]
    stats["kept"] = len(kept)
    return kept, stats


def enrich(lang: str, words: list[str], charset: set[str], rng: random.Random) -> list[str]:
    """Add Title/UPPER and attached-punctuation variants."""
    extra = [
        v
        for w in words
        if rng.random() < CASE_VARIANT_PROB and (v := random_case(w, lang, charset, rng, 0.2, 0.8)) != w
    ]
    words = words + extra
    extra = [v for w in words if rng.random() < PUNCT_VARIANT_PROB and (v := attach_punct(w, lang, charset, rng)) != w]
    return list(dict.fromkeys(words + extra))


def page_weights(langs: list[str]) -> dict[str, float]:
    """Share of pages per language."""
    groups: dict[str, list[str]] = defaultdict(list)
    for lang in langs:
        groups[LANGUAGES[lang][1]].append(lang)
    w = {}
    for g, members in groups.items():
        tot = sum(LANG_MULT.get(m, 1.0) for m in members)
        for m in members:
            w[m] = GROUP_SHARE[g] * LANG_MULT.get(m, 1.0) / tot
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def prepare(args) -> None:
    """Word pools and vetted fonts per language (written to ``OUT/meta``)."""
    require_harfbuzz()
    meta = Path(args.out, "meta")
    (meta / "pools").mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    langs = resolve_languages(args.languages)
    print(f"{len(langs)} languages: {' '.join(langs)}")

    print("Word pools ...")
    corpus = Corpus(Path(args.cache) / "corpora", full_lists=False, offline=args.offline)
    pools, stats = {}, {}
    for lang in langs:
        words, st = clean_pool(lang, corpus.raw_words(lang, rng), rng)
        if words:
            pools[lang], stats[lang] = words, dict(st)
        else:
            print(f"  {lang:8s} no usable words - skipped")
    langs = [lang for lang in langs if lang in pools]

    print("Fonts ...")
    cache = Path(args.cache)
    paths = collect_fonts(Path(args.font_src), cache / "google_fonts", download_extra=not args.offline)
    fonts = scan_fonts(paths, cache / "font_scan.json")
    freq = {lang: Counter(c for w in ws for c in w) for lang, ws in pools.items()}
    lang_fonts, impossible = assign_fonts(fonts, freq, langs)
    by_lang = write_font_dir(fonts, lang_fonts, meta / "fonts")
    selector = ManifestFontSelector(meta / "fonts", handwriting=False)
    print(f"  {len({f for v in by_lang.values() for f in v})} faces kept; fewest per language: "
          f"{sorted((len(v), k) for k, v in by_lang.items())[:8]}")  # fmt: skip

    print("Coverage, variants, renderability ...")
    summary = {}
    for lang in langs:
        if not by_lang.get(lang):
            print(f"  {lang:8s} no face qualifies - skipped")
            continue
        key = LANGUAGES[lang][0]
        target = {c for c in VOCABS[key] if not c.isspace() and selector.candidates(c, lang)}
        words = [w for w in pools[lang] if selector.candidates(w, lang)]
        counts = Counter(c for w in words for c in w)
        for c in sorted(target):  # punctuation and symbols in the contexts documents use
            if unicodedata.category(c)[0] in "PS" and counts[c] < COVERAGE_MIN and words:
                tokens = (symbol_token(c, rng.choice(words), rng) for _ in range(COVERAGE_MIN * 4))
                words += [t for t in tokens if set(t) <= target and selector.candidates(t, lang)][:COVERAGE_MIN]
        words, n_cov, _ = augment_words_for_coverage(
            words, target, min_count=COVERAGE_MIN, seed=SEED, accept=lambda t: selector.renders(t, rng=rng)
        )
        present = {c for w in words for c in w}
        missing = sorted(c for c in target if c not in present and c.isalpha())
        if missing:  # huge scripts (CJK, Hangul) are skipped by the coverage helper
            rng.shuffle(missing)
            words += ["".join(missing[i : i + 4]) for i in range(0, len(missing), 4)]
            n_cov += math.ceil(len(missing) / 4)
        words = enrich(lang, words, target, rng)
        before = len(words)
        vet_shaping = any(unicodedata.category(c).startswith("M") for c in target)
        words = [w for w in words if (selector.renders(w, lang, rng) if vet_shaping else selector.candidates(w, lang))]
        nums = numeric_tokens(lang, 1500, target | set("0123456789"), rng, 0.35, allow_spaces=True)
        nums = [t for t in nums if selector.candidates(t, lang)]
        (meta / "pools" / f"{lang}.txt").write_text("\n".join(words), encoding="utf-8")
        (meta / "pools" / f"{lang}.num.txt").write_text("\n".join(nums), encoding="utf-8")
        seen = {c for w in words for c in w}
        summary[lang] = {
            "vocab": key,
            "group": LANGUAGES[lang][1],
            "pool": len(words),
            "coverage_tokens": n_cov,
            "dropped_unrenderable": before - len(words),
            "fonts": len(by_lang[lang]),
            "font_files": by_lang[lang],
            "vocab_chars_not_on_pages": "".join(c for c in VOCABS[key] if c not in seen and not c.isspace()),
            "vocab_chars_no_face": impossible.get(lang, ""),
            "cleaning": stats[lang],
        }
        not_on_pages = "".join(c for c in VOCABS[key] if c not in seen and not c.isspace())
        flag = f"  !! not on pages: {not_on_pages[:30]}" if not_on_pages else ""
        print(f"  {lang:8s} pool {len(words):6d}  fonts {len(by_lang[lang]):3d}  cov+{n_cov:<5d}{flag}")

    weights = page_weights(list(summary))
    for lang in summary:
        summary[lang]["page_share"] = round(weights[lang], 5)
        summary[lang]["train_pages"] = round(weights[lang] * args.num_train)
    (meta / "languages.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(summary)} languages ready.")


# =============================================================================
# render
# =============================================================================


def text_heights_at_input(
    polygons: np.ndarray, size: tuple[int, int], input_size: int = TRAIN_INPUT_SIZE
) -> np.ndarray:
    """Text height of each word once docTR resizes the page to ``input_size`` (long edge).

    Horizontal words are measured by their height. Tall boxes are either vertical
    words or single narrow glyphs ('l', '1') whose width says nothing about type
    size, so they only count when the page has hardly any horizontal words (a
    vertical CJK page), measured by their width.
    """
    scale = input_size / max(size)
    xs, ys = polygons[:, :, 0], polygons[:, :, 1]
    w, h = (xs.max(1) - xs.min(1)) * scale, (ys.max(1) - ys.min(1)) * scale
    horizontal = w >= 0.6 * h
    if horizontal.sum() >= 0.3 * len(h):
        return h[horizontal]
    return w[h >= 3 * w]


def legible_at_input_size(
    polygons: np.ndarray,
    size: tuple[int, int],
    min_px: float = MIN_TEXT_PX,
    max_tiny: float = MAX_TINY_SHARE,
    input_size: int = TRAIN_INPUT_SIZE,
) -> bool:
    """Whether no more than ``max_tiny`` of the words fall below ``min_px`` at the training input size."""
    heights = text_heights_at_input(polygons, size, input_size)
    return not len(heights) or float((heights < min_px).mean()) <= max_tiny


def _input_size(args) -> int:
    return getattr(args, "input_size", TRAIN_INPUT_SIZE)


class PageWorker(Worker):
    """Renders one page in faces of its language."""

    def __init__(
        self,
        cfg: GenerationConfig,
        font_dir: str,
        max_long_edge: int | None = None,
        max_boxes: int | None = None,
        density: str = "low",
        input_size: int = TRAIN_INPUT_SIZE,
    ):
        self.selector = ManifestFontSelector(
            font_dir, handwriting=cfg.resources.auto_download_fonts, cache_dir=cfg.resources.font_cache_dir
        )
        self.gen = PageGenerator(cfg, font_selector=self.selector)
        self.fixed_long_edge, self.fixed_max_boxes = max_long_edge, max_boxes
        self.density = density
        self.input_size = input_size  # the docTR --input_size the pages must stay legible at
        self.max_long_edge, self.max_boxes = max_long_edge or MAX_LONG_EDGE, max_boxes or MAX_BOXES_PER_PAGE
        self.min_px, self.max_tiny = MIN_TEXT_PX, MAX_TINY_SHARE
        self.dense = False
        self.save = ("JPEG", {"quality": cfg.core.output_jpeg_quality}) if cfg.core.output_jpeg else ("PNG", {})

    def _apply_density(self) -> None:
        """Pick this page's typography preset (``mixed``: drawn per page) and its limits."""
        preset = self.density
        if preset == "mixed":
            preset = random.choices(list(DENSITY_MIX), weights=list(DENSITY_MIX.values()))[0]
        apply_density(self.gen.config, preset)
        self.dense = preset == "high"
        boxes, long_edge, self.min_px, self.max_tiny = DENSITY_LIMITS[preset]
        self.max_boxes = self.fixed_max_boxes or boxes
        self.max_long_edge = self.fixed_long_edge or max(long_edge, self.input_size)
        self.gen.config.media.delivery_long_edge_range = (DELIVERY_LONG_EDGE[0], float(self.max_long_edge))

    def page(self, lang: str, words: list[str], negative: bool = False) -> tuple[Image.Image, np.ndarray] | None:
        """Render a page whose text stays legible at docTR's input size (``None`` if none did).

        ``negative`` renders a page without any labelled text (see
        :meth:`PageGenerator.generate_page`); ``words`` then only feed its show-through.
        """
        self._apply_density()
        if negative:
            self.selector.language = lang
            page, _ = self.gen.generate_page(words, negative=True)
            if self.max_long_edge and max(page.size) > self.max_long_edge:
                scale = self.max_long_edge / max(page.size)
                page = page.resize(
                    (max(1, round(page.width * scale)), max(1, round(page.height * scale))), Image.Resampling.LANCZOS
                )
            return page, np.zeros((0, 4, 2))
        for _ in range(PAGE_ATTEMPTS):
            result = self._page_once(lang, words)
            if result is not None and legible_at_input_size(
                result[1], result[0].size, self.min_px, self.max_tiny, self.input_size
            ):
                return result
        return None

    def _page_once(self, lang: str, words: list[str]) -> tuple[Image.Image, np.ndarray] | None:
        det = self.gen.config.detection
        cjk = lang in CJK_VERTICAL
        det.layout_weights = LAYOUT_CJK_VERTICAL if cjk else LAYOUT_DENSE if self.dense else LAYOUT_DEFAULT
        # Stacked vertical text emits one box per character: fine for CJK, a box
        # explosion anywhere else.
        det.vertical_stacked_prob = 0.7 if cjk else 0.05
        self.selector.language = lang
        page, polys = self.gen.generate_page(words)
        if not polys:
            return None
        arr = np.asarray(polys, dtype=np.float64).reshape(-1, 4, 2)
        if self.max_long_edge and max(page.size) > self.max_long_edge:
            scale = self.max_long_edge / max(page.size)
            size = (max(1, round(page.width * scale)), max(1, round(page.height * scale)))
            page = page.resize(size, Image.Resampling.LANCZOS)
            arr *= scale
        if self.max_boxes and len(arr) > self.max_boxes:
            return None
        return page, arr

    def render(self, payload, path):
        """Render ``(lang, words, negative)`` to ``path``; the record is the docTR label entry."""
        lang, words, *rest = payload
        negative = bool(rest and rest[0])
        result = self.page(lang, words, negative)
        if result is None:
            return None
        page, arr = result
        page.save(path, self.save[0], **self.save[1])
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        # Integer pixels: docTR normalises by image size and resizes to 1024, so sub-pixel
        # corners are noise - and they cost ~35% of the file.
        polygons = np.rint(arr).astype(np.int32).tolist()
        record = {"img_dimensions": [page.height, page.width], "img_hash": digest, "polygons": polygons}
        return {"lang": "none (no text)" if negative else lang, **record}


def _load_meta(meta: Path):
    langs = json.loads((meta / "languages.json").read_text(encoding="utf-8"))
    pools = {k: (meta / "pools" / f"{k}.txt").read_text(encoding="utf-8").split("\n") for k in langs}
    nums = {
        k: [t for t in (meta / "pools" / f"{k}.num.txt").read_text(encoding="utf-8").split("\n") if t] for k in langs
    }
    weights = {k: v["page_share"] for k, v in langs.items()}
    return pools, nums, weights


def page_plan(n: int, seed: int, pools, nums, weights, negative_share: float = 0.0):
    """Deterministic language, candidate list and negative flag per page (identical on resume).

    A negative page carries no labelled text; its words only feed the show-through.
    The flag has its own random stream, so the text pages of a plan do not change
    with ``negative_share``.
    """
    rng = random.Random(seed)
    neg_rng = random.Random(seed + 104_729)
    langs = list(weights)
    en = pools.get("en", [])
    for lang in rng.choices(langs, weights=[weights[k] for k in langs], k=n):
        pool, num = pools[lang], nums[lang]
        n_num = int(CANDIDATES_PER_PAGE * NUMERIC_SHARE) if num else 0
        mix = en and LANGUAGES[lang][1] not in ("latin", "cjk") and rng.random() < MIX_PROB
        n_mix = int(CANDIDATES_PER_PAGE * MIX_SHARE) if mix else 0
        words = [rng.choice(pool) for _ in range(CANDIDATES_PER_PAGE - n_num - n_mix)]
        words += [rng.choice(num) for _ in range(n_num)] + [rng.choice(en) for _ in range(n_mix)]
        rng.shuffle(words)  # the RTL decision looks at the first candidates
        yield lang, words, neg_rng.random() < negative_share


def apply_density(cfg: GenerationConfig, preset: str) -> None:
    """Set the typography of one ``PAGE_DENSITY`` preset on ``cfg``."""
    body, fine_prob, news_px, max_blocks, table_rows, news_cols = PAGE_DENSITY[preset]
    cfg.detection.body_point_range = body
    cfg.detection.fine_print_prob = fine_prob
    cfg.detection.newspaper_font_size_range = news_px
    cfg.detection.max_blocks = max_blocks
    cfg.detection.table_rows_range = table_rows
    cfg.detection.newspaper_columns_range = news_cols


def _config(args) -> GenerationConfig:
    cfg = GenerationConfig.flat(
        task="detection",
        output_dir=str(args.out),
        num_workers=args.workers,
        auto_download_fonts=not getattr(args, "offline", False),  # handwriting faces only
        bg_image_dir=args.bg_dir,
        output_jpeg=not args.png,
        output_jpeg_quality=JPEG_QUALITY,
        det_max_words_per_page=CANDIDATES_PER_PAGE,
        det_punctuation_prob=0.0,  # pools already carry script-appropriate punctuation
        media_delivery_long_edge_range=DELIVERY_LONG_EDGE,
        media_resample_prob=1.0,
    )
    apply_density(cfg, "low" if args.density == "mixed" else args.density)
    return cfg


def _finalize(out: Path, split: str) -> None:
    """Move every part's images into ``<split>/images`` and stream one ``labels.json``."""
    final = out / split
    parts = sorted((out / "_parts" / split).glob("part_*"))
    if (final / "labels.json").exists() or not parts:
        return
    (final / "images").mkdir(parents=True, exist_ok=True)
    langs: Counter = Counter()
    pages = boxes = 0
    tmp = final / "labels.json.tmp"
    with open(tmp, "w", encoding="utf-8") as out_fh:
        out_fh.write("{")
        for part in parts:
            records = json.loads((part / "labels.json").read_text(encoding="utf-8"))
            for name, rec in sorted(records.items()):
                src = part / "images" / name
                if src.exists():
                    os.replace(src, final / "images" / name)
                elif not (final / "images" / name).exists():
                    continue  # neither moved by an interrupted finalize nor present
                lang = rec.pop("lang")
                out_fh.write(("," if pages else "") + json.dumps(name) + ":")
                json.dump(rec, out_fh, separators=(",", ":"))
                langs[lang] += 1
                boxes += len(rec["polygons"])
                pages += 1
        out_fh.write("}")
    os.replace(tmp, final / "labels.json")
    (final / "languages.json").write_text(json.dumps(dict(langs.most_common()), indent=1), encoding="utf-8")
    shutil.rmtree(out / "_parts" / split, ignore_errors=True)
    size = (final / "labels.json").stat().st_size
    print(f"  -> {final}: {pages:,} pages, {boxes:,} boxes, labels.json {size / 1e9:.2f} GB "
          f"({boxes / max(pages, 1):.0f} boxes/page)")  # fmt: skip


def render(args) -> None:
    """Render val + train in resumable parts, then merge each split into one folder."""
    require_harfbuzz()
    out, meta = Path(args.out), Path(args.out, "meta")
    if not (meta / "languages.json").exists():
        sys.exit("Run `prepare` first.")
    pools, nums, weights = _load_meta(meta)
    cfg = _config(args)
    SyntheticDatasetGenerator(cfg).resolve_backgrounds()
    ext = "png" if args.png else "jpg"

    jobs = []
    for split, total, seed_offset in (("val", args.num_val, 999_983), ("train", args.num_train, 0)):
        if (out / split / "labels.json").exists():
            continue
        for i, start in enumerate(range(0, total, args.part_size)):
            part = out / "_parts" / split / f"part_{i:04d}"
            if not (part / "labels.json").exists():
                jobs.append((split, min(args.part_size, total - start), start, SEED + seed_offset + i, part))
    progress = Progress(sum(j[1] for j in jobs), unit="pages")
    if jobs:
        print(f"Rendering {progress.total:,} pages in {len(jobs)} part(s) with {args.workers} workers ...")
        factory_args = (cfg, str(meta / "fonts"), None, None, args.density, _input_size(args))
        with make_pool(args.workers, PageWorker, factory_args) as pool:
            for split, n, start, seed, part in jobs:
                items = [
                    (f"{start + i:07d}.{ext}", payload)
                    for i, payload in enumerate(
                        page_plan(n, seed, pools, nums, weights, getattr(args, "negative_share", NEGATIVE_SHARE))
                    )
                ]
                records = render_part(pool, items, part, CHUNK, progress, f"{split}/{part.name}")
                tmp = part / "labels.json.tmp"
                tmp.write_text(json.dumps(records, separators=(",", ":")), encoding="utf-8")
                os.replace(tmp, part / "labels.json")
                (part / "progress.jsonl").unlink(missing_ok=True)
                print(f"\n  {split}/{part.name}: {len(records):,} pages, {n - len(records)} dropped", flush=True)
    print("Merging parts ...")
    for split in ("val", "train"):
        _finalize(out, split)
    shutil.rmtree(out / "_parts", ignore_errors=True)
    print(f"Done in {(time.time() - progress.t0) / 3600:.2f} h.")


def preview(args) -> Path:
    """Render a handful of pages with their polygons drawn on top."""
    from .preview import detection_sheet

    meta = Path(args.out, "meta")
    pools, nums, weights = _load_meta(meta)
    cfg = _config(args)
    SyntheticDatasetGenerator(cfg).resolve_backgrounds()
    random.seed(args.seed)
    np.random.seed(args.seed)
    worker = PageWorker(cfg, str(meta / "fonts"), None, None, args.density, _input_size(args))
    wanted = args.preview_languages.split(",") if args.preview_languages else None
    pages = []
    for lang, words, _ in page_plan(args.samples * 8, args.seed, pools, nums, weights):
        if wanted and lang not in wanted:
            continue
        result = worker.page(lang, words)
        if result:
            pages.append((result[0], result[1].tolist(), lang))
        if len(pages) == args.samples:
            break
    return detection_sheet(pages, Path(args.out, "preview", "detection.jpg"))


def main(argv: list[str] | None = None) -> None:
    """Command line entry point (``python -m generator.recipes.detection``)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["prepare", "render", "all", "preview"])
    p.add_argument("--out", required=True, help="dataset directory")
    p.add_argument("--languages", default="all", help="preset (european, multilingual, all) or codes: de,fr,ru")
    p.add_argument("--num-train", type=int, default=NUM_TRAIN)
    p.add_argument("--num-val", type=int, default=NUM_VAL)
    p.add_argument("--part-size", type=int, default=PART_SIZE)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument(
        "--density",
        choices=[*sorted(PAGE_DENSITY), "mixed"],
        default="mixed",
        help="typography preset; 'mixed' draws low/medium/high per page (50/30/20%%) - high is broadsheet-dense",
    )
    p.add_argument(
        "--input-size",
        type=int,
        default=TRAIN_INPUT_SIZE,
        help="the docTR --input_size you train at; every page stays legible at it (larger: denser pages allowed)",
    )
    p.add_argument(
        "--negative-share",
        type=float,
        default=NEGATIVE_SHARE,
        help="share of pages without any text (blank, empty forms, graphics, photos, show-through only, "
        "creased sheets photographed on cloth or wood, fabric close-ups)",
    )
    p.add_argument("--font-src", default="fonts_v1", help="unzipped fonts_v1 release (downloaded if missing)")
    p.add_argument("--cache", default=".recipe_cache", help="corpora, extra fonts and the font scan")
    p.add_argument("--bg-dir", default=None, help="background images (default: the curated download)")
    p.add_argument("--offline", action="store_true", help="never download; use cached corpora and fonts only")
    p.add_argument("--png", action="store_true", help="PNG instead of JPEG")
    p.add_argument("--samples", type=int, default=4, help="preview: number of pages")
    p.add_argument("--preview-languages", default="", help="preview: only these codes")
    p.add_argument("--seed", type=int, default=0, help="preview: page plan seed")
    args = p.parse_args(argv)
    if args.step in ("prepare", "all"):
        prepare(args)
    if args.step in ("render", "all"):
        render(args)
    if args.step == "preview":
        print(preview(args))


if __name__ == "__main__":
    main()
