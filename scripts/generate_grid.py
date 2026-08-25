#!/usr/bin/env python3
# Copyright (C) 2021-2026, Felix Dittrich.
#
# Render an example grid (detection pages + recognition crops) like the one in
# the README. Words, fonts and backgrounds are downloaded/cached on first run.
#
#   python make_examples_grid.py                      # -> docs/examples_grid.png
#   python make_examples_grid.py -o my_grid.png --seed 7

from __future__ import annotations

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from generator.components import GenerationConfig, PageGenerator, TextImageGenerator
from generator.components.background_downloader import BackgroundDownloader
from generator.components.corpus_downloader import CorpusDownloader

# (layout, languages, seed, vertical_prob) - one detection page per layout;
# Arabic id_card shows RTL, the Japanese page shows fully vertical typesetting.
DETECTION = [
    ("paragraph", ["en", "de"], 3, 0.0),
    ("newspaper", ["en", "de"], 2, 1.0),
    ("form", ["en", "de"], 7, 0.0),
    ("id_card", ["ar"], 4, 0.0),
    ("vertical", ["ja"], 11, 0.0),
]
DETECTION_CAPTIONS = ["paragraph", "newspaper (+ banner)", "form", "id_card (RTL)", "vertical (JA)"]

# (languages, seed) - recognition crops across scripts/fonts/colours.
RECOGNITION = [
    (["en"], 1),
    (["de"], 2),
    (["fr"], 3),
    (["ru"], 4),
    (["ar"], 5),
    (["hi"], 6),
    (["el"], 7),
    (["en"], 8),
    (["de"], 9),
    (["es"], 10),
    (["th"], 11),
    (["en"], 12),
]

CACHE = dict(
    corpus_cache_dir="/tmp/synth_corpus", font_cache_dir="/tmp/synth_fonts", background_cache_dir="/tmp/synth_bg"
)


def _font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def resolve_backgrounds():
    """Download the curated background set once and return its directory (or None).

    ``auto_download_backgrounds`` only fires inside the full dataset generator, so
    when rendering pages directly we trigger the download ourselves and pass the
    resulting directory as ``bg_image_dir``.
    """
    downloader = BackgroundDownloader(cache_dir=CACHE["background_cache_dir"], enabled=True)
    paths = downloader.download_all()
    if not paths:
        print("No backgrounds downloaded - falling back to generated/blank backgrounds.")
        return None
    print(f"Using {len(paths)} downloaded backgrounds from {CACHE['background_cache_dir']}")
    return CACHE["background_cache_dir"]


def detection_page(corpus, languages, layout, seed, vertical_prob=0.0, bg_dir=None, w=620, h=860):
    """Render one detection page (PIL image) for a given layout."""
    words = corpus.build_vocabulary(languages, words_per_language=12000)
    cfg = GenerationConfig.flat(
        task="detection",
        languages=languages,
        det_layout=layout,
        det_vertical_prob=vertical_prob,
        det_rotation_prob=0.0,
        det_page_width_range=(w, w),
        det_page_height_range=(h, h),
        bg_image_dir=bg_dir,  # use the downloaded backgrounds
        det_plain_background_prob=0.0 if bg_dir else 0.4,  # show them off in the demo
        auto_download_backgrounds=False,  # already resolved above
        **CACHE,
    )
    random.seed(seed)
    np.random.seed(seed)
    page, _ = PageGenerator(cfg).generate_page(random.choices(words, k=600))
    return page


def recognition_crop(corpus, languages, seed, bg_dir=None):
    """Render one recognition word crop (PIL image)."""
    words = corpus.build_vocabulary(languages, words_per_language=6000)
    gen = TextImageGenerator(
        GenerationConfig.flat(
            task="recognition",
            languages=languages,
            auto_download_fonts=True,
            auto_download_backgrounds=False,
            bg_image_dir=bg_dir,
            **CACHE,
        )
    )
    random.seed(seed * 13)
    np.random.seed(seed * 13)
    for _ in range(30):  # skip the rare blank render
        crop = gen.generate_image(random.choice(words))
        if crop is not None and crop.width > 20:
            return crop.convert("RGB")
    return Image.new("RGB", (60, 24), (255, 255, 255))


def build_grid(out_path: str, seed: int = 0):
    """Render the full example grid and save it to ``out_path``."""
    corpus = CorpusDownloader(cache_dir=CACHE["corpus_cache_dir"])
    random.seed(seed)
    bg_dir = resolve_backgrounds()

    # --- detection row (uniform height) ---
    th = 360
    pages = [
        detection_page(corpus, langs, layout, s, vertical_prob=vp, bg_dir=bg_dir) for layout, langs, s, vp in DETECTION
    ]
    pages = [p.resize((int(p.width * th / p.height), th)) for p in pages]

    # --- recognition crops ---
    crops = [recognition_crop(corpus, langs, s, bg_dir=bg_dir) for langs, s in RECOGNITION]

    pad, cap = 16, 24
    det_w = sum(p.width for p in pages) + pad * (len(pages) + 1)
    cols, rows = 4, 3
    cell_w = (det_w - pad * (cols + 1)) // cols
    cell_h = 66
    rec_h = rows * (cell_h + pad) + pad + cap
    canvas = Image.new("RGB", (det_w, cap + th + pad * 2 + rec_h + 24), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = _font(15)

    draw.text(
        (pad, 5),
        "Detection - full synthetic pages, horizontal and vertical (every word is boxed in the labels)",
        fill=(35, 35, 35),
        font=font,
    )
    x, y = pad, cap
    for page, caption in zip(pages, DETECTION_CAPTIONS):
        canvas.paste(page, (x, y))
        draw.rectangle([x, y, x + page.width, y + page.height], outline=(205, 205, 205))
        draw.text((x + 4, y + page.height + 3), caption, fill=(95, 95, 95), font=font)
        x += page.width + pad

    ry = cap + th + pad * 2 + 12
    draw.text(
        (pad, ry - cap + 2),
        "Recognition - word crops across scripts, fonts, colours and degradations",
        fill=(35, 35, 35),
        font=font,
    )
    for i, crop in enumerate(crops):
        cx = pad + (i % cols) * (cell_w + pad)
        cy = ry + (i // cols) * (cell_h + pad)
        tile = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
        scale = min((cell_w - 14) / crop.width, (cell_h - 14) / crop.height, 2.2)
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
        tile.paste(crop, ((cell_w - crop.width) // 2, (cell_h - crop.height) // 2))
        canvas.paste(tile, (cx, cy))
        draw.rectangle([cx, cy, cx + cell_w, cy + cell_h], outline=(222, 222, 222))

    canvas.save(out_path)
    print(f"Saved {out_path}  ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render an example grid (detection pages + recognition crops).")
    ap.add_argument("-o", "--out", default="docs/examples_grid_x.png", help="output PNG path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    build_grid(args.out, seed=args.seed)
