#!/usr/bin/env python3
# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Render the README example image from two prepared recipe builds.

    python -m generator.recipes.detection prepare --languages en,de,pl,ar,ru,hi,ja,he --out /tmp/det
    python -m generator.recipes.recognition prepare --languages multilingual --out /tmp/rec --num-train 300000
    python scripts/generate_grid.py --det /tmp/det --rec /tmp/rec -o docs/examples.jpg

Two rows of detection pages (one layout and language each), then recognition crops in
many scripts, each with its exact label.
"""

import argparse
import random
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from generator.components.config import GenerationConfig
from generator.dataset_generator import SyntheticDatasetGenerator
from generator.recipes import detection
from generator.recipes.fonts import ManifestFontSelector
from generator.recipes.languages import script_of
from generator.recipes.recognition import CropWorker, _read_tsv

# (language, layout, capture, caption) - two rows of four. ``capture`` picks how the page
# reaches the camera or scanner, so the image shows the range: a colour scan, a grayscale
# scan, a black-and-white copy and phone photos (bent sheet, lighting, colour cast).
PAGES = [
    ("en", "paragraph", "photo", "Report · English · phone photo"),
    ("de", "receipt", "scan", "Receipt · German · scan"),
    ("pl", "newspaper", "gray", "Newspaper · Polish · grayscale scan"),
    ("ar", "id_card", "photo", "ID card · Arabic · phone photo"),
    ("ru", "form", "bw", "Form · Russian · B&W copy"),
    ("hi", "table", "scan", "Table · Hindi · scan"),
    ("ja", "vertical", "photo", "Vertical · Japanese · phone photo"),
    ("he", "paragraph", "gray", "Letter · Hebrew · grayscale scan"),
]
# (script group, language) for the crop grid.
CROPS = [
    ("LATIN", "de"), ("LATIN", "vi"), ("LATIN", "pl"), ("LATIN", "es"), ("LATIN", "tr"), ("CYRILLIC", "ru"),
    ("CYRILLIC", "uk"), ("GREEK", "el"), ("ARABIC", "ar"), ("ARABIC", "fa"), ("ARABIC", "ur"), ("HEBREW", "he"),
    ("DEVANAGARI", "hi"), ("BENGALI", "bn"), ("GURMUKHI", "pa"), ("GUJARATI", "gu"), ("TAMIL", "ta"),
    ("TELUGU", "te"), ("KANNADA", "kn"), ("MALAYALAM", "ml"), ("SINHALA", "si"), ("THAI", "th"), ("LAO", "lo"),
    ("KHMER", "km"), ("MYANMAR", "my"), ("GEORGIAN", "ka"), ("ARMENIAN", "hy"), ("ETHIOPIC", "am"),
]  # fmt: skip
WIDTH, MARGIN, GAP, RADIUS = 2400, 40, 22, 14
PAGE_H = 440
CROP_H, CELL_H, COLS = 46, 104, 7
BG, CARD, INK, MUTED, LABEL = (236, 239, 243), (255, 255, 255), (32, 36, 46), (104, 112, 128), (196, 44, 44)


def _group(text: str) -> str:
    return next((script_of(c) for c in text if c.isalpha()), "NUMBERS")


def detection_pages(det_out: Path, bg_dir: str | None, seed: int) -> list[tuple[Image.Image, str]]:
    """One page per entry of ``PAGES``, rendered exactly like the detection recipe."""
    args = Namespace(out=str(det_out), density="low", workers=1, bg_dir=bg_dir, png=False, offline=False)
    cfg = detection._config(args)
    cfg.detection.partial_page_prob = 0.0  # show each layout in full
    SyntheticDatasetGenerator(cfg).resolve_backgrounds()
    pools, nums, _ = detection._load_meta(det_out / "meta")
    worker = detection.PageWorker(cfg, str(det_out / "meta" / "fonts"), detection.MAX_LONG_EDGE, 1000)
    pages = []
    plain = cfg.detection.plain_background_prob
    for i, (lang, layout, capture, caption) in enumerate(PAGES):
        if lang not in pools:
            continue
        cfg.detection.layout = layout
        cfg.capture.prob = 1.0 if capture == "photo" else 0.0
        cfg.realism.grayscale_prob = 1.0 if capture == "gray" else 0.0
        cfg.realism.bitonal_prob = 1.0 if capture == "bw" else 0.0
        # A copier sheet is plain paper; on a textured background thresholding would
        # erase words, and the generator then (rightly) falls back to grayscale.
        cfg.detection.plain_background_prob = 1.0 if capture == "bw" else plain
        for attempt in range(8):
            random.seed(seed + 31 * i + attempt)
            np.random.seed(seed + 31 * i + attempt)
            _, words = next(detection.page_plan(1, seed + i + attempt, pools, nums, {lang: 1.0}))
            result = worker.page(lang, words)
            if result and capture == "bw" and len(np.unique(np.asarray(result[0].convert("L")))) > 2:
                continue  # fell back to grayscale: the caption would be wrong
            if result:
                pages.append((result[0], caption))
                break
    return pages


def recognition_crops(rec_out: Path, bg_dir: str | None, seed: int) -> list[tuple[Image.Image, str]]:
    """One crop per entry of ``CROPS``, drawn from the prepared label pool."""
    rows = _read_tsv(rec_out / "meta" / "train_labels.tsv")
    rng = random.Random(seed)
    pool: dict[tuple[str, str], list[str]] = defaultdict(list)
    for lang, text in rng.sample(rows, min(len(rows), 400_000)):
        if 4 <= len(text) <= 13:
            pool[(_group(text), lang)].append(text)
    cfg = GenerationConfig.flat(task="recognition", output_dir=str(rec_out), bg_image_dir=bg_dir)
    SyntheticDatasetGenerator(cfg).resolve_backgrounds()
    random.seed(seed)
    np.random.seed(seed)
    worker = CropWorker(cfg, str(rec_out / "meta" / "fonts"))
    tmp = rec_out / "preview" / "_grid_crop.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    crops = []
    for group, lang in CROPS:
        for text in pool.get((group, lang), [])[:20]:
            if worker.render((lang, text), str(tmp)) is not None:
                crops.append((Image.open(tmp).convert("RGB"), text))
                break
    return crops


def _card(sheet: Image.Image, box: tuple[int, int, int, int]) -> None:
    """A white rounded card with a soft shadow."""
    x0, y0, x1, y1 = box
    shadow = Image.new("L", sheet.size, 0)
    ImageDraw.Draw(shadow).rounded_rectangle((x0 + 2, y0 + 5, x1 + 2, y1 + 5), RADIUS, fill=70)
    shadow = shadow.filter(ImageFilter.GaussianBlur(8))
    sheet.paste(Image.new("RGB", sheet.size, (190, 196, 206)), (0, 0), shadow)
    ImageDraw.Draw(sheet).rounded_rectangle(box, RADIUS, fill=CARD)


def compose(pages, crops, ui_font: str, selector: ManifestFontSelector) -> Image.Image:
    """Headings, two rows of pages and the crop grid on one sheet."""
    title = ImageFont.truetype(ui_font, 34)
    small = ImageFont.truetype(ui_font, 19)
    inner = WIDTH - 2 * MARGIN
    layout = []
    for row in (pages[:4], pages[4:8]):
        if not row:
            continue
        scaled = [
            (p.resize((max(1, int(p.width * PAGE_H / p.height)), PAGE_H), Image.Resampling.LANCZOS), c) for p, c in row
        ]
        total = sum(p.width for p, _ in scaled) + GAP * 2 * (len(scaled) - 1)
        fit = min(1.35, inner / total)  # fill the row width
        scaled = [(p.resize((int(p.width * fit), int(p.height * fit)), Image.Resampling.LANCZOS), c) for p, c in scaled]
        layout.append(scaled)

    crop_rows = (len(crops) + COLS - 1) // COLS
    height = MARGIN + 58 + sum(r[0][0].height + 46 + GAP for r in layout)
    height += 70 + crop_rows * (CELL_H + GAP) + MARGIN
    sheet = Image.new("RGB", (WIDTH, height), BG)

    def heading(y: int, text: str, sub: str) -> None:
        draw = ImageDraw.Draw(sheet)
        draw.text((MARGIN, y), text, fill=INK, font=title)
        draw.text((MARGIN + draw.textlength(text, font=title) + 18, y + 12), sub, fill=MUTED, font=small)

    y = MARGIN
    heading(y, "Text detection", "pages with a polygon for every word")
    y += 58
    for row in layout:
        row_w = sum(p.width for p, _ in row) + GAP * 2 * (len(row) - 1)
        x = MARGIN + (inner - row_w) // 2
        for page, caption in row:
            _card(sheet, (x - 10, y - 10, x + page.width + 10, y + page.height + 38))
            sheet.paste(page, (x, y))
            ImageDraw.Draw(sheet).text((x + 2, y + page.height + 8), caption, fill=MUTED, font=small)
            x += page.width + 2 * GAP
        y += row[0][0].height + 46 + GAP

    y += 12
    heading(y, "Text recognition", "word crops in 25 scripts, each with its exact label")
    y += 58
    cell_w = (inner - GAP * (COLS - 1)) // COLS
    for i, (crop, text) in enumerate(crops):
        cx = MARGIN + (i % COLS) * (cell_w + GAP)
        cy = y + (i // COLS) * (CELL_H + GAP)
        _card(sheet, (cx, cy, cx + cell_w, cy + CELL_H))
        draw = ImageDraw.Draw(sheet)
        im = crop.resize((max(1, int(crop.width * CROP_H / crop.height)), CROP_H), Image.Resampling.LANCZOS)
        if im.width > cell_w - 20:
            im = im.resize((cell_w - 20, max(1, int(im.height * (cell_w - 20) / im.width))), Image.Resampling.LANCZOS)
        sheet.paste(im, (cx + (cell_w - im.width) // 2, cy + 12 + (CROP_H - im.height) // 2))
        path = selector.get_font_for_text(text)
        font = ImageFont.truetype(path, 19) if path else small
        tw = draw.textlength(text, font=font)
        draw.text((cx + (cell_w - tw) / 2, cy + CROP_H + 24), text, fill=LABEL, font=font)
    return sheet


def main() -> None:
    """Command line entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--det", required=True, help="prepared detection build (generator.recipes.detection prepare)")
    p.add_argument("--rec", required=True, help="prepared recognition build (generator.recipes.recognition prepare)")
    p.add_argument("--bg-dir", default=None, help="background images (default: the curated download)")
    p.add_argument("--ui-font", default=None, help="font for headings and captions (default: Inter from --rec)")
    p.add_argument("-o", "--output", default="docs/examples.jpg")
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()
    rec = Path(args.rec)
    candidates = sorted(rec.glob("meta/fonts/*Inter_500.ttf")) + sorted(rec.glob("meta/fonts/*Noto_Sans_500.ttf"))
    ui_font = args.ui_font or (str(candidates[0]) if candidates else None)
    if ui_font is None:
        p.error("no Inter / Noto Sans face in the recognition build - pass --ui-font")
    pages = detection_pages(Path(args.det), args.bg_dir, args.seed)
    crops = recognition_crops(rec, args.bg_dir, args.seed)
    selector = ManifestFontSelector(rec / "meta" / "fonts", handwriting=False)
    sheet = compose(pages, crops, ui_font, selector)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=88, optimize=True)
    print(args.output, sheet.size, f"{len(pages)} pages, {len(crops)} crops")


if __name__ == "__main__":
    main()
