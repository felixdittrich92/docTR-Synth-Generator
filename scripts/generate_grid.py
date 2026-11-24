#!/usr/bin/env python3
# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Render the README example image from two prepared recipe builds.

    python -m generator.recipes.detection prepare --languages en,de,ru,ar,hi,ja,he,el --out /tmp/det
    python -m generator.recipes.recognition prepare --languages all --out /tmp/rec --num-train 300000
    python scripts/generate_grid.py --det /tmp/det --rec /tmp/rec -o docs/examples.jpg

Two rows of detection pages (one layout and language each, plus a text-free hard
negative), then printed and handwritten recognition crops in many scripts, each with
its exact label.
"""

import argparse
import random
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from generator.components.config import GenerationConfig
from generator.dataset_generator import SyntheticDatasetGenerator
from generator.recipes import detection
from generator.recipes.fonts import ManifestFontSelector
from generator.recipes.languages import script_of
from generator.recipes.recognition import CropWorker, _read_tsv

# (language, layout, density, capture, caption) - two rows of five. ``capture`` picks how the
# page reaches the camera or scanner, so the image shows the range: colour, grayscale and
# black-and-white scans and phone photos (bent sheet, fabric or wood under it, colour cast).
# ``layout=None`` is a text-free hard negative: a crumpled blank sheet on a tablecloth.
PAGES: list[tuple] = [  # (lang, layout, density, capture, caption[, pinned seed])
    ("en", "paragraph", "low", "photo", "Report · English · phone photo"),
    ("de", "receipt", "low", "photo", "Receipt · German · photo"),
    ("ru", "newspaper", "high", "gray", "Dense newspaper · Russian · gray scan"),
    ("ar", "id_card", "low", "photo", "ID card · Arabic · phone photo"),
    ("de", "form", "low", "bw", "Form, by hand · German · B&W copy"),
    ("hi", "table", "low", "scan", "Table · Hindi · scan"),
    ("ja", "vertical", "low", "photo", "Vertical · Japanese · phone photo"),
    ("he", "paragraph", "medium", "design", "Magazine · Hebrew · generated design", 11),
    ("el", "id_card", "low", "scan", "ID card, chip + MRZ · Greek · scan"),
    ("en", None, "low", "photo", "No text (hard negative) · photo"),
]
# (script group, language) for the crop grid.
CROPS = [
    ("LATIN", "de"), ("LATIN", "vi"), ("LATIN", "pl"), ("LATIN", "es"), ("CJK", "zh"), ("CYRILLIC", "ru"),
    ("CYRILLIC", "uk"), ("GREEK", "el"), ("ARABIC", "ar"), ("ARABIC", "fa"), ("HANGUL", "ko"), ("HEBREW", "he"),
    ("DEVANAGARI", "hi"), ("BENGALI", "bn"), ("GURMUKHI", "pa"), ("GUJARATI", "gu"), ("TAMIL", "ta"),
    ("TELUGU", "te"), ("KANNADA", "kn"), ("MALAYALAM", "ml"), ("SINHALA", "si"), ("THAI", "th"), ("LAO", "lo"),
    ("KHMER", "km"), ("MYANMAR", "my"), ("GEORGIAN", "ka"), ("ARMENIAN", "hy"), ("ETHIOPIC", "am"),
]  # fmt: skip
# Handwritten crops (filled-in forms, notes): one row.
HAND_CROPS = [
    ("LATIN", "de"), ("CYRILLIC", "ru"), ("GREEK", "el"), ("ARABIC", "ar"), ("THAI", "th"), ("CJK", "ja"),
    ("HANGUL", "ko"),
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
    worker = detection.PageWorker(cfg, str(det_out / "meta" / "fonts"), detection.MAX_LONG_EDGE)
    pages = []
    plain = cfg.detection.plain_background_prob
    for i, (lang, layout, density, capture, caption, *pinned) in enumerate(PAGES):
        # An entry may pin its own seed: the page then does not change when others do.
        base_rand, base_plan = (pinned[0], pinned[0]) if pinned else (seed + 31 * i, seed + i)
        if lang not in pools:
            continue
        worker.density = density
        cfg.capture.prob = 1.0 if capture == "photo" else 0.0
        cfg.capture.textured_surface_prob = 1.0  # show the tablecloths and tables
        cfg.realism.grayscale_prob = 1.0 if capture == "gray" else 0.0
        cfg.realism.bitonal_prob = 1.0 if capture == "bw" else 0.0
        cfg.detection.crease_prob = 1.0 if layout == "receipt" else 0.0
        cfg.detection.picture_prob = 1.0 if layout == "paragraph" and capture == "photo" else 0.0
        cfg.detection.handwriting_prob = 1.0 if layout == "form" else 0.35
        # A copier sheet is plain paper; on a textured background thresholding would
        # erase words, and the generator then (rightly) falls back to grayscale.
        cfg.detection.plain_background_prob = 1.0 if capture == "bw" else 0.0 if capture == "design" else plain
        # A magazine page on a generated design; everywhere else the default mix.
        worker.gen.background_manager.procedural_prob = 1.0 if capture == "design" else 0.0
        cfg.detection.picture_prob = 1.0 if capture == "design" else cfg.detection.picture_prob
        for attempt in range(8):
            random.seed(base_rand + attempt)
            np.random.seed(base_rand + attempt)
            _, words, _ = next(detection.page_plan(1, base_plan + attempt, pools, nums, {lang: 1.0}))
            if layout is None:
                worker._apply_density()
                worker.selector.language = lang
                pages.append((
                    worker.gen.generate_page(words, negative=True, negative_kind="creased_sheet")[0],
                    caption,
                ))
                break
            cfg.detection.layout = layout
            result = worker.page(lang, words)
            if result and capture == "bw" and len(np.unique(np.asarray(result[0].convert("L")))) > 2:
                continue  # fell back to grayscale: the caption would be wrong
            if result and capture == "design" and result[0].width < 0.6 * result[0].height:
                continue  # a strip, not a magazine spread
            if result:
                pages.append((result[0], caption))
                break
    return pages


def recognition_crops(rec_out: Path, bg_dir: str | None, seed: int) -> list[tuple[Image.Image, str, bool]]:
    """One crop per entry of ``CROPS`` and ``HAND_CROPS``, drawn from the prepared label pool."""
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
    worker = CropWorker(cfg, str(rec_out / "meta" / "fonts"), handwriting_share=1.0)
    crops = []
    for group, lang in CROPS:
        candidates = []
        for text in pool.get((group, lang), [])[:20]:
            worker.selector.language = lang
            if (img := worker.gen.generate_image(text)) is not None:
                candidates.append((img.convert("RGB"), text, False))
            if len(candidates) == 3:
                break
        if candidates:  # the training data keeps its hard cases; the showcase shows a clear one
            crops.append(max(candidates, key=lambda c: _clarity(c[0])))
    for group, lang in HAND_CROPS:
        candidates = []
        for text in pool.get((group, lang), [])[20:60]:
            worker.selector.language = lang
            hand = worker.selector.get_handwriting_font(text)
            if hand and (img := worker.gen.generate_image(text, font_path=hand)) is not None:
                candidates.append((img.convert("RGB"), text, True))
            if len(candidates) == 3:
                break
        if candidates:
            crops.append(max(candidates, key=lambda c: _clarity(c[0])))
    return crops


def _clarity(img: Image.Image) -> float:
    """Contrast x sharpness at docTR's crop height: picks a readable example for the sheet."""
    g = np.asarray(img.convert("L").resize((max(8, img.width * 32 // img.height), 32)), dtype=np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(np.percentile(g, 95) - np.percentile(g, 5)) * float(np.abs(lap).mean())


_LABEL_FACES: dict[str, set[str]] = {}


def _label_font(text: str, selector: ManifestFontSelector) -> str | None:
    """A plain Noto face for a CJK/Hangul label (display faces make those unreadable), else the script's face."""
    if _group(text) not in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL"):
        return selector.get_font_for_text(text)  # shaped (Arabic, Indic) faces of the script itself
    for path in selector.paths:
        if "Noto" not in Path(path).name or "Serif" in Path(path).name:
            continue
        if path not in _LABEL_FACES:
            _LABEL_FACES[path] = {chr(c) for c in TTFont(path, lazy=True).getBestCmap()}
        if set(text) <= _LABEL_FACES[path]:
            return path
    return selector.get_font_for_text(text)


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

    def _fit(text: str, width: int) -> ImageFont.FreeTypeFont:
        """The caption face, shrunk until ``text`` fits under its page."""
        for size in range(19, 11, -1):
            face = ImageFont.truetype(ui_font, size)
            if face.getlength(text) <= width - 4:
                return face
        return face

    ui_chars = {chr(c) for c in TTFont(ui_font).getBestCmap()}
    inner = WIDTH - 2 * MARGIN
    layout = []
    for row in (pages[:5], pages[5:10]):
        if not row:
            continue
        scaled = [
            (p.resize((max(1, int(p.width * PAGE_H / p.height)), PAGE_H), Image.Resampling.LANCZOS), c) for p, c in row
        ]
        total = sum(p.width for p, _ in scaled) + GAP * 2 * (len(scaled) - 1)
        fit = min(1.35, inner / total)  # fill the row width
        scaled = [(p.resize((int(p.width * fit), int(p.height * fit)), Image.Resampling.LANCZOS), c) for p, c in scaled]
        layout.append(scaled)

    printed = [c for c in crops if not c[2]]
    written = [c for c in crops if c[2]]
    crop_rows = (len(printed) + COLS - 1) // COLS + (len(written) + COLS - 1) // COLS
    height = MARGIN + 58 + sum(r[0][0].height + 46 + GAP for r in layout)
    height += 70 + crop_rows * (CELL_H + GAP) + (40 if written else 0) + MARGIN
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
            ImageDraw.Draw(sheet).text(
                (x + 2, y + page.height + 8), caption, fill=MUTED, font=_fit(caption, page.width)
            )
            x += page.width + 2 * GAP
        y += row[0][0].height + 46 + GAP

    y += 12
    scripts = len({_group(text) for _, text, _ in crops} - {"NUMBERS"})
    heading(
        y, "Text recognition", f"printed and handwritten word crops in {scripts} scripts, each with its exact label"
    )
    y += 58
    cell_w = (inner - GAP * (COLS - 1)) // COLS
    slots = [(i, c) for i, c in enumerate(printed)]
    first_written = ((len(printed) + COLS - 1) // COLS) * COLS
    slots += [(first_written + i, c) for i, c in enumerate(written)]
    if written:
        label_y = y + (first_written // COLS) * (CELL_H + GAP) - 4
        ImageDraw.Draw(sheet).text((MARGIN, label_y), "handwritten", fill=MUTED, font=small)
    for i, (crop, text, hand) in slots:
        cx = MARGIN + (i % COLS) * (cell_w + GAP)
        cy = y + (i // COLS) * (CELL_H + GAP) + (40 if hand else 0)
        _card(sheet, (cx, cy, cx + cell_w, cy + CELL_H))
        draw = ImageDraw.Draw(sheet)
        im = crop.resize((max(1, int(crop.width * CROP_H / crop.height)), CROP_H), Image.Resampling.LANCZOS)
        if im.width > cell_w - 20:
            im = im.resize((cell_w - 20, max(1, int(im.height * (cell_w - 20) / im.width))), Image.Resampling.LANCZOS)
        sheet.paste(im, (cx + (cell_w - im.width) // 2, cy + 12 + (CROP_H - im.height) // 2))
        # The UI face where it covers the label (readable, same size everywhere), else a
        # face of the label's script.
        path = None if set(text) <= ui_chars else _label_font(text, selector)
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
