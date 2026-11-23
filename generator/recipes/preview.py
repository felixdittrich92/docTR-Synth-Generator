# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Contact sheets of a prepared build, to eyeball it before rendering millions of samples."""

import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .languages import script_of

__all__ = ["recognition_sheet", "detection_sheet"]


def _group(text: str) -> str:
    for c in text:
        if c.isalpha():
            return script_of(c)
    return "NUMBERS_SYMBOLS"


def _label_font(selector, text: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = selector.get_font_for_text(text) or selector.get_font_for_text("Abc")
    return ImageFont.truetype(path, size) if path else ImageFont.load_default()


def recognition_sheet(out: Path, per_group: int = 6, seed: int = 0, bg_dir: str | None = None, cols: int = 4) -> Path:
    """Render ``per_group`` crops per script from ``OUT/meta/train_labels.tsv``.

    Writes ``OUT/preview/recognition.png`` (crop above, its exact label below) and
    the crops themselves to ``OUT/preview/crops/``.
    """
    from ..components.config import GenerationConfig
    from ..dataset_generator import SyntheticDatasetGenerator
    from .recognition import CropWorker, _read_tsv

    meta = Path(out, "meta")
    rows = _read_tsv(meta / "train_labels.tsv")
    rng = random.Random(seed)
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for lang, text in rng.sample(rows, min(len(rows), 300_000)):
        groups[_group(text)].append((lang, text))
    cfg = GenerationConfig.flat(task="recognition", output_dir=str(out), auto_download_fonts=True, bg_image_dir=bg_dir)
    SyntheticDatasetGenerator(cfg).resolve_backgrounds()
    random.seed(seed)
    worker = CropWorker(cfg, str(meta / "fonts"))
    crops_dir = Path(out, "preview", "crops")
    crops_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for group in sorted(groups):
        for i, (lang, text) in enumerate(groups[group][:per_group]):
            path = crops_dir / f"{group.lower()}_{i}.png"
            if worker.render((lang, text), str(path)) is not None:
                tiles.append((f"{lang}", text, Image.open(path).convert("RGB")))

    cell_w, cell_h = 300, 96
    rows_n = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows_n), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for i, (lang, text, img) in enumerate(tiles):
        x, y = (i % cols) * cell_w, (i // cols) * cell_h
        img.thumbnail((cell_w - 10, cell_h - 30))
        sheet.paste(img, (x + 5, y + 4))
        code_font = _label_font(worker.selector, f"{lang}:", 13)
        draw.text((x + 5, y + cell_h - 22), f"{lang}:", fill=(90, 90, 90), font=code_font)
        offset = int(draw.textlength(f"{lang}: ", font=code_font))
        draw.text(
            (x + 5 + offset, y + cell_h - 24), text, fill=(180, 0, 0), font=_label_font(worker.selector, text, 15)
        )
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(215, 215, 215))
    target = Path(out, "preview", "recognition.png")
    sheet.save(target)
    return target


def detection_sheet(pages: list[tuple[Image.Image, list, str]], target: Path, height: int = 900) -> Path:
    """Side-by-side pages with their word polygons drawn on top."""
    tiles = []
    for page, polygons, lang in pages:
        overlay = page.convert("RGB").copy()
        draw = ImageDraw.Draw(overlay)
        for poly in polygons:
            draw.polygon([tuple(pt) for pt in poly], outline=(230, 30, 30))
        scale = height / overlay.height
        overlay = overlay.resize((max(1, int(overlay.width * scale)), height), Image.Resampling.LANCZOS)
        ImageDraw.Draw(overlay).text((8, 8), f"{lang} - {len(polygons)} boxes", fill=(0, 90, 200))
        tiles.append(overlay)
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + 10 * (len(tiles) - 1), height), (90, 90, 90))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 10
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, quality=90)
    return target
