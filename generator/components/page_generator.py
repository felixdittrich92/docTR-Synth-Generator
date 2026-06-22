# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import hashlib
import math
import multiprocessing as mp
import random
from dataclasses import dataclass, field
from queue import Empty

import numpy as np
from PIL import Image, ImageDraw

from .background_manager import BackgroundManager
from .config import GenerationConfig
from .font_selector import FontSelector
from .text_renderer import TextRenderer
from .text_styling import apply_final_degradations, build_final_augmentations, decide_text_style, recolor_coverage

__all__ = ["PageGenerator", "DetectionTask"]

Polygon = list[list[float]]


@dataclass
class DetectionTask:
    """A single page to render for the detection dataset.

    Attributes:
        words (list[str]): Candidate words to lay out (as many as fit are used).
        save_path (str): Where to write the page image.
        filename (str): Filename key for the labels file.
        split (str): "train" or "val".
    """

    words: list[str] = field(default_factory=list)
    save_path: str = ""
    filename: str = ""
    split: str = "train"


class PageGenerator:
    """Generates document-like pages with per-word polygons.

    Args:
        config (GenerationConfig): Configuration (see the ``det_*`` fields).
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.font_selector = FontSelector(
            config.font_dir,
            auto_download=config.auto_download_fonts,
            font_cache_dir=config.font_cache_dir,
            download_timeout=config.font_download_timeout,
        )
        self.text_renderer = TextRenderer(config)
        self.background_manager = BackgroundManager(
            config.bg_image_dir,
            cache_size=config.bg_cache_size,
            max_dimension=config.bg_max_dimension,
        )
        self.final_augs = build_final_augmentations(config)

    # -- layout -----------------------------------------------------------

    def generate_page(self, words: list[str]) -> tuple[Image.Image, list[Polygon]]:
        """Lay out ``words`` on a page and return the page image + word polygons.

        The layout (paragraph, dense newspaper columns, a label/value form or an
        ID-card with fields) is chosen per :attr:`GenerationConfig.det_layout`, so
        a single run can mimic the variety of real documents.
        """
        cfg = self.config
        width = random.randint(*cfg.det_page_width_range)
        height = random.randint(*cfg.det_page_height_range)
        layout = self._choose_layout()

        # Detection ground truth only contains the words we place, so any text in
        # a background photo becomes an unlabelled false negative. Forms/ID-cards
        # always use clean generated paper; others mix in textures.
        if layout in ("form", "id_card") or random.random() < cfg.det_plain_background_prob:
            page = self._paper_background((width, height)).convert("RGBA")
        else:
            page = self.background_manager.get_page_background((width, height)).convert("RGBA")

        if not words:
            return page.convert("RGB"), []

        take_word = self._word_supplier(words)
        rtl = self._is_rtl(words)
        margin = max(4, int(min(width, height) * cfg.det_margin_ratio))
        area = (margin, margin, width - margin, height - margin)
        polygons: list[Polygon] = []

        if layout == "newspaper":
            self._layout_newspaper(page, area, take_word, rtl, polygons)
        elif layout == "form":
            self._layout_form(page, area, take_word, rtl, polygons)
        elif layout == "id_card":
            self._layout_id_card(page, area, take_word, rtl, polygons)
        else:
            self._layout_paragraph(page, area, take_word, rtl, polygons)

        page = page.convert("RGB")
        if polygons and random.random() < cfg.det_rotation_prob:
            angle = random.uniform(*cfg.det_rotation_range)
            page, polygons = self._rotate(page, polygons, angle)

        page = apply_final_degradations(cfg, page, self.final_augs)
        return page, polygons

    # -- layout building blocks ------------------------------------------

    def _choose_layout(self) -> str:
        cfg = self.config
        if cfg.det_layout != "mixed":
            return cfg.det_layout
        weights = cfg.det_layout_weights or {"paragraph": 0.4, "newspaper": 0.25, "form": 0.2, "id_card": 0.15}
        names = list(weights)
        return random.choices(names, weights=[weights[n] for n in names], k=1)[0]

    @staticmethod
    def _word_supplier(words: list[str]):
        """Return a ``take_word()`` that recycles words so any region fills fully."""
        shuffled = list(words)
        n = len(shuffled)
        state = {"i": 0}

        def take_word() -> str:
            word = shuffled[state["i"] % n]
            state["i"] += 1
            if state["i"] % n == 0:
                random.shuffle(shuffled)
            return word

        return take_word

    def _style_at(self, page: Image.Image, x: float, y: float, w: float, h: float, bold_width: int):
        x0, y0 = int(max(0, x)), int(max(0, y))
        x1, y1 = int(min(page.width, x + max(20, w))), int(min(page.height, y + max(6, h)))
        if x1 <= x0 or y1 <= y0:
            x1, y1 = x0 + 1, y0 + 1
        sample = page.crop((x0, y0, x1, y1)).convert("RGB")
        return decide_text_style(self.config, sample, bold_width=bold_width, outline_width=0)

    def _render_word(self, word: str, font_size: int, bold_width: int):
        font_path = self.font_selector.get_font_for_text(word)
        if not font_path:
            return None
        return self.text_renderer.render_coverage(word, font_path, font_size, bold_width)

    @staticmethod
    def _emit(page, coverage, style, paste_x: float, y: float, polygons: list[Polygon]) -> bool:
        px, py = int(paste_x), int(y)
        page.alpha_composite(recolor_coverage(coverage, style), (px, py))
        bbox = coverage.getchannel("A").getbbox()
        if bbox:
            left, top, right, bottom = bbox
            polygons.append([
                [px + left, py + top],
                [px + right, py + top],
                [px + right, py + bottom],
                [px + left, py + bottom],
            ])
            return True
        return False

    def _place_token(self, page, token, x, y, font_size, style, bold_width, polygons, align_right=False):
        """Place a single pre-built token (e.g. a field label) at (x, y)."""
        coverage = self._render_word(token, font_size, bold_width)
        if coverage is None:
            return 0
        paste_x = x - coverage.width if align_right else x
        self._emit(page, coverage, style, paste_x, y, polygons)
        return coverage.width

    def _fill_box(
        self,
        page,
        box,
        take_word,
        polygons,
        font_size,
        style,
        bold_width,
        rtl,
        max_lines=10**9,
        first_indent=0,
        line_spacing=(1.15, 1.4),
    ) -> int:
        """Fill a rectangular box with wrapped text; return the y reached."""
        bx0, by0, bx1, by1 = box
        line_height = max(font_size + 1, int(font_size * random.uniform(*line_spacing)))
        space = max(2, int(font_size * 0.33))
        y, lines = by0, 0
        while y + line_height <= by1 and lines < max_lines:
            indent = first_indent if lines == 0 else 0
            cursor = (bx1 - indent) if rtl else (bx0 + indent)
            line_has_word = False
            for _ in range(400):  # bounded attempts per line
                coverage = self._render_word(take_word(), font_size, bold_width)
                if coverage is None:
                    continue
                ww = coverage.width
                paste_x = cursor - ww if rtl else cursor
                overflow = (paste_x < bx0) if rtl else (paste_x + ww > bx1)
                if overflow:
                    if line_has_word:
                        break
                    continue  # single token wider than the box: try another
                if self._emit(page, coverage, style, paste_x, y, polygons):
                    line_has_word = True
                cursor = (paste_x - space) if rtl else (paste_x + ww + space)
            if not line_has_word:
                break  # nothing fits this box width
            y += line_height
            lines += 1
        return y

    def _body_font(self) -> int:
        lo, hi = self.config.det_font_size_range
        return random.randint(lo, max(lo, (lo + hi) // 2))  # smaller end -> denser

    def _layout_paragraph(self, page, area, take_word, rtl, polygons) -> None:
        cfg = self.config
        bx0, by0, bx1, by1 = area
        y, blocks = by0, 0
        while y < by1 and blocks < cfg.det_max_blocks:
            blocks += 1
            base = random.randint(*cfg.det_font_size_range)
            heading = random.random() < cfg.det_heading_prob
            font_size = int(base * random.uniform(1.4, 1.9)) if heading else base
            bold_width = self._heading_bold(font_size) if heading else self._maybe_bold(font_size)
            style = self._style_at(page, bx0, y, 240, font_size * 2, bold_width)
            indent = random.randint(0, int((bx1 - bx0) * 0.08)) if (not heading and random.random() < 0.25) else 0
            max_lines = 2 if heading else random.randint(2, 8)
            y = self._fill_box(
                page, (bx0, y, bx1, by1), take_word, polygons, font_size, style, bold_width, rtl, max_lines, indent
            )
            y += int(font_size * random.uniform(*cfg.det_block_gap_range))

    def _layout_newspaper(self, page, area, take_word, rtl, polygons) -> None:
        cfg = self.config
        bx0, by0, bx1, by1 = area
        y = by0
        # Masthead / headline spanning the full width.
        if random.random() < 0.9:
            hfs = int(random.randint(*cfg.det_font_size_range) * random.uniform(2.0, 3.2))
            hbw = self._heading_bold(hfs)
            hstyle = self._style_at(page, bx0, y, bx1 - bx0, hfs * 2, hbw)
            y = self._fill_box(
                page,
                (bx0, y, bx1, by1),
                take_word,
                polygons,
                hfs,
                hstyle,
                hbw,
                rtl,
                max_lines=random.randint(1, 2),
                line_spacing=(1.05, 1.2),
            )
            y += int(hfs * 0.3)
            if random.random() < 0.5:  # sub-deck under the headline
                dfs = max(cfg.det_newspaper_font_size_range[1], int(hfs * 0.4))
                dstyle = self._style_at(page, bx0, y, bx1 - bx0, dfs * 2, 0)
                y = self._fill_box(page, (bx0, y, bx1, by1), take_word, polygons, dfs, dstyle, 0, rtl, max_lines=1)
                y += int(dfs * 0.5)

        # Dense columns: many narrow columns of small, tightly-leaded body text.
        lo, hi = cfg.det_newspaper_columns_range
        gutter = max(6, int((bx1 - bx0) * 0.018))
        max_cols = max(2, int((bx1 - bx0 + gutter) / (90 + gutter)))  # keep columns legible
        ncols = max(2, min(random.randint(lo, hi), max_cols))
        col_w = ((bx1 - bx0) - (ncols - 1) * gutter) / ncols
        spacing = cfg.det_newspaper_line_spacing_range
        for c in range(ncols):
            cx0 = bx0 + c * (col_w + gutter)
            cx1 = cx0 + col_w
            cy, cblocks = y, 0
            while cy < by1 and cblocks < 80:
                cblocks += 1
                sub = random.random() < 0.18  # column sub-heading
                if sub:
                    font_size = int(random.randint(*cfg.det_newspaper_font_size_range) * random.uniform(1.4, 1.8))
                    bold_width = self._heading_bold(font_size)
                    style = self._style_at(page, cx0, cy, col_w, font_size * 2, bold_width)
                    cy = self._fill_box(
                        page,
                        (cx0, cy, cx1, by1),
                        take_word,
                        polygons,
                        font_size,
                        style,
                        bold_width,
                        rtl,
                        max_lines=random.randint(1, 2),
                        line_spacing=(1.1, 1.25),
                    )
                else:
                    font_size = random.randint(*cfg.det_newspaper_font_size_range)
                    bold_width = self._maybe_bold(font_size)
                    style = self._style_at(page, cx0, cy, col_w, font_size * 2, bold_width)
                    cy = self._fill_box(
                        page,
                        (cx0, cy, cx1, by1),
                        take_word,
                        polygons,
                        font_size,
                        style,
                        bold_width,
                        rtl,
                        max_lines=random.randint(5, 16),
                        line_spacing=spacing,
                    )
                cy += int(font_size * random.uniform(0.2, 0.5))

    def _layout_form(self, page, area, take_word, rtl, polygons) -> None:
        bx0, by0, bx1, by1 = area
        draw = ImageDraw.Draw(page)
        y = by0
        # Title.
        tfs = int(random.randint(*self.config.det_font_size_range) * random.uniform(1.6, 2.2))
        tstyle = self._style_at(page, bx0, y, bx1 - bx0, tfs * 2, self._heading_bold(tfs))
        y = self._fill_box(
            page, (bx0, y, bx1, by1), take_word, polygons, tfs, tstyle, self._heading_bold(tfs), rtl, max_lines=1
        )
        y += int(tfs * 0.7)

        label_col = int((bx1 - bx0) * random.uniform(0.28, 0.42))
        while y < by1:
            font_size = self._body_font()
            row_h = int(font_size * 2.0)
            if y + row_h > by1:
                break
            if random.random() < 0.12:  # section header
                sfs = int(font_size * 1.25)
                sstyle = self._style_at(page, bx0, y, bx1 - bx0, sfs * 2, self._heading_bold(sfs))
                self._fill_box(
                    page,
                    (bx0, y, bx1, y + sfs * 1.5),
                    take_word,
                    polygons,
                    sfs,
                    sstyle,
                    self._heading_bold(sfs),
                    rtl,
                    max_lines=1,
                )
                y += row_h
                continue
            # "Label:" + value, mirrored for RTL.
            label = take_word().capitalize() + ":"
            lstyle = self._style_at(page, bx0, y, label_col, font_size * 2, 0)
            if rtl:
                self._place_token(page, label, bx1, y, font_size, lstyle, 0, polygons, align_right=True)
                vbox = (bx0, y, bx1 - label_col, y + row_h)
                line_x0, line_x1 = bx0, bx1 - label_col
            else:
                self._place_token(page, label, bx0, y, font_size, lstyle, 0, polygons)
                vbox = (bx0 + label_col, y, bx1, y + row_h)
                line_x0, line_x1 = bx0 + label_col, bx1
            vstyle = self._style_at(page, vbox[0], y, vbox[2] - vbox[0], font_size * 2, 0)
            self._fill_box(page, vbox, take_word, polygons, font_size, vstyle, 0, rtl, max_lines=1)
            if random.random() < 0.5:  # field underline
                ly = int(y + font_size * 1.35)
                draw.line((line_x0, ly, line_x1, ly), fill=(165, 165, 175), width=1)
            y += row_h

    def _layout_id_card(self, page, area, take_word, rtl, polygons) -> None:
        bx0, by0, bx1, by1 = area
        aw, ah = bx1 - bx0, by1 - by0
        draw = ImageDraw.Draw(page)

        card_w = int(aw * random.uniform(0.74, 0.96))
        card_h = int(min(ah * random.uniform(0.5, 0.72), card_w * 0.64))
        cx0 = bx0 + (aw - card_w) // 2
        cy0 = by0 + int(ah * random.uniform(0.05, 0.18))
        cx1, cy1 = cx0 + card_w, cy0 + card_h
        tone = random.randint(222, 248)
        draw.rounded_rectangle(
            (cx0, cy0, cx1, cy1),
            radius=int(card_h * 0.06),
            fill=(tone, tone, max(0, tone - 6)),
            outline=(140, 145, 160),
            width=2,
        )
        pad = int(card_h * 0.08)

        # Title line.
        tfs = max(12, int(card_h * 0.1))
        tstyle = self._style_at(page, cx0 + pad, cy0 + pad, card_w - 2 * pad, tfs * 1.6, self._heading_bold(tfs))
        self._fill_box(
            page,
            (cx0 + pad, cy0 + pad, cx1 - pad, cy0 + pad + int(tfs * 1.5)),
            take_word,
            polygons,
            tfs,
            tstyle,
            self._heading_bold(tfs),
            rtl,
            max_lines=1,
        )

        # Photo placeholder.
        photo_w = int(card_w * 0.24)
        photo_h = int(card_h * 0.55)
        px0 = cx0 + pad
        py0 = cy0 + pad + int(tfs * 1.7)
        draw.rectangle((px0, py0, px0 + photo_w, py0 + photo_h), fill=(198, 200, 208), outline=(150, 150, 162))

        # Field rows beside the photo.
        fx0 = px0 + photo_w + pad
        fx1 = cx1 - pad
        fs = max(11, int(card_h * 0.062))
        label_col = int((fx1 - fx0) * 0.42)
        fy = py0
        while fy + int(fs * 1.8) <= cy1 - pad - int(fs * 2.4):
            label = take_word().capitalize() + ":"
            lstyle = self._style_at(page, fx0, fy, label_col, fs * 2, 0)
            self._place_token(page, label, fx0, fy, fs, lstyle, self._maybe_bold(fs), polygons)
            vstyle = self._style_at(page, fx0 + label_col, fy, fx1 - fx0 - label_col, fs * 2, 0)
            self._fill_box(
                page,
                (fx0 + label_col, fy, fx1, fy + int(fs * 1.7)),
                take_word,
                polygons,
                fs,
                vstyle,
                0,
                rtl,
                max_lines=1,
            )
            fy += int(fs * 1.95)

        # Two MRZ-like lines across the bottom, sized to fit the card width.
        mfs = max(10, int(card_h * 0.07))
        avail = cx1 - cx0 - 2 * pad
        n_chars = max(12, int(avail / (mfs * 0.62)))
        mstyle = self._style_at(page, cx0 + pad, cy1 - pad - int(mfs * 2.4), avail, mfs * 2, 0)
        my = cy1 - pad - int(mfs * 2.4)
        for _ in range(2):
            mrz = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<<<<") for _ in range(n_chars))
            self._place_token(page, mrz, cx0 + pad, my, mfs, mstyle, 0, polygons)
            my += int(mfs * 1.15)

    @staticmethod
    def _is_rtl(words: list[str]) -> bool:
        """Decide whether the page should be laid out right-to-left.

        True when the majority of the candidate words use a right-to-left script
        (Arabic, Hebrew, Syriac, ...), so Arabic/Hebrew pages read naturally.
        """

        def char_is_rtl(cp: int) -> bool:
            return (
                0x0590 <= cp <= 0x05FF  # Hebrew
                or 0x0600 <= cp <= 0x06FF  # Arabic
                or 0x0700 <= cp <= 0x074F  # Syriac
                or 0x0750 <= cp <= 0x077F  # Arabic Supplement
                or 0x08A0 <= cp <= 0x08FF  # Arabic Extended-A
                or 0xFB1D <= cp <= 0xFEFC  # Hebrew/Arabic presentation forms
            )

        sample = words[:200]
        if not sample:
            return False
        rtl = sum(1 for w in sample if any(char_is_rtl(ord(c)) for c in w))
        return rtl > len(sample) / 2

    @staticmethod
    def _paper_background(size: tuple[int, int]) -> Image.Image:
        """A clean, subtly-shaded paper background."""
        width, height = size
        tone = random.randint(236, 252)
        arr = np.full((height, width, 3), float(tone), dtype=np.float32)
        # Gentle low-frequency shading across the page + faint grain.
        gx = np.linspace(random.uniform(-7, 7), random.uniform(-7, 7), width)
        gy = np.linspace(random.uniform(-7, 7), random.uniform(-7, 7), height)
        arr += gx[None, :, None] + gy[:, None, None]
        arr += np.random.normal(0.0, 2.0, (height, width, 3))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

    def _maybe_bold(self, font_size: int) -> int:
        if random.random() < self.config.bold_prob:
            frac = random.uniform(*self.config.bold_width_frac_range)
            return max(1, round(font_size * self.text_renderer.supersample * frac))
        return 0

    def _heading_bold(self, font_size: int) -> int:
        frac = self.config.bold_width_frac_range[1]
        return max(1, round(font_size * self.text_renderer.supersample * frac))

    @staticmethod
    def _rotate(page: Image.Image, polygons: list[Polygon], angle: float) -> tuple[Image.Image, list[Polygon]]:
        """Rotate the page and its polygons together about the page centre."""
        width, height = page.size
        rotated = page.rotate(
            angle,
            expand=True,
            resample=Image.BICUBIC,  # type: ignore[attr-defined]
            fillcolor=(255, 255, 255),
        )
        new_w, new_h = rotated.size
        cx, cy = width / 2.0, height / 2.0
        ncx, ncy = new_w / 2.0, new_h / 2.0
        # PIL rotates the image counter-clockwise; with the y-axis pointing down
        # this maps a source point (x, y) to the destination below.
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        def transform(px: float, py: float) -> list[float]:
            dx, dy = px - cx, py - cy
            return [dx * cos_a + dy * sin_a + ncx, -dx * sin_a + dy * cos_a + ncy]

        new_polys = [[transform(px, py) for px, py in poly] for poly in polygons]
        return rotated, new_polys

    # -- multiprocessing worker ------------------------------------------

    @staticmethod
    def detection_worker_process(
        task_queue: mp.Queue, result_queue: mp.Queue, config: GenerationConfig, worker_id: int
    ):
        """Worker process: render pages and return their polygon annotations."""
        print(f"Detection worker {worker_id} starting...")
        save_format = "JPEG" if config.output_jpeg else "PNG"
        save_kwargs = {"quality": config.output_jpeg_quality} if config.output_jpeg else {}

        try:
            generator = PageGenerator(config)
            processed = 0
            while True:
                try:
                    task = task_queue.get()
                    if task is None:
                        break
                    success: bool = False
                    dims: list[int] | None = None
                    polygons: list[Polygon] = []
                    try:
                        page, polygons = generator.generate_page(task.words)
                        if polygons:
                            page.save(task.save_path, save_format, **save_kwargs)
                            with open(task.save_path, "rb") as fh:
                                img_hash = hashlib.sha256(fh.read()).hexdigest()
                            dims = [page.height, page.width]
                            success = True
                            processed += 1
                            if processed % 200 == 0:
                                print(f"Detection worker {worker_id}: {processed} pages")
                        else:
                            img_hash = ""
                    except Exception as e:
                        img_hash = ""
                        print(f"Detection worker {worker_id}: error on '{task.filename}': {e}")

                    result_queue.put((task.filename, task.split, dims, img_hash, polygons, success))
                except Empty:
                    continue
                except Exception as e:
                    print(f"Detection worker {worker_id}: unexpected error: {e}")
                    break
        except Exception as e:
            print(f"Detection worker {worker_id}: failed to initialize: {e}")
        print(f"Detection worker {worker_id} finished. Rendered {processed} pages.")
