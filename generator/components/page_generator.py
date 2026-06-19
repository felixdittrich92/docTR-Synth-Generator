# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Synthetic *detection* dataset generation.

Where :class:`TextImageGenerator` produces single word/line crops for
recognition training, :class:`PageGenerator` composes whole document-like pages
- multiple paragraph blocks of real words laid out with margins, line wrapping,
the occasional heading and indent - and records a 4-point polygon for every word
in the docTR detection format::

    {"page_0001.jpg": {"img_dimensions": [H, W], "img_hash": "...",
                       "polygons": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]}}

It reuses the recognition pipeline's font resolution, ink styling, background
textures and degradations, so detection pages look like the same "world" as the
recognition crops. An optional small global rotation skews the whole page and
the polygons together for skew robustness.

The output is consumed directly by docTR detection training:
https://github.com/mindee/doctr/tree/main/references/detection
"""

import hashlib
import math
import multiprocessing as mp
import random
from dataclasses import dataclass, field
from queue import Empty

import numpy as np
from PIL import Image

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

        Returns:
            tuple[Image.Image, list[Polygon]]: RGB page and a list of 4-point
            polygons (absolute pixel coordinates) - one per placed word.
        """
        cfg = self.config
        width = random.randint(*cfg.det_page_width_range)
        height = random.randint(*cfg.det_page_height_range)
        # Detection ground truth only contains the words we place, so any text
        # already present in a background photo would become an unlabelled false
        # negative. Plain generated paper avoids that; texture images are used
        # for the rest (and should themselves be text-free - see the README).
        if random.random() < cfg.det_plain_background_prob:
            page = self._paper_background((width, height)).convert("RGBA")
        else:
            page = self.background_manager.get_page_background((width, height)).convert("RGBA")

        margin = max(4, int(min(width, height) * cfg.det_margin_ratio))
        x_left, x_right = margin, width - margin
        y = margin

        polygons: list[Polygon] = []
        target_words = random.randint(*cfg.det_words_per_page_range)
        wi, n_words, placed, blocks = 0, len(words), 0, 0
        rtl = self._is_rtl(words)

        while y < height - margin and wi < n_words and placed < target_words and blocks < cfg.det_max_blocks:
            blocks += 1
            base_size = random.randint(*cfg.det_font_size_range)
            heading = random.random() < cfg.det_heading_prob
            font_size = int(base_size * random.uniform(1.4, 1.9)) if heading else base_size
            bold_width = self._heading_bold(font_size) if heading else self._maybe_bold(font_size)

            # One ink style per block, decided from the local background.
            sample = page.crop((x_left, y, min(x_right, x_left + 240), min(height, y + font_size * 2))).convert("RGB")
            style = decide_text_style(cfg, sample, bold_width=bold_width, outline_width=0)

            line_height = int(font_size * random.uniform(1.15, 1.4))
            space = max(2, int(font_size * 0.33))
            max_lines = 2 if heading else random.randint(1, 6)

            for line_idx in range(max_lines):
                if y + line_height > height - margin or wi >= n_words or placed >= target_words:
                    break
                indent = 0
                if line_idx == 0 and not heading and random.random() < 0.25:
                    indent = random.randint(0, int((x_right - x_left) * 0.08))  # paragraph indent
                # Cursor starts at the leading edge: left for LTR, right for RTL.
                cursor = (x_right - indent) if rtl else (x_left + indent)
                line_has_word = False

                while wi < n_words and placed < target_words:
                    word = words[wi]
                    font_path = self.font_selector.get_font_for_text(word)
                    if not font_path:
                        wi += 1
                        continue
                    coverage = self.text_renderer.render_coverage(word, font_path, font_size, bold_width)
                    ww, wh = coverage.size

                    # Determine the paste x for the word in the current direction.
                    paste_x = cursor - ww if rtl else cursor
                    overflow = (paste_x < x_left) if rtl else (paste_x + ww > x_right)
                    if overflow:
                        if line_has_word:
                            break  # wrap to next line, retry this word
                        wi += 1  # single word wider than the column: skip it
                        continue

                    glyph = recolor_coverage(coverage, style)
                    page.alpha_composite(glyph, (paste_x, y))

                    bbox = coverage.getchannel("A").getbbox()
                    if bbox:
                        left, top, right, bottom = bbox
                        polygons.append([
                            [paste_x + left, y + top],
                            [paste_x + right, y + top],
                            [paste_x + right, y + bottom],
                            [paste_x + left, y + bottom],
                        ])
                        placed += 1
                        line_has_word = True
                    cursor = (paste_x - space) if rtl else (paste_x + ww + space)
                    wi += 1
                y += line_height

            y += int(line_height * random.uniform(*cfg.det_block_gap_range))

        page = page.convert("RGB")
        if polygons and random.random() < cfg.det_rotation_prob:
            angle = random.uniform(*cfg.det_rotation_range)
            page, polygons = self._rotate(page, polygons, angle)

        page = apply_final_degradations(cfg, page, self.final_augs)
        return page, polygons

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
        """A clean, subtly-shaded paper background (guaranteed text-free)."""
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
