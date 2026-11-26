# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import multiprocessing as mp
import random
from dataclasses import dataclass
from queue import Empty

import numpy as np
from PIL import Image, ImageDraw

from .background_manager import BackgroundManager
from .config import GenerationConfig
from .font_selector import FontSelector
from .font_styles import StyleSpec, color_name, hand_warp, paint_highlight
from .legibility import DegradationBudget, local_ink_contrast
from .scan import apply_scan_tone
from .text_renderer import TextRenderer, TextStyle
from .text_styling import (
    apply_final_degradations,
    build_final_augmentations,
    calm_texture,
    decide_text_style,
    recolor_coverage,
)

__all__ = ["TextImageGenerator", "GenerationTask"]


@dataclass
class GenerationTask:
    """Task for the queue.

    Attributes:
        text (str): Text to render
        save_path (str): Path to save the generated image
        filename (str): Filename for the saved image
        worker_id (int): ID of the worker processing this task
    """

    text: str
    save_path: str
    filename: str
    worker_id: int = 0


class TextImageGenerator:
    """Generates text overlay images with backgrounds.

    Args:
        config (GenerationConfig): Configuration for text image generation
        font_selector (FontSelector | None): use this selector instead of building one from the config
    """

    def __init__(self, config: GenerationConfig, font_selector: FontSelector | None = None):
        self.config = config
        # A caller may supply its own selector (e.g. a vetted, per-language font set).
        self.font_selector = font_selector or FontSelector(
            config.resources.font_dir,
            auto_download=config.resources.auto_download_fonts,
            font_cache_dir=config.resources.font_cache_dir,
            download_timeout=config.resources.font_download_timeout,
            verify_shaping=config.resources.verify_shaping,
        )
        self.text_renderer = TextRenderer(config)
        self._last_ink: tuple[int, ...] = (0, 0, 0)  # ink of the last styled compose (for its colour label)
        self.background_manager = BackgroundManager(
            config.resources.bg_image_dir,
            cache_size=config.resources.bg_cache_size,
            max_dimension=config.resources.bg_max_dimension,
            procedural_prob=config.resources.procedural_bg_prob,
        )

        # Image-space degradations applied AFTER compositing text onto the
        # background, mirroring how a real capture degrades the whole frame.
        self.final_augs = build_final_augmentations(config)

    def is_text_visible(self, image: Image.Image, alpha_thresh: int = 20, min_visible_ratio: float = 0.02) -> bool:
        """Check if rendered text is sufficiently visible."""
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        alpha_np = np.array(image.getchannel("A"))
        ratio = np.count_nonzero(alpha_np > alpha_thresh) / alpha_np.size
        return bool(ratio >= min_visible_ratio)

    def _decide_style(self, bg_img: Image.Image, bold_width: int, outline_width: int) -> TextStyle:
        """Decide ink colour, polarity and contrast from the background."""
        return decide_text_style(self.config, bg_img, bold_width, outline_width)

    def _recolor(self, coverage: Image.Image, style: TextStyle) -> Image.Image:
        """Recolour a black coverage glyph to the ink colour, reusing its alpha."""
        return recolor_coverage(coverage, style)

    def _apply_final_degradations(self, image: Image.Image, text_height: float | None = None) -> Image.Image:
        """Brightness/contrast jitter + sensor noise + JPEG artifacts on the crop.

        Recognition crops are budgeted by glyph height exactly as detection pages
        are: a blur that is invisible on a 40px crop closes the counters on a
        12px one, and the model is asked to read the result either way.
        """
        budget = DegradationBudget.for_text_height(text_height) if text_height else None
        return apply_final_degradations(self.config, image, self.final_augs, budget)

    def generate_image(self, text: str, font_path: str | None = None, hand: float = 0.0) -> Image.Image | None:
        """Generate a single text overlay image.

        Args:
            text (str): Text to render
            font_path (str | None): use this face (e.g. a handwriting face) instead of
                resolving one through the font selector.
            hand (float): strength of the pen variation (:func:`hand_warp`) for handwriting, 0 for print.

        Returns:
            Image.Image | None: RGB image with realistic text overlay, or None if
            no suitable font/background was found.
        """
        font_path = font_path or self.font_selector.get_font_for_text(text)
        if not font_path:
            return None
        result = self._generate(text, font_path, None, hand)
        return result[0] if result else None

    def generate_styled(self, text: str, font_path: str, spec: StyleSpec) -> tuple[Image.Image, dict] | None:
        """A crop of ``text`` in ``font_path`` drawn in ``spec``, and what it shows.

        For font-style data: no faux bold, no outline, and of the page context the
        neighbouring lines and the borders of a cell, but never a rule below the word (it would
        look like an underline). Returns the image
        and ``{"color": ...}``, the ink colour measured on the finished crop.
        """
        return self._generate(text, font_path, spec)

    def _generate(
        self, text: str, font_path: str, spec: StyleSpec | None, hand: float = 0.0
    ) -> tuple[Image.Image, dict] | None:
        """The re-draw loop shared by :meth:`generate_image` and :meth:`generate_styled`."""
        # Background and ink are re-drawn until the finished crop is readable along
        # its whole length: a dark band or busy patch of paper under part of the
        # word makes a label the image no longer supports. A second round re-rolls
        # size and weight too, since a hairline at 12px may never clear the bar.
        min_contrast = self.config.realism.min_local_contrast
        for _ in range(2 if min_contrast > 0 else 1):
            font_size, bold_width, outline_width = self.text_renderer.sample_style()
            if spec is not None:  # the face's own weight is the label: no stroke tricks
                bold_width = outline_width = 0
            # Render the glyph once as a black coverage map (single getbbox/draw).
            coverage = None
            for _ in range(self.config.core.max_attempts):
                candidate = self.text_renderer.render_coverage(
                    text,
                    font_path,
                    font_size,
                    bold_width,
                    weight=spec.weight if spec else None,
                    decorations=spec.decorations if spec else None,
                )
                if self.is_text_visible(candidate):
                    coverage = candidate
                    break
            if coverage is None:
                continue
            strength = spec.hand if spec is not None else hand
            if strength > 0:
                coverage = hand_warp(coverage, strength, np.random.default_rng(random.getrandbits(32)))
            for _ in range(max(1, self.config.core.max_attempts)):
                image, alpha = self._compose(text, font_path, font_size, coverage, outline_width, bold_width, spec)
                if min_contrast <= 0 or local_ink_contrast(image, alpha) >= min_contrast:
                    return image, (self._measure_ink(image, alpha, self._last_ink) if spec is not None else {})
        return None

    @staticmethod
    def _measure_ink(image: Image.Image, alpha: Image.Image, ink_drawn: tuple[int, ...]) -> dict:
        """The ink colour a reader names, judged against the paper around the word.

        A crop that went through a grayscale or black-and-white scan has lost its
        colour, so it is named from its pixels. Otherwise the ink that was drawn is
        named: faded ink over a yellow marker *measures* olive but reads as black.
        """
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        a = np.asarray(alpha, dtype=np.uint8)
        core = a > 200 if (a > 200).sum() >= 12 else a > 100
        paper = a == 0
        if not core.any() or not paper.any():
            return {"color": color_name(np.asarray(ink_drawn, np.float32), np.full(3, 255.0))}
        bg = np.median(arr[paper], axis=0)
        gray = float(np.abs(arr[core] - arr[core].mean(axis=1, keepdims=True)).max()) < 3.0
        ink = np.median(arr[core], axis=0) if gray else np.asarray(ink_drawn, np.float32)
        return {"color": color_name(ink, bg)}

    def _margins(self, text_height: int) -> tuple[int, int, int, int]:
        """Extra (left, top, right, bottom) margin, like an expanded detector box."""
        rec = self.config.recognition
        if random.random() >= rec.margin_prob:
            return 0, 0, 0, 0
        lo, hi = rec.margin_range
        return tuple(int(round(text_height * random.uniform(lo, hi))) for _ in range(4))  # type: ignore[return-value]

    def _context(
        self,
        size: tuple[int, int],
        box: tuple[int, int, int, int],
        text: str,
        font_path: str,
        font_size: int,
        style,
        rules: bool = True,
        weight: int | None = None,
        hand: float = 0.0,
        cells: bool = False,
    ) -> Image.Image | None:
        """What a real crop picks up around a word: neighbouring lines and rules, in its margins only.

        ``box`` is where the word sits. Line fragments are placed so they leave the
        crop through its top or bottom edge - only the descenders of the line above
        or the tops of the line below show, never a readable character. They use the
        word's own font, ``weight``, ink and pen variation (``hand``), so they are
        rendered (and shaped) like the word. ``rules=False`` leaves out the underlines
        and table rules (a font-style label tells an underline); ``cells=True`` draws
        the borders of a form or table cell instead: its sides, and its edge above
        the word - never one below, where it would read as an underline.
        """
        width, height = size
        x0, y0, x1, y1 = box
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        placed = False
        if cells and random.random() < 0.5:
            thickness = max(1, font_size // 18)
            border = (*style.fill_color, style.opacity) if random.random() < 0.5 else (90, 90, 90, 255)
            draw = ImageDraw.Draw(layer)
            for side, room in (("left", x0), ("right", width - x1)):
                if room > thickness and random.random() < 0.6:
                    x = (
                        random.randint(0, room - thickness)
                        if side == "left"
                        else x1 + random.randint(0, room - thickness)
                    )
                    draw.rectangle((x, 0, x + thickness - 1, height), fill=border)
                    placed = True
            if y0 > thickness and random.random() < 0.4:
                y = random.randint(0, y0 - thickness)
                draw.rectangle((0, y, width, y + thickness - 1), fill=border)
                placed = True
        for edge, room in (("top", y0), ("bottom", height - y1)):
            if room < 3 or random.random() >= 0.6:
                continue
            if rules and random.random() < 0.3:  # an underline or a table rule
                thickness = max(1, font_size // 18)
                y = random.randint(0, room - thickness) if edge == "top" else y1 + random.randint(0, room - thickness)
                left, right = (
                    (0, width) if random.random() < 0.6 else (random.randint(0, x0), random.randint(x1, width))
                )
                ImageDraw.Draw(layer).rectangle(
                    (left, y, right, y + thickness - 1), fill=(*style.fill_color, style.opacity)
                )
                placed = True
                continue
            fragment = self.text_renderer.render_coverage(
                text if random.random() < 0.5 else f"{text} {text}",
                font_path,
                font_size,
                style.bold_width,
                weight=weight,
            )
            if hand > 0:
                fragment = hand_warp(fragment, hand, np.random.default_rng(random.getrandbits(32)))
            ink = recolor_coverage(fragment, style)
            visible = random.randint(max(1, room // 3), room)  # never past the word's own line
            y = visible - ink.height if edge == "top" else height - visible
            x = (
                random.randint(min(0, width - ink.width), 0)
                if ink.width > width
                else random.randint(-ink.width // 3, 0)
            )
            layer.paste(ink, (x, y), ink)  # runs off the crop edge by design
            placed = True
        return layer if placed else None

    def _compose(
        self,
        text: str,
        font_path: str,
        font_size: int,
        coverage: Image.Image,
        outline_width: int,
        bold_width: int,
        spec: StyleSpec | None = None,
    ) -> tuple[Image.Image, Image.Image]:
        """Background + ink (+ page context) + final degradations for one rendered glyph map."""
        left, top, right, bottom = self._margins(coverage.height)
        size = (coverage.width + left + right, coverage.height + top + bottom)
        bg_crop = self.background_manager.get_background_crop(size)
        # Calm the texture *before* choosing ink, so the decision is made against
        # the background the glyph will actually sit on.
        bg_crop = calm_texture(bg_crop, self.config.realism.crop_texture_std)
        if spec is not None and spec.highlight is not None and (ink_box := coverage.getchannel("A").getbbox()):
            # A marker only shows on light paper (and then takes dark ink): lift a dark crop first.
            arr = np.asarray(bg_crop, dtype=np.float32)
            if arr.mean() < 170:
                arr = 200 + (arr - arr.mean()) * 0.3
                bg_crop = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            x0, y0, x1, y1 = ink_box
            bg_crop = paint_highlight(bg_crop, (left + x0, top + y0, left + x1, top + y1), spec.highlight)
        if spec is not None:
            style = decide_text_style(
                self.config, bg_crop, bold_width, outline_width, ink_hue=spec.ink_hue, neutral=spec.neutral
            )
            self._last_ink = style.fill_color
        else:
            style = self._decide_style(bg_crop, bold_width, outline_width)

        if outline_width > 0 and style.outline_color is not None:
            # Rarer two-tone path: re-render with the styled fill + outline.
            glyph = self.text_renderer.render_text_to_image(text, font_path, style, font_size)
            size = (glyph.width + left + right, glyph.height + top + bottom)
            bg_crop = calm_texture(
                self.background_manager.get_background_crop(size), self.config.realism.crop_texture_std
            )
        else:
            glyph = self._recolor(coverage, style)

        word = Image.new("RGBA", size, (0, 0, 0, 0))
        word.paste(glyph, (left, top))
        canvas = bg_crop.convert("RGBA")
        if random.random() < self.config.recognition.context_prob:
            # Styled crops (font style labels): the neighbouring lines and cell borders, never a rule below the word
            # (it would read as an underline)
            context = self._context(
                size,
                (left, top, left + glyph.width, top + glyph.height),
                text,
                font_path,
                font_size,
                style,
                rules=spec is None,
                weight=spec.weight if spec is not None else None,
                hand=spec.hand if spec is not None else 0.0,
                cells=spec is not None,  # form labels sit in boxes: their sides and top edge, never a rule below
            )
            if context is not None:
                canvas.alpha_composite(context)
        composed = Image.alpha_composite(canvas, word).convert("RGB")
        image = self._apply_final_degradations(composed, text_height=coverage.height)
        # Grayscale / black-and-white like a scanned page; the caller's legibility
        # check rejects a crop that thresholding erased.
        image = apply_scan_tone(self.config, image, text_height=coverage.height)
        return image, word.getchannel("A")

    @staticmethod
    def worker_process(task_queue: mp.Queue, result_queue: mp.Queue, config: GenerationConfig, worker_id: int):
        """Worker process function that processes tasks from the queue."""
        print(f"Worker {worker_id} starting...")

        save_format = "JPEG" if config.core.output_jpeg else "PNG"
        save_kwargs = {"quality": config.core.output_jpeg_quality} if config.core.output_jpeg else {}

        try:
            generator = TextImageGenerator(config)
            processed_count = 0

            while True:
                try:
                    task = task_queue.get()
                    if task is None:  # Poison pill to stop worker
                        break

                    success = False
                    try:
                        img = generator.generate_image(task.text)
                        if img is not None:
                            img.save(task.save_path, save_format, **save_kwargs)
                            success = True
                            processed_count += 1
                            if processed_count % 1000 == 0:
                                print(f"Worker {worker_id}: processed {processed_count} images")
                        else:
                            print(f"Worker {worker_id}: Skipping '{task.text}' - no suitable font")
                    except Exception as e:
                        print(f"Worker {worker_id}: Error generating '{task.text}': {e}")

                    result_queue.put((task.text, task.filename, success))

                except Empty:
                    continue
                except Exception as e:
                    print(f"Worker {worker_id}: Unexpected error: {e}")
                    break

        except Exception as e:
            print(f"Worker {worker_id}: Failed to initialize: {e}")

        print(f"Worker {worker_id} finished. Processed {processed_count} images.")
