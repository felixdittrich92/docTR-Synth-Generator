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
        self.background_manager = BackgroundManager(
            config.resources.bg_image_dir,
            cache_size=config.resources.bg_cache_size,
            max_dimension=config.resources.bg_max_dimension,
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

    def generate_image(self, text: str) -> Image.Image | None:
        """Generate a single text overlay image.

        Args:
            text (str): Text to render

        Returns:
            Image.Image | None: RGB image with realistic text overlay, or None if
            no suitable font/background was found.
        """
        font_path = self.font_selector.get_font_for_text(text)
        if not font_path:
            return None

        # Background and ink are re-drawn until the finished crop is readable along
        # its whole length: a dark band or busy patch of paper under part of the
        # word makes a label the image no longer supports. A second round re-rolls
        # size and weight too, since a hairline at 12px may never clear the bar.
        min_contrast = self.config.realism.min_local_contrast
        for _ in range(2 if min_contrast > 0 else 1):
            font_size, bold_width, outline_width = self.text_renderer.sample_style()
            # Render the glyph once as a black coverage map (single getbbox/draw).
            coverage = None
            for _ in range(self.config.core.max_attempts):
                candidate = self.text_renderer.render_coverage(text, font_path, font_size, bold_width)
                if self.is_text_visible(candidate):
                    coverage = candidate
                    break
            if coverage is None:
                continue
            for _ in range(max(1, self.config.core.max_attempts)):
                image, alpha = self._compose(text, font_path, font_size, coverage, outline_width, bold_width)
                if min_contrast <= 0 or local_ink_contrast(image, alpha) >= min_contrast:
                    return image
        return None

    def _margins(self, text_height: int) -> tuple[int, int, int, int]:
        """Extra (left, top, right, bottom) margin, like an expanded detector box."""
        rec = self.config.recognition
        if random.random() >= rec.margin_prob:
            return 0, 0, 0, 0
        lo, hi = rec.margin_range
        return tuple(int(round(text_height * random.uniform(lo, hi))) for _ in range(4))  # type: ignore[return-value]

    def _context(
        self, size: tuple[int, int], box: tuple[int, int, int, int], text: str, font_path: str, font_size: int, style
    ) -> Image.Image | None:
        """What a real crop picks up around a word: neighbouring lines and rules, in its margins only.

        ``box`` is where the word sits. Line fragments are placed so they leave the
        crop through its top or bottom edge - only the descenders of the line above
        or the tops of the line below show, never a readable character. They use the
        word's own font and ink, so they are rendered (and shaped) like the word.
        """
        width, height = size
        x0, y0, x1, y1 = box
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        placed = False
        for edge, room in (("top", y0), ("bottom", height - y1)):
            if room < 3 or random.random() >= 0.6:
                continue
            if random.random() < 0.3:  # an underline or a table rule
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
                text if random.random() < 0.5 else f"{text} {text}", font_path, font_size, style.bold_width
            )
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
        self, text: str, font_path: str, font_size: int, coverage: Image.Image, outline_width: int, bold_width: int
    ) -> tuple[Image.Image, Image.Image]:
        """Background + ink (+ page context) + final degradations for one rendered glyph map."""
        left, top, right, bottom = self._margins(coverage.height)
        size = (coverage.width + left + right, coverage.height + top + bottom)
        bg_crop = self.background_manager.get_background_crop(size)
        # Calm the texture *before* choosing ink, so the decision is made against
        # the background the glyph will actually sit on.
        bg_crop = calm_texture(bg_crop, self.config.realism.crop_texture_std)
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
            context = self._context(
                size, (left, top, left + glyph.width, top + glyph.height), text, font_path, font_size, style
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
