# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import multiprocessing as mp
from dataclasses import dataclass
from queue import Empty

import numpy as np
from PIL import Image

from ..components.config import GenerationConfig
from .background_manager import BackgroundManager
from .font_selector import FontSelector
from .text_renderer import TextRenderer, TextStyle
from .text_styling import (
    apply_final_degradations,
    build_final_augmentations,
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

    def _apply_final_degradations(self, image: Image.Image) -> Image.Image:
        """Brightness/contrast jitter + sensor noise + JPEG artifacts on the crop."""
        return apply_final_degradations(self.config, image, self.final_augs)

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

        font_size, bold_width, outline_width = self.text_renderer.sample_style()

        # Render the glyph once as a black coverage map (single getbbox/draw).
        coverage = None
        for _ in range(self.config.max_attempts):
            candidate = self.text_renderer.render_coverage(text, font_path, font_size, bold_width)
            if self.is_text_visible(candidate):
                coverage = candidate
                break
        if coverage is None:
            return None

        bg_crop = self.background_manager.get_background_crop(coverage.size)
        style = self._decide_style(bg_crop, bold_width, outline_width)

        if outline_width > 0 and style.outline_color is not None:
            # Rarer two-tone path: re-render with the styled fill + outline.
            glyph = self.text_renderer.render_text_to_image(text, font_path, style, font_size)
            bg_crop = self.background_manager.get_background_crop(glyph.size)
        else:
            glyph = self._recolor(coverage, style)

        composed = Image.alpha_composite(bg_crop.convert("RGBA"), glyph).convert("RGB")
        return self._apply_final_degradations(composed)

    @staticmethod
    def worker_process(task_queue: mp.Queue, result_queue: mp.Queue, config: GenerationConfig, worker_id: int):
        """Worker process function that processes tasks from the queue."""
        print(f"Worker {worker_id} starting...")

        save_format = "JPEG" if config.output_jpeg else "PNG"
        save_kwargs = {"quality": config.output_jpeg_quality} if config.output_jpeg else {}

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
