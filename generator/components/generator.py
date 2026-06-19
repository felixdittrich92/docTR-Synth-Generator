# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import multiprocessing as mp
import random
from dataclasses import dataclass
from queue import Empty

import numpy as np
from PIL import Image, ImageEnhance

from ..augmentations import AugmentationPipeline, RandomBlur, RandomGaussianNoise, RandomJpegCompression
from .background_manager import BackgroundManager
from .config import GenerationConfig
from .font_selector import FontSelector
from .text_renderer import TextRenderer, TextStyle

__all__ = ["TextImageGenerator", "GenerationTask"]


def _luminance(rgb) -> float:
    return float(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])


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
        self.final_augs = AugmentationPipeline([
            RandomBlur(radius_range=config.final_blur_radius_range, prob=config.final_blur_prob),
            RandomGaussianNoise(std_range=config.noise_std_range, prob=config.noise_prob),
            RandomJpegCompression(quality_range=config.jpeg_quality_range, prob=config.jpeg_prob),
        ])

    def is_text_visible(self, image: Image.Image, alpha_thresh: int = 20, min_visible_ratio: float = 0.02) -> bool:
        """Check if rendered text is sufficiently visible."""
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        alpha_np = np.array(image.getchannel("A"))
        ratio = np.count_nonzero(alpha_np > alpha_thresh) / alpha_np.size
        return bool(ratio >= min_visible_ratio)

    def _decide_style(self, bg_img: Image.Image, bold_width: int, outline_width: int) -> TextStyle:
        """Decide ink colour, polarity and contrast from the background.

        Unlike the previous logic - which always produced a darkened tint of the
        background - this can render dark-on-light *and* light-on-dark text,
        across a controllable (and often deliberately low) contrast range, with
        neutral or colourful ink. That variety is what real captures contain.
        """
        cfg = self.config
        bg_np = np.array(bg_img.convert("RGB"), dtype=np.float32).reshape(-1, 3)
        avg = bg_np.mean(axis=0)
        bg_lum = _luminance(avg)

        # Polarity: keep contrast by inking away from the background luminance,
        # with an occasional (legible) inversion on mid-tone backgrounds.
        if bg_lum > 150:
            dark_text = True
        elif bg_lum < 105:
            dark_text = False
        else:
            dark_text = bg_lum >= 128
            if random.random() < cfg.invert_prob:
                dark_text = not dark_text

        contrast = random.uniform(cfg.min_contrast, cfg.max_contrast)
        if dark_text:
            base = avg * (1.0 - contrast)
        else:
            base = avg + (255.0 - avg) * contrast

        if random.random() < cfg.colored_ink_prob:
            # Random hue, rescaled so its luminance matches the contrast target.
            rnd = np.random.uniform(20, 235, size=3)
            rnd_lum = _luminance(rnd)
            target_lum = max(8.0, _luminance(base))
            rnd = rnd * (target_lum / rnd_lum) if rnd_lum > 0 else base
            ink = rnd
        else:
            ink = base.copy()

        ink = ink + np.random.normal(0.0, cfg.ink_color_jitter, size=3)
        fill_color = tuple(int(np.clip(c, 0, 255)) for c in ink)

        outline_color = None
        if outline_width > 0:
            # Outline contrasts the fill (light fill -> dark outline and vice versa).
            outline_color = (245, 245, 245) if dark_text else (15, 15, 15)

        opacity = random.randint(*cfg.text_opacity_range)
        return TextStyle(
            fill_color=fill_color,
            opacity=opacity,
            bold_width=bold_width,
            outline_color=outline_color,
            outline_width=outline_width,
        )

    def _recolor(self, coverage: Image.Image, style: TextStyle) -> Image.Image:
        """Recolour a black coverage glyph to the ink colour, in place.

        Much cheaper than re-rendering: the anti-aliased alpha channel is reused
        and only RGB (and optionally alpha, for faded ink) are rewritten.
        """
        arr = np.array(coverage, dtype=np.uint8)
        arr[:, :, 0] = style.fill_color[0]
        arr[:, :, 1] = style.fill_color[1]
        arr[:, :, 2] = style.fill_color[2]
        if style.opacity < 255:
            arr[:, :, 3] = (arr[:, :, 3].astype(np.uint16) * style.opacity // 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGBA")

    def _apply_final_degradations(self, image: Image.Image) -> Image.Image:
        """Brightness/contrast jitter + sensor noise + JPEG artifacts on the crop."""
        cfg = self.config
        if cfg.brightness_jitter > 0:
            factor = 1.0 + random.uniform(-cfg.brightness_jitter, cfg.brightness_jitter)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if cfg.contrast_jitter > 0:
            factor = 1.0 + random.uniform(-cfg.contrast_jitter, cfg.contrast_jitter)
            image = ImageEnhance.Contrast(image).enhance(factor)
        return self.final_augs(image)

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
