# Copyright (C) 2021-2025, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import multiprocessing as mp
import random
from dataclasses import dataclass
from queue import Empty

import numpy as np
from PIL import Image

from .background_manager import BackgroundManager
from .config import GenerationConfig
from .font_selector import FontSelector
from .text_renderer import TextRenderer

__all__ = ["TextImageGenerator", "GenerationTask"]


@dataclass
class GenerationTask:
    """Task for the queue

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
    """Generates text overlay images with backgrounds

    Args:
        config (GenerationConfig): Configuration for text image generation
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.font_selector = FontSelector(config.font_dir)
        self.text_renderer = TextRenderer(config)
        self.background_manager = BackgroundManager(config.bg_image_dir)

    def is_text_visible(self, image: Image.Image, alpha_thresh: int = 20, min_visible_ratio: float = 0.02) -> bool:
        """Check if rendered text is sufficiently visible

        Args:
            image (Image.Image): Rendered text image
            alpha_thresh (int): Minimum alpha value to consider pixel visible
            min_visible_ratio (float): Minimum ratio of visible pixels to total pixels

        Returns:
            bool: True if text is sufficiently visible, False otherwise
        """
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        alpha = image.getchannel("A")
        alpha_np = np.array(alpha)
        visible_pixels = np.count_nonzero(alpha_np > alpha_thresh)
        total_pixels = alpha_np.size
        ratio = visible_pixels / total_pixels
        return bool(ratio >= min_visible_ratio)

    def _match_text_to_background(self, text_img: Image.Image, bg_img: Image.Image) -> Image.Image:
        """Match text color to background for better visibility

        Args:
            text_img (Image.Image): Rendered text image
            bg_img (Image.Image): Background image to match against

        Returns:
            Image.Image: Text image with adjusted colors for visibility
        """
        bg_np = np.array(bg_img.convert("RGB"))
        avg_color = np.mean(bg_np.reshape(-1, 3), axis=0)

        text_np = np.array(text_img)
        mask = text_np[:, :, 3] > 0  # alpha mask where text is drawn

        # Slightly darken compared to background
        target_color = avg_color * random.uniform(0.3, 0.6)
        text_np[mask, :3] = target_color

        return Image.fromarray(text_np)

    def generate_image(self, text: str) -> Image.Image | None:
        """Generate a single text overlay image

        Args:
            text (str): Text to render

        Returns:
            Image.Image: Generated image with text overlay, or None if no suitable font/background found
        """
        font_path = self.font_selector.get_font_for_text(text)
        if not font_path:
            return None

        # Try to render visible text
        for _ in range(self.config.max_attempts):
            rendered = self.text_renderer.render_text_to_image(text, font_path)
            if self.is_text_visible(rendered):
                break
        else:
            raise ValueError(f"Failed to render visible text for: {text}")

        bg_crop = self.background_manager.get_background_crop(rendered.size)

        # Smooth overlayed text with background and composite
        bg_crop = bg_crop.convert("RGBA")
        rendered = self._match_text_to_background(rendered, bg_crop)
        return Image.alpha_composite(bg_crop, rendered)

    @staticmethod
    def worker_process(task_queue: mp.Queue, result_queue: mp.Queue, config: GenerationConfig, worker_id: int):
        """Worker process function that processes tasks from the queue

        Args:
            task_queue (mp.Queue): Queue containing tasks to process
            result_queue (mp.Queue): Queue to put results into
            config (GenerationConfig): Configuration for text image generation
            worker_id (int): ID of the worker process
        """
        print(f"Worker {worker_id} starting...")

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
                            img.save(task.save_path, "PNG")
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
                    # Timeout - check if we should continue
                    continue
                except Exception as e:
                    print(f"Worker {worker_id}: Unexpected error: {e}")
                    break

        except Exception as e:
            print(f"Worker {worker_id}: Failed to initialize: {e}")

        print(f"Worker {worker_id} finished. Processed {processed_count} images.")
