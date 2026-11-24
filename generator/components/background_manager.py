# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
import random
import warnings
from collections import OrderedDict

from PIL import Image

from .designs import design_background

__all__ = ["BackgroundManager"]


class BackgroundManager:
    """Handles background image loading and cropping.

    Decoded backgrounds are kept in a small bounded LRU cache so the same image
    is not re-decoded from disk on every single sample (previously the dominant
    I/O cost, ~40% of runtime). Very large images are downscaled once on load so
    the cache stays memory-light regardless of source resolution.

    Args:
        bg_image_dir (str | None): Directory containing background images.
        cache_size (int): Max number of decoded images held in memory at once.
        max_dimension (int): Longest side a cached image may have; larger images
            are downscaled on load to bound per-image memory.
        procedural_prob (float): Share of backgrounds that are generated
            magazine/brochure/poster designs (see :mod:`.designs`) instead of an
            image. Crops are cut from a small pool of designs that is refreshed
            as it is used, so a design costs one render per ~50 crops.
    """

    DESIGN_POOL = 6
    DESIGN_SIZE = (1200, 1600)
    DESIGN_REFRESH = 0.02

    def __init__(
        self,
        bg_image_dir: str | None = None,
        cache_size: int = 16,
        max_dimension: int = 2000,
        procedural_prob: float = 0.0,
    ):
        self.bg_image_dir = bg_image_dir
        self.procedural_prob = procedural_prob
        self._designs: list[Image.Image] = []
        self.cache_size = max(1, cache_size)
        self.max_dimension = max_dimension
        self.bg_images = self._load_background_images()
        self._cache: OrderedDict[str, Image.Image] = OrderedDict()
        if not self.bg_images and procedural_prob < 1.0:
            warnings.warn(
                f"No background images found in {bg_image_dir}. Using a blank background instead.", UserWarning
            )

    def _load_background_images(self) -> list[str]:
        """Load all background image paths."""
        return (
            [
                os.path.join(self.bg_image_dir, f)
                for f in os.listdir(self.bg_image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
            ]
            if self.bg_image_dir
            else []
        )

    def _get_image(self, path: str) -> Image.Image | None:
        """Return a decoded RGB background from cache, loading it on miss."""
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached
        try:
            with Image.open(path) as bg:
                bg = bg.convert("RGB")
                longest = max(bg.size)
                if self.max_dimension and longest > self.max_dimension:
                    scale = self.max_dimension / longest
                    bg = bg.resize((max(1, int(bg.width * scale)), max(1, int(bg.height * scale))), Image.LANCZOS)  # type: ignore
                bg.load()
        except Exception as e:
            print(f"Error loading background {path}: {e}")
            return None
        self._cache[path] = bg
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return bg

    def get_background_crop(self, size: tuple[int, int]) -> Image.Image:
        """Get a random background crop of the specified size.

        Args:
            size (tuple[int, int]): Desired crop size (width, height)

        Returns:
            Image.Image: Cropped background image (RGB).
        """
        if random.random() < self.procedural_prob:
            return self._design_crop(size)
        if not self.bg_images:
            return Image.new("RGB", size, (255, 255, 255))

        crop_width, crop_height = size
        for _ in range(10):
            bg = self._get_image(random.choice(self.bg_images))
            if bg is None:
                continue
            bg_width, bg_height = bg.size
            if crop_width <= bg_width and crop_height <= bg_height:
                x = random.randint(0, bg_width - crop_width)
                y = random.randint(0, bg_height - crop_height)
                return bg.crop((x, y, x + crop_width, y + crop_height))

        return Image.new("RGB", size, (255, 255, 255))

    def _design_crop(self, size: tuple[int, int]) -> Image.Image:
        """A crop of a generated design; the pool is filled lazily and refreshed as it is used."""
        if len(self._designs) < self.DESIGN_POOL:
            self._designs.append(design_background(self.DESIGN_SIZE))
        elif random.random() < self.DESIGN_REFRESH:
            self._designs[random.randrange(self.DESIGN_POOL)] = design_background(self.DESIGN_SIZE)
        design = random.choice(self._designs)
        w, h = min(size[0], design.width), min(size[1], design.height)
        x, y = random.randint(0, design.width - w), random.randint(0, design.height - h)
        crop = design.crop((x, y, x + w, y + h))
        return crop if crop.size == size else crop.resize(size, Image.Resampling.BICUBIC)

    def get_page_background(self, size: tuple[int, int]) -> Image.Image:
        """Get a full-page background of the requested size.

        Unlike :meth:`get_background_crop` (which only crops and falls back to a
        blank when the source is too small), this always fills the whole page:
        a large enough image is randomly cropped, a smaller one is resized to
        cover the page, and with no images a subtly tinted "paper" is returned.

        Args:
            size (tuple[int, int]): Desired page size (width, height).

        Returns:
            Image.Image: RGB page-sized background.
        """
        width, height = size
        if random.random() < self.procedural_prob:
            return design_background(size)
        if self.bg_images:
            for _ in range(10):
                bg = self._get_image(random.choice(self.bg_images))
                if bg is None:
                    continue
                bw, bh = bg.size
                if bw >= width and bh >= height:
                    x = random.randint(0, bw - width)
                    y = random.randint(0, bh - height)
                    return bg.crop((x, y, x + width, y + height))
                scale = max(width / bw, height / bh)
                rw, rh = max(width, int(bw * scale) + 1), max(height, int(bh * scale) + 1)
                resized = bg.resize((rw, rh), Image.LANCZOS)  # type: ignore[attr-defined]
                x = random.randint(0, rw - width)
                y = random.randint(0, rh - height)
                return resized.crop((x, y, x + width, y + height))

        tone = random.randint(238, 252)
        return Image.new("RGB", size, (tone, tone, max(0, tone - 3)))
