# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import random

import numpy as np
from PIL import Image, ImageEnhance

from ..augmentations import AugmentationPipeline, RandomBlur, RandomGaussianNoise, RandomJpegCompression
from .config import GenerationConfig
from .text_renderer import TextStyle

__all__ = [
    "luminance",
    "decide_text_style",
    "recolor_coverage",
    "build_final_augmentations",
    "apply_final_degradations",
]


def luminance(rgb) -> float:
    """Return the perceptual luminance (0-255) of an RGB triple."""
    return float(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])


def decide_text_style(
    config: GenerationConfig,
    bg_img: Image.Image,
    bold_width: int = 0,
    outline_width: int = 0,
) -> TextStyle:
    """Decide ink colour, polarity and contrast from the background.

    Supports dark-on-light *and* light-on-dark text across a controllable
    contrast range, with neutral or colourful ink - the variety real captures
    contain.
    """
    bg_np = np.array(bg_img.convert("RGB"), dtype=np.float32).reshape(-1, 3)
    avg = bg_np.mean(axis=0)
    bg_lum = luminance(avg)

    if bg_lum > 150:
        dark_text = True
    elif bg_lum < 105:
        dark_text = False
    else:
        dark_text = bg_lum >= 128
        if random.random() < config.realism.invert_prob:
            dark_text = not dark_text

    contrast = random.uniform(config.realism.min_contrast, config.realism.max_contrast)
    if dark_text:
        base = avg * (1.0 - contrast)
    else:
        base = avg + (255.0 - avg) * contrast

    if random.random() < config.realism.colored_ink_prob:
        rnd = np.random.uniform(20, 235, size=3)
        rnd_lum = luminance(rnd)
        target_lum = max(8.0, luminance(base))
        rnd = rnd * (target_lum / rnd_lum) if rnd_lum > 0 else base
        ink = rnd
    else:
        ink = base.copy()

    ink = ink + np.random.normal(0.0, config.realism.ink_color_jitter, size=3)
    fill_color = tuple(int(np.clip(c, 0, 255)) for c in ink)

    outline_color = None
    if outline_width > 0:
        outline_color = (245, 245, 245) if dark_text else (15, 15, 15)

    opacity = random.randint(*config.realism.text_opacity_range)
    return TextStyle(
        fill_color=fill_color,
        opacity=opacity,
        bold_width=bold_width,
        outline_color=outline_color,
        outline_width=outline_width,
    )


def recolor_coverage(coverage: Image.Image, style: TextStyle) -> Image.Image:
    """Recolour a black coverage glyph to the ink colour, reusing its alpha."""
    arr = np.array(coverage, dtype=np.uint8)
    arr[:, :, 0] = style.fill_color[0]
    arr[:, :, 1] = style.fill_color[1]
    arr[:, :, 2] = style.fill_color[2]
    if style.opacity < 255:
        arr[:, :, 3] = (arr[:, :, 3].astype(np.uint16) * style.opacity // 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def build_final_augmentations(config: GenerationConfig) -> AugmentationPipeline:
    """Image-space degradations applied after compositing (noise, JPEG, blur)."""
    return AugmentationPipeline([
        RandomBlur(radius_range=config.realism.final_blur_radius_range, prob=config.realism.final_blur_prob),
        RandomGaussianNoise(std_range=config.realism.noise_std_range, prob=config.realism.noise_prob),
        RandomJpegCompression(quality_range=config.realism.jpeg_quality_range, prob=config.realism.jpeg_prob),
    ])


def apply_final_degradations(
    config: GenerationConfig,
    image: Image.Image,
    final_augs: AugmentationPipeline,
) -> Image.Image:
    """Brightness/contrast jitter + sensor noise + JPEG artifacts on the image."""
    if config.realism.brightness_jitter > 0:
        factor = 1.0 + random.uniform(-config.realism.brightness_jitter, config.realism.brightness_jitter)
        image = ImageEnhance.Brightness(image).enhance(factor)
    if config.realism.contrast_jitter > 0:
        factor = 1.0 + random.uniform(-config.realism.contrast_jitter, config.realism.contrast_jitter)
        image = ImageEnhance.Contrast(image).enhance(factor)
    return final_augs(image)
