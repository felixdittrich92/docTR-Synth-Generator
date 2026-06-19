import random

import numpy as np
from PIL import Image

from generator.components import GenerationConfig, TextImageGenerator, TextStyle


def _gen(tiny_font_dir, temp_image_dir, **kw):
    base = dict(
        font_dir=tiny_font_dir,
        bg_image_dir=temp_image_dir,
        output_dir="ds",
        num_images=1,
        auto_download_fonts=False,
        languages=None,
        font_size_range=(12, 14),
        padding=1,
        supersample=2,
        colored_ink_prob=0.0,
        ink_color_jitter=0.0,
        rotation_prob=0.0,
        blur_prob=0.0,
        perspective_prob=0.0,
        pixel_dropout_prob=0.0,
        final_blur_prob=0.0,
        noise_prob=0.0,
        jpeg_prob=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
    )
    base.update(kw)
    return TextImageGenerator(GenerationConfig(**base))


def test_is_text_visible(tiny_font_dir, temp_image_dir):
    gen = _gen(tiny_font_dir, temp_image_dir)
    opaque = Image.new("RGBA", (20, 20), (0, 0, 0, 255))
    transparent = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    assert gen.is_text_visible(opaque) is True
    assert gen.is_text_visible(transparent) is False


def test_decide_style_dark_on_light(tiny_font_dir, temp_image_dir):
    gen = _gen(tiny_font_dir, temp_image_dir)
    random.seed(0)
    np.random.seed(0)
    white = Image.new("RGB", (20, 20), (255, 255, 255))
    style = gen._decide_style(white, bold_width=0, outline_width=0)
    lum = 0.299 * style.fill_color[0] + 0.587 * style.fill_color[1] + 0.114 * style.fill_color[2]
    assert lum < 200  # ink is clearly darker than a white background


def test_decide_style_light_on_dark(tiny_font_dir, temp_image_dir):
    gen = _gen(tiny_font_dir, temp_image_dir)
    random.seed(0)
    np.random.seed(0)
    black = Image.new("RGB", (20, 20), (0, 0, 0))
    style = gen._decide_style(black, bold_width=0, outline_width=0)
    lum = 0.299 * style.fill_color[0] + 0.587 * style.fill_color[1] + 0.114 * style.fill_color[2]
    assert lum > 50  # ink is clearly lighter than a black background


def test_recolor_applies_fill_and_opacity(tiny_font_dir, temp_image_dir):
    gen = _gen(tiny_font_dir, temp_image_dir)
    coverage = Image.new("RGBA", (4, 4), (0, 0, 0, 255))
    style = TextStyle(fill_color=(200, 50, 60), opacity=128)
    out = gen._recolor(coverage, style)
    arr = np.array(out)
    assert tuple(arr[0, 0, :3]) == (200, 50, 60)
    assert arr[0, 0, 3] == 128  # alpha scaled by opacity/255 from 255


def test_generate_image_end_to_end(tiny_font_dir, temp_image_dir):
    gen = _gen(tiny_font_dir, temp_image_dir)
    img = gen.generate_image("Test")
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size[0] > 0 and img.size[1] > 0


def test_generate_image_none_without_font(tiny_font_dir, temp_image_dir):
    gen = _gen(tiny_font_dir, temp_image_dir)
    # Cyrillic is not covered by the tiny font and auto-download is disabled.
    assert gen.generate_image("Привет") is None


def test_generate_image_outline_path(tiny_font_dir, temp_image_dir):
    # outline_prob=1.0 forces the rarer two-tone re-render branch.
    gen = _gen(tiny_font_dir, temp_image_dir, outline_prob=1.0, outline_width_frac_range=(0.05, 0.05))
    img = gen.generate_image("Test")
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
