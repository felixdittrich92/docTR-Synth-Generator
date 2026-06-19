from PIL import Image

from generator.components import GenerationConfig, TextRenderer, TextStyle


def _config(**kw):
    base = dict(
        output_dir="ds",
        num_images=1,
        font_size_range=(20, 20),
        padding=2,
        supersample=2,
        rotation_prob=0.0,
        blur_prob=0.0,
        perspective_prob=0.0,
        pixel_dropout_prob=0.0,
        bold_prob=0.0,
        outline_prob=0.0,
    )
    base.update(kw)
    return GenerationConfig(**base)


def test_font_cache_reuse(tiny_font):
    r = TextRenderer(_config())
    f1 = r._get_font(tiny_font, 30)
    f2 = r._get_font(tiny_font, 30)
    assert f1 is f2  # cached, not re-created
    assert (tiny_font, 30) in r._font_cache


def test_font_cache_eviction(tiny_font):
    r = TextRenderer(_config())
    r._font_cache_size = 3
    for size in range(10, 20):
        r._get_font(tiny_font, size)
    assert len(r._font_cache) <= 3


def test_render_coverage_is_visible_rgba(tiny_font):
    r = TextRenderer(_config())
    img = r.render_coverage("Test", tiny_font, font_size=20, bold_width=0)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGBA"
    alpha = img.getchannel("A")
    assert alpha.getextrema()[1] > 0  # has opaque ink


def test_measure_size_positive(tiny_font):
    r = TextRenderer(_config())
    w, h = r.measure_size("Test", tiny_font, 20)
    assert w > 0 and h > 0


def test_render_styled_with_outline(tiny_font):
    r = TextRenderer(_config())
    style = TextStyle(fill_color=(10, 20, 30), opacity=255, outline_color=(250, 250, 250), outline_width=2)
    img = r.render_text_to_image("Te", tiny_font, style, font_size=20)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGBA"


def test_sample_style_ranges(tiny_font):
    r = TextRenderer(_config(font_size_range=(15, 40), bold_prob=1.0, bold_width_range=(1, 1)))
    font_size, bold_width, outline_width = r.sample_style()
    assert 15 <= font_size <= 40
    assert bold_width == r.supersample  # 1 * supersample
    assert outline_width == 0
