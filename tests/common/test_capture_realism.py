import random

import numpy as np
from PIL import Image, ImageDraw

from generator.components import GenerationConfig
from generator.components.capture import _curl
from generator.components.generator import TextImageGenerator
from generator.components.scan import apply_scan_tone


def _page_with_words(ink=(20, 20, 20), paper=(245, 245, 240)):
    page = Image.new("RGB", (400, 300), paper)
    draw = ImageDraw.Draw(page)
    polygons = []
    for row in range(5):
        for col in range(4):
            x0, y0 = 20 + col * 95, 30 + row * 50
            # a "word": a few strokes, so the box holds ink and paper
            for k in range(5):
                draw.rectangle((x0 + k * 14, y0, x0 + k * 14 + 6, y0 + 20), fill=ink)
            polygons.append([[x0, y0], [x0 + 62, y0], [x0 + 62, y0 + 20], [x0, y0 + 20]])
    return page, polygons


def _is_bitonal(image):
    values = np.unique(np.asarray(image.convert("L")))
    return set(values.tolist()) <= {0, 255}


def test_black_and_white_keeps_every_word():
    cfg = GenerationConfig.flat(bitonal_prob=1.0, grayscale_prob=0.0)
    page, polygons = _page_with_words()
    random.seed(0)
    out = apply_scan_tone(cfg, page, polygons)
    assert _is_bitonal(out)
    arr = np.asarray(out.convert("L")) < 128
    for poly in polygons:
        (x0, y0), (x1, y1) = poly[0], poly[2]
        assert arr[y0:y1, x0:x1].mean() > 0.2  # ink survived thresholding


def test_black_and_white_falls_back_when_it_would_erase_text():
    cfg = GenerationConfig.flat(bitonal_prob=1.0, grayscale_prob=0.0)
    page, polygons = _page_with_words(ink=(242, 242, 242), paper=(245, 245, 245))  # nearly invisible ink
    random.seed(0)
    out = apply_scan_tone(cfg, page, polygons)
    assert not _is_bitonal(out)  # delivered in grayscale instead of blank boxes
    assert out.mode == "RGB"


def test_curl_keeps_polygons_on_their_ink():
    page, polygons = _page_with_words()
    random.seed(3)
    bent, moved = _curl(page.convert("RGBA"), polygons, 0.03)
    assert bent.height > page.height  # room for the displacement
    arr = np.asarray(bent.convert("L"))
    for poly in moved:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        box = arr[int(min(ys)) + 2 : int(max(ys)) - 1, int(min(xs)) + 1 : int(max(xs))]
        assert (box < 100).mean() > 0.3  # the box still sits on the word's strokes


def test_crop_context_stays_out_of_the_word_band(tiny_font_dir, temp_image_dir):
    cfg = GenerationConfig.flat(task="recognition", font_dir=tiny_font_dir, bg_image_dir=temp_image_dir)
    gen = TextImageGenerator(cfg)
    style = gen._decide_style(Image.new("RGB", (40, 20), (240, 240, 240)), 0, 0)
    font = next(iter(gen.font_selector.font_support_table))
    random.seed(1)
    for _ in range(30):
        layer = gen._context((120, 60), (10, 18, 110, 42), "word", font, 20, style)
        if layer is None:
            continue
        alpha = np.asarray(layer.getchannel("A"))
        assert not alpha[18:42].any()  # nothing between the word's top and bottom


def test_small_text_is_never_thresholded():
    cfg = GenerationConfig.flat(bitonal_prob=1.0, grayscale_prob=0.0)
    page, polygons = _page_with_words()
    out = apply_scan_tone(cfg, page, polygons, text_height=9.0)
    assert not _is_bitonal(out)  # grayscale instead of glyphs broken into blobs
