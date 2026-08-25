import random

import numpy as np
from PIL import Image

from generator.components import DetectionTask, GenerationConfig, PageGenerator

WORDS = ["test", "hello", "world", "abc", "Datum", "Konto", "number", "value", "page", "text"] * 6


def _cfg(tiny_font_dir, **kw):
    base = dict(
        task="detection",
        font_dir=tiny_font_dir,
        bg_image_dir=None,
        output_dir="ds",
        num_images=1,
        auto_download_fonts=False,
        languages=None,
        det_page_width_range=(400, 400),
        det_page_height_range=(520, 520),
        det_font_size_range=(16, 16),
        det_max_blocks=4,
        det_plain_background_prob=1.0,
        det_heading_prob=0.5,
        det_rotation_prob=0.0,
        noise_prob=0.0,
        jpeg_prob=0.0,
        final_blur_prob=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        supersample=2,
    )
    base.update(kw)
    return GenerationConfig.flat(**base)


def test_generate_page_returns_image_and_polygons(tiny_font_dir):
    page, polygons = PageGenerator(_cfg(tiny_font_dir)).generate_page(WORDS)
    assert isinstance(page, Image.Image)
    assert page.mode == "RGB"
    assert page.size == (400, 520)
    assert len(polygons) > 0
    for poly in polygons:
        assert len(poly) == 4
        assert all(len(point) == 2 for point in poly)


def test_page_fills_most_of_its_height(tiny_font_dir):
    # Regression: pages used to stop after a small word target, leaving the
    # bottom ~2/3 empty. With enough candidate words a dense layout should fill
    # down to near the bottom margin. (Form/id_card layouts are intentionally
    # sparse, so this guard pins the paragraph layout.)
    cfg = _cfg(
        tiny_font_dir,
        det_layout="paragraph",
        det_max_blocks=60,
        det_font_size_range=(14, 14),
        det_heading_prob=0.0,
    )
    page, polygons = PageGenerator(cfg).generate_page(["word", "text", "page", "fill"] * 200)
    height = page.height
    lowest = max(pt[1] for poly in polygons for pt in poly)
    assert lowest > 0.6 * height


def test_polygons_within_page_bounds(tiny_font_dir):
    page, polygons = PageGenerator(_cfg(tiny_font_dir)).generate_page(WORDS)
    w, h = page.size
    for poly in polygons:
        for x, y in poly:
            assert 0 <= x <= w
            assert 0 <= y <= h


def test_polygons_are_doctr_parseable(tiny_font_dir):
    # Mirror doctr.datasets.DetectionDataset.format_polygons: (N, 4, 2) -> boxes.
    _, polygons = PageGenerator(_cfg(tiny_font_dir)).generate_page(WORDS)
    arr = np.asarray(polygons, dtype=np.float32)
    assert arr.ndim == 3
    assert arr.shape[1:] == (4, 2)
    boxes = np.concatenate((arr.min(axis=1), arr.max(axis=1)), axis=1)
    assert boxes.shape == (len(polygons), 4)


def test_rotation_keeps_polygons_in_bounds(tiny_font_dir):
    page, polygons = PageGenerator(_cfg(tiny_font_dir, det_rotation_prob=1.0, det_rotation_range=(6, 6))).generate_page(
        WORDS
    )
    assert len(polygons) > 0
    w, h = page.size
    for poly in polygons:
        for x, y in poly:
            assert -1 <= x <= w + 1
            assert -1 <= y <= h + 1


def test_paper_background_size_and_mode():
    img = PageGenerator._paper_background((123, 77))
    assert img.size == (123, 77)
    assert img.mode == "RGB"


def test_empty_words_yield_no_polygons(tiny_font_dir):
    page, polygons = PageGenerator(_cfg(tiny_font_dir)).generate_page([])
    assert isinstance(page, Image.Image)
    assert polygons == []


def test_detection_task_dataclass():
    t = DetectionTask(words=["a", "b"], save_path="/tmp/x.png", filename="x.png", split="val")
    assert t.words == ["a", "b"]
    assert t.split == "val"


def test_is_rtl_detection():
    assert PageGenerator._is_rtl(["مرحبا", "كتاب", "سلام"]) is True  # Arabic
    assert PageGenerator._is_rtl(["שלום", "עולם"]) is True  # Hebrew
    assert PageGenerator._is_rtl(["hello", "world", "test"]) is False  # Latin
    assert PageGenerator._is_rtl([]) is False


def test_rtl_layout_places_words_from_the_right(tiny_font_dir, monkeypatch):
    # Force RTL so the latin tiny font can still exercise the right-to-left path:
    # in reading order the first word should sit further right than the second.
    pg = PageGenerator(_cfg(tiny_font_dir, det_max_blocks=2))
    monkeypatch.setattr(pg, "_is_rtl", lambda words: True)
    _, polygons = pg.generate_page(["aa", "bb", "cc", "dd", "ee", "ff"])
    assert len(polygons) >= 2
    x_first = polygons[0][0][0]
    x_second = polygons[1][0][0]
    assert x_first > x_second


def test_choose_layout_respects_explicit(tiny_font_dir):
    for layout in ("paragraph", "newspaper", "form", "id_card", "vertical"):
        assert PageGenerator(_cfg(tiny_font_dir, det_layout=layout))._choose_layout() == layout


def test_each_layout_produces_valid_in_bounds_polygons(tiny_font_dir):
    for layout in ("paragraph", "newspaper", "form", "id_card", "vertical"):
        page, polygons = PageGenerator(_cfg(tiny_font_dir, det_layout=layout, det_rotation_prob=0.0)).generate_page(
            WORDS
        )
        assert len(polygons) > 0, layout
        w, h = page.size
        for poly in polygons:
            assert len(poly) == 4
            for x, y in poly:
                assert 0 <= x <= w and 0 <= y <= h, layout


def test_paragraph_fills_vertically_with_few_candidates(tiny_font_dir):
    # Word recycling must fill the page even from a handful of candidates.
    page, polygons = PageGenerator(
        _cfg(
            tiny_font_dir,
            det_layout="paragraph",
            det_page_height_range=(800, 800),
            det_font_size_range=(14, 14),
            det_max_blocks=40,
            det_rotation_prob=0.0,
        )
    ).generate_page(["alpha", "beta", "gamma", "delta"])
    assert len(polygons) > 30
    lowest = max(pt[1] for poly in polygons for pt in poly)
    assert lowest > 0.6 * 800


def test_newspaper_is_dense_and_denser_than_paragraph(tiny_font_dir):
    import random as _r

    base = dict(det_page_width_range=(640, 640), det_page_height_range=(860, 860), det_rotation_prob=0.0)
    _r.seed(0)
    news = PageGenerator(_cfg(tiny_font_dir, det_layout="newspaper", **base)).generate_page(WORDS)[1]
    _r.seed(0)
    para = PageGenerator(_cfg(tiny_font_dir, det_layout="paragraph", **base)).generate_page(WORDS)[1]
    assert len(news) > 200  # genuinely dense newsprint
    assert len(news) > len(para)  # denser than running paragraphs


# -- vertical text --------------------------------------------------------

VERTICAL_WORDS = ["vertical", "spine", "margin", "rotated", "column", "seite", "rand"] * 8


def _aspect_split(polygons):
    """Split polygons into (portrait, landscape) by their aspect ratio."""
    portrait, landscape = [], []
    for poly in polygons:
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        (portrait if (max(ys) - min(ys)) > (max(xs) - min(xs)) else landscape).append(poly)
    return portrait, landscape


def test_render_vertical_word_transposes_the_horizontal_render(tiny_font_dir):
    pg = PageGenerator(_cfg(tiny_font_dir, rotation_prob=0.0, perspective_prob=0.0, blur_prob=0.0))
    flat = pg._render_word("vertical", 18, 0)
    for mode in ("cw", "ccw"):
        rotated = pg._render_vertical_word("vertical", 18, 0, mode)
        assert rotated is not None
        assert (rotated.width, rotated.height) == (flat.height, flat.width)


def test_stacked_word_is_a_tall_column(tiny_font_dir):
    pg = PageGenerator(_cfg(tiny_font_dir, rotation_prob=0.0, perspective_prob=0.0, blur_prob=0.0))
    stacked = pg._render_vertical_word("abcd", 16, 0, "stacked")
    assert stacked is not None
    assert stacked.height > stacked.width
    # Skipped rather than rendered into an unrealistically long column.
    assert pg._render_vertical_word("a" * 40, 16, 0, "stacked") is None
    assert pg._render_vertical_word("", 16, 0, "stacked") is None


def test_vertical_layout_yields_mostly_portrait_polygons(tiny_font_dir):
    _, polygons = PageGenerator(_cfg(tiny_font_dir, det_layout="vertical")).generate_page(VERTICAL_WORDS)
    portrait, landscape = _aspect_split(polygons)
    assert len(polygons) > 10
    assert len(portrait) > len(landscape)  # an optional masthead may stay horizontal


def test_vertical_prob_zero_keeps_pages_horizontal(tiny_font_dir):
    for layout in ("paragraph", "newspaper", "form"):
        _, polygons = PageGenerator(_cfg(tiny_font_dir, det_layout=layout, det_vertical_prob=0.0)).generate_page(
            VERTICAL_WORDS
        )
        portrait, _ = _aspect_split(polygons)
        assert portrait == [], layout


def test_vertical_regions_appear_on_horizontal_pages(tiny_font_dir):
    _, polygons = PageGenerator(
        _cfg(tiny_font_dir, det_layout="paragraph", det_vertical_prob=1.0, det_vertical_stacked_prob=0.0)
    ).generate_page(VERTICAL_WORDS)
    portrait, landscape = _aspect_split(polygons)
    assert portrait, "vertical_prob=1.0 must add a vertical region"
    assert landscape, "the body text must still be laid out horizontally"


def test_vertical_regions_never_overlap_the_body(tiny_font_dir):
    # The strip is reserved before the body is laid out, so the two must occupy
    # disjoint x-ranges - any overlap would mean unreadable, double-boxed text.
    for seed in range(8):
        random.seed(seed)
        _, polygons = PageGenerator(
            _cfg(tiny_font_dir, det_layout="paragraph", det_vertical_prob=1.0, det_vertical_stacked_prob=0.0)
        ).generate_page(VERTICAL_WORDS)
        portrait, landscape = _aspect_split(polygons)
        if not portrait or not landscape:
            continue
        spans = lambda poly: (min(pt[0] for pt in poly), max(pt[0] for pt in poly))
        body = [spans(poly) for poly in landscape]
        for vx0, vx1 in (spans(poly) for poly in portrait):
            for bx0, bx1 in body:
                assert bx1 <= vx0 + 1 or bx0 >= vx1 - 1, seed


def test_vertical_polygons_stay_doctr_parseable(tiny_font_dir):
    _, polygons = PageGenerator(_cfg(tiny_font_dir, det_layout="vertical")).generate_page(VERTICAL_WORDS)
    arr = np.asarray(polygons, dtype=np.float32)
    assert arr.ndim == 3 and arr.shape[1:] == (4, 2)


def test_ccw_columns_read_bottom_to_top(tiny_font_dir):
    pg = PageGenerator(
        _cfg(
            tiny_font_dir,
            det_layout="vertical",
            det_vertical_stacked_prob=0.0,
            det_vertical_ccw_prob=1.0,
        )
    )
    _, polygons = pg.generate_page(VERTICAL_WORDS)
    portrait, _ = _aspect_split(polygons)
    assert len(portrait) >= 2
    # Consecutive words in a counter-clockwise column stack upwards.
    assert portrait[1][0][1] < portrait[0][0][1]
