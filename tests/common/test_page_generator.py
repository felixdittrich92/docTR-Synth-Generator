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
    return GenerationConfig(**base)


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
    for layout in ("paragraph", "newspaper", "form", "id_card"):
        assert PageGenerator(_cfg(tiny_font_dir, det_layout=layout))._choose_layout() == layout


def test_each_layout_produces_valid_in_bounds_polygons(tiny_font_dir):
    for layout in ("paragraph", "newspaper", "form", "id_card"):
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
