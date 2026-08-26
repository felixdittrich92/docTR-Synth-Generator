import random
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from generator.components import DetectionTask, GenerationConfig, PageGenerator
from generator.components.legibility import DegradationBudget
from generator.components.media import PageMedia, apply_delivery_resample
from generator.components.text_styling import build_budgeted_augmentations

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
        capture_prob=0.0,
        det_bleed_through_prob=0.0,
        # Pixel mode: these tests assert exact page sizes. Media mode is
        # exercised separately below.
        media_enabled=False,
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
    # Uses _render_stacked_word directly: "abcd" is Latin, so the public
    # _render_vertical_word would (correctly) rotate it instead of stacking.
    pg = PageGenerator(_cfg(tiny_font_dir, rotation_prob=0.0, perspective_prob=0.0, blur_prob=0.0))
    stacked = pg._render_stacked_word("abcd", 16, 0)
    assert stacked is not None
    assert stacked.height > stacked.width
    # Skipped rather than rendered into an unrealistically long column.
    assert pg._render_stacked_word("a" * 40, 16, 0) is None
    assert pg._render_stacked_word("", 16, 0) is None


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
    # Seeded and repeated: a strip narrower than legible type is skipped by
    # design, so a single unseeded page is not a fair test.
    generator = PageGenerator(
        _cfg(tiny_font_dir, det_layout="paragraph", det_vertical_prob=1.0, det_vertical_stacked_prob=0.0)
    )
    with_strip = 0
    for seed in range(8):
        random.seed(seed)
        _, polygons = generator.generate_page(VERTICAL_WORDS)
        portrait, landscape = _aspect_split(polygons)
        assert landscape, "the body text must still be laid out horizontally"
        with_strip += bool(portrait)
    assert with_strip >= 4, f"vertical_prob=1.0 produced a strip on only {with_strip}/8 pages"


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


# -- realism: page coherence, bleed-through, capture ----------------------


def test_page_pins_one_font_per_role(tiny_font_dir):
    pg = PageGenerator(_cfg(tiny_font_dir))
    pg._begin_page()
    pg._page_fonts.pop("__free__", None)  # force the coherent path
    first = pg._font_for("hello", "body")
    assert first is not None
    for word in ("world", "another", "word"):
        assert pg._font_for(word, "body") == first


def test_font_coherence_zero_restores_per_word_resolution(tiny_font_dir):
    pg = PageGenerator(_cfg(tiny_font_dir, det_page_font_coherence=0.0))
    pg._begin_page()
    assert "__free__" in pg._page_fonts
    assert pg._font_for("hello") is not None


def test_page_palette_keeps_blocks_in_one_ink(tiny_font_dir):
    # With deviation disabled every block must land on the same ink colour.
    pg = PageGenerator(_cfg(tiny_font_dir, det_ink_deviation_prob=0.0, colored_ink_prob=1.0, ink_color_jitter=0.0))
    pg._begin_page()
    page = Image.new("RGB", (60, 30), (250, 250, 250))
    colors = {pg._style_at(page, 0, 0, 60, 30, 0).fill_color for _ in range(12)}
    assert len(colors) == 1  # ink_color_jitter is what varies blocks, not a fresh roll


def test_ink_deviation_one_restores_per_block_rolls(tiny_font_dir):
    pg = PageGenerator(_cfg(tiny_font_dir, det_ink_deviation_prob=1.0, colored_ink_prob=1.0))
    pg._begin_page()
    page = Image.new("RGB", (60, 30), (250, 250, 250))
    colors = {pg._style_at(page, 0, 0, 60, 30, 0).fill_color for _ in range(12)}
    assert len(colors) > 1


def test_bleed_through_darkens_the_page_without_adding_labels(tiny_font_dir):
    pg = PageGenerator(_cfg(tiny_font_dir, det_bleed_through_alpha_range=(0.3, 0.3)))
    pg._begin_page()
    page = Image.new("RGBA", (300, 400), (250, 250, 250, 255))
    before = np.asarray(page.convert("RGB"), dtype=np.float32).mean()
    polygons = []
    take = pg._word_supplier(["bleed", "through", "reverse", "sheet"])
    pg._add_bleed_through(page, (10, 10, 290, 390), take, False)
    after = np.asarray(page.convert("RGB"), dtype=np.float32).mean()
    assert after < before  # show-through is visible
    assert polygons == []  # ... but never labelled


def test_bleed_through_is_faint_relative_to_real_ink(tiny_font_dir):
    # It must stay well below the real text, or the detector should box it.
    pg = PageGenerator(_cfg(tiny_font_dir, det_bleed_through_alpha_range=(0.1, 0.1)))
    pg._begin_page()
    take = pg._word_supplier(["bleed", "through", "reverse", "sheet"])

    bled = Image.new("RGBA", (300, 400), (250, 250, 250, 255))
    pg._add_bleed_through(bled, (10, 10, 290, 390), take, False)
    inked = Image.new("RGBA", (300, 400), (250, 250, 250, 255))
    style = pg._style_at(inked, 0, 0, 300, 400, 0)
    pg._fill_box(inked, (10, 10, 290, 390), take, [], 16, style, 0, False)

    drop = lambda img: 250.0 - np.asarray(img.convert("RGB"), dtype=np.float32).mean()
    assert drop(bled) < drop(inked) / 2


def test_capture_enlarges_the_frame_and_moves_polygons(tiny_font_dir):
    # Built from sub-configs rather than flat(), so nothing counts as an
    # explicitly pinned page size - pinning one deliberately disables capture.
    cfg = _cfg(tiny_font_dir, capture_prob=1.0)
    cfg.explicit_options = frozenset()
    page, polygons = PageGenerator(cfg).generate_page(WORDS)
    assert len(polygons) > 0
    assert page.size != (400, 520)  # the sheet is now an object in a scene
    assert page.width > 400 and page.height > 520
    for poly in polygons:
        assert len(poly) == 4
        for x, y in poly:
            assert 0 <= x <= page.width and 0 <= y <= page.height


def test_capture_polygons_track_the_homography(tiny_font_dir):
    # A perspective warp must keep boxes on their words: the captured polygon
    # area should stay in the same ballpark, not collapse or explode.
    random.seed(2)
    flat_page, flat_polys = PageGenerator(_cfg(tiny_font_dir, capture_prob=0.0)).generate_page(WORDS)
    random.seed(2)
    shot, shot_polys = PageGenerator(_cfg(tiny_font_dir, capture_prob=1.0)).generate_page(WORDS)
    assert len(shot_polys) == len(flat_polys)

    def area(poly):
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    flat_area = sum(area(p) for p in flat_polys) / len(flat_polys)
    shot_area = sum(area(p) for p in shot_polys) / len(shot_polys)
    assert 0.4 < shot_area / flat_area < 2.5


def test_capture_disabled_leaves_the_page_flat(tiny_font_dir):
    page, _ = PageGenerator(_cfg(tiny_font_dir, capture_prob=0.0)).generate_page(WORDS)
    assert page.size == (400, 520)


# -- layouts, furniture and content statistics ----------------------------


def test_table_and_receipt_layouts_produce_valid_polygons(tiny_font_dir):
    for layout in ("table", "receipt"):
        page, polygons = PageGenerator(_cfg(tiny_font_dir, det_layout=layout)).generate_page(WORDS)
        assert len(polygons) > 5, layout
        for poly in polygons:
            assert len(poly) == 4
            for x, y in poly:
                assert 0 <= x <= page.width and 0 <= y <= page.height, layout


def test_receipt_overrides_the_page_geometry(tiny_font_dir):
    # A thermal roll is far narrower than any det_page_*_range can produce.
    page, _ = PageGenerator(
        _cfg(
            tiny_font_dir,
            det_layout="receipt",
            det_receipt_width_range=(320, 320),
            det_receipt_height_range=(1200, 1200),
        )
    ).generate_page(WORDS)
    assert page.size == (320, 1200)
    flat, _ = PageGenerator(_cfg(tiny_font_dir, det_layout="paragraph")).generate_page(WORDS)
    assert flat.size == (400, 520)  # other layouts keep page_*_range


def test_table_right_aligns_its_numeric_columns(tiny_font_dir):
    page, polygons = PageGenerator(
        _cfg(tiny_font_dir, det_layout="table", det_page_width_range=(700, 700), det_table_columns_range=(3, 3))
    ).generate_page(WORDS)
    right_edges = [max(pt[0] for pt in poly) for poly in polygons]
    # Right-aligned cells share an edge, so the same x recurs across rows.
    common = Counter(round(edge) for edge in right_edges).most_common(1)[0][1]
    assert common >= 3


def test_numeric_cells_look_like_ledger_values(tiny_font_dir):
    values = [PageGenerator._numeric_cell() for _ in range(200)]
    assert all(any(ch.isdigit() for ch in v) for v in values)
    assert any("%" in v for v in values)
    assert any("." in v for v in values)


def test_redaction_removes_the_labels_it_covers(tiny_font_dir):
    box = (10, 10, 100, 40)
    covered = [[10, 12], [60, 12], [60, 30], [10, 30]]
    outside = [[200, 200], [260, 200], [260, 220], [200, 220]]
    assert PageGenerator._mostly_inside(covered, box) is True
    assert PageGenerator._mostly_inside(outside, box) is False


def test_clip_polygons_trims_at_the_edge_and_drops_slivers(tiny_font_dir):
    inside = [[10, 10], [50, 10], [50, 30], [10, 30]]
    straddling = [[-20, 10], [40, 10], [40, 30], [-20, 30]]
    outside = [[-80, 10], [-40, 10], [-40, 30], [-80, 30]]
    sliver = [[-40, 10], [1, 10], [1, 30], [-40, 30]]
    kept = PageGenerator._clip_polygons([inside, straddling, outside, sliver], 400, 520)
    assert len(kept) == 2
    assert kept[1][0][0] == 0  # clamped to the page edge, not dropped
    assert all(0 <= x <= 400 and 0 <= y <= 520 for poly in kept for x, y in poly)


def test_token_sampler_clusters_short_words(tiny_font_dir):
    from generator.components.token_sampler import TokenSampler

    words = ["a", "the", "of", "in"] + ["consideration", "developments", "unternehmen"] * 8
    biased = TokenSampler(words, function_word_ratio=0.45, punctuation_prob=0.0)
    uniform = TokenSampler(words, function_word_ratio=0.0, punctuation_prob=0.0)
    short_biased = sum(len(biased.take()) <= 4 for _ in range(600))
    short_uniform = sum(len(uniform.take()) <= 4 for _ in range(600))
    assert short_biased > short_uniform * 2


def test_token_sampler_attaches_punctuation(tiny_font_dir):
    from generator.components.token_sampler import TokenSampler

    sampler = TokenSampler(["word"], function_word_ratio=0.0, punctuation_prob=1.0)
    tokens = {sampler.take() for _ in range(200)}
    assert all(token != "word" for token in tokens)
    assert any(token.endswith((",", ".", ")")) for token in tokens)
    assert any(token.startswith(("(", '"', "'", "-")) for token in tokens)


def test_punctuation_zero_restores_bare_words(tiny_font_dir):
    from generator.components.token_sampler import TokenSampler

    sampler = TokenSampler(["word", "text"], punctuation_prob=0.0)
    assert {sampler.take() for _ in range(100)} <= {"word", "text"}


# -- legibility -----------------------------------------------------------


def _readability(img, polygons):
    """Edge sharpness inside word boxes, normalised by local ink/paper contrast.

    ~0.7 is crisp type; below ~0.3 the strokes have merged into a smudge.
    """
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(arr)
    grad = np.hypot(gx, gy)
    scores = []
    for poly in polygons:
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        x0, x1, y0, y1 = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
        if y1 - y0 < 2 or x1 - x0 < 3:
            continue
        crop, gcrop = arr[y0:y1, x0:x1], grad[y0:y1, x0:x1]
        contrast = np.percentile(crop, 90) - np.percentile(crop, 10)
        scores.append(0.0 if contrast < 8 else float(np.percentile(gcrop, 99) / contrast))
    return scores


def test_layouts_never_render_below_the_configured_font_floor(tiny_font_dir):
    # font_size_range[0] is a promise: a layout that cannot fit type at that
    # size must change its geometry, not silently shrink the text.
    seen = []
    generator = PageGenerator(_cfg(tiny_font_dir, det_font_size_range=(14, 20), det_furniture_prob=1.0))
    original = generator._render_word

    def spy(word, font_size, bold_width, role="body"):
        seen.append(font_size)
        return original(word, font_size, bold_width, role)

    generator._render_word = spy
    # newspaper is excluded on purpose: it has its own newspaper_font_size_range,
    # which is a deliberate, separately configured choice.
    for layout in ("paragraph", "form", "id_card", "vertical", "table", "receipt"):
        generator.config.detection.layout = layout
        generator.generate_page(WORDS)
    assert seen and min(seen) >= 14, f"smallest rendered font was {min(seen)}"


def test_newspaper_respects_its_own_font_range(tiny_font_dir):
    seen = []
    generator = PageGenerator(_cfg(tiny_font_dir, det_layout="newspaper", det_newspaper_font_size_range=(11, 15)))
    original = generator._render_word

    def spy(word, font_size, bold_width, role="body"):
        seen.append(font_size)
        return original(word, font_size, bold_width, role)

    generator._render_word = spy
    generator.generate_page(WORDS)
    assert seen and min(seen) >= 11


def test_text_stays_readable_across_layouts(tiny_font_dir):
    for layout in ("paragraph", "table", "receipt", "form", "vertical"):
        scores = []
        for seed in range(4):
            random.seed(seed)
            page, polygons = PageGenerator(
                _cfg(tiny_font_dir, det_layout=layout, capture_prob=0.0, final_blur_prob=1.0, jpeg_prob=1.0)
            ).generate_page(WORDS)
            scores += _readability(page, polygons)
        assert scores, layout
        assert np.median(scores) > 0.35, f"{layout}: median sharpness {np.median(scores):.2f}"


def test_budget_tightens_degradations_for_small_text():
    small = DegradationBudget.for_text_height(9)
    large = DegradationBudget.for_text_height(30)
    assert small.max_blur_radius < large.max_blur_radius
    assert small.max_motion_length < large.max_motion_length
    assert small.min_jpeg_quality > large.min_jpeg_quality
    assert small.max_noise_std < large.max_noise_std


def test_budget_clamps_the_final_blur_range(tiny_font_dir):
    cfg = _cfg(tiny_font_dir, final_blur_radius_range=(3.0, 4.0), final_blur_prob=1.0)
    pipeline = build_budgeted_augmentations(cfg, DegradationBudget.for_text_height(12))
    assert pipeline.augmentations[0].radius_range[1] <= 1.0  # 12 / 12


def test_capture_motion_blur_respects_the_budget(tiny_font_dir):
    # A long smear on small type is what turned captured pages into mush.
    scores = []
    for seed in range(4):
        random.seed(seed)
        page, polygons = PageGenerator(
            _cfg(
                tiny_font_dir, capture_prob=1.0, capture_motion_blur_prob=1.0, capture_motion_blur_length_range=(14, 18)
            )
        ).generate_page(WORDS)
        scores += _readability(page, polygons)
    assert np.median(scores) > 0.30, f"median sharpness {np.median(scores):.2f}"


def test_glare_lifts_toward_white_without_erasing_ink(tiny_font_dir):
    # Multiplying and clipping sent paper past 255 while ink stayed put, which
    # silently wiped out any text the highlight covered.
    scores = []
    for seed in range(4):
        random.seed(seed)
        page, polygons = PageGenerator(
            _cfg(tiny_font_dir, capture_prob=1.0, capture_glare_prob=1.0, capture_glare_strength=0.9)
        ).generate_page(WORDS)
        scores += _readability(page, polygons)
    assert np.median(scores) > 0.30


# -- physical media model -------------------------------------------------


def _media_cfg(font_dir, **overrides):
    base = dict(
        media_enabled=True,
        media_dpi_range=(200.0, 300.0),
        capture_prob=0.0,
        det_bleed_through_prob=0.0,
        noise_prob=0.0,
        jpeg_prob=0.0,
        final_blur_prob=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        det_plain_background_prob=1.0,
    )
    base.update(overrides)
    return _cfg(font_dir, **base)


def test_media_pages_have_real_aspect_ratios(tiny_font_dir):
    generator = PageGenerator(_media_cfg(tiny_font_dir, det_layout="paragraph"))
    ratios = []
    for seed in range(12):
        random.seed(seed)
        page, _ = generator.generate_page(WORDS)
        ratios.append(page.width / page.height)
    # A4 is 0.707, Letter 0.773, and landscape flips them.
    assert any(0.6 < r < 0.85 for r in ratios)
    assert min(ratios) > 0.35 and max(ratios) < 3.0


def test_points_convert_at_the_page_resolution():
    media = PageMedia("a4", dpi=300.0, width_px=2480, height_px=3508, delivery_long_edge=3508)
    assert media.points_to_px(12) == 50  # 12pt at 300 DPI
    assert media.points_to_px(6) == 25  # 6pt is small, not illegible
    low = PageMedia("a4", dpi=150.0, width_px=1240, height_px=1754, delivery_long_edge=1754)
    assert low.points_to_px(12) == 25


def test_floor_scales_with_the_delivery_downscale():
    # Halving the image on delivery means the smallest type must render twice
    # as large to arrive at the same size.
    full = PageMedia("a4", 300.0, 2480, 3508, 3508)
    halved = PageMedia("a4", 300.0, 2480, 3508, 1754)
    assert full.min_render_px_for_delivery(9) == 9
    assert halved.min_render_px_for_delivery(9) == 18


def test_media_produces_genuinely_small_text(tiny_font_dir):
    # The point of the exercise: fine print that is small *and* sharp.
    generator = PageGenerator(
        _media_cfg(tiny_font_dir, det_layout="paragraph", det_fine_print_prob=1.0, media_resample_prob=0.0)
    )
    heights = []
    for seed in range(6):
        random.seed(seed)
        page, polygons = generator.generate_page(WORDS)
        heights += [max(pt[1] for pt in p) - min(pt[1] for pt in p) for p in polygons]
        assert np.median(_readability(page, polygons)) > 0.35
    relative = np.array(heights) / 3000.0
    assert np.percentile(relative, 5) < 0.006  # fine print is a small share of the page


def test_delivery_resample_scales_polygons_with_the_image(tiny_font_dir):
    cfg = _media_cfg(tiny_font_dir, media_resample_prob=1.0, media_upscale_after_prob=0.0)
    media = PageMedia("a4", 300.0, 1000, 1400, 700)
    page = Image.new("RGB", (1000, 1400), (255, 255, 255))
    polygons = [[[100, 200], [300, 200], [300, 240], [100, 240]]]
    out, moved, scale = apply_delivery_resample(cfg, page, polygons, media)
    assert out.size == (500, 700)
    assert abs(scale - 0.5) < 1e-6
    assert moved[0][0] == [50.0, 100.0]
    assert moved[0][2] == [150.0, 120.0]


def test_resample_disabled_leaves_the_page_untouched(tiny_font_dir):
    cfg = _media_cfg(tiny_font_dir, media_resample_prob=0.0)
    media = PageMedia("a4", 300.0, 1000, 1400, 700)
    page = Image.new("RGB", (1000, 1400), (255, 255, 255))
    polygons = [[[10, 20], [30, 20], [30, 24], [10, 24]]]
    out, moved, scale = apply_delivery_resample(cfg, page, polygons, media)
    assert out.size == (1000, 1400) and moved == polygons and scale == 1.0


def test_media_polygons_stay_in_bounds(tiny_font_dir):
    for layout in ("paragraph", "table", "receipt", "form", "id_card", "vertical"):
        generator = PageGenerator(_media_cfg(tiny_font_dir, det_layout=layout, media_resample_prob=1.0))
        random.seed(1)
        page, polygons = generator.generate_page(WORDS)
        assert polygons, layout
        for poly in polygons:
            for x, y in poly:
                assert 0 <= x <= page.width and 0 <= y <= page.height, layout


def test_media_disabled_keeps_pixel_page_sizes(tiny_font_dir):
    page, _ = PageGenerator(_cfg(tiny_font_dir)).generate_page(WORDS)
    assert page.size == (400, 520)


# -- regressions reported from real runs ----------------------------------


def test_ink_is_chosen_over_the_whole_block_not_just_its_first_line(tiny_font_dir):
    # A block's ink used to be picked from a thin strip at its top. On a texture
    # that shades light-to-dark, the rest of the block was then unreadable while
    # every individual style decision still looked correct.
    calls = []
    generator = PageGenerator(_cfg(tiny_font_dir, det_layout="paragraph"))
    original = generator._style_at

    def spy(page, x, y, w, h, bold_width, extent=None):
        calls.append((h, extent))
        return original(page, x, y, w, h, bold_width, extent)

    generator._style_at = spy
    generator.generate_page(WORDS)
    multiline = [(h, e) for h, e in calls if e is not None]
    assert multiline, "paragraph blocks must pass the extent they will fill"
    assert any(e > h for h, e in multiline)


def test_pinned_page_size_pins_the_output_size(tiny_font_dir):
    # Capture, the delivery resample and the expanding rotation all change the
    # dimensions, so pinning the page must switch them off - otherwise batching
    # the docTR dataset fails with a shape mismatch.
    generator = PageGenerator(
        _cfg(tiny_font_dir, capture_prob=1.0, det_rotation_prob=1.0, media_enabled=True, media_resample_prob=1.0)
    )
    sizes = set()
    for seed in range(6):
        random.seed(seed)
        page, _ = generator.generate_page(WORDS)
        sizes.add(page.size)
    assert sizes == {(400, 520)}


def test_pinned_receipt_geometry_beats_the_page_pin(tiny_font_dir):
    page, _ = PageGenerator(
        _cfg(
            tiny_font_dir,
            det_layout="receipt",
            det_receipt_width_range=(320, 320),
            det_receipt_height_range=(1100, 1100),
        )
    ).generate_page(WORDS)
    assert page.size == (320, 1100)


def test_collate_reports_mismatched_page_sizes_clearly(tiny_font_dir):
    pytest.importorskip("torch")
    import torch

    from generator.doctr_dataset import SyntheticDetectionDataset

    samples = [
        (torch.zeros(3, 100, 80), {"words": np.zeros((1, 4), dtype=np.float32)}),
        (torch.zeros(3, 120, 90), {"words": np.zeros((1, 4), dtype=np.float32)}),
    ]
    with pytest.raises(RuntimeError, match="different sizes"):
        SyntheticDetectionDataset.collate_fn(samples)


def test_handwriting_font_names_resolve(tiny_font_dir):
    # Every handwriting family needs at least one filename that exists upstream;
    # a rename there used to silently drop the face with a 404.
    from generator.components.font_downloader import _HANDWRITING_FONTS

    for faces in _HANDWRITING_FONTS.values():
        for _family, filenames in faces:
            assert filenames, "each family needs a filename"
            assert all(name.endswith(".ttf") for name in filenames)
            assert len(filenames) >= 1


def test_page_contrast_is_drawn_from_the_upper_range(tiny_font_dir):
    # Contrast is pinned per page, so a faint draw fades the whole document.
    from generator.components.text_styling import sample_page_palette

    cfg = _cfg(tiny_font_dir, min_contrast=0.45, max_contrast=0.95)
    values = [sample_page_palette(cfg)["contrast"] for _ in range(200)]
    assert min(values) >= 0.45 + (0.95 - 0.45) * cfg.realism.page_contrast_bias - 1e-9
    assert max(values) <= 0.95


def test_ink_stays_separable_from_the_paper(tiny_font_dir):
    # Contrast, hue scaling, jitter and opacity each look fine alone and compound
    # into text that is drawn but invisible.
    from generator.components.text_styling import decide_text_style, luminance

    cfg = _cfg(tiny_font_dir, min_contrast=0.02, max_contrast=0.05, colored_ink_prob=1.0)
    paper = Image.new("RGB", (40, 20), (250, 250, 250))
    for _ in range(60):
        style = decide_text_style(cfg, paper, bold_width=0, outline_width=0)
        alpha = style.opacity / 255.0
        effective = abs(luminance(style.fill_color) - 250.0) * alpha
        assert effective > cfg.realism.min_ink_separation * 0.75


def test_recognition_crops_are_budgeted_by_glyph_size(tiny_font_dir):
    from generator.components.generator import TextImageGenerator

    generator = TextImageGenerator(_cfg(tiny_font_dir, final_blur_prob=1.0, final_blur_radius_range=(4.0, 4.0)))
    image = generator.generate_image("small")
    assert image is not None
    arr = np.asarray(image.convert("L"), dtype=np.float32)
    assert arr.max() - arr.min() > 40  # a 4px blur would have flattened the crop


def test_numeric_tokens_cover_symbols_and_currencies(tiny_font_dir):
    # Frequency word lists contain no digits, operators or currency at all.
    from generator.components.corpus_downloader import generate_numeric_tokens

    tokens = generate_numeric_tokens(800, seed=0)
    symbols = {ch for token in tokens for ch in token if not ch.isalnum() and ch != " "}
    assert len(symbols) >= 25, sorted(symbols)
    for expected in "+-%=@#&/:()[]":
        assert expected in symbols, expected
    with_currency = [t for t in tokens if any(c in t for c in "€$£¥₹")]
    assert len(with_currency) / len(tokens) > 0.05
    standalone = [t for t in tokens if len(t) == 1 and not t.isalnum()]
    assert standalone, "crops of a single symbol must occur"


def test_ordinal_suffixes_match_their_number():
    import re

    from generator.components.corpus_downloader import generate_numeric_tokens

    for token in generate_numeric_tokens(2000, seed=1):
        match = re.match(r"^(\d+)(st|nd|rd|th)$", token)
        if not match:
            continue
        number, suffix = int(match.group(1)), match.group(2)
        expected = {1: "st", 2: "nd", 3: "rd"}.get(number if number < 20 else number % 10, "th")
        if 11 <= number <= 13:
            expected = "th"
        assert suffix == expected, token


def test_texture_behind_a_crop_is_calmed(tiny_font_dir):
    # A photo crop carries structure at glyph scale, which camouflages strokes
    # no matter what ink is chosen.
    from generator.components.text_styling import calm_texture

    rng = np.random.default_rng(0)
    noisy = Image.fromarray(rng.integers(0, 255, (24, 60, 3), dtype=np.uint8), mode="RGB")
    calmed = calm_texture(noisy, 10.0)
    assert np.asarray(calmed, dtype=np.float32).std() < np.asarray(noisy, dtype=np.float32).std() / 2
    assert calm_texture(noisy, 0.0) is noisy  # disabled


def test_handwriting_filenames_have_no_invented_fallbacks():
    # Every alternate here is only reached *after* the real file 404s, so an
    # unverified guess cannot rescue anything and guarantees a 404 on the way.
    from generator.components.font_downloader import _HANDWRITING_FONTS

    known = {
        "Caveat[wght].ttf",
        "IndieFlower-Regular.ttf",
        "ShadowsIntoLight.ttf",
        "ArchitectsDaughter-Regular.ttf",
        "PatrickHand-Regular.ttf",
        "Kalam-Regular.ttf",
    }
    for faces in _HANDWRITING_FONTS.values():
        for _family, filenames in faces:
            assert set(filenames) <= known, filenames


def test_only_cjk_is_stacked_upright(tiny_font_dir):
    # Stacking is a CJK convention. Stacking Latin spells a word
    # letter-under-letter, which no document outside a shop sign does.
    assert PageGenerator._is_stackable("設計師")
    assert PageGenerator._is_stackable("日本語")
    assert PageGenerator._is_stackable("12")  # numerals read fine upright
    assert PageGenerator._is_stackable("%")
    assert not PageGenerator._is_stackable("NUTRITIOUS")
    assert not PageGenerator._is_stackable("consider")
    assert not PageGenerator._is_stackable("Büsche")
    assert not PageGenerator._is_stackable("Привет")
    assert not PageGenerator._is_stackable("12月")  # mixed: rotate the whole run
    assert not PageGenerator._is_stackable("")


def test_latin_falls_back_to_rotation_in_a_stacked_column(tiny_font_dir):
    pg = PageGenerator(_cfg(tiny_font_dir, rotation_prob=0.0, perspective_prob=0.0, blur_prob=0.0))
    pg._begin_page()
    stacked = pg._render_vertical_word("NUTRITIOUS", 18, 0, "stacked", fallback="ccw")
    rotated = pg._render_vertical_word("NUTRITIOUS", 18, 0, "ccw")
    assert stacked is not None and rotated is not None
    assert stacked.size == rotated.size  # it rotated instead of stacking


def test_no_latin_word_is_ever_stacked_on_a_vertical_page(tiny_font_dir):
    stacked_words = []
    generator = PageGenerator(_cfg(tiny_font_dir, det_layout="vertical", det_vertical_stacked_prob=1.0))
    original = generator._render_stacked_word

    def spy(word, font_size, bold_width, role="body"):
        stacked_words.append(word)
        return original(word, font_size, bold_width, role)

    generator._render_stacked_word = spy
    for seed in range(4):
        random.seed(seed)
        generator.generate_page(WORDS)
    assert not [w for w in stacked_words if any(c.isalpha() and c.isascii() for c in w)]


def test_rotated_columns_leave_more_room_between_words(tiny_font_dir):
    # Rotated words run into each other and read as one long token when spaced
    # like stacked CJK glyphs.
    cfg = _cfg(tiny_font_dir)
    lo, hi = cfg.detection.vertical_word_gap_range
    assert lo >= 0.35 and hi > lo


def test_lighting_cannot_flatten_local_contrast(tiny_font_dir):
    # Falloff and glare each scale local contrast, and it is their product that
    # erases text, so the pair is bounded together rather than one at a time.
    cfg = _cfg(
        tiny_font_dir,
        capture_glare_prob=1.0,
        capture_glare_strength=1.0,
        capture_illumination_prob=1.0,
        capture_illumination_strength=1.0,
        capture_vignette_prob=1.0,
        capture_vignette_strength=1.0,
    )
    floor = cfg.capture.min_contrast_factor
    assert 0.0 < floor < 1.0

    from generator.components.capture import _illuminate

    bars = np.zeros((80, 80, 3), dtype=np.uint8)
    bars[::4] = 255  # alternating ink/paper rows
    before = np.asarray(Image.fromarray(bars, mode="RGB").convert("L"), dtype=np.float32)
    for seed in range(6):
        random.seed(seed)
        np.random.seed(seed)
        lit = _illuminate(cfg, Image.fromarray(bars, mode="RGB"))
        after = np.asarray(lit.convert("L"), dtype=np.float32)
        kept = (np.percentile(after, 95) - np.percentile(after, 5)) / (
            np.percentile(before, 95) - np.percentile(before, 5)
        )
        assert kept > floor * 0.85, f"lighting kept only {kept:.2f} of the contrast"
