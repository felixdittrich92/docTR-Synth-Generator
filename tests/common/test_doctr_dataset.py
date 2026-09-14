# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import random

import numpy as np
import pytest

from generator.components.config import GenerationConfig
from generator.components.generator import TextImageGenerator
from generator.components.page_generator import PageGenerator
from generator.components.vocabs import VOCABS
from generator.doctr_dataset import (
    CLASS_NAME,
    Sample,
    SyntheticDetectionDataset,
    SyntheticRecognitionDataset,
    polygons_to_target,
    render_detection_sample,
    render_recognition_sample,
)

REC_POOL = ["alpha", "beta", "gamma", "delta", "epsilon"]
DET_POOL = ["word", "text", "page", "fill"]


def _cfg(font_dir, task, **kw):
    base = dict(
        task=task,
        font_dir=font_dir,
        bg_image_dir=None,
        output_dir="ds",
        num_images=1,
        auto_download_fonts=False,
        languages=None,
        noise_prob=0.0,
        jpeg_prob=0.0,
        final_blur_prob=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        supersample=2,
    )
    base.update(kw)
    return GenerationConfig.flat(**base)


def _det_cfg(font_dir, **kw):
    return _cfg(
        font_dir,
        "detection",
        det_layout="paragraph",
        det_plain_background_prob=1.0,
        det_rotation_prob=0.0,
        det_page_width_range=(420, 420),
        det_page_height_range=(560, 560),
        det_font_size_range=(14, 16),
        det_max_words_per_page=300,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Framework-agnostic core (no torch required).
# --------------------------------------------------------------------------- #
def test_polygons_to_target_boxes_and_polygons():
    polys = [[[10, 10], [30, 10], [30, 20], [10, 20]], [[5, 40], [25, 40], [25, 55], [5, 55]]]
    boxes = polygons_to_target(polys, use_polygons=False)
    quads = polygons_to_target(polys, use_polygons=True)
    assert set(boxes) == {CLASS_NAME} and set(quads) == {CLASS_NAME}
    assert boxes[CLASS_NAME].shape == (2, 4)  # [xmin, ymin, xmax, ymax]
    assert boxes[CLASS_NAME][0].tolist() == [10.0, 10.0, 30.0, 20.0]
    assert quads[CLASS_NAME].shape == (2, 4, 2)
    assert boxes[CLASS_NAME].dtype == np.float32


def test_polygons_to_target_empty():
    assert polygons_to_target([], use_polygons=False)[CLASS_NAME].shape == (0, 4)
    assert polygons_to_target([], use_polygons=True)[CLASS_NAME].shape == (0, 4, 2)


def test_render_recognition_sample(tiny_font_dir):
    gen = TextImageGenerator(_cfg(tiny_font_dir, "recognition"))
    random.seed(0)
    img, label = render_recognition_sample(gen, REC_POOL)
    assert img.mode == "RGB" and img.width > 0 and img.height > 0
    assert label in REC_POOL


def test_render_detection_sample_in_bounds(tiny_font_dir):
    pg = PageGenerator(_det_cfg(tiny_font_dir))
    random.seed(1)
    img, target = render_detection_sample(pg, DET_POOL, words_per_page=300, use_polygons=False)
    geoms = target[CLASS_NAME]
    w, h = img.size
    assert img.mode == "RGB"
    assert geoms.shape[0] > 0 and geoms.shape[1] == 4
    assert geoms[:, 0].min() >= 0 and geoms[:, 1].min() >= 0
    assert geoms[:, 2].max() <= w and geoms[:, 3].max() <= h
    assert (geoms[:, 0] < geoms[:, 2]).all() and (geoms[:, 1] < geoms[:, 3]).all()


# --------------------------------------------------------------------------- #
# Torch glue (skipped automatically when torch is not installed).
# --------------------------------------------------------------------------- #
def test_recognition_dataset_getitem_and_collate(tiny_font_dir):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    # docTR always pairs recognition with a fixed-size resize img_transform; raw
    # crops vary in size, so a resize must run before collate (as in training).
    def resize(t):
        image = F.interpolate(t.image.unsqueeze(0), size=(32, 128), mode="bilinear", align_corners=False).squeeze(0)
        return Sample(image=image, target=t.target)

    ds = SyntheticRecognitionDataset(REC_POOL, _cfg(tiny_font_dir, "recognition"), num_samples=8, img_transforms=resize)
    assert len(ds) == 8
    img, label = ds[0]
    assert img.shape == (3, 32, 128) and img.dtype == torch.float32
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    assert isinstance(label, str)
    images, labels = ds.collate_fn([ds[0], ds[1], ds[2]])
    assert images.shape == (3, 3, 32, 128) and isinstance(labels, list) and len(labels) == 3


def test_detection_dataset_getitem_and_collate(tiny_font_dir):
    pytest.importorskip("torch")
    ds = SyntheticDetectionDataset(DET_POOL, _det_cfg(tiny_font_dir), num_samples=6, use_polygons=False)
    img, target = ds[0]
    assert img.ndim == 3 and img.shape[0] == 3
    assert set(target) == {CLASS_NAME} and target[CLASS_NAME].shape[1] == 4
    images, targets = ds.collate_fn([ds[0], ds[1]])
    assert images.shape[0] == 2 and isinstance(targets, list) and len(targets) == 2
    assert set(targets[0]) == {CLASS_NAME}


def test_detection_dataset_use_polygons(tiny_font_dir):
    pytest.importorskip("torch")
    ds = SyntheticDetectionDataset(DET_POOL, _det_cfg(tiny_font_dir), num_samples=3, use_polygons=True)
    _, target = ds[0]
    assert target[CLASS_NAME].ndim == 3 and target[CLASS_NAME].shape[1:] == (4, 2)


def test_val_seed_reproducible_train_fresh(tiny_font_dir):
    pytest.importorskip("torch")
    cfg = _det_cfg(tiny_font_dir)
    val = SyntheticDetectionDataset(DET_POOL, cfg, num_samples=4, seed=99)
    a, b = val[0][1][CLASS_NAME], val[0][1][CLASS_NAME]
    assert a.shape == b.shape and np.array_equal(a, b)  # reproducible per index
    train = SyntheticDetectionDataset(DET_POOL, cfg, num_samples=4, seed=None)
    c, d = train[0][1][CLASS_NAME], train[0][1][CLASS_NAME]
    assert not (c.shape == d.shape and np.array_equal(c, d))  # fresh every access


def test_empty_pool_raises(tiny_font_dir):
    pytest.importorskip("torch")
    with pytest.raises(ValueError):
        SyntheticRecognitionDataset([], _cfg(tiny_font_dir, "recognition"), num_samples=4)
    with pytest.raises(ValueError):
        SyntheticDetectionDataset([], _det_cfg(tiny_font_dir), num_samples=4)


def test_recognition_dataset_restricts_to_vocab(tiny_font_dir):
    pytest.importorskip("torch")
    from generator.components.vocab_coverage import resolve_vocab_charset

    pool = ["hello", "world", "naive", "привет", "日本語"]  # last two are out-of-vocab
    ds = SyntheticRecognitionDataset(pool, _cfg(tiny_font_dir, "recognition"), num_samples=6, vocab="english")
    charset = resolve_vocab_charset("english")
    assert ds.pool and all(set(w) <= charset for w in ds.pool)  # pool filtered to the vocab
    assert "привет" not in ds.pool and "日本語" not in ds.pool
    _, label = ds[0]
    assert set(label) <= charset  # every emitted label is encodable by the model


def test_recognition_dataset_empty_after_vocab_filter_raises(tiny_font_dir):
    pytest.importorskip("torch")
    with pytest.raises(ValueError):
        SyntheticRecognitionDataset(
            ["日本語", "привет"], _cfg(tiny_font_dir, "recognition"), num_samples=4, vocab="english"
        )


def test_build_recognition_datasets_sizes_from_config(monkeypatch):
    import generator.dataset_generator as dg
    from generator.components import GenerationConfig
    from generator.doctr_dataset import build_recognition_datasets

    captured = {}

    def fake_pools(self):
        captured["languages"] = list(self.config.core.languages)
        return (["alpha", "beta", "gamma", "delta"], ["one", "two"])

    monkeypatch.setattr(dg.SyntheticDatasetGenerator, "build_recognition_pools", fake_pools)
    cfg = GenerationConfig.flat(task="recognition", num_images=100, val_percent=0.2)
    train, val = build_recognition_datasets(cfg, vocab=["german", "urdu"])
    assert len(train) == 80 and len(val) == 20  # derived from num_images + val_percent
    # vocab-derived languages combine every word list that fits the vocab, so the
    # two named languages are present (plus others fully encodable in it).
    assert {"de", "ur"} <= set(captured["languages"])


def test_build_detection_datasets_sizes_and_language_names(monkeypatch):
    import generator.dataset_generator as dg
    from generator.components import GenerationConfig
    from generator.doctr_dataset import build_detection_datasets

    captured = {}

    def fake_pool(self):
        captured["languages"] = list(self.config.core.languages)
        return ["alpha", "beta", "gamma"]

    monkeypatch.setattr(dg.SyntheticDatasetGenerator, "build_detection_pool", fake_pool)
    cfg = GenerationConfig.flat(task="detection", num_images=50, val_percent=0.1)
    train, val = build_detection_datasets(cfg, languages=["german", "english", "malay"], use_polygons=True)
    assert len(train) == 45 and len(val) == 5  # derived from num_images + val_percent
    assert captured["languages"] == ["de", "en", "ms"]  # incl. malay -> ms
    assert train.use_polygons is True


def test_corpus_languages_for_vocab_combines_and_handles_scriptless():
    from generator.doctr_dataset import _corpus_languages_for_vocab

    multi = _corpus_languages_for_vocab("multilingual")
    assert "de" in multi and "ru" in multi and len(multi) > 10  # combines many word lists
    assert _corpus_languages_for_vocab("khmer") == []  # nothing fits -> synthesise downstream
    assert _corpus_languages_for_vocab("odia") == ["or"]
    german = _corpus_languages_for_vocab("german")
    assert "de" in german and "en" in german  # english fits within the german vocab


def test_latex_formulas_are_valid_and_in_vocab():
    from generator.components.latex_generator import generate_latex_formulas, is_valid_latex

    formulas = generate_latex_formulas(40, seed=1)
    assert len(formulas) >= 30  # most attempts yield distinct valid formulas
    vocab = set(VOCABS["latex"])
    for f in formulas:
        assert is_valid_latex(f)  # mathtext can render it
        assert set(f) <= vocab  # label stays within VOCABS["latex"]


def test_build_latex_recognition_datasets(monkeypatch):
    from generator.components import GenerationConfig
    from generator.components.vocabs import VOCABS
    from generator.doctr_dataset import build_recognition_datasets

    cfg = GenerationConfig.flat(task="recognition", num_images=20, val_percent=0.25, corpus_seed=5)
    train, val = build_recognition_datasets(cfg, vocab="latex")  # routed to the latex builder
    assert len(train) == 15 and len(val) == 5  # sizes from num_images + val_percent
    assert set(train.formulas).isdisjoint(val.formulas)  # disjoint train/val formulas
    img, label = train[0]
    assert img.shape[0] == 3 and set(label) <= set(VOCABS["latex"])


def test_vocab_restriction_falls_back_to_the_languages_vocabs(tiny_font_dir):
    # restrict_to_vocab defaults to True but target_vocab defaults to None, which
    # used to restrict nothing at all - so labels could carry characters the
    # docTR model for those languages cannot encode.
    from generator.components import GenerationConfig
    from generator.dataset_generator import SyntheticDatasetGenerator

    cfg = GenerationConfig.flat(
        task="recognition", languages=["en", "de"], output_dir="/tmp/_vocab_check", num_images=4
    )
    charset = SyntheticDatasetGenerator(cfg)._vocab_charset(["en", "de"])
    assert charset and len(charset) > 50
    assert "a" in charset and "ä" in charset

    off = GenerationConfig.flat(
        task="recognition",
        languages=["en", "de"],
        output_dir="/tmp/_vocab_check",
        num_images=4,
        restrict_to_vocab=False,
    )
    assert SyntheticDatasetGenerator(off)._vocab_charset(["en", "de"]) is None


def test_numeric_tokens_outside_the_vocab_are_dropped(tiny_font_dir):
    from generator.components.corpus_downloader import generate_numeric_tokens
    from generator.components.vocab_coverage import filter_in_vocab, resolve_target_vocab

    allowed: set[str] = set()
    for lang in ("en", "de"):
        resolved = resolve_target_vocab(lang, None)
        if resolved:
            allowed |= resolved
    tokens = generate_numeric_tokens(500, seed=0)
    kept = filter_in_vocab(tokens, allowed)
    assert 0 < len(kept) < len(tokens), "some tokens must survive, and some must be dropped"
    assert all(all(ch in allowed for ch in token) for token in kept)


def test_val_percent_must_be_a_fraction():
    # The name says percent but the value is a fraction. Passing 15 for "15%"
    # silently produced a 77% validation split instead of failing.
    from generator.components import GenerationConfig

    with pytest.raises(ValueError, match="fraction"):
        GenerationConfig.flat(val_percent=15)
    with pytest.raises(ValueError, match="fraction"):
        GenerationConfig.flat(val_percent=1.0)
    with pytest.raises(ValueError, match="fraction"):
        GenerationConfig.flat(val_percent=-0.1)
    assert GenerationConfig.flat(val_percent=0.15).core.val_percent == 0.15
    assert GenerationConfig.flat(val_percent=0.0).core.val_percent == 0.0
