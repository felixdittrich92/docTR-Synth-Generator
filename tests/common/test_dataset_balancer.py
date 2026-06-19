from collections import Counter

from generator.components import BalanceResult, DatasetBalancer


def _pools():
    return {
        "en": [f"e{i}" for i in range(1000)],
        "de": [f"d{i}" for i in range(1000)],
        "el": [f"g{i}" for i in range(200)],  # low-resource language
    }


def test_balanced_strategy_equalizes_languages():
    res = DatasetBalancer(seed=1).allocate_and_split(
        _pools(), num_images=600, val_percent=0.2, strategy="balanced", report=False
    )
    assert isinstance(res, BalanceResult)
    assert len(res.train) + len(res.val) == 600
    tr = res.summary["per_bucket"]
    # Each language bucket gets an (almost) equal total of 200.
    for lang in ("en", "de", "el"):
        assert sum(tr[lang]) == 200


def test_proportional_strategy_tracks_pool_sizes():
    res = DatasetBalancer(seed=1).allocate_and_split(
        _pools(), num_images=660, val_percent=0.2, strategy="proportional", report=False
    )
    counts = {lang: sum(tr) for lang, tr in res.summary["per_bucket"].items()}
    # en and de pools are equal and 5x larger than el.
    assert counts["en"] == counts["de"]
    assert counts["en"] > counts["el"]


def test_explicit_weights():
    res = DatasetBalancer(seed=1).allocate_and_split(
        _pools(),
        num_images=1000,
        val_percent=0.2,
        weights={"en": 0.6, "de": 0.3, "el": 0.1},
        report=False,
    )
    counts = {lang: sum(tr) for lang, tr in res.summary["per_bucket"].items()}
    assert counts["en"] == 600
    assert counts["de"] == 300
    assert counts["el"] == 100


def test_stratified_split_val_percent_per_bucket():
    res = DatasetBalancer(seed=3).allocate_and_split(
        _pools(), num_images=600, val_percent=0.25, strategy="balanced", report=False
    )
    for _, (tr, va) in res.summary["per_bucket"].items():
        total = tr + va
        assert abs(va - round(total * 0.25)) <= 1


def test_numeric_bucket():
    res = DatasetBalancer(seed=4).allocate_and_split(
        {"en": [f"e{i}" for i in range(500)]},
        num_images=200,
        val_percent=0.2,
        numeric_tokens=["123", "45,90", "2024"],
        numeric_ratio=0.25,
        report=False,
    )
    assert "_numeric" in res.summary["per_bucket"]
    numeric_total = sum(res.summary["per_bucket"]["_numeric"])
    assert numeric_total == 50  # 25% of 200


def test_char_coverage_enforcement():
    # Many common words make their characters abundant; only 'qjz' is rare, so
    # the bounded top-up has ample budget to lift q/j/z to the threshold.
    common = [
        "the",
        "and",
        "you",
        "for",
        "are",
        "was",
        "his",
        "her",
        "one",
        "our",
        "day",
        "new",
        "get",
        "out",
        "see",
        "two",
        "how",
        "its",
        "who",
        "yes",
    ]
    pools = {"en": common + ["qjz"]}
    res = DatasetBalancer(seed=5).allocate_and_split(
        pools, num_images=80, val_percent=0.0, min_char_coverage=3, report=False
    )
    chars = Counter("".join(res.train + res.val))
    assert chars and min(chars.values()) >= 3
    assert res.summary["coverage_extra_added"] > 0


def test_reproducible_with_seed():
    a = DatasetBalancer(seed=7).allocate_and_split(_pools(), 300, 0.2, report=False)
    b = DatasetBalancer(seed=7).allocate_and_split(_pools(), 300, 0.2, report=False)
    assert a.train == b.train
    assert a.val == b.val


def test_report_prints(capsys):
    DatasetBalancer(seed=1).allocate_and_split(_pools(), 120, 0.2, report=True)
    out = capsys.readouterr().out
    assert "Dataset balance report" in out
    assert "per bucket" in out


def test_summary_fields():
    res = DatasetBalancer(seed=1).allocate_and_split(_pools(), 300, 0.2, report=False)
    for key in ("total", "train", "val", "distinct_chars", "length_mean", "train_val_overlap"):
        assert key in res.summary
