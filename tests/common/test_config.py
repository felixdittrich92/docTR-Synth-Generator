from generator.components import GenerationConfig


def test_defaults_language_when_no_wordlist():
    cfg = GenerationConfig.flat(output_dir="ds", num_images=10)
    # No wordlist and no explicit languages -> defaults to English.
    assert cfg.resources.wordlist_path is None
    assert cfg.core.languages == ["en"]
    assert cfg.resources.auto_download_fonts is True


def test_explicit_languages_preserved():
    cfg = GenerationConfig.flat(output_dir="ds", num_images=10, languages=["de", "ru"])
    assert cfg.core.languages == ["de", "ru"]


def test_wordlist_skips_language_default():
    cfg = GenerationConfig.flat(wordlist_path="words.txt", output_dir="ds", num_images=10)
    # When a wordlist is given, language auto-defaulting is not applied.
    assert cfg.core.languages is None


def test_balancing_and_cache_defaults():
    cfg = GenerationConfig.flat(output_dir="ds", num_images=10)
    assert cfg.balance.language_balance == "balanced"
    assert cfg.balance.language_weights is None
    assert cfg.balance.min_char_coverage == 0
    assert cfg.resources.bg_cache_size == 16
    assert cfg.realism.supersample >= 1


def test_detection_defaults():
    cfg = GenerationConfig.flat(output_dir="ds", num_images=10)
    # Recognition stays the default task; detection is opt-in.
    assert cfg.core.task == "recognition"
    assert cfg.detection.page_width_range[0] < cfg.detection.page_width_range[1]
    assert 0.0 <= cfg.detection.plain_background_prob <= 1.0
    assert cfg.detection.max_blocks >= 1


def test_background_download_defaults():
    cfg = GenerationConfig.flat(output_dir="ds", num_images=10)
    # Backgrounds auto-download by default, but an explicit dir always wins.
    assert cfg.resources.auto_download_backgrounds is True
    assert cfg.resources.bg_image_dir is None
    assert cfg.resources.background_cache_dir is None


def test_vocab_coverage_and_layout_defaults():
    cfg = GenerationConfig.flat(output_dir="ds", num_images=10)
    assert cfg.coverage.ensure_vocab_coverage is True
    assert cfg.coverage.target_vocab is None
    assert cfg.coverage.vocab_coverage_min_count >= 1
    assert cfg.detection.layout == "mixed"


def test_nested_construction():
    from generator.components import CoreConfig, DetectionConfig

    cfg = GenerationConfig(
        core=CoreConfig(num_images=42, task="detection", languages=["en", "de"]),
        detection=DetectionConfig(layout="newspaper"),
    )
    assert cfg.core.num_images == 42 and cfg.core.task == "detection"
    assert cfg.detection.layout == "newspaper"
    assert cfg.realism.supersample == 3  # untouched sub-config keeps its defaults


def test_flat_routes_keywords_into_subconfigs():
    cfg = GenerationConfig.flat(
        num_images=7,
        task="detection",
        det_layout="form",
        det_max_words_per_page=120,
        supersample=2,
        target_vocab=["german"],
        corpus_seed=11,
        num_workers=3,
    )
    assert cfg.core.num_images == 7 and cfg.core.num_workers == 3
    assert cfg.detection.layout == "form" and cfg.detection.max_words_per_page == 120
    assert cfg.realism.supersample == 2
    assert cfg.coverage.target_vocab == ["german"]
    assert cfg.corpus.seed == 11  # corpus_seed -> corpus.seed


def test_flat_rejects_unknown_option():
    import pytest

    with pytest.raises(TypeError):
        GenerationConfig.flat(output_dir="ds", not_a_real_option=1)


def test_subconfigs_are_exported():
    import generator

    for name in (
        "CoreConfig",
        "ResourceConfig",
        "CorpusConfig",
        "BalanceConfig",
        "CoverageConfig",
        "RecognitionConfig",
        "RealismConfig",
        "DetectionConfig",
    ):
        assert hasattr(generator, name), name
