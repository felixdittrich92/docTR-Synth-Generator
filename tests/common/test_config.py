from generator.components import GenerationConfig


def test_defaults_language_when_no_wordlist():
    cfg = GenerationConfig(output_dir="ds", num_images=10)
    # No wordlist and no explicit languages -> defaults to English.
    assert cfg.wordlist_path is None
    assert cfg.languages == ["en"]
    assert cfg.auto_download_fonts is True


def test_explicit_languages_preserved():
    cfg = GenerationConfig(output_dir="ds", num_images=10, languages=["de", "ru"])
    assert cfg.languages == ["de", "ru"]


def test_wordlist_skips_language_default():
    cfg = GenerationConfig(wordlist_path="words.txt", output_dir="ds", num_images=10)
    # When a wordlist is given, language auto-defaulting is not applied.
    assert cfg.languages is None


def test_balancing_and_cache_defaults():
    cfg = GenerationConfig(output_dir="ds", num_images=10)
    assert cfg.language_balance == "balanced"
    assert cfg.language_weights is None
    assert cfg.min_char_coverage == 0
    assert cfg.bg_cache_size == 16
    assert cfg.supersample >= 1


def test_detection_defaults():
    cfg = GenerationConfig(output_dir="ds", num_images=10)
    # Recognition stays the default task; detection is opt-in.
    assert cfg.task == "recognition"
    assert cfg.det_page_width_range[0] < cfg.det_page_width_range[1]
    assert 0.0 <= cfg.det_plain_background_prob <= 1.0
    assert cfg.det_max_blocks >= 1


def test_background_download_defaults():
    cfg = GenerationConfig(output_dir="ds", num_images=10)
    # Backgrounds auto-download by default, but an explicit dir always wins.
    assert cfg.auto_download_backgrounds is True
    assert cfg.bg_image_dir is None
    assert cfg.background_cache_dir is None


def test_vocab_coverage_and_layout_defaults():
    cfg = GenerationConfig(output_dir="ds", num_images=10)
    assert cfg.ensure_vocab_coverage is True
    assert cfg.target_vocab is None
    assert cfg.vocab_coverage_min_count >= 1
    assert cfg.det_layout == "mixed"
