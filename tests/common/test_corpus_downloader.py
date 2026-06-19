import tempfile

from generator.components import CorpusDownloader, apply_casing_variants, generate_numeric_tokens


def test_clean_filters_punctuation_length_and_script():
    cd = CorpusDownloader(cache_dir=tempfile.mkdtemp(), min_word_length=2, max_word_length=10)
    raw = "\n".join([
        "hallo 100",  # ok
        "die 90",  # ok
        ", 80",  # pure punctuation -> dropped
        "a 70",  # too short -> dropped
        "superlongword12345 60",  # too long -> dropped
        "привет 50",  # Cyrillic in a German list -> dropped by script filter
        "die 40",  # duplicate -> dropped
    ])
    words = cd._clean(raw, "de")
    assert "hallo" in words
    assert "die" in words
    assert words.count("die") == 1
    assert "," not in words
    assert "a" not in words
    assert "привет" not in words


def test_clean_without_script_filter_keeps_foreign():
    cd = CorpusDownloader(cache_dir=tempfile.mkdtemp(), filter_by_script=False, min_word_length=2)
    words = cd._clean("hello 10\nпривет 5", "de")
    assert "привет" in words


def test_fetch_and_build_vocabulary_offline(monkeypatch):
    cd = CorpusDownloader(cache_dir=tempfile.mkdtemp(), min_word_length=2)

    fake = {
        "en": "the 100\nand 90\nyou 80",
        "de": "und 100\nist 90\ndas 80",
    }
    monkeypatch.setattr(cd, "_download_raw", lambda lang: fake.get(lang))

    en = cd.fetch("en")
    assert en == ["the", "and", "you"]
    # cached on second call
    assert cd.fetch("en") is en

    vocab = cd.build_vocabulary(["en", "de"], words_per_language=2)
    assert vocab == ["the", "and", "und", "ist"]


def test_fetch_missing_language_returns_empty(monkeypatch):
    cd = CorpusDownloader(cache_dir=tempfile.mkdtemp())
    monkeypatch.setattr(cd, "_download_raw", lambda lang: None)
    assert cd.fetch("xx") == []


def test_apply_casing_variants_adds_capitalized():
    words = ["hallo", "welt"]
    out = apply_casing_variants(words, prob=1.0, seed=1)
    assert "hallo" in out and "welt" in out
    # At prob=1.0 every cased word yields one extra Title/UPPER variant.
    assert len(out) == 4
    assert any(w[0].isupper() for w in out)


def test_apply_casing_variants_skips_caseless():
    # Caseless script (digits) yields no extra variants.
    out = apply_casing_variants(["123", "456"], prob=1.0, seed=1)
    assert out == ["123", "456"]


def test_generate_numeric_tokens():
    toks = generate_numeric_tokens(50, seed=3)
    assert len(toks) == 50
    assert all(isinstance(t, str) and t for t in toks)
    # at least some contain a digit
    assert any(any(c.isdigit() for c in t) for t in toks)
    # reproducible
    assert generate_numeric_tokens(50, seed=3) == toks


def test_required_scripts_static():
    assert CorpusDownloader  # smoke - imported
