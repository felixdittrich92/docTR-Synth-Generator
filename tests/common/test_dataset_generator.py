import os
import tempfile
from collections import Counter

from generator import GenerationConfig, SyntheticDatasetGenerator


def test_prepare_train_val_uses_wordlist(tmp_path):
    p = os.path.join(tempfile.mkdtemp(), "words.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(f"w{i}" for i in range(50)))

    cfg = GenerationConfig(wordlist_path=p, output_dir="ds", num_images=20, val_percent=0.25)
    train, val = SyntheticDatasetGenerator(cfg)._prepare_train_val()

    assert len(train) + len(val) == 20
    assert len(val) == 5
    assert len(set(train) & set(val)) == 0  # unique sampling -> no leakage


def test_prepare_train_val_downloads_balanced_corpus(monkeypatch):
    # Offline fake corpus: patch the downloader used inside the orchestrator.
    pools = {
        "en": [f"e{i}" for i in range(200)],
        "de": [f"d{i}" for i in range(200)],
    }
    monkeypatch.setattr(
        "generator.dataset_generator.CorpusDownloader.fetch",
        lambda self, lang: pools.get(lang, []),
    )

    cfg = GenerationConfig(
        output_dir="ds",
        num_images=200,
        languages=["en", "de"],
        language_balance="balanced",
        casing_variant_prob=0.0,
        numeric_token_ratio=0.0,
        print_balance_report=False,
        corpus_seed=0,
    )
    train, val = SyntheticDatasetGenerator(cfg)._prepare_train_val()

    assert len(train) + len(val) == 200
    prefixes = Counter(w[0] for w in train + val)
    # Balanced strategy -> roughly equal counts per language.
    assert abs(prefixes["e"] - prefixes["d"]) <= 2


def test_prepare_train_val_corpus_with_numerics(monkeypatch):
    # Digit-free fake words so injected numeric tokens are distinguishable.
    fake_words = [f"abc{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(300)]
    monkeypatch.setattr(
        "generator.dataset_generator.CorpusDownloader.fetch",
        lambda self, lang: fake_words,
    )
    cfg = GenerationConfig(
        output_dir="ds",
        num_images=100,
        languages=["en"],
        casing_variant_prob=0.0,
        numeric_token_ratio=0.2,
        print_balance_report=False,
        corpus_seed=0,
    )
    train, val = SyntheticDatasetGenerator(cfg)._prepare_train_val()
    assert len(train) + len(val) == 100
    # ~20% should be numeric tokens (contain a digit).
    numeric = sum(1 for w in train + val if any(c.isdigit() for c in w))
    assert 10 <= numeric <= 30
