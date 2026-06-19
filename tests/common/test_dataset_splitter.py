import os
import random
import tempfile

from generator.components import DatasetSplitter


def test_load_vocabulary(tmp_path=None):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "words.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("alpha\nbeta\n\n  gamma  \n")
    words = DatasetSplitter.load_vocabulary(p)
    assert words == ["alpha", "beta", "gamma"]


def test_prepare_splits_respects_num_images_cap():
    # Vocabulary larger than num_images must not blow past the requested count.
    vocab = [f"w{i}" for i in range(1000)]
    random.seed(0)
    train, val = DatasetSplitter.prepare_splits(vocab, num_images=100, val_percent=0.2)
    assert len(train) + len(val) == 100
    assert len(val) == 20
    # When sampling from a larger unique vocab, words are unique (no leakage).
    assert len(set(train) & set(val)) == 0


def test_prepare_splits_repeats_when_vocab_small():
    vocab = ["a", "b", "c"]
    random.seed(1)
    train, val = DatasetSplitter.prepare_splits(vocab, num_images=30, val_percent=0.2)
    assert len(train) + len(val) == 30
    # Every unique word is represented at least once across the dataset.
    assert set(train + val) == set(vocab)


def test_prepare_splits_empty():
    train, val = DatasetSplitter.prepare_splits([], num_images=10, val_percent=0.2)
    assert train == []
    assert val == []


def test_prepare_splits_dedupes_input():
    vocab = ["a", "a", "b", "b", "c"]
    random.seed(2)
    train, val = DatasetSplitter.prepare_splits(vocab, num_images=3, val_percent=0.0)
    assert len(train) == 3
    assert set(train) == {"a", "b", "c"}
