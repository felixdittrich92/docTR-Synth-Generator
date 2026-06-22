from collections import Counter

from generator.components import VOCABS
from generator.components.vocab_coverage import augment_words_for_coverage, resolve_target_vocab


def test_resolve_by_language():
    target = resolve_target_vocab("de", None)
    assert target == set(VOCABS["german"])


def test_resolve_explicit_vocab_name_overrides_language():
    target = resolve_target_vocab("de", "french")
    assert target == set(VOCABS["french"])


def test_resolve_literal_characters():
    target = resolve_target_vocab(None, "abc€")
    assert target == set("abc€")


def test_resolve_unknown_returns_none():
    assert resolve_target_vocab("zz", None) is None
    assert resolve_target_vocab(None, None) is None


def test_augment_covers_missing_characters():
    words = ["the", "quick", "brown", "fox"]  # missing many letters/punctuation
    target = set("abcdefghijklmnopqrstuvwxyz!?€")
    augmented, added, missing = augment_words_for_coverage(words, target, min_count=2, seed=0)
    assert added > 0
    assert missing > 0
    counts = Counter(c for w in augmented for c in w)
    for ch in target:
        assert counts[ch] >= 2  # every target character now well represented


def test_augment_noop_when_already_covered():
    target = set("abc")
    words = ["abc", "cab", "bca", "abcabc"]  # already plenty
    augmented, added, _ = augment_words_for_coverage(words, target, min_count=2, seed=0)
    assert added == 0
    assert augmented == words


def test_augment_empty_target_is_noop():
    augmented, added, missing = augment_words_for_coverage(["x"], set(), min_count=3)
    assert added == 0 and missing == 0
    assert augmented == ["x"]
