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


def test_synthesized_tokens_are_single_script():
    from generator.components.vocab_coverage import _script_of

    target = set()
    for lang in ("en", "ru", "el", "he", "ar", "th", "hi"):
        resolved = resolve_target_vocab(lang, None)
        if resolved:
            target |= resolved
    base = ["hello", "привет", "γειά", "שלום", "سلام", "สวัสดี", "नमस्ते"]
    augmented, _, _ = augment_words_for_coverage(base, target, min_count=2, seed=0)
    for token in augmented[len(base) :]:
        scripts = {s for s in (_script_of(c) for c in token) if s != "COMMON"}
        assert len(scripts) <= 1, (token, scripts)  # never mix scripts in one token


def test_large_scripts_left_to_corpus():
    # Japanese includes thousands of CJK ideographs - synthesis must not try to
    # cover them all (that would bloat the dataset); only its small sub-scripts.
    target = resolve_target_vocab("ja", None)
    _, added, _ = augment_words_for_coverage([], target, min_count=3, seed=0)
    assert added < 2000


def test_combining_marks_always_follow_a_base_letter():
    import unicodedata

    target = set()
    for lang in ("ar", "he", "hi", "th", "bn", "ta", "te", "ml", "kn"):
        resolved = resolve_target_vocab(lang, None)
        if resolved:
            target |= resolved
    base = ["سلام", "שלום", "नमस्ते", "สวัสดี", "আমি", "தமிழ்"]
    augmented, _, _ = augment_words_for_coverage(base, target, min_count=3, seed=0)
    for token in augmented[len(base) :]:
        assert token  # no empty tokens
        seen_letter = False
        for ch in token:
            cat = unicodedata.category(ch)
            if cat.startswith("M"):
                assert seen_letter, f"mark {ch!r} not preceded by a base letter in {token!r}"
            if cat.startswith("L"):
                seen_letter = True


def test_present_rare_chars_are_covered_with_real_words():
    # A rare-but-present character must be topped up by repeating a REAL word
    # (an attested combination), not a synthesised token. Absent chars are from
    # another script so their synthesised tokens never share letters.
    words = ["naive", "cafe", "cafe", "cafe"]  # 'n','i','v' appear once (rare)
    target = set("niv") | set("αβγ")  # Latin rare-present + Greek absent
    augmented, _, _ = augment_words_for_coverage(words, target, min_count=3, seed=0)
    added = augmented[len(words) :]
    latin_tokens = [t for t in added if any("a" <= c <= "z" for c in t)]
    assert latin_tokens
    assert all(t in words for t in latin_tokens)  # covered by repeating real words
