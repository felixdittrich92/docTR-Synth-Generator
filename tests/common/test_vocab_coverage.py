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


def test_resolve_vocab_charset_union_and_filter():
    from generator.components.vocab_coverage import (
        VOCAB_TO_LANGUAGE,
        filter_in_vocab,
        resolve_vocab_charset,
    )

    only_en = resolve_vocab_charset("english")
    union = resolve_vocab_charset(["english", "german"])
    assert only_en and union and only_en <= union  # union is a superset
    # literal strings are accepted too
    assert resolve_vocab_charset("abc") == set("abc")
    # filtering keeps only words fully inside the charset
    kept = filter_in_vocab(["hello", "world", "naïve", "привет"], only_en)
    assert "hello" in kept and "world" in kept
    assert "привет" not in kept  # out-of-vocab script dropped
    # reverse mapping resolves the example keys to corpus languages
    assert VOCAB_TO_LANGUAGE["german"] == "de" and VOCAB_TO_LANGUAGE["urdu"] == "ur"


def test_synthesized_virama_is_never_dangling():
    # A virama/halant (combining class 9) must sit between two base letters, not
    # at the end of a token (which renders an invalid dotted circle). Burmese has
    # no frequency corpus, so every token here is synthesised.
    import unicodedata

    from generator.components.vocab_coverage import resolve_target_vocab

    target = resolve_target_vocab("my", None)  # Burmese
    augmented, _, _ = augment_words_for_coverage([], target, min_count=3, seed=2)
    viramas = {c for c in target if unicodedata.combining(c) == 9}
    assert viramas  # sanity: Burmese vocab includes U+1039
    for token in augmented:
        for i, ch in enumerate(token):
            if ch in viramas:
                assert i + 1 < len(token), f"dangling virama in {token!r}"
                assert unicodedata.category(token[i + 1]).startswith("L"), (
                    f"virama not followed by a letter in {token!r}"
                )


def test_all_vocabs_keys_synthesize_valid_clusters():
    # Sweep EVERY VOCABS key (worst case: empty corpus -> pure synthesis) and
    # assert each synthesised coverage token is a structurally valid cluster:
    # no leading combining mark, no dangling virama, every mark on a same-script
    # base letter, and full coverage of letters/marks (CJK/Hangul excepted).
    import unicodedata
    from collections import defaultdict

    from generator.components.vocab_coverage import (
        _LARGE_SCRIPT_THRESHOLD,
        _script_of,
        augment_words_for_coverage,
        resolve_vocab_charset,
    )
    from generator.components.vocabs import VOCABS

    def is_mark(ch):
        return unicodedata.category(ch).startswith("M")

    for key in VOCABS:
        charset = set(resolve_vocab_charset([key]))
        by_script = defaultdict(int)
        for c in charset:
            s = _script_of(c)
            if s != "COMMON":
                by_script[s] += 1
        oversized = {s for s, n in by_script.items() if n > _LARGE_SCRIPT_THRESHOLD}
        aug, _, _ = augment_words_for_coverage([], charset, min_count=2, seed=7)
        covered = set()
        for tok in aug:
            assert tok, f"{key}: empty token emitted"
            covered.update(tok)
            assert not is_mark(tok[0]), f"{key}: token starts with a combining mark: {tok!r}"
            for i, ch in enumerate(tok):
                if unicodedata.combining(ch) == 9:
                    assert i + 1 < len(tok) and unicodedata.category(tok[i + 1]).startswith("L"), (
                        f"{key}: dangling virama in {tok!r}"
                    )
                if is_mark(ch):
                    j = i - 1
                    while j >= 0 and is_mark(tok[j]):
                        j -= 1
                    assert (
                        j >= 0
                        and unicodedata.category(tok[j]).startswith("L")
                        and (_script_of(tok[j]) == _script_of(ch))
                    ), f"{key}: mark not on a same-script base letter in {tok!r}"
        gaps = [
            c
            for c in charset
            if c not in covered
            and unicodedata.category(c)[0] in "LM"
            and _script_of(c) not in oversized
            and _script_of(c) != "?"
        ]
        assert not gaps, f"{key}: uncovered letters/marks {[hex(ord(c)) for c in gaps]}"


def test_all_vocabs_synthesis_is_structurally_valid():
    # Every VOCABS key must synthesise coverage tokens that respect the script's
    # diacritic/vowel/matra/virama rules: no token may start with a combining
    # mark, a virama must sit between two letters, and a Thai/Lao pre-base vowel
    # must be followed by a consonant. Forcing an empty corpus exercises the
    # pure-synthesis path for every script at once.
    import unicodedata

    from generator.components.vocab_coverage import _PREBASE_VOWELS
    from generator.components.vocabs import VOCABS

    def _is_letter(c: str) -> bool:
        return unicodedata.category(c).startswith("L")

    for key, chars in VOCABS.items():
        augmented, _, _ = augment_words_for_coverage([], chars, min_count=2, seed=7)
        viramas = {c for c in chars if unicodedata.combining(c) == 9}
        for token in augmented:
            assert not unicodedata.category(token[0]).startswith("M"), (key, token)
            for i, c in enumerate(token):
                follows_letter = i + 1 < len(token) and _is_letter(token[i + 1])
                if c in viramas:
                    assert follows_letter, f"dangling virama in {key}: {token!r}"
                if c in _PREBASE_VOWELS:
                    assert follows_letter, f"orphaned pre-base vowel in {key}: {token!r}"
