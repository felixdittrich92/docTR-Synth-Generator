import json
import random
import unicodedata
from argparse import Namespace
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import build_font
from PIL import Image, features

from generator.components.shaping import HAS_HARFBUZZ
from generator.components.vocab_coverage import augment_words_for_coverage
from generator.components.vocabs import VOCABS
from generator.recipes import render as render_mod
from generator.recipes.corpus import Corpus, catalan_interpunct, fix_prebase_matras, sr_latn_to_cyrl, uzbek_apostrophes
from generator.recipes.languages import PRESETS, primary_script, resolve_languages, script_of
from generator.recipes.recognition import clean_words, group_weights, quotas
from generator.recipes.text import (
    attach_punct,
    capitalize,
    cycle_sample,
    digit_system,
    is_ambiguous,
    mixes_digit_systems,
    native_digit_systems,
    native_number_tokens,
    upper,
)

needs_shaping = pytest.mark.skipif(
    not (HAS_HARFBUZZ and features.check("raqm")), reason="recipes need uharfbuzz and Pillow with raqm"
)

# -- languages -----------------------------------------------------------------


def test_presets_resolve_and_european_stays_in_its_scripts():
    langs = resolve_languages("european")
    assert "de" in langs and "ru" in langs and "el" in langs
    assert {primary_script(lang) for lang in langs} == {"LATIN", "CYRILLIC", "GREEK"}
    assert not {"ja", "ko", "zh"} & set(PRESETS["multilingual"])
    assert resolve_languages("de, fr,de") == ["de", "fr"]
    with pytest.raises(ValueError):
        resolve_languages("klingon")


def test_modifier_letters_belong_to_their_host_script():
    assert script_of("ʻ") == "LATIN"  # Uzbek oʻ
    assert script_of("ჼ") == "GEORGIAN"


# -- corpus repairs ------------------------------------------------------------


def test_prebase_matras_are_moved_behind_their_consonant():
    assert fix_prebase_matras("िकया") == "किया"
    assert fix_prebase_matras("(िक") == "(कि"
    assert fix_prebase_matras("किया") == "किया"  # already correct
    assert fix_prebase_matras("েকান") == "কোন"  # and re-composed (NFC)


def test_orthography_conversions():
    assert sr_latn_to_cyrl("Njegoš ljubav") == "Његош љубав"
    assert uzbek_apostrophes(["o'zbek"]) == ["oʻzbek"]
    assert catalan_interpunct(["coŀlecció"]) == ["col·lecció"]


def test_corpus_reads_cached_lists_offline(tmp_path):
    rows = [f"word{i} {100 if i < 50_000 else 3}" for i in range(50_010)]
    (tmp_path / "hermit_xx_full.txt").write_text("\n".join(rows), encoding="utf-8")
    words = Corpus(tmp_path, full_lists=True, offline=True).frequency_list("xx")
    assert len(words) == 50_000  # past rank 50k only frequent words are kept
    assert Corpus(tmp_path, offline=True).frequency_list("yy") == []  # offline: no download


# -- text helpers ---------------------------------------------------------------


def test_language_aware_casing():
    assert upper("istanbul", "tr") == "İSTANBUL"
    assert upper("άνθρωπος", "el") == "ΑΝΘΡΩΠΟΣ"  # all-caps Greek drops the tonos
    assert capitalize("ijsland", "nl") == "IJsland"
    assert capitalize("it's", "en") == "It's"


def test_punctuation_never_leaves_the_charset():
    rng = random.Random(0)
    charset = set("abc,.")
    for _ in range(200):
        assert set(attach_punct("abc", "en", charset, rng)) <= charset


def test_homoglyph_words_are_ambiguous():
    assert is_ambiguous("се", "CYRILLIC")  # renders exactly like Latin "ce"
    assert not is_ambiguous("так", "CYRILLIC")
    assert is_ambiguous("ΤΟ", "GREEK")
    assert not is_ambiguous("ce", "LATIN")


def test_digit_systems():
    assert digit_system("7") == "ASCII"
    assert digit_system("٧") == "٠"
    assert digit_system("a") is None
    assert mixes_digit_systems("۰42") and not mixes_digit_systems("۰۴۲")
    # Persian has both the Arabic-Indic and the extended Arabic-Indic digits.
    systems = native_digit_systems("fa", set(VOCABS["persian"]))
    assert len(systems) == 2
    tokens = native_number_tokens(systems[0], 300, random.Random(0))
    assert all(not mixes_digit_systems(t) for t in tokens)
    assert set("".join(tokens)) >= set(systems[0].values())  # every digit is used


def test_cycle_sample_repeats_evenly():
    counts = Counter(cycle_sample(list("abcde"), 52, random.Random(0)))
    assert max(counts.values()) - min(counts.values()) <= 1


# -- coverage synthesis ----------------------------------------------------------


def test_modifier_letters_are_placed_inside_latin_words():
    words = ["kitob", "maktab", "olma"] * 3
    out, added, _ = augment_words_for_coverage(words, {"ʻ", "ʼ"}, min_count=5, seed=0)
    synthetic = out[len(words) :]
    assert added and all(any(c.isascii() and c.isalpha() for c in t) for t in synthetic)


def test_capitals_start_a_word():
    words = ["guichet", "maison", "école"] * 3
    out, _, _ = augment_words_for_coverage(words, {"Ổ"}, min_count=10, seed=0)
    assert all(t.startswith("Ổ") for t in out[len(words) :])


def test_accept_gate_rejects_tokens():
    words = ["क", "कम"]
    out, _, _ = augment_words_for_coverage(words, {"ि"}, min_count=3, seed=0, accept=lambda t: "ि" not in t[:1])
    assert all(not t.startswith("ि") for t in out)


# -- sampling --------------------------------------------------------------------


def test_clean_words_filters_scripts_vocab_and_homoglyphs():
    rng = random.Random(0)
    charset = set(VOCABS["russian"])
    words, stats = clean_words("ru", ["дом", "dом", "се", "дом", "ok", "a" * 40], charset, 100, rng)
    assert words == ["дом"]
    assert stats["wrong_or_mixed_script"] == 2 and stats["ambiguous_homoglyph"] == 1


def test_weights_and_quotas():
    w = group_weights({"de": 1000, "fr": 500, "ru": 800})
    assert abs(sum(w.values()) - 1) < 1e-9
    assert w["ru"] == pytest.approx(0.12 / 0.46)  # cyrillic family share, renormalised
    q = quotas(w, 1001)
    assert sum(q.values()) == 1001


# -- resumable rendering ------------------------------------------------------------


class _Echo(render_mod.Worker):
    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)

    def render(self, payload, path):
        if payload in self.fail_on:
            return None
        Image.new("RGB", (4, 4)).save(path, "PNG")
        return payload


def test_render_part_resumes_and_removes_stale_images(tmp_path, monkeypatch):
    part = tmp_path / "part"
    items = [(f"{i}.png", f"t{i}") for i in range(10)]
    # A previous run finished three samples and left a stale image for a fourth.
    (part / "images").mkdir(parents=True)
    with open(part / "progress.jsonl", "w", encoding="utf-8") as fh:
        for i in range(3):
            fh.write(json.dumps({"f": f"{i}.png", "r": f"t{i}"}) + "\n")
            Image.new("RGB", (4, 4)).save(part / "images" / f"{i}.png")
        fh.write('{"f": "3.png", "r"')  # torn line from a crash
    Image.new("RGB", (4, 4)).save(part / "images" / "5.png")

    worker = _Echo(fail_on={"t5"})
    monkeypatch.setattr(render_mod, "_WORKER", worker)
    calls = []
    original = render_mod._work
    monkeypatch.setattr(render_mod, "_work", lambda items: calls.append(len(items)) or original(items))
    with ThreadPoolExecutor(1) as pool:
        labels = render_mod.render_part(pool, items, part, chunk=4)

    assert sum(calls) == 7  # only the unfinished samples were rendered
    assert set(labels) == {f"{i}.png" for i in range(10)} - {"5.png"}
    assert not (part / "images" / "5.png").exists()  # failed sample: stale file removed


# -- end to end ---------------------------------------------------------------------


@needs_shaping
def test_recognition_recipe_end_to_end(tmp_path, temp_image_dir):
    from generator.recipes import recognition

    fonts = tmp_path / "fonts_src"
    fonts.mkdir()
    for name in ("A", "B", "C"):
        build_font(str(fonts / f"{name}.ttf"), VOCABS["english"] + " ")
    corpus = tmp_path / "cache" / "corpora"
    corpus.mkdir(parents=True)
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "pack", "my", "box", "with", "five"]
    words += ["dozen", "liquor", "jugs", "sphinx", "of", "black", "quartz", "judge", "vow"] * 10
    (corpus / "frek_en.txt").write_text("\n".join(f"{w}{i}" if i else w for i, w in enumerate(words)), encoding="utf-8")

    args = Namespace(
        out=str(tmp_path / "ds"), languages="en", num_train=300, num_val=60, shard_size=200, workers=1,
        words_per_lang=1000, min_char_count=3, min_char_count_val=1, extra_chars="", font_src=str(fonts),
        cache=str(tmp_path / "cache"), bg_dir=temp_image_dir, max_fonts_per_lang=0, full_lists=False,
        offline=True, png=False,
    )  # fmt: skip
    recognition.prepare(args)
    vocab = (tmp_path / "ds" / "meta" / "vocab.txt").read_text(encoding="utf-8")
    assert set(vocab) == set(VOCABS["english"])
    # Real crops are full of short words ('of', 'my', '1'); they must reach the pool.
    rows = recognition._read_tsv(tmp_path / "ds" / "meta" / "train_labels.tsv")
    assert sum(len(t.strip("()\"'.,;:!?")) <= 3 for _, t in rows) / len(rows) > 0.15

    recognition.render(args)
    assert recognition.verify(args)  # every vocab character is in the rendered labels
    shards = sorted((tmp_path / "ds" / "train").glob("shard_*"))
    assert len(shards) == 2
    labels = json.loads((shards[0] / "labels.json").read_text(encoding="utf-8"))
    assert all((shards[0] / "images" / name).exists() for name in labels)
    assert all(unicodedata.is_normalized("NFC", t) for t in labels.values())


def test_symbol_tokens_use_documents_contexts():
    from generator.recipes.text import symbol_token

    rng = random.Random(0)
    amp = [symbol_token("&", "Haus", rng) for _ in range(60)]
    assert any(len(t) > 1 for t in amp)  # 'R&D', 'Haus&Hau' ... not only a lone '&'
    degrees = [symbol_token("°", "Haus", rng) for _ in range(30)]
    assert all(t[0].isdigit() for t in degrees)  # '25°C', '12°'
    assert symbol_token("‰", "Haus", random.Random(1))  # unknown symbols still get a token


def test_tiny_text_pages_are_rejected():
    import numpy as np

    from generator.recipes.detection import legible_at_input_size, text_heights_at_input

    def boxes(height, n, width=60):
        return np.array([[[0, 0], [width, 0], [width, height], [0, height]]] * n, dtype=float)

    assert legible_at_input_size(boxes(30, 100), (2048, 2048))  # 15 px at 1024: body text
    assert not legible_at_input_size(boxes(8, 100), (2048, 2048))  # 4 px at 1024: unreadable
    # narrow single glyphs ('l', '1') do not count as tiny vertical text
    mixed = np.concatenate([boxes(30, 100), boxes(30, 10, width=4)])
    assert len(text_heights_at_input(mixed, (2048, 2048))) == 100


@needs_shaping
def test_detection_recipe_end_to_end(tmp_path, temp_image_dir):
    from generator.recipes import detection

    fonts = tmp_path / "fonts_src"
    fonts.mkdir()
    for name in ("A", "B", "C"):
        build_font(str(fonts / f"{name}.ttf"), VOCABS["english"] + " ")
    corpus = tmp_path / "cache" / "corpora"
    corpus.mkdir(parents=True)
    words = [f"word{i}" for i in range(400)] + ["alpha", "beta", "gamma", "delta"] * 20
    (corpus / "frek_en.txt").write_text("\n".join(words), encoding="utf-8")
    args = Namespace(
        out=str(tmp_path / "det"), languages="en", num_train=3, num_val=1, part_size=2, workers=1, density="low",
        font_src=str(fonts), cache=str(tmp_path / "cache"), bg_dir=temp_image_dir, offline=True, png=False,
        negative_share=0.4,
    )  # fmt: skip
    detection.prepare(args)
    detection.render(args)
    negatives = 0
    for split, n in (("train", 3), ("val", 1)):
        labels = json.loads((tmp_path / "det" / split / "labels.json").read_text(encoding="utf-8"))
        assert len(labels) == n
        counts = json.loads((tmp_path / "det" / split / "languages.json").read_text(encoding="utf-8"))
        negatives += counts.get("none (no text)", 0)
        for name, entry in labels.items():
            image = Image.open(tmp_path / "det" / split / "images" / name)
            assert [image.height, image.width] == entry["img_dimensions"]
            assert all(0 <= x <= image.width and 0 <= y <= image.height for poly in entry["polygons"] for x, y in poly)
    empty = sum(
        not e["polygons"]
        for split in ("train", "val")
        for e in json.loads((tmp_path / "det" / split / "labels.json").read_text(encoding="utf-8")).values()
    )
    assert empty == negatives  # a page without boxes is exactly a planned negative


def test_page_plan_negatives_do_not_change_text_pages():
    from generator.recipes.detection import page_plan

    pools, nums, weights = {"en": ["a", "b", "c"]}, {"en": ["1"]}, {"en": 1.0}
    plain = list(page_plan(50, 3, pools, nums, weights))
    mixed = list(page_plan(50, 3, pools, nums, weights, negative_share=0.3))
    assert [p[:2] for p in plain] == [m[:2] for m in mixed]
    assert not any(p[2] for p in plain) and 5 <= sum(m[2] for m in mixed) <= 25


def test_mrz_tokens_follow_the_machine_readable_alphabet():
    from generator.recipes.recognition import mrz_tokens

    tokens = mrz_tokens(200, random.Random(0))
    assert all(len(t) == 30 and set(t) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<") for t in tokens)
    assert any("<<" in t for t in tokens)


def test_legibility_is_measured_at_the_training_input_size():
    import numpy as np

    from generator.recipes.detection import legible_at_input_size

    # 10 px words on a 2048 px page: 5 px at 1024 (too small for the dense floor of 6),
    # 7.5 px at 1536 - legible for a model trained at that size.
    boxes = np.array([[[0, 0], [60, 0], [60, 10], [0, 10]]] * 50, dtype=np.float64)
    assert not legible_at_input_size(boxes, (2048, 2048), min_px=6.0, max_tiny=0.03)
    assert legible_at_input_size(boxes, (2048, 2048), min_px=6.0, max_tiny=0.03, input_size=1536)


def test_shared_signs_reach_every_language():
    import random

    from generator.recipes.languages import SHARED_SIGNS, lang_chars
    from generator.recipes.text import mixes_digit_systems, symbol_token

    assert SHARED_SIGNS and all(c in lang_chars(lang) for lang in ("de", "ru", "ar", "th", "ja") for c in SHARED_SIGNS)
    assert set(VOCABS["german"]) <= set(lang_chars("de"))  # on top of the language's own vocab
    # Typographic numbers sit next to any digits: not a second digit system
    assert not any(mixes_digit_systems(t) for t in ("10²", "1½", "H₂O", "Ca²⁺", "10⁻³"))
    assert mixes_digit_systems("۰42")
    rng = random.Random(0)
    for c in SHARED_SIGNS:  # each sign is written the way documents write it, and carries it
        assert all(c in symbol_token(c, "Wort", rng) for _ in range(20))
