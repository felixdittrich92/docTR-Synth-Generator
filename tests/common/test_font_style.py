import json
import random
from argparse import Namespace
from collections import Counter

import numpy as np
import pytest
from conftest import build_font
from PIL import Image, features

from generator.components import GenerationConfig
from generator.components.font_styles import (
    COLOR_NAMES,
    Decorations,
    StyleSpec,
    _standard_weight,
    color_name,
    font_traits,
    paint_highlight,
    weight_name,
)
from generator.components.shaping import HAS_HARFBUZZ
from generator.components.text_renderer import TextRenderer
from generator.components.vocabs import VOCABS

needs_shaping = pytest.mark.skipif(not (HAS_HARFBUZZ and features.check("raqm")), reason="needs uharfbuzz and raqm")


def test_font_traits_read_the_tables(tmp_path):
    bold = build_font(str(tmp_path / "b.ttf"), "abc", family="Fira Sans Bold", style="Bold", weight=700)
    it = build_font(str(tmp_path / "i.ttf"), "abc", family="Fira Sans", style="Light Italic", weight=300, italic=True)
    tb, ti = font_traits(bold), font_traits(it)
    assert (tb.family, tb.weight, tb.italic) == ("Fira Sans", 700, False)  # style word stripped from the family
    assert (ti.family, ti.weight, ti.italic) == ("Fira Sans", 300, True)
    assert tb.weight_range is None and tb.underline == (-100, 50)


@pytest.mark.parametrize(("angle", "italic"), [(-14.0, True), (17.0, False), (-0.2, False)])
def test_only_a_right_leaning_angle_makes_an_upright_regular_italic(tmp_path, angle, italic):
    from fontTools.ttLib import TTFont

    path = str(tmp_path / f"r{angle}.ttf")
    build_font(path, "abc", family="Slant Test")
    font = TTFont(path)
    font["post"].italicAngle = angle  # italic bit unset, Regular subfamily: only the angle speaks
    font.save(path)
    assert font_traits(path).italic is italic


def test_weights_snap_to_the_css_scale():
    assert _standard_weight(250) == 200 and _standard_weight(275) == 200  # Windows ExtraLight stand-ins
    assert _standard_weight(5) == 500 and _standard_weight(700) == 700
    assert weight_name(700) == "Bold" and weight_name(640) == "SemiBold" and weight_name(100) == "Thin"


@pytest.mark.parametrize(
    "ink, paper, name",
    [
        ((20, 20, 25), (245, 245, 240), "black"),
        ((40, 30, 20), (245, 245, 240), "black"),  # near-black with a warm tint is still black
        ((120, 120, 120), (245, 245, 240), "gray"),
        ((235, 235, 235), (30, 30, 30), "white"),  # light ink on a dark band
        ((200, 30, 30), (245, 245, 240), "red"),
        ((110, 20, 25), (245, 245, 240), "red"),  # maroon
        ((110, 60, 30), (245, 245, 240), "brown"),
        ((30, 60, 190), (245, 245, 240), "blue"),
        ((30, 140, 60), (245, 245, 240), "green"),
        ((130, 50, 170), (245, 245, 240), "purple"),
    ],
)
def test_colour_names(ink, paper, name):
    assert color_name(np.array(ink, float), np.array(paper, float)) == name
    assert name in COLOR_NAMES


def test_underline_and_strike_follow_the_font_metrics(tmp_path):
    path = build_font(str(tmp_path / "f.ttf"), "abc ")
    renderer = TextRenderer(GenerationConfig.flat(task="recognition", output_dir=str(tmp_path)))
    font = renderer._get_font(path, 100)
    baseline = font.getmetrics()[0]
    double = renderer._decoration_lines("abc", font, path, Decorations(underline=2, strikethrough=True))
    assert len(double) == 3
    under, second, strike = double
    assert baseline < under[1] < second[1]  # both rules below the baseline, one under the other
    assert strike[3] < baseline  # strike-out above it
    assert under[2] == pytest.approx(font.getlength("abc"))  # exactly as wide as the text


def test_decorations_are_drawn_into_the_coverage(tmp_path):
    path = build_font(str(tmp_path / "f.ttf"), "abc ")
    cfg = GenerationConfig.flat(task="recognition", output_dir=str(tmp_path), rotation_prob=0.0, perspective_prob=0.0)
    renderer = TextRenderer(cfg, word_augmentations=False)
    plain = np.asarray(renderer.render_coverage("abc", path, 40).getchannel("A"))
    lined = np.asarray(renderer.render_coverage("abc", path, 40, decorations=Decorations(underline=1)).getchannel("A"))
    assert lined.shape[0] > plain.shape[0]  # the canvas grew to hold the rule
    assert ((lined > 128).mean(axis=1) > 0.8).any()  # a row of ink across the whole word


def test_highlight_tints_only_the_word_box():
    random.seed(0)
    bg = Image.new("RGB", (200, 60), (250, 250, 250))
    out = np.asarray(paint_highlight(bg, (50, 20, 150, 40), (255, 240, 60)), dtype=int)
    assert out[30, 100, 2] < 200  # yellow: the blue channel drops under the marker
    assert out[5, 5].tolist() == [250, 250, 250]  # paper outside stays as it was


@needs_shaping
def test_styled_crops_report_their_ink(tmp_path, temp_image_dir):
    from generator.components.font_selector import FontSelector
    from generator.components.generator import TextImageGenerator

    path = build_font(str(tmp_path / "f.ttf"), VOCABS["english"] + " ")
    cfg = GenerationConfig.flat(
        task="recognition", output_dir=str(tmp_path), font_dir=str(tmp_path), bg_image_dir=temp_image_dir,
        grayscale_prob=0.0, bitonal_prob=0.0,
    )  # fmt: skip
    gen = TextImageGenerator(cfg, font_selector=FontSelector(str(tmp_path), auto_download=False))
    random.seed(1)
    np.random.seed(1)
    result = gen.generate_styled("Invoice", path, StyleSpec(ink_hue=(215, 35, 35), decorations=Decorations(1)))
    assert result is not None
    image, info = result
    assert image.mode == "RGB" and info["color"] in COLOR_NAMES


def test_styled_crops_carry_neighbouring_lines_but_never_a_rule(tmp_path, temp_image_dir):
    from generator.components.font_selector import FontSelector
    from generator.components.generator import TextImageGenerator
    from generator.components.text_renderer import TextStyle

    path = build_font(str(tmp_path / "f.ttf"), VOCABS["english"] + " ")
    cfg = GenerationConfig.flat(task="recognition", output_dir=str(tmp_path), font_dir=str(tmp_path),
                                bg_image_dir=temp_image_dir)  # fmt: skip
    gen = TextImageGenerator(cfg, font_selector=FontSelector(str(tmp_path), auto_download=False))
    style = TextStyle(fill_color=(0, 0, 0), opacity=255, bold_width=0)

    def full_rows(rules):  # a rule spans the crop (or the word): fragments of text never do
        random.seed(0)
        found = 0
        for _ in range(60):
            layer = gen._context((200, 80), (20, 25, 180, 55), "Text", path, 24, style, rules=rules)
            if layer is not None:
                alpha = np.asarray(layer.getchannel("A")) > 0
                found += int((alpha.mean(axis=1) > 0.75).any())
        return found

    assert full_rows(True) > 0 and full_rows(False) == 0

    random.seed(0)
    sides = below = 0
    for _ in range(60):  # cell borders: sides and the edge above, never a rule below the word (an underline)
        layer = gen._context((200, 80), (20, 25, 180, 55), "Text", path, 24, style, rules=False, cells=True)
        if layer is not None:
            alpha = np.asarray(layer.getchannel("A")) > 0
            sides += int((alpha.mean(axis=0) > 0.95).any())
            below += int((alpha[55:].mean(axis=1) > 0.75).any())
    assert sides > 0 and below == 0


def test_typeset_hands_are_written_or_printed():
    from generator.recipes.font_style import StyleWorker

    rng = random.Random(0)
    written = [StyleWorker._spec(None, rng, handwritten=True, written=True).hand for _ in range(200)]
    hands = [StyleWorker._spec(None, rng, handwritten=True).hand for _ in range(200)]
    assert min(written) >= 0.4  # what a hand wrote always shows the hand
    assert 0 < sum(h == 0 for h in hands) < 100  # the other hands: some clean renders, as before


def test_plan_reaches_bold_and_italic_shares():
    from generator.recipes.font_style import BOLD_SHARE, ITALIC_SHARE, _plan

    variable = [
        {"file": "v.ttf", "italic": False, "weight": 400, "weight_range": [100, 900]},
        {"file": "vi.ttf", "italic": True, "weight": 400, "weight_range": [100, 900]},
    ]
    static = [{"file": "s.ttf", "italic": False, "weight": 400, "weight_range": None}]
    families = {
        "Var": {"variants": variable, "languages": ["en"]},
        **{f"Static{i}": {"variants": static, "languages": ["en", "de"]} for i in range(8)},
    }
    rows = _plan(4000, families, {"en": 0.7, "de": 0.3}, random.Random(0))
    bold = np.mean([w >= 600 for _, _, w, _, _ in rows])
    italic = np.mean([f == "vi.ttf" for _, f, _, _, _ in rows])
    assert abs(bold - BOLD_SHARE) < 0.03 and abs(italic - ITALIC_SHARE) < 0.03
    statics = [r for r in rows if r[0].startswith("Static")]
    counts = np.bincount([int(r[0][-1]) for r in statics])
    assert counts.min() > 0.8 * counts.max()  # families take turns


def test_copied_latin_is_found_and_duplicates_merge():
    from generator.recipes.font_style import _merge_duplicates, _same_look

    img = np.zeros((96, 300), np.float32)
    img[50:80, 10:290] = 1.0
    assert _same_look((280.0, img), (280.5, img.copy()))
    other = img.copy()
    other[50:80, 150:290] = 0.0
    assert not _same_look((280.0, img), (280.0, other))  # same advance, different letters
    assert not _same_look((280.0, img), (310.0, img))  # a condensed cut is its own design
    fams = {
        "Mplus 1p": {"languages": {"ja"}, "variants": {"a.ttf": {}}},
        "M PLUS 1p": {"languages": {"en"}, "variants": {"b.ttf": {}}},
    }
    merged = _merge_duplicates(fams)
    assert list(merged) == ["M PLUS 1p"] and merged["M PLUS 1p"]["languages"] == {"en", "ja"}


@needs_shaping
def test_font_style_recipe_end_to_end(tmp_path, temp_image_dir):
    from generator.recipes import font_style

    src = tmp_path / "fonts_src"
    src.mkdir()
    chars = VOCABS["english"] + " "
    build_font(str(src / "AlphaSans-Regular.ttf"), chars, family="Alpha Sans")
    build_font(str(src / "AlphaSans-Bold.ttf"), chars, family="Alpha Sans", style="Bold", weight=700)
    build_font(str(src / "AlphaSans-Italic.ttf"), chars, family="Alpha Sans", style="Italic", italic=True)
    build_font(str(src / "BetaSerif-Regular.ttf"), chars, family="Beta Serif", glyph=(200, 440))  # its own design
    corpus = tmp_path / "cache" / "corpora"
    corpus.mkdir(parents=True)
    words = (
        [f"word{i}" for i in range(300)]
        + ["Invoice", "total", "amount", "due"] * 20
        + ["two words"] * 50
        + ["a", "I", "of", "to"] * 10
    )
    (corpus / "frek_en.txt").write_text("\n".join(words), encoding="utf-8")
    args = Namespace(
        out=str(tmp_path / "fs"), languages="en", num_train=40, num_val=8, shard_size=25, workers=1,
        font_src=str(src), cache=str(tmp_path / "cache"), bg_dir=temp_image_dir, offline=True, png=False,
        samples=8, seed=0,
    )  # fmt: skip
    font_style.prepare(args)
    meta = tmp_path / "fs" / "meta"
    classes = json.loads((meta / "classes.json").read_text(encoding="utf-8"))["class_to_idx"]
    assert classes == {"Alpha Sans": 0, "Beta Serif": 1}
    families = json.loads((meta / "families.json").read_text(encoding="utf-8"))
    assert len(families["Alpha Sans"]["variants"]) == 3 and families["Beta Serif"]["category"] == "SERIF"

    font_style.render(args)
    records = []
    for folder in [tmp_path / "fs" / "val", *sorted((tmp_path / "fs" / "train").glob("shard_*"))]:
        labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
        assert all((folder / "images" / name).exists() for name in labels)
        records += labels.values()
    assert len(records) >= 40
    for r in records:
        assert r["family"] in classes and r["color"] in COLOR_NAMES
        assert 1 <= len(r["text"]) <= 32 and not any(c.isspace() for c in r["text"])  # one word, as docTR boxes it
        assert r["underline"] in ("none", "single", "double")
        assert r["bold"] == (r["weight"] >= 600)
        if r["family"] == "Beta Serif":
            assert not r["bold"] and not r["italic"]  # it has no such faces: never labelled so
    assert any(r["bold"] for r in records) and any(r["italic"] for r in records)
    assert font_style.preview(args).exists()


def test_copied_letters_lose_their_languages_in_every_script(tmp_path):
    from generator.recipes.font_style import _borrowed

    latin = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
    hebrew = "".join(chr(c) for c in range(0x05D0, 0x05EB))
    build_font(str(tmp_path / "base.ttf"), latin, family="Base Sans")
    build_font(str(tmp_path / "heb.ttf"), latin + hebrew, family="Base Sans Hebrew")  # Latin copied from Base Sans
    build_font(str(tmp_path / "own.ttf"), latin + hebrew, family="Own", glyph=(200, 440))  # its own design
    manifest = {"fonts": {n: {"chars": c} for n, c in (("base.ttf", latin), ("heb.ttf", latin + hebrew),
                                                       ("own.ttf", latin + hebrew))}}  # fmt: skip
    (tmp_path / "fonts.json").write_text(json.dumps(manifest), encoding="utf-8")

    def fam(file, langs):
        return {"languages": set(langs), "variants": {file: {"file": file, "italic": False, "weight": 400,
                                                             "weight_range": None}}}  # fmt: skip

    families = {
        "Base Sans": fam("base.ttf", ["en"]),
        "Base Sans Hebrew": fam("heb.ttf", ["en", "he"]),
        "Own": fam("own.ttf", ["en", "he"]),
    }
    probes = {"LATIN": ("Hamburgefonstiv", ["en"]), "HEBREW": (hebrew[:8], ["he"])}
    drop = _borrowed(families, tmp_path, probes)
    assert drop == {"Base Sans Hebrew": {"en"}}  # Base Sans keeps Latin; both Hebrew designs differ


def test_scaled_copies_lose_their_languages(tmp_path):
    from generator.recipes.font_style import _borrowed

    latin = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
    hebrew = "".join(chr(c) for c in range(0x05D0, 0x05EB))
    build_font(str(tmp_path / "base.ttf"), latin, family="Base Sans")
    # The Latin of Base Sans drawn 10% larger: a copy (Noto Sans JP draws Source Sans larger)
    build_font(str(tmp_path / "big.ttf"), latin + hebrew, family="Base Sans Hebrew", scale=1.1)
    # Another design, also drawn larger: not a copy
    build_font(str(tmp_path / "own.ttf"), latin + hebrew, family="Own", glyph=(200, 440), scale=1.1)
    files = {"base.ttf": latin, "big.ttf": latin + hebrew, "own.ttf": latin + hebrew}
    manifest = {"fonts": {n: {"chars": c} for n, c in files.items()}}
    (tmp_path / "fonts.json").write_text(json.dumps(manifest), encoding="utf-8")

    def fam(file, langs):
        return {"languages": set(langs), "variants": {file: {"file": file, "italic": False, "weight": 400,
                                                             "weight_range": None}}}  # fmt: skip

    families = {
        "Base Sans": fam("base.ttf", ["en"]),
        "Base Sans Hebrew": fam("big.ttf", ["en", "he"]),
        "Own": fam("own.ttf", ["en", "he"]),
    }
    drop = _borrowed(families, tmp_path, {"LATIN": ("Hamburgefonstiv", ["en"])})
    assert drop == {"Base Sans Hebrew": {"en"}}


def test_latin_lookalikes_of_document_fonts_keep_only_their_script():
    from generator.recipes.font_style import _foreign_alphabets

    families = {
        "Scheherazade New": {"languages": {"ar", "en", "de", "ru"}},  # its Latin and Cyrillic read as Times
        "Noto Sans Arabic": {"languages": {"ar", "en"}},  # not listed: a copy, which _borrowed drops
        "Kalam": {"languages": {"hi", "en"}},  # its Latin is a hand of its own
        "Roboto": {"languages": {"en", "ru"}},
    }
    families["Aparajita"] = {"languages": {"hi", "en"}}  # a serif made for Devanagari: Times-like Latin
    families["Poppins"] = {"languages": {"hi", "en"}}  # a sans made for Devanagari, a Latin design of its own
    families["Lora"] = {"languages": {"en", "ru"}}
    categories = {"Aparajita": "SERIF", "Poppins": "SANS_SERIF", "Lora": "SERIF", "Kalam": "HANDWRITING"}
    scripts = {"Aparajita": "Deva", "Poppins": "Deva", "Lora": "Latn", "Kalam": "Deva"}
    drop = _foreign_alphabets(families, categories, scripts)
    assert drop == {"Scheherazade New": {"en", "de", "ru"}, "Aparajita": {"en"}}


def test_designs_merge_with_the_languages_of_their_faces():
    from generator.recipes.font_style import _merge_designs, _plan

    def fam(file, langs):
        face = {"file": file, "italic": False, "weight": 400, "weight_range": None, "languages": set(langs)}
        return {"languages": set(langs), "variants": {file: face}}

    families = {
        "Arimo": fam("arimo.ttf", ["en", "ru", "el"]),
        "Arial": fam("arial.ttf", ["en", "ar"]),
        "Fira Sans": fam("fira.ttf", ["en"]),
        "Fira Sans Condensed": fam("firac.ttf", ["en", "ru"]),
    }
    _merge_designs(families, {"Fira Sans": ("Fira Sans Condensed",)})
    _merge_designs(families, {"Arimo": ("Arial",)}, frozenset({"LATIN", "CYRILLIC", "GREEK"}))
    assert families["Fira Sans"]["languages"] == {"en", "ru"} and not families["Fira Sans Condensed"]["languages"]
    assert families["Arial"]["languages"] == {"ar"}  # its Arabic stays its own
    assert set(families["Arimo"]["variants"]) == {"arimo.ttf", "arial.ttf"}
    variants = [{**v, "languages": sorted(v["languages"] & families["Arimo"]["languages"])}
                for v in families["Arimo"]["variants"].values()]  # fmt: skip
    rows = _plan(400, {"Arimo": {"languages": ["el", "en", "ru"], "variants": variants}}, {}, random.Random(0))
    assert all(lang == "en" for _, file, _, lang, _ in rows if file == "arial.ttf")  # never Greek in Arial


def test_plan_draws_document_fonts_more_often():
    from generator.recipes.font_style import _plan

    variant = [{"file": "s.ttf", "italic": False, "weight": 400, "weight_range": None}]
    families = {name: {"variants": variant, "languages": ["en"]} for name in ("Doc", "A", "B", "C")}
    rows = _plan(3000, families, {"en": 1.0}, random.Random(0), {"Doc": 3})
    counts = Counter(r[0] for r in rows)
    assert 2.7 < counts["Doc"] / counts["A"] < 3.3 and abs(counts["A"] - counts["B"]) < 0.1 * counts["A"]


def test_document_fonts_come_from_the_cache_offline(tmp_path, monkeypatch):
    import zipfile

    from generator.recipes import fonts as font_style

    folder = tmp_path / "doc_fonts"
    folder.mkdir()
    monkeypatch.setattr(font_style, "DOC_FONTS", [("tinos", "Tinos-Regular.ttf")])
    (folder / "tinos__Tinos-Regular.ttf").write_bytes(b"ttf")
    url = "https://example.org/fonts/lm.zip"
    monkeypatch.setattr(
        font_style, "DOC_FONT_ARCHIVES", [(url, ["lm/otf/lmroman10-regular.otf", "lm/otf/missing.otf"])]
    )
    with zipfile.ZipFile(folder / "lm.zip", "w") as z:
        z.writestr("lm/otf/lmroman10-regular.otf", b"otf")
    with pytest.raises(KeyError):  # a member missing from the archive is an error, not a silent gap
        font_style.doc_fonts(tmp_path, offline=True)
    monkeypatch.setattr(font_style, "DOC_FONT_ARCHIVES", [(url, ["lm/otf/lmroman10-regular.otf"])])
    paths = font_style.doc_fonts(tmp_path, offline=True)
    assert [p.name for p in paths] == ["tinos__Tinos-Regular.ttf", "lmroman10-regular.otf"]
    assert (folder / "lmroman10-regular.otf").read_bytes() == b"otf"


def test_known_categories_are_valid():
    from generator.recipes.font_style import CATEGORIES, KNOWN_CATEGORIES

    assert set(KNOWN_CATEGORIES.values()) <= set(CATEGORIES)
    assert KNOWN_CATEGORIES["TeX Gyre Termes"] == "SERIF" and KNOWN_CATEGORIES["TeX Gyre Heros"] == "SANS_SERIF"


def test_hand_warp_moves_the_ink_like_a_hand(tmp_path):
    from generator.components.font_styles import hand_warp

    coverage = Image.new("RGBA", (300, 80), (0, 0, 0, 0))
    alpha = np.zeros((80, 300), np.uint8)
    alpha[20:60, 20:280:20] = 255  # vertical strokes, like letters
    alpha[66:69, 20:280] = 255  # an underline
    coverage.putalpha(Image.fromarray(alpha))
    assert hand_warp(coverage, 0.0) is coverage
    warped = hand_warp(coverage, 1.0, np.random.default_rng(0))
    out = np.asarray(warped.getchannel("A"), np.float32)
    pad = (warped.width - coverage.width) // 2
    assert warped.mode == "RGBA" and warped.height - coverage.height == 2 * pad > 0
    # the ink moves (the underline is no longer a straight row) but none of it is lost
    rows = out[:, 20 + pad : 280 + pad]
    underline_rows = {int(np.argmax(rows[60 + pad :, x])) for x in range(0, 260, 10)}
    assert len(underline_rows) > 1
    assert 0.6 * alpha.sum() < out.sum() <= alpha.sum() * 1.05


def test_handwritten_families():
    from generator.recipes.font_style import PRINTED_LOOKALIKES, is_handwritten

    assert is_handwritten("Caveat", "HANDWRITING")
    assert not is_handwritten("Comic Neue", "HANDWRITING") and "Comic Neue" in PRINTED_LOOKALIKES
    assert not is_handwritten("Roboto", "SANS_SERIF")


def test_handwriting_fonts_come_from_the_cache_offline(tmp_path, monkeypatch):
    from generator.recipes import font_style

    folder = tmp_path / "handwriting_fonts"
    folder.mkdir()
    (folder / "caveat__Caveat_wght_.ttf").write_bytes(b"ttf")
    monkeypatch.setattr(font_style, "_HANDWRITING_FONTS", {"latin": [("caveat", ["Caveat[wght].ttf"])]})
    monkeypatch.setattr(font_style, "_SIGNATURE_FONTS", {"latin": [("allura", ["Allura-Regular.ttf"])]})
    paths = font_style._handwriting_fonts(tmp_path, offline=True)
    assert [p.name for p in paths] == ["caveat__Caveat_wght_.ttf"]  # the signature face is not cached


def test_merged_classes_keep_the_share_of_their_faces():
    from generator.recipes.font_style import DOC_FONT_WEIGHT, _family_weights

    weights = _family_weights({"Tinos", "Lato", "Fira Sans", "Lora"}, {"Tinos", "TeX Gyre Termes", "Carlito"})
    assert weights["Tinos"] == 2 * DOC_FONT_WEIGHT + 1  # Tinos and TeX Gyre Termes (documents), Roman Unicode
    assert weights["Lato"] == DOC_FONT_WEIGHT + 1  # Carlito (Calibri) is drawn from Lato
    assert weights["Fira Sans"] == 3 and weights["Lora"] == 1  # two condensed cuts; no merge
