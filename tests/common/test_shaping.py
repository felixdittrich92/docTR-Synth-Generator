import os

import pytest
from conftest import build_font

from generator.components.font_selector import FontSelector
from generator.components.shaping import HAS_HARFBUZZ, ShapingChecker

pytestmark = pytest.mark.skipif(not HAS_HARFBUZZ, reason="uharfbuzz not installed")


@pytest.fixture
def fonts(tmp_path):
    ascii_letters = "abcdefghijklmnopqrstuvwxyz "
    good = build_font(str(tmp_path / "Good.ttf"), ascii_letters + "कखि◌")
    hollow = build_font(str(tmp_path / "Hollow.ttf"), "abcdefghijklmnopqrstuvwxy ", empty="z")
    return {"good": good, "hollow": hollow, "dir": str(tmp_path)}


def test_clean_text_passes(fonts):
    assert ShapingChecker().problem("abc def", fonts["good"]) is None


def test_missing_glyph_is_notdef(fonts):
    assert ShapingChecker().problem("abc€", fonts["good"]) == "notdef"


def test_empty_glyph_is_invisible(fonts):
    checker = ShapingChecker()
    assert checker.problem("zzz", fonts["hollow"]) == "invisible"
    assert checker.problem("abc", fonts["hollow"]) is None


def test_broken_cluster_draws_dotted_circle(fonts):
    checker = ShapingChecker()
    assert checker.problem("कि", fonts["good"]) is None  # matra after its consonant
    assert checker.problem("िक", fonts["good"]) == "dotted_circle"  # matra in visual order


def test_disabled_checker_accepts_everything(fonts):
    assert ShapingChecker(enabled=False).ok("abc€", fonts["good"])


def test_font_selector_rejects_faces_that_would_lie(fonts):
    selector = FontSelector(fonts["dir"], auto_download=False)
    # Both faces map 'z' in their cmap, only one draws it.
    assert selector.get_font_for_text("zebra") == fonts["good"]
    assert selector.supports("zebra", fonts["good"])
    assert not selector.supports("zebra", fonts["hollow"])
    assert selector.get_font_for_text("िक") is None  # malformed in every face


def test_font_selector_without_verification_uses_cmap_only(fonts):
    selector = FontSelector(fonts["dir"], auto_download=False, verify_shaping=False)
    assert selector.supports("zebra", fonts["hollow"])
    assert os.path.exists(selector.get_font_for_text("zebra"))
