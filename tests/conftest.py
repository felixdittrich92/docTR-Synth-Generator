import os
import shutil
import tempfile

import pytest
from PIL import Image


@pytest.fixture
def sample_image():
    return Image.new("RGBA", (10, 10), color="red")


@pytest.fixture
def temp_image_dir():
    """Create a temporary directory with a sample image."""
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, "bg.png")
    img = Image.new("RGB", (50, 50), (100, 100, 100))
    img.save(img_path)
    yield temp_dir
    shutil.rmtree(temp_dir)


def _build_tiny_font(path, chars):
    """Build a minimal but valid TTF (filled box glyphs) covering ``chars``.

    Lets the renderer/generator tests run fully offline without depending on any
    system or downloaded font.
    """
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    def box_glyph():
        pen = TTGlyphPen(None)
        pen.moveTo((80, 0))
        pen.lineTo((80, 700))
        pen.lineTo((560, 700))
        pen.lineTo((560, 0))
        pen.closePath()
        return pen.glyph()

    fb = FontBuilder(1024, isTTF=True)
    order = [".notdef"] + [f"g{i}" for i in range(len(chars))]
    fb.setupGlyphOrder(order)
    glyphs = {".notdef": box_glyph()}
    metrics = {".notdef": (640, 80)}
    cmap = {}
    for i, ch in enumerate(chars):
        glyphs[f"g{i}"] = box_glyph()
        metrics[f"g{i}"] = (640, 80)
        cmap[ord(ch)] = f"g{i}"
    fb.setupGlyf(glyphs)
    fb.setupCharacterMap(cmap)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "Tiny", "styleName": "Regular"})
    fb.setupOS2()
    fb.setupPost()
    fb.save(path)


def build_font(path, inked, empty="", family="Shape", style="Regular", weight=400, italic=False, glyph=(80, 560)):
    """A TTF with box glyphs for ``inked`` and outline-less glyphs for ``empty``.

    ``family`` / ``style`` / ``weight`` / ``italic`` set the name, OS/2 and post tables
    the way a real face of that style has them; ``glyph`` is the x-extent of the box
    glyph, so two test families can look different.
    """
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    def box():
        pen = TTGlyphPen(None)
        pen.moveTo((glyph[0], 0))
        pen.lineTo((glyph[0], 700))
        pen.lineTo((glyph[1], 700))
        pen.lineTo((glyph[1], 0))
        pen.closePath()
        return pen.glyph()

    chars = inked + empty
    fb = FontBuilder(1024, isTTF=True)
    fb.setupGlyphOrder([".notdef"] + [f"g{i}" for i in range(len(chars))])
    glyphs = {".notdef": box()}
    for i, ch in enumerate(chars):
        glyphs[f"g{i}"] = box() if ch in inked else TTGlyphPen(None).glyph()
    fb.setupGlyf(glyphs)
    fb.setupCharacterMap({ord(ch): f"g{i}" for i, ch in enumerate(chars)})
    fb.setupHorizontalMetrics(dict.fromkeys(glyphs, (640, 80)))
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": family, "styleName": style})
    fb.setupOS2(usWeightClass=weight, fsSelection=0x01 if italic else 0x40)
    fb.setupPost(italicAngle=-12 if italic else 0, underlinePosition=-100, underlineThickness=50)
    fb.save(path)
    return path


@pytest.fixture
def tiny_font():
    """Path to a minimal offline TTF covering ASCII letters, digits and basics."""
    import string

    chars = string.ascii_letters + string.digits + " .,-"
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "Tiny-Regular.ttf")
    _build_tiny_font(path, chars)
    yield path
    shutil.rmtree(temp_dir)


@pytest.fixture
def tiny_font_dir():
    """Directory containing a single minimal offline TTF."""
    import string

    chars = string.ascii_letters + string.digits + " .,-"
    temp_dir = tempfile.mkdtemp()
    _build_tiny_font(os.path.join(temp_dir, "Tiny-Regular.ttf"), chars)
    yield temp_dir
    shutil.rmtree(temp_dir)
