# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Font style classification dataset: word crops labelled with how they are set.

    python -m generator.recipes.font_style prepare --out font_style --languages all
    python -m generator.recipes.font_style render  --out font_style
    python -m generator.recipes.font_style preview --out font_style

For a classifier next to docTR's page/crop orientation models (mindee/doctr#2025):
"is this word bold, italic, underlined, red?" - and, as in Lens (mixfont/lens),
"which font family is it?". Every crop gets one record in ``labels.json``::

    "000042.jpg": {"text": "Rechnung", "lang": "de", "family": "Noto Serif", "category": "SERIF",
                   "font": "Noto Serif Bold Italic", "weight": 700, "bold": true, "italic": true,
                   "underline": "double", "strikethrough": false, "highlight": null, "color": "red"}

``meta/classes.json`` holds ``class_to_idx`` for the families (as Lens ships it),
``meta/families.json`` each family's category, languages and variants, and
``meta/attributes.json`` the value set of every attribute.

The labels are ground truth by construction:

* **family / weight / italic** come from the font files - real bold and light from
  variable fonts (``wght`` 100-900), real italics from italic faces. No faux bold, no
  slanted regular faces, and only slight crop skew, so neither label can be faked.
* **underline / strike-out** are drawn from each face's own ``post`` / ``OS/2``
  metrics, the way word processors draw them; double underlines are two rules.
* **colour** names the ink a reader sees (``black`` ... ``purple``, ``white`` for
  light-on-dark); a crop that went through a grayscale scan is named from its pixels.
* **multilingual**: a family only renders languages whose letters it inks and
  (for complex scripts) shapes; every text is HarfBuzz-checked in its exact face.
* **one class per design**: a family whose letters for a script are a copy of another
  family's (Heebo's Latin is Roboto's) does not render that script.
* **document fonts**: the faces most pages are set in - Times, Helvetica, Courier,
  Cambria, Computer Modern, ... as free metric-compatible clones (:data:`DOC_FONTS`).

Families are sampled uniformly (balanced classes), the document fonts
:data:`DOC_FONT_WEIGHT` times more often; within a family, the language follows the
recipe's script shares. Train and val share the families, not the words.
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..components.config import GenerationConfig
from ..components.font_downloader import FontDownloader
from ..components.font_styles import (
    COLOR_NAMES,
    HIGHLIGHT_COLORS,
    Decorations,
    StyleSpec,
    font_traits,
    set_weight,
    weight_name,
)
from ..components.generator import TextImageGenerator
from ..components.vocabs import VOCABS
from ..dataset_generator import SyntheticDatasetGenerator
from .corpus import Corpus, http_get
from .detection import clean_pool, enrich, page_weights
from .fonts import ManifestFontSelector, assign_fonts, collect_fonts, require_harfbuzz, scan_fonts, write_font_dir
from .languages import LANGUAGES, primary_script, resolve_languages, script_of
from .render import Progress, Worker, atomic_write_json, make_pool, render_part
from .text import numeric_tokens

__all__ = ["prepare", "render", "preview", "main", "STYLE_FONTS"]

NUM_TRAIN, NUM_VAL = 300_000, 15_000
SHARD_SIZE = 100_000
CHUNK = 128
SEED = 2025
JPEG_QUALITY = 92
MIN_TEXT_LEN, MAX_TEXT_LEN = 1, 32  # characters per word (crop): from 'a' or '5' to long compounds
# Short words ('a', 'и', '5', 'of') are rare in a word list but everywhere on a page.
SHORT_SHARE, SHORT_MAX_LEN = 0.15, 3
VAL_EVERY = 20  # one word in 20 is held out for val (stable hash)

# Variable families with true weights 100-900 and their italics, for Latin, Cyrillic
# and Greek (the non-Latin scripts get theirs from the recipe's font scan). Without
# them almost no face in the collection is heavier than 500.
STYLE_FONTS = [
    ("notosans", "NotoSans[wdth,wght].ttf"), ("notosans", "NotoSans-Italic[wdth,wght].ttf"),
    ("notoserif", "NotoSerif[wdth,wght].ttf"), ("notoserif", "NotoSerif-Italic[wdth,wght].ttf"),
    ("opensans", "OpenSans[wdth,wght].ttf"), ("opensans", "OpenSans-Italic[wdth,wght].ttf"),
    ("roboto", "Roboto[wdth,wght].ttf"), ("roboto", "Roboto-Italic[wdth,wght].ttf"),
    ("inter", "Inter[opsz,wght].ttf"), ("inter", "Inter-Italic[opsz,wght].ttf"),
    ("montserrat", "Montserrat[wght].ttf"), ("montserrat", "Montserrat-Italic[wght].ttf"),
    ("lora", "Lora[wght].ttf"), ("lora", "Lora-Italic[wght].ttf"),
    ("sourceserif4", "SourceSerif4[opsz,wght].ttf"), ("sourceserif4", "SourceSerif4-Italic[opsz,wght].ttf"),
    ("robotomono", "RobotoMono[wght].ttf"), ("robotomono", "RobotoMono-Italic[wght].ttf"),
    ("raleway", "Raleway[wght].ttf"), ("raleway", "Raleway-Italic[wght].ttf"),
    ("merriweather", "Merriweather[opsz,wdth,wght].ttf"), ("merriweather", "Merriweather-Italic[opsz,wdth,wght].ttf"),
    ("playfairdisplay", "PlayfairDisplay[wght].ttf"), ("playfairdisplay", "PlayfairDisplay-Italic[wght].ttf"),
    ("notosansmono", "NotoSansMono[wdth,wght].ttf"),
    ("ibmplexsans", "IBMPlexSans[wdth,wght].ttf"), ("ibmplexsans", "IBMPlexSans-Italic[wdth,wght].ttf"),
    ("ebgaramond", "EBGaramond[wght].ttf"), ("ebgaramond", "EBGaramond-Italic[wght].ttf"),
    ("arimo", "Arimo[wght].ttf"), ("arimo", "Arimo-Italic[wght].ttf"),
]  # fmt: skip

# The faces most documents are set in, or metric-compatible clones of them, under free
# licenses: Times (Tinos, TeX Gyre Termes), Helvetica (TeX Gyre Heros), Courier (Cousine),
# Cambria (Caladea), Palatino (TeX Gyre Pagella), Bookman (TeX Gyre Bonum), Century
# Schoolbook (TeX Gyre Schola), Verdana-like DejaVu Sans, Comic Sans (Comic Neue) and the
# Computer Modern of LaTeX papers (Latin Modern). Without them the classifier never sees the
# faces of most real pages, and names the closest look-alike of another script instead.
_FACES = ("Regular", "Bold", "Italic", "BoldItalic")
DOC_FONTS = [
    *((family, f"{name}-{face}.ttf") for family, name in (("tinos", "Tinos"), ("cousine", "Cousine"),
                                                           ("caladea", "Caladea"), ("comicneue", "ComicNeue"))
      for face in _FACES),
]  # fmt: skip
# Fonts outside Google Fonts: (archive URL, members) - TeX Gyre and Latin Modern (GUST font
# license), DejaVu (Bitstream Vera license).
DOC_FONT_ARCHIVES = [
    ("https://mirrors.ctan.org/fonts/tex-gyre.zip",
     [f"tex-gyre/opentype/texgyre{name}-{face}.otf" for name in ("termes", "heros", "pagella", "bonum", "schola")
      for face in ("regular", "bold", "italic", "bolditalic")]),
    ("https://mirrors.ctan.org/fonts/lm.zip",
     [f"lm/fonts/opentype/public/lm/{name}.otf" for name in (
         "lmroman10-regular", "lmroman10-bold", "lmroman10-italic", "lmroman10-bolditalic",
         "lmsans10-regular", "lmsans10-bold", "lmsans10-oblique", "lmsans10-boldoblique",
         "lmmono10-regular", "lmmono10-italic")]),
    ("https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip",
     [f"dejavu-fonts-ttf-2.37/ttf/{name}.ttf" for name in (
         "DejaVuSans", "DejaVuSans-Bold", "DejaVuSans-Oblique", "DejaVuSans-BoldOblique",
         "DejaVuSansMono", "DejaVuSansMono-Bold", "DejaVuSansMono-Oblique", "DejaVuSansMono-BoldOblique")]),
]  # fmt: skip
# Real pages are mostly set in these faces: they are drawn this many times more often than the
# other families (the classifier's prior then follows documents, not the font collection).
DOC_FONT_WEIGHT = 3

# What a crop is set in. Weights follow documents (regular and bold most often) but
# every weight a face offers appears; underlines, strike-outs and highlights are the
# emphasis marks of issue #2025, coloured ink one crop in three. Bold stays a minority as on
# real pages: with a third of the crops bold, a classifier calls the short words of a
# regular page (rendered at 144 dpi, ~20 px high) bold whenever their strokes are ambiguous.
BOLD_SHARE = 0.2
ITALIC_SHARE = 0.3
WEIGHT_PRIOR = {100: 0.03, 200: 0.05, 300: 0.12, 400: 0.3, 500: 0.12, 600: 0.1, 700: 0.18, 800: 0.06, 900: 0.04}
UNDERLINE_PRIOR = {0: 0.84, 1: 0.12, 2: 0.04}
STRIKE_PROB = 0.04
HIGHLIGHT_PROB = 0.06
COLORED_PROB = 0.35
INK_HUES = {
    "red": (215, 35, 35),
    "blue": (35, 70, 200),
    "green": (30, 140, 60),
    "orange": (235, 120, 20),
    "purple": (130, 50, 170),
    "brown": (120, 70, 35),
    "yellow": (240, 200, 30),
}
CATEGORIES = ("SANS_SERIF", "SERIF", "DISPLAY", "HANDWRITING", "MONOSPACE", "UNKNOWN")
# Families outside Google Fonts have no category there, and their names do not tell it:
# without these, TeX Gyre Termes (Times) would be UNKNOWN next to Tinos (Times) as SERIF.
KNOWN_CATEGORIES = {
    **dict.fromkeys(("TeX Gyre Termes", "TeX Gyre Pagella", "TeX Gyre Bonum", "TeX Gyre Schola", "Latin Modern Roman",
                     "Aparajita", "Roman Unicode", "Code2000", "Kelvinch", "Javanese Text", "站酷小薇体"), "SERIF"),
    **dict.fromkeys(("TeX Gyre Heros", "Latin Modern Sans", "Arial", "BPG Arial", "Aksaramatee", "Rounded Mplus 1c",
                     "TH Sarabun New", "lipipragatuchhi"), "SANS_SERIF"),
    **dict.fromkeys(("TeX Gyre Cursor", "Latin Modern Mono"), "MONOSPACE"),
    "Nanum Pen": "HANDWRITING",
}  # fmt: skip


# =============================================================================
# prepare
# =============================================================================


def _style_fonts(cache: Path, offline: bool) -> list[Path]:
    """The :data:`STYLE_FONTS` variable families (downloaded once into ``cache/style_fonts``)."""
    downloader = FontDownloader(cache_dir=str(cache / "style_fonts"), enabled=not offline)
    out = []
    for family, filename in STYLE_FONTS:
        local = cache / "style_fonts" / f"{family}__{filename}".replace("[", "_").replace("]", "_").replace(",", "_")
        path = str(local) if local.exists() else (downloader._download(family, filename) if not offline else None)
        if path:
            out.append(Path(path))
    return out


def _doc_fonts(cache: Path, offline: bool) -> list[Path]:
    """The :data:`DOC_FONTS` and :data:`DOC_FONT_ARCHIVES` faces (downloaded once into ``cache/doc_fonts``)."""
    folder = cache / "doc_fonts"
    folder.mkdir(parents=True, exist_ok=True)
    downloader = FontDownloader(cache_dir=str(folder), enabled=not offline)
    out = []
    for family, filename in DOC_FONTS:
        local = folder / f"{family}__{filename}"
        path = str(local) if local.exists() else (downloader._download(family, filename) if not offline else None)
        if path:
            out.append(Path(path))
    for url, members in DOC_FONT_ARCHIVES:
        missing = [m for m in members if not (folder / m.rsplit("/", 1)[-1]).exists()]
        archive = folder / url.rsplit("/", 1)[-1]
        if missing and not archive.exists() and not offline:
            data = http_get(url, timeout=600)
            if data:
                archive.write_bytes(data)
            else:
                print(f"  WARNING: could not download {url}")
        if missing and archive.exists():
            with zipfile.ZipFile(archive) as z:
                for member in missing:
                    (folder / member.rsplit("/", 1)[-1]).write_bytes(z.read(member))
        out += [folder / m.rsplit("/", 1)[-1] for m in members if (folder / m.rsplit("/", 1)[-1]).exists()]
    return out


def _category(family: str, cache_file: Path, offline: bool) -> str:
    """Google Fonts category of ``family`` (``METADATA.pb``), cached; a name heuristic otherwise."""
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    if family not in cache:
        found = None
        if not offline:
            slug = "".join(c for c in family.lower() if c.isalnum())
            for licence in ("ofl", "apache", "ufl"):
                data = http_get(f"https://raw.githubusercontent.com/google/fonts/main/{licence}/{slug}/METADATA.pb", 30)
                if data:
                    for line in data.decode("utf-8", "replace").splitlines():
                        if line.startswith("category:"):
                            found = line.split('"')[1] if '"' in line else None
                            break
                    break
        if found is None:
            low = family.lower()
            found = (
                "MONOSPACE"
                if "mono" in low
                else "SERIF"
                if "serif" in low and "sans" not in low
                else "SANS_SERIF"
                if "sans" in low
                else "UNKNOWN"
            )
        cache[family] = found if found in CATEGORIES else "UNKNOWN"
        atomic_write_json(cache_file, cache, indent=0)
    return cache[family]


def _merge_duplicates(families: dict[str, dict]) -> dict[str, dict]:
    """One class per design: "Mplus 1p" and "M PLUS 1p" (two sources, one font) merge."""
    merged: dict[str, dict] = {}
    names: dict[str, str] = {}
    for name in sorted(families, key=lambda n: (-sum(c.isupper() for c in n), n)):  # keep the GF spelling
        key = "".join(c for c in name.casefold() if c.isalnum())
        if key in names:
            target = merged[names[key]]
            target["languages"] |= families[name]["languages"]
            target["variants"].update(families[name]["variants"])
        else:
            names[key] = name
            merged[name] = families[name]
    return merged


# Probes per script; scripts not listed take frequent words of their own languages.
_PROBES = {"LATIN": "Hamburgefonstiv", "CYRILLIC": "Шрифтовобразец", "GREEK": "Γραμματοσειρά"}


def _look(font_dir: Path, variants: dict, text: str, size: float = 48) -> tuple[float, np.ndarray] | None:
    """``text`` in the family at weight ~400, upright, ``size`` px, baseline-aligned: (advance, bitmap)."""

    def score(v: dict) -> tuple[bool, int]:
        lo, hi = v["weight_range"] or (v["weight"], v["weight"])
        return bool(v["italic"]), 0 if lo <= 400 <= hi else abs(int(v["weight"]) - 400)

    variant = min(variants.values(), key=score)
    font = ImageFont.truetype(str(font_dir / variant["file"]), size)
    if variant["weight_range"]:
        set_weight(font, 400)
    width = font.getlength(text)
    if width < 10:
        return None
    # One canvas for every face, baseline at y = 150: tall scripts (Myanmar, Tibetan) fit.
    img = Image.new("L", (int(width) + 8, 240), 0)
    ImageDraw.Draw(img).text((4, 150), text, font=font, fill=255, anchor="ls")
    return width, np.asarray(img, dtype=np.float32) / 255.0


# Widest scaling between a family and a copy of its letters (Noto Sans JP draws the Latin of
# Source Sans 12% larger): the copy is then compared rendered at the size of the original.
MAX_COPY_SCALE = 1.15


def _same_look(a: tuple[float, np.ndarray], b: tuple[float, np.ndarray]) -> bool:
    """Same advance (within 0.5%) and nearly the same ink: a copy, not a look-alike.

    Measured on Latin, the copies rendered at the size of the original: copies 0.00-0.126
    (Noto Sans Arabic = Noto Sans, Khula = Open Sans 0.08, Heebo = Roboto 0.10, Mada and
    Assistant = the Source Sans of Noto Sans JP 0.11, the Times-like Latin of Roman Unicode,
    Tai Heritage Pro and Scheherazade New 0.06-0.13), metric-compatible designs from 0.15
    (Arial vs Arimo 0.15, Arimo vs TeX Gyre Heros 0.18); condensed cuts differ in shape once
    scaled to the same advance.
    """
    (wa, img_a), (wb, img_b) = a, b
    if abs(wa - wb) > 0.005 * max(wa, wb):
        return False
    w = max(img_a.shape[1], img_b.shape[1])
    img_a = np.pad(img_a, ((0, 0), (0, w - img_a.shape[1])))
    img_b = np.pad(img_b, ((0, 0), (0, w - img_b.shape[1])))
    ink = np.maximum(img_a, img_b) > 0.3
    return bool(ink.any()) and float(np.abs(img_a - img_b)[ink].mean()) < 0.135


def _probes(langs: list[str], pools: dict[str, list[str]]) -> dict[str, tuple[str, list[str]]]:
    """``{script: (probe text, its languages)}``: a fixed pangram-like probe or frequent words."""
    by_script: dict[str, list[str]] = defaultdict(list)
    for lang in langs:
        by_script[primary_script(lang)].append(lang)
    out = {}
    for script, members in by_script.items():
        text = _PROBES.get(script)
        if text is None:
            words = [w for lang in members for w in pools.get(lang, [])[:200] if len(w) >= 3][:40]
            text, i = "", 0
            while len(text) < 12 and i < len(words):
                text = (text + " " + words[i]).strip()
                i += 1
        if text:
            out[script] = (text, members)
    return out


def _borrowed(
    families: dict[str, dict], font_dir: Path, probes: dict[str, tuple[str, list[str]]]
) -> dict[str, set[str]]:
    """Languages each family must not render, because its letters for them are another family's.

    Per script, families that draw the probe the same (:func:`_same_look`) form a
    group: German in "Noto Sans Arabic" *is* Noto Sans, Hindi in "Baloo Paaji 2" is
    Baloo 2. The member with the broadest repertoire of that script keeps its
    languages - the family the letters were copied *from* - the others lose them, so
    no two classes share a look. A family whose letters are its own design (Arimo,
    Rubik, Secular One) keeps them even when it also covers other scripts.
    """
    manifest = json.loads((font_dir / "fonts.json").read_text(encoding="utf-8"))["fonts"]
    chars = {n: set().union(*(set(manifest[f]["chars"]) for f in fam["variants"])) for n, fam in families.items()}
    drop: dict[str, set[str]] = defaultdict(set)
    for script, (text, members) in probes.items():
        member_set = set(members)
        looks, repertoire = {}, {}
        for name, fam in families.items():
            if not (fam["languages"] & member_set) or not set(text) - {" "} <= chars[name]:
                continue
            look = _look(font_dir, fam["variants"], text)
            if look is not None:
                looks[name] = look
                repertoire[name] = sum(1 for c in chars[name] if c.isalpha() and script_of(c) == script)
        names = sorted(looks, key=lambda n: looks[n][0])  # by width: copies are neighbours
        parent = {n: n for n in names}

        def root(n: str, parent: dict = parent) -> str:
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        for i, name in enumerate(names):
            for other in names[i + 1 :]:
                scale = looks[other][0] / looks[name][0]
                if scale > MAX_COPY_SCALE:
                    break
                # A scaled copy is compared at the size where it has the advance of ``name``
                look = (
                    looks[other] if scale <= 1.005 else _look(font_dir, families[other]["variants"], text, 48 / scale)
                )
                if look is not None and _same_look(looks[name], look):
                    parent[root(other)] = root(name)
        groups: dict[str, list[str]] = defaultdict(list)
        for name in names:
            groups[root(name)].append(name)
        for group in groups.values():
            if len(group) > 1:
                keeper = min(group, key=lambda n: (-repertoire[n], len(n), n))
                for name in set(group) - {keeper}:
                    drop[name] |= member_set
    return drop


def _held_out(word: str) -> bool:
    return zlib.crc32(word.casefold().encode("utf-8")) % VAL_EVERY == 0


def _can_be(variant: dict, bold: bool) -> bool:
    """Whether ``variant`` can be set bold (>= 600) - or, for ``bold=False``, not bold."""
    lo, hi = variant["weight_range"] or (variant["weight"], variant["weight"])
    return hi >= 600 if bold else lo < 600


def _pick_weight(variant: dict, bold: bool, rng: random.Random) -> int:
    if variant["weight_range"] is None:
        return int(variant["weight"])
    lo, hi = variant["weight_range"]
    ws = [w for w in WEIGHT_PRIOR if lo <= w <= hi and (w >= 600) == bold] or [max(lo, min(hi, 700 if bold else 400))]
    return rng.choices(ws, weights=[WEIGHT_PRIOR[w] for w in ws])[0]


def _plan(
    n: int,
    families: dict,
    lang_weights: dict[str, float],
    rng: random.Random,
    family_weights: dict[str, int] | None = None,
) -> list[tuple]:
    """``(family, file, weight, lang, seed)`` per crop.

    Bold and italic are drawn first (:data:`BOLD_SHARE`, :data:`ITALIC_SHARE`): most
    static faces stop at weight 500 and many families have no italic, so balancing
    families alone would leave both rare. The family is then taken in turns from
    those that can provide the combination (all families when none can), each family
    ``family_weights[family]`` turns per round (one by default).
    """
    queues: dict[tuple, list[str]] = {}

    def next_family(names: list[str], key: tuple) -> str:
        queue = queues.setdefault(key, [])
        if not queue:
            turns = [name for name in names for _ in range((family_weights or {}).get(name, 1))]
            queue.extend(rng.sample(turns, len(turns)))
        return queue.pop()

    rows = []
    for _ in range(n):
        want_bold, want_italic = rng.random() < BOLD_SHARE, rng.random() < ITALIC_SHARE
        for bold, italic in ((want_bold, want_italic), (want_bold, None), (None, None)):

            def fits(v: dict, bold=bold, italic=italic) -> bool:
                return (italic is None or v["italic"] == italic) and (bold is None or _can_be(v, bold))

            names = sorted(f for f, fam in families.items() if any(fits(v) for v in fam["variants"]))
            if names:
                break
        family = next_family(names, (bold, italic))
        fam = families[family]
        variant = rng.choice([v for v in fam["variants"] if fits(v)])
        weight = _pick_weight(variant, want_bold if bold is None else bold, rng)
        langs = fam["languages"]
        lang = rng.choices(langs, weights=[lang_weights.get(lg, 1e-3) for lg in langs])[0]
        rows.append((family, variant["file"], weight, lang, rng.getrandbits(31)))
    return rows


def prepare(args) -> None:
    """Word pools, vetted fonts, the family inventory and the crop plan (in ``OUT/meta``)."""
    require_harfbuzz()
    meta = Path(args.out, "meta")
    (meta / "pools").mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    rng = random.Random(SEED)
    langs = resolve_languages(args.languages)
    print(f"{len(langs)} languages: {' '.join(langs)}")

    print("Word pools ...")
    corpus = Corpus(cache / "corpora", full_lists=False, offline=args.offline)
    pools = {}
    for lang in langs:
        words, _ = clean_pool(lang, corpus.raw_words(lang, rng), rng, max_len=MAX_TEXT_LEN)
        if words:
            pools[lang] = words
    langs = [lang for lang in langs if lang in pools]

    print("Fonts ...")
    paths = collect_fonts(Path(args.font_src), cache / "google_fonts", download_extra=not args.offline)
    paths += _style_fonts(cache, args.offline)
    doc_paths = _doc_fonts(cache, args.offline)
    paths += doc_paths
    fonts = scan_fonts(paths, cache / "font_scan.json")
    freq = {lang: Counter(c for w in ws for c in w) for lang, ws in pools.items()}
    lang_fonts, _ = assign_fonts(fonts, freq, langs)
    by_lang = write_font_dir(fonts, lang_fonts, meta / "fonts")
    selector = ManifestFontSelector(meta / "fonts", handwriting=False)

    print("Families ...")
    families: dict[str, dict] = {}
    for lang, names in by_lang.items():
        for name in names:
            traits = font_traits(str(meta / "fonts" / name))
            fam = families.setdefault(traits.family, {"variants": {}, "languages": set()})
            fam["languages"].add(lang)
            fam["variants"][name] = {
                "file": name,
                "italic": traits.italic,
                "weight": traits.weight,
                "weight_range": list(traits.weight_range) if traits.weight_range else None,
            }
    families = _merge_duplicates(families)
    borrowed = _borrowed(families, meta / "fonts", _probes(langs, pools))
    for name, lost in borrowed.items():
        families[name]["languages"] -= lost
    by_script = Counter(primary_script(lg) for lost in borrowed.values() for lg in lost)
    print(f"  copied letters: {len(borrowed)} families drop languages whose letters are another family's "
          f"({', '.join(f'{k.lower()} {v}' for k, v in by_script.most_common(8))})")  # fmt: skip
    families = {k: v for k, v in families.items() if v["languages"] and v["variants"]}
    for name, fam in families.items():
        fam["languages"] = sorted(fam["languages"])
        fam["variants"] = sorted(fam["variants"].values(), key=lambda v: v["file"])
        fam["category"] = KNOWN_CATEGORIES.get(name) or _category(name, cache / "gf_categories.json", args.offline)

    print("Texts ...")
    for lang in langs:
        if not by_lang.get(lang):
            continue
        target = {c for c in VOCABS[LANGUAGES[lang][0]] if not c.isspace() and selector.candidates(c, lang)}
        words = enrich(lang, [w for w in pools[lang] if selector.candidates(w, lang)], target, rng)
        words += numeric_tokens(lang, 600, target | set("0123456789"), rng, 0.35)
        train = [w for w in words if not _held_out(w)]
        val = [w for w in words if _held_out(w)] or train[: max(1, len(train) // VAL_EVERY)]
        (meta / "pools" / f"{lang}.train.txt").write_text("\n".join(train), encoding="utf-8")
        (meta / "pools" / f"{lang}.val.txt").write_text("\n".join(val), encoding="utf-8")

    lang_weights = page_weights([lg for lg in langs if by_lang.get(lg)])
    classes = {name: i for i, name in enumerate(sorted(families))}
    atomic_write_json(meta / "classes.json", {"class_to_idx": classes}, indent=1)
    atomic_write_json(meta / "families.json", families, indent=1)
    atomic_write_json(
        meta / "attributes.json",
        {
            "family": "see classes.json",
            "category": list(CATEGORIES),
            "weight": [100, 200, 300, 400, 500, 600, 700, 800, 900],
            "bold": "weight >= 600",
            "italic": [False, True],
            "underline": ["none", "single", "double"],
            "strikethrough": [False, True],
            "highlight": [None, *HIGHLIGHT_COLORS],
            "color": list(COLOR_NAMES),
        },
        indent=1,
    )
    doc_families = {font_traits(str(p)).family for p in doc_paths} & set(families)
    print(f"  document fonts: {', '.join(sorted(doc_families))} (x{DOC_FONT_WEIGHT})")
    weights = dict.fromkeys(doc_families, DOC_FONT_WEIGHT)
    for split, n in (("train", args.num_train), ("val", args.num_val)):
        rows = _plan(n, families, lang_weights, random.Random(SEED + (split == "val")), weights)
        lines = ["\t".join(map(str, r)) for r in rows]
        (meta / f"{split}_plan.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_lang = Counter(len(f["languages"]) for f in families.values())
    print(
        f"{len(families)} families ({sum(len(f['variants']) for f in families.values())} faces) | "
        f"categories {dict(Counter(f['category'] for f in families.values()))} | "
        f"families per language: min {min(sum(lg in f['languages'] for f in families.values()) for lg in langs)}"
    )
    print(f"plan: {args.num_train:,} train / {args.num_val:,} val crops | families covering >1 language: "
          f"{sum(v for k, v in n_lang.items() if k > 1)}")  # fmt: skip


# =============================================================================
# render
# =============================================================================


def _config(args) -> GenerationConfig:
    return GenerationConfig.flat(
        task="recognition",
        output_dir=str(args.out),
        num_workers=args.workers,
        bg_image_dir=args.bg_dir,
        output_jpeg=not args.png,
        output_jpeg_quality=JPEG_QUALITY,
        font_size_range=(18, 56),  # the weight and the serifs have to be visible
        bold_prob=0.0,  # the face's own weight is the label
        outline_prob=0.0,
        perspective_prob=0.1,  # a strong skew would look like an italic
        grayscale_prob=0.1,  # colour is a label; most crops keep it
        bitonal_prob=0.02,
        # Generated designs carry stripes and bands; one through a word reads as a
        # strike-out the label does not have. Ruled paper stays (lines across the whole
        # crop are what an underline is *not*).
        procedural_bg_prob=0.1,
        # Always some paper around the word: an underline or strike-out spans the text,
        # a ruled line on the paper spans the crop - without a margin they look the same.
        margin_prob=1.0,
        margin_range=(0.06, 0.3),
    )


class StyleWorker(Worker):
    """Renders one styled crop; the record is the crop's style label."""

    def __init__(self, cfg: GenerationConfig, meta: str):
        meta_dir = Path(meta)
        self.font_dir = meta_dir / "fonts"
        self.selector = ManifestFontSelector(self.font_dir, handwriting=False)
        self.gen = TextImageGenerator(cfg, font_selector=self.selector)
        manifest = json.loads((self.font_dir / "fonts.json").read_text(encoding="utf-8"))
        self.chars = {name: set(info["chars"]) for name, info in manifest["fonts"].items()}
        self.families = json.loads((meta_dir / "families.json").read_text(encoding="utf-8"))
        self.pools: dict[str, dict[str, list[str]]] = {"train": {}, "val": {}}
        for f in (meta_dir / "pools").glob("*.txt"):
            lang, split = f.stem.rsplit(".", 1)
            self.pools[split][lang] = f.read_text(encoding="utf-8").split("\n")
        self.short = {
            split: {lang: [w for w in words if 0 < len(w) <= SHORT_MAX_LEN] for lang, words in pools.items()}
            for split, pools in self.pools.items()
        }
        self.save = ("JPEG", {"quality": cfg.core.output_jpeg_quality}) if cfg.core.output_jpeg else ("PNG", {})

    def _text(self, lang: str, split: str, rng: random.Random) -> str:
        """One word, as docTR's detector boxes it (a space-free run for Thai, CJK ...).

        Over-long entries are skipped rather than cut: a cut can split a cluster.
        """
        pool = self.pools[split].get(lang) or self.pools["train"][lang]
        short = self.short[split].get(lang) or self.short["train"].get(lang)
        if short and rng.random() < SHORT_SHARE:
            pool = short
        for _ in range(20):
            word = rng.choice(pool)
            if MIN_TEXT_LEN <= len(word) <= MAX_TEXT_LEN and not any(c.isspace() for c in word):
                return word
        return ""

    @staticmethod
    def _spec(weight: int | None, rng: random.Random) -> StyleSpec:
        underline = rng.choices(list(UNDERLINE_PRIOR), weights=list(UNDERLINE_PRIOR.values()))[0]
        highlight = rng.choice(list(HIGHLIGHT_COLORS)) if rng.random() < HIGHLIGHT_PROB else None
        colored = rng.random() < COLORED_PROB
        hue = INK_HUES[rng.choice(list(INK_HUES))] if colored else None
        return StyleSpec(
            weight=weight,
            decorations=Decorations(underline=underline, strikethrough=rng.random() < STRIKE_PROB),
            highlight=HIGHLIGHT_COLORS[highlight] if highlight else None,
            ink_hue=hue,
            neutral=not colored,
        )

    def render(self, payload, path):
        """Render ``(family, file, weight, lang, seed, split)``; the record is its label."""
        family, file, weight, lang, seed, split = payload
        rng = random.Random(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        font_path = str(self.font_dir / file)
        chars = self.chars[file]
        for _ in range(30):  # a text this exact face inks and shapes cleanly
            text = self._text(lang, split, rng)
            if text and set(text) - {" "} <= chars and self.selector.shaping.ok(text, font_path):
                break
        else:
            return None
        traits = font_traits(font_path)
        weight = int(weight) if traits.weight_range else traits.weight
        spec = self._spec(weight if traits.weight_range else None, rng)
        result = self.gen.generate_styled(text, font_path, spec)
        if result is None:
            return None
        image, info = result
        image.save(path, self.save[0], **self.save[1])
        highlight = next((k for k, v in HIGHLIGHT_COLORS.items() if v == spec.highlight), None)
        style_name = weight_name(weight) + (" Italic" if traits.italic else "")
        return {
            "text": text,
            "lang": lang,
            "family": family,
            "category": self.families[family]["category"],
            "font": f"{family} {style_name.replace('Regular Italic', 'Italic')}",
            "weight": weight,
            "bold": weight >= 600,
            "italic": traits.italic,
            "underline": ("none", "single", "double")[spec.decorations.underline],
            "strikethrough": spec.decorations.strikethrough,
            "highlight": highlight,
            "color": info["color"],
        }


def _read_plan(path: Path, split: str) -> list[tuple]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            family, file, weight, lang, seed = line.split("\t")
            rows.append((family, file, int(weight), lang, int(seed), split))
    return rows


def _finish_part(part: Path, final: Path, labels: dict) -> None:
    atomic_write_json(part / "labels.json", dict(sorted(labels.items())))
    (part / "progress.jsonl").unlink(missing_ok=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(part, final)


def render(args) -> None:
    """Render val and every unfinished train shard (resumable)."""
    require_harfbuzz()
    out, meta = Path(args.out), Path(args.out, "meta")
    if not (meta / "train_plan.tsv").exists():
        sys.exit("Run `prepare` first.")
    train, val = _read_plan(meta / "train_plan.tsv", "train"), _read_plan(meta / "val_plan.tsv", "val")
    cfg = _config(args)
    SyntheticDatasetGenerator(cfg).resolve_backgrounds()
    ext = "png" if args.png else "jpg"
    jobs = []
    if not (out / "val" / "labels.json").exists():
        jobs.append((val, out / "val", out / "_partial" / "val"))
    for s, start in enumerate(range(0, len(train), args.shard_size)):
        final = out / "train" / f"shard_{s:04d}"
        if not (final / "labels.json").exists():
            jobs.append((train[start : start + args.shard_size], final, out / "_partial" / final.name))
    if not jobs:
        print("Nothing to do - all shards are complete.")
    else:
        progress = Progress(sum(len(j[0]) for j in jobs))
        print(f"Rendering {progress.total:,} crops in {len(jobs)} part(s) with {args.workers} workers ...")
        with make_pool(args.workers, StyleWorker, (cfg, str(meta))) as pool:
            for rows, final, part in jobs:
                width = max(6, len(str(len(rows))))
                items = [(f"{i:0{width}d}.{ext}", row) for i, row in enumerate(rows)]
                labels = render_part(pool, items, part, CHUNK, progress, final.name)
                _finish_part(part, final, labels)
                print(f"\n  -> {final}: {len(labels):,} images, {len(rows) - len(labels)} dropped", flush=True)
        shutil.rmtree(out / "_partial", ignore_errors=True)
        print(f"Done in {(time.time() - progress.t0) / 3600:.2f} h.")
    report(out)


def _labels(out: Path) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {"train": [], "val": []}
    for split, folders in (("val", [out / "val"]), ("train", sorted((out / "train").glob("shard_*")))):
        for folder in folders:
            if (folder / "labels.json").exists():
                found[split] += json.loads((folder / "labels.json").read_text(encoding="utf-8")).values()
    return found


def report(out: Path) -> None:
    """Class balance and attribute shares of the rendered labels."""
    labels = _labels(out)
    recs = labels["train"]
    if not recs:
        return
    n = len(recs)
    fam = Counter(r["family"] for r in recs)
    n_langs = len({r["lang"] for r in recs})
    print(f"\ntrain {n:,} / val {len(labels['val']):,} crops | {len(fam)} families, "
          f"{min(fam.values())}-{max(fam.values())} crops each | {n_langs} languages")  # fmt: skip
    for key in ("bold", "italic", "underline", "strikethrough", "highlight", "color", "category"):
        shares = Counter(str(r[key]) for r in recs)
        print(f"  {key:13s} " + "  ".join(f"{k} {v / n:.1%}" for k, v in shares.most_common()))
    weights = Counter(r["weight"] for r in recs)
    print("  weight        " + "  ".join(f"{k} {v / n:.1%}" for k, v in sorted(weights.items())))


# =============================================================================
# preview
# =============================================================================


def preview(args) -> Path:
    """A contact sheet of rendered crops with their labels (``OUT/preview.jpg``)."""
    out = Path(args.out)
    folders = [out / "val", *sorted((out / "train").glob("shard_*"))]
    folder = next((f for f in folders if (f / "labels.json").exists()), None)
    if folder is None:
        sys.exit("Nothing rendered yet - run `render` first.")
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    names = rng.sample(sorted(labels), min(args.samples, len(labels)))
    ui_path = next(iter(sorted((out / "meta" / "fonts").glob("*Inter*"))), None)
    ui = ImageFont.truetype(str(ui_path), 14) if ui_path else ImageFont.load_default()
    tiles = []
    for name in names:
        r = labels[name]
        img = Image.open(folder / "images" / name).convert("RGB")
        img = img.resize((max(1, int(img.width * 56 / img.height)), 56))
        img = img if img.width <= 460 else img.resize((460, 56))
        marks = [r["font"], r["color"]]
        marks += [f"underline {r['underline']}"] if r["underline"] != "none" else []
        marks += ["strike"] if r["strikethrough"] else []
        marks += [f"highlight {r['highlight']}"] if r["highlight"] else []
        tile = Image.new("RGB", (470, 92), (255, 255, 255))
        tile.paste(img, (5, 5))
        ImageDraw.Draw(tile).text((5, 66), " | ".join(marks), fill=(40, 70, 180), font=ui)
        tiles.append(tile)
    cols = 4
    sheet = Image.new("RGB", (cols * 474, ((len(tiles) + cols - 1) // cols) * 96), (225, 228, 233))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 474, (i // cols) * 96))
    target = out / "preview.jpg"
    sheet.save(target, quality=90)
    return target


def main(argv: list[str] | None = None) -> None:
    """Command line entry point (``python -m generator.recipes.font_style``)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["prepare", "render", "all", "preview", "report"])
    p.add_argument("--out", required=True, help="dataset directory")
    p.add_argument("--languages", default="all", help="preset (european, multilingual, all) or codes: de,fr,ru")
    p.add_argument("--num-train", type=int, default=NUM_TRAIN)
    p.add_argument("--num-val", type=int, default=NUM_VAL)
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--font-src", default="fonts_v1", help="unzipped fonts_v1 release (downloaded if missing)")
    p.add_argument("--cache", default=".recipe_cache", help="corpora, extra fonts and the font scan")
    p.add_argument("--bg-dir", default=None, help="background images (default: the curated download)")
    p.add_argument("--offline", action="store_true", help="never download; use cached corpora and fonts only")
    p.add_argument("--png", action="store_true", help="PNG instead of JPEG")
    p.add_argument("--samples", type=int, default=24, help="preview: crops on the sheet")
    p.add_argument("--seed", type=int, default=0, help="preview: which crops")
    args = p.parse_args(argv)
    if args.step in ("prepare", "all"):
        prepare(args)
    if args.step in ("render", "all"):
        render(args)
    if args.step == "report":
        report(Path(args.out))
    if args.step == "preview":
        print(preview(args))


if __name__ == "__main__":
    main()
