# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Font collection, vetting and per-language font assignment for the recipes.

Every font is scanned once with HarfBuzz:

* **inked characters** - codepoints whose glyph actually draws something. A cmap
  entry pointing at an empty glyph would otherwise produce a crop whose label
  names a character the image does not show.
* **complex scripts it can shape** - Arabic, Indic and South-East Asian scripts
  need the font's GSUB rules; without them letters come out unjoined and
  conjuncts as bare halants.

A font *serves* a language when it inks every letter, mark and digit of that
language's own script (plus the shaping rules, for complex scripts). Rendering
then only uses a language's own fonts, so a typeface is never seen dropping a
diacritic it cannot draw - the model would learn "this typeface => no ő".
"""

import json
import os
import random
import shutil
import sys
import unicodedata
import urllib.parse
import zipfile
from collections import Counter
from pathlib import Path

from ..components.font_selector import FontSelector
from ..components.shaping import HAS_HARFBUZZ
from ..components.vocabs import VOCABS
from .corpus import http_get
from .languages import COMPLEX_SCRIPT_TAGS, LANGUAGES, allowed_scripts, script_of

if HAS_HARFBUZZ:
    import uharfbuzz as hb

__all__ = ["collect_fonts", "scan_fonts", "assign_fonts", "write_font_dir", "ManifestFontSelector", "require_harfbuzz"]

FONTS_V1_URL = "https://github.com/felixdittrich92/docTR-Synth-Generator/releases/download/v0.0.1/fonts_v1.zip"

# google/fonts (OFL) faces for scripts fonts_v1 covers thinly or not at all: it has no
# simplified-Chinese face, no Sundanese face, only ~6 faces that can set Chinese/Japanese
# and single digits of faces for Hebrew, Arabic, Ethiopic and most Indic scripts.
EXTRA_FONTS = [
    # Chinese (simplified)
    "notosanssc/NotoSansSC[wght].ttf", "notoserifsc/NotoSerifSC[wght].ttf",
    "zcoolxiaowei/ZCOOLXiaoWei-Regular.ttf", "zcoolqingkehuangyou/ZCOOLQingKeHuangYou-Regular.ttf",
    "zcoolkuaile/ZCOOLKuaiLe-Regular.ttf", "mashanzheng/MaShanZheng-Regular.ttf",
    "zhimangxing/ZhiMangXing-Regular.ttf", "longcang/LongCang-Regular.ttf",
    "liujianmaocao/LiuJianMaoCao-Regular.ttf",
    # Japanese
    "notosansjp/NotoSansJP[wght].ttf", "notoserifjp/NotoSerifJP[wght].ttf",
    "zenmarugothic/ZenMaruGothic-Regular.ttf", "zenkakugothicnew/ZenKakuGothicNew-Regular.ttf",
    "zenoldmincho/ZenOldMincho-Regular.ttf", "sawarabimincho/SawarabiMincho-Regular.ttf",
    "shipporimincho/ShipporiMincho-Regular.ttf", "bizudpgothic/BIZUDPGothic-Regular.ttf",
    "bizudpmincho/BIZUDPMincho-Regular.ttf", "kleeone/KleeOne-Regular.ttf", "kiwimaru/KiwiMaru-Regular.ttf",
    "mplusrounded1c/MPLUSRounded1c-Regular.ttf", "hinamincho/HinaMincho-Regular.ttf",
    "yujisyuku/YujiSyuku-Regular.ttf", "rocknrollone/RocknRollOne-Regular.ttf",
    "dotgothic16/DotGothic16-Regular.ttf", "reggaeone/ReggaeOne-Regular.ttf",
    # Korean
    "notosanskr/NotoSansKR[wght].ttf", "notoserifkr/NotoSerifKR[wght].ttf",
    "nanumgothic/NanumGothic-Regular.ttf", "nanummyeongjo/NanumMyeongjo-Regular.ttf",
    "nanumpenscript/NanumPenScript-Regular.ttf", "blackhansans/BlackHanSans-Regular.ttf",
    "dohyeon/DoHyeon-Regular.ttf", "gowundodum/GowunDodum-Regular.ttf", "gowunbatang/GowunBatang-Regular.ttf",
    "jua/Jua-Regular.ttf", "songmyung/SongMyung-Regular.ttf", "gothica1/GothicA1-Regular.ttf",
    "ibmplexsanskr/IBMPlexSansKR-Regular.ttf",
    # Khmer
    "notosanskhmer/NotoSansKhmer[wdth,wght].ttf", "notoserifkhmer/NotoSerifKhmer[wdth,wght].ttf",
    "battambang/Battambang-Regular.ttf", "hanuman/Hanuman[wght].ttf", "moul/Moul-Regular.ttf",
    "bokor/Bokor-Regular.ttf", "content/Content-Regular.ttf", "dangrek/Dangrek-Regular.ttf",
    "freehand/Freehand-Regular.ttf", "kantumruypro/KantumruyPro[wght].ttf", "koulen/Koulen-Regular.ttf",
    "nokora/Nokora[wght].ttf", "odormeanchey/OdorMeanChey-Regular.ttf",
    "suwannaphum/Suwannaphum-Regular.ttf", "taprom/Taprom-Regular.ttf",
    # Myanmar, Javanese, Sundanese
    "padauk/Padauk-Regular.ttf", "notoserifmyanmar/NotoSerifMyanmar-Regular.ttf",
    "notosansmyanmar/NotoSansMyanmar[wdth,wght].ttf",
    "notosansjavanese/NotoSansJavanese[wght].ttf", "notosanssundanese/NotoSansSundanese[wght].ttf",
    # Hebrew
    "notosanshebrew/NotoSansHebrew[wdth,wght].ttf", "notoserifhebrew/NotoSerifHebrew[wdth,wght].ttf",
    "notorashihebrew/NotoRashiHebrew[wght].ttf", "davidlibre/DavidLibre-Regular.ttf",
    "frankruhllibre/FrankRuhlLibre[wght].ttf", "heebo/Heebo[wght].ttf", "assistant/Assistant[wght].ttf",
    "rubik/Rubik[wght].ttf", "secularone/SecularOne-Regular.ttf", "suezone/SuezOne-Regular.ttf",
    "bellefair/Bellefair-Regular.ttf", "varelaround/VarelaRound-Regular.ttf",
    # Arabic script (naskh, kufi, nastaliq-ish and modern sans)
    "notosansarabic/NotoSansArabic[wdth,wght].ttf", "notonaskharabic/NotoNaskhArabic[wght].ttf",
    "notokufiarabic/NotoKufiArabic[wght].ttf", "amiri/Amiri-Regular.ttf", "cairo/Cairo[slnt,wght].ttf",
    "tajawal/Tajawal-Regular.ttf", "almarai/Almarai-Regular.ttf", "elmessiri/ElMessiri[wght].ttf",
    "lateef/Lateef-Regular.ttf", "scheherazadenew/ScheherazadeNew-Regular.ttf",
    "markazitext/MarkaziText[wght].ttf", "reemkufi/ReemKufi[wght].ttf", "harmattan/Harmattan-Regular.ttf",
    # Devanagari
    "notosansdevanagari/NotoSansDevanagari[wdth,wght].ttf",
    "notoserifdevanagari/NotoSerifDevanagari[wdth,wght].ttf",
    "tirodevanagarimarathi/TiroDevanagariMarathi-Regular.ttf", "hind/Hind-Regular.ttf",
    "mukta/Mukta-Regular.ttf", "poppins/Poppins-Regular.ttf", "rozhaone/RozhaOne-Regular.ttf",
    "baloo2/Baloo2[wght].ttf", "khand/Khand-Regular.ttf", "teko/Teko[wght].ttf",
    "yatraone/YatraOne-Regular.ttf", "kalam/Kalam-Regular.ttf",
    # Bengali, Gujarati, Gurmukhi, Odia
    "notosansbengali/NotoSansBengali[wdth,wght].ttf", "notoserifbengali/NotoSerifBengali[wdth,wght].ttf",
    "hindsiliguri/HindSiliguri-Regular.ttf",
    "notosansgujarati/NotoSansGujarati[wdth,wght].ttf", "notoserifgujarati/NotoSerifGujarati[wght].ttf",
    "hindvadodara/HindVadodara-Regular.ttf", "shrikhand/Shrikhand-Regular.ttf",
    "notosansgurmukhi/NotoSansGurmukhi[wdth,wght].ttf", "notoserifgurmukhi/NotoSerifGurmukhi[wght].ttf",
    "baloopaaji2/BalooPaaji2[wght].ttf",
    "notosansoriya/NotoSansOriya[wdth,wght].ttf", "notoseriforiya/NotoSerifOriya[wght].ttf",
    "baloobhaina2/BalooBhaina2[wght].ttf",
    # Tamil, Telugu, Kannada, Malayalam, Sinhala
    "notosanstamil/NotoSansTamil[wdth,wght].ttf", "notoseriftamil/NotoSerifTamil[wdth,wght].ttf",
    "hindmadurai/HindMadurai-Regular.ttf", "catamaran/Catamaran[wght].ttf",
    "notosanstelugu/NotoSansTelugu[wdth,wght].ttf", "notoseriftelugu/NotoSerifTelugu[wght].ttf",
    "hindguntur/HindGuntur-Regular.ttf", "ramabhadra/Ramabhadra-Regular.ttf", "gidugu/Gidugu-Regular.ttf",
    "notosanskannada/NotoSansKannada[wdth,wght].ttf", "notoserifkannada/NotoSerifKannada[wght].ttf",
    "baloochettan2/BalooChettan2[wght].ttf",
    "notosansmalayalam/NotoSansMalayalam[wdth,wght].ttf",
    "notoserifmalayalam/NotoSerifMalayalam[wght].ttf", "gayathri/Gayathri-Regular.ttf",
    "manjari/Manjari-Regular.ttf", "chilanka/Chilanka-Regular.ttf",
    "notosanssinhala/NotoSansSinhala[wdth,wght].ttf", "notoserifsinhala/NotoSerifSinhala[wdth,wght].ttf",
    "abhayalibre/AbhayaLibre-Regular.ttf", "gemunulibre/GemunuLibre[wght].ttf",
    "yaldevicolombo/YaldeviColombo-Regular.ttf",
    # Thai, Lao
    "notosansthai/NotoSansThai[wdth,wght].ttf", "notoserifthai/NotoSerifThai[wdth,wght].ttf",
    "kanit/Kanit-Regular.ttf", "sarabun/Sarabun-Regular.ttf", "prompt/Prompt-Regular.ttf",
    "mitr/Mitr-Regular.ttf", "chonburi/Chonburi-Regular.ttf", "charmonman/Charmonman-Regular.ttf",
    "taviraj/Taviraj-Regular.ttf", "baijamjuree/BaiJamjuree-Regular.ttf", "k2d/K2D-Regular.ttf",
    "maitree/Maitree-Regular.ttf",
    "notosanslao/NotoSansLao[wdth,wght].ttf", "notoseriflao/NotoSerifLao[wdth,wght].ttf",
    "phetsarath/Phetsarath-Regular.ttf",
    # Armenian, Georgian, Ethiopic
    "notosansarmenian/NotoSansArmenian[wdth,wght].ttf",
    "notoserifarmenian/NotoSerifArmenian[wdth,wght].ttf",
    "notosansgeorgian/NotoSansGeorgian[wdth,wght].ttf",
    "notoserifgeorgian/NotoSerifGeorgian[wdth,wght].ttf",
    "notosansethiopic/NotoSansEthiopic[wdth,wght].ttf",
    "notoserifethiopic/NotoSerifEthiopic[wdth,wght].ttf", "abyssinicasil/AbyssinicaSIL-Regular.ttf",
]  # fmt: skip
_GOOGLE_FONTS_RAW = "https://raw.githubusercontent.com/google/fonts/main/ofl/"

# Fallback for languages whose own-script letter set no face covers completely: keep
# faces that render this share of the language's text (by character occurrence).
OCCURRENCE_COVERAGE = 0.995
# Letters rarer than this share of a language's characters need not be in every face.
RARE_LETTER_SHARE = 2e-4
# Below this many full-coverage faces, the occurrence rule is used as well.
MIN_FONTS_PER_LANGUAGE = 3
_SCAN_VERSION = 1
_INKLESS = {"Zs", "Zl", "Zp", "Cc", "Cf"}


def require_harfbuzz() -> None:
    """Exit with an explanation when the recipes cannot verify shaping."""
    from PIL import features

    problems = []
    if not HAS_HARFBUZZ:
        problems.append('uharfbuzz is missing: pip install "doctr-synth-generator[recipes]"')
    if not features.check("raqm"):
        problems.append(
            "Pillow has no libraqm - Arabic, Hebrew and every Indic script would be shaped wrong.\n"
            "  Install libraqm/libfribidi (e.g. apt install libraqm0 libfribidi0) and check PIL.features.check('raqm')."
        )
    if problems:
        sys.exit(
            "Cannot build a dataset whose images are guaranteed to match their labels:\n  " + "\n  ".join(problems)
        )


# -----------------------------------------------------------------------------
# Collection
# -----------------------------------------------------------------------------


def collect_fonts(font_src: Path, extra_dir: Path, download_extra: bool = True) -> list[Path]:
    """All candidate font files: ``fonts_v1`` (downloaded if missing) + :data:`EXTRA_FONTS`."""
    if not font_src.exists():
        print(f"Downloading fonts_v1 -> {font_src} ...")
        data = http_get(FONTS_V1_URL, timeout=600)
        if not data:
            sys.exit(f"Could not download fonts_v1 ({FONTS_V1_URL}); download it manually and pass --font-src")
        zpath = font_src.with_suffix(".zip")
        zpath.write_bytes(data)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(font_src)
    extra_dir.mkdir(parents=True, exist_ok=True)
    if download_extra:
        for rel in EXTRA_FONTS:
            dst = extra_dir / rel.split("/")[-1].replace("[", "_").replace("]", "").replace(",", "_")
            if not dst.exists():
                data = http_get(_GOOGLE_FONTS_RAW + urllib.parse.quote(rel), timeout=600)
                if data:
                    dst.write_bytes(data)
                else:
                    print(f"  WARNING: could not download {rel}")
    files = list(font_src.rglob("*")) + list(extra_dir.iterdir())
    return sorted(p for p in files if p.is_file() and p.suffix.lower() in (".ttf", ".otf"))


# -----------------------------------------------------------------------------
# Scanning
# -----------------------------------------------------------------------------


def _scan_one(path: Path) -> dict | None:
    """Inked characters and shapeable complex scripts of one font file."""
    try:
        face = hb.Face(hb.Blob.from_file_path(str(path)))
        font = hb.Font(face)
        inked = []
        for cp in face.unicodes:
            ch = chr(cp)
            if unicodedata.category(ch) in _INKLESS:
                inked.append(ch)  # spaces/format characters draw nothing by design
                continue
            gid = font.get_nominal_glyph(cp)
            ext = font.get_glyph_extents(gid) if gid else None
            if ext and ext.width and ext.height:
                inked.append(ch)
        tags = set(face.get_table_script_tags("GSUB"))
        complex_ok = sorted(s for s, need in COMPLEX_SCRIPT_TAGS.items() if tags & need)
        return {"chars": "".join(inked), "complex": complex_ok}
    except Exception:
        return None


def scan_fonts(paths: list[Path], cache_file: Path) -> dict[Path, dict]:
    """Scan every font (cached across builds, keyed by path, size and mtime)."""
    cache: dict = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}
    if cache.get("version") != _SCAN_VERSION:
        cache = {"version": _SCAN_VERSION, "fonts": {}}
    out: dict[Path, dict] = {}
    dirty = False
    for i, p in enumerate(paths, 1):
        st = p.stat()
        key = f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"
        info = cache["fonts"].get(key)
        if info is None:
            info = _scan_one(p) or {"chars": "", "complex": [], "broken": True}
            cache["fonts"][key] = info
            dirty = True
        if not info.get("broken"):
            out[p] = {"chars": set(info["chars"]), "complex": set(info["complex"])}
        if i % 200 == 0:
            print(f"  scanned {i}/{len(paths)} fonts")
    if dirty:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return out


# -----------------------------------------------------------------------------
# Assignment
# -----------------------------------------------------------------------------


def _own_letters(lang: str) -> set[str]:
    """Letters, marks and digits of the language's own script(s) - what a face must draw."""
    scripts = allowed_scripts(lang)
    own = set()
    for c in VOCABS[LANGUAGES[lang][0]]:
        cat = unicodedata.category(c)
        if cat[0] in "LM" and script_of(c) in scripts:
            own.add(c)
        elif cat == "Nd":
            own.add(c)
    return own


def assign_fonts(
    fonts: dict[Path, dict], char_freq: dict[str, Counter], languages: list[str]
) -> tuple[dict[str, list[Path]], dict[str, str]]:
    """Faces per language, plus the language's letters no face can draw at all.

    A face qualifies when it inks every own-script letter/mark/digit of the language
    and - for complex scripts - carries the script's shaping rules. Scripts whose
    letter set no face covers in full (CJK: tens of thousands of ideographs) fall
    back to faces rendering :data:`OCCURRENCE_COVERAGE` of the language's text.
    """
    out: dict[str, list[Path]] = {}
    impossible: dict[str, str] = {}
    for lang in languages:
        scripts = allowed_scripts(lang)
        need_shaping = scripts & set(COMPLEX_SCRIPT_TAGS)
        shapers = [p for p, f in fonts.items() if need_shaping <= f["complex"]]
        own = _own_letters(lang)
        drawable = set().union(*(fonts[p]["chars"] for p in shapers)) if shapers else set()
        impossible[lang] = "".join(sorted(own - drawable))
        freq = char_freq.get(lang, Counter())
        total = sum(freq.values())
        # Letters a reader meets in real text. A face lacking one of them would only
        # ever be shown without it; archaic or rare letters (Devanagari ॻ, Latin ŉ)
        # are drawn only by the faces that have them, which costs no realism.
        required = {c for c in own & drawable if not total or freq[c] >= RARE_LETTER_SHARE * total}
        full = [p for p in shapers if required <= fonts[p]["chars"]]
        if len(full) < MIN_FONTS_PER_LANGUAGE:
            if total:
                partial = [
                    p
                    for p in shapers
                    if p not in full
                    and sum(n for c, n in freq.items() if c in fonts[p]["chars"]) >= OCCURRENCE_COVERAGE * total
                ]
                full += partial
        out[lang] = full
    return out, impossible


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_font_dir(fonts: dict[Path, dict], lang_fonts: dict[str, list[Path]], font_dir: Path) -> dict[str, list[str]]:
    """Materialise the used faces in ``font_dir`` plus a ``fonts.json`` manifest.

    Returns ``{lang: [file names]}``.
    """
    used = sorted({p for ps in lang_fonts.values() for p in ps})
    names = {p: p.name for p in used}
    if len(set(names.values())) != len(names):  # the same file name from two sources
        names = {p: f"{i:04d}_{p.name}" for i, p in enumerate(used)}
    shutil.rmtree(font_dir, ignore_errors=True)
    font_dir.mkdir(parents=True)
    for p in used:
        _link_or_copy(p, font_dir / names[p])
    by_lang = {lang: [names[p] for p in ps] for lang, ps in lang_fonts.items()}
    manifest = {
        "fonts": {
            names[p]: {"chars": "".join(sorted(fonts[p]["chars"])), "complex": sorted(fonts[p]["complex"])}
            for p in used
        },
        "languages": by_lang,
    }
    (font_dir / "fonts.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return by_lang


# -----------------------------------------------------------------------------
# Selection at render time
# -----------------------------------------------------------------------------


class ManifestFontSelector(FontSelector):
    """:class:`FontSelector` over a vetted font directory with a per-language restriction.

    Character support comes from ``fonts.json`` (inked glyphs only, no font parsing at
    start-up). Set :attr:`language` before each sample: candidates are then that
    language's faces first, and any face covering the text otherwise (numbers and
    punctuation, or English words on a non-Latin page). A face is never used for a
    complex script it has no shaping rules for, and every choice is confirmed by
    HarfBuzz (see :class:`~generator.components.shaping.ShapingChecker`).

    Args:
        font_dir: directory written by :func:`write_font_dir`.
        verify_shaping: shape each text before accepting a face.
        handwriting: allow downloading handwriting faces (detection forms).
        cache_dir: where downloaded handwriting faces are cached.
    """

    def __init__(
        self, font_dir: str | Path, verify_shaping: bool = True, handwriting: bool = True, cache_dir: str | None = None
    ):
        # The base class refuses an empty font table without downloads, which is what
        # a manifest-backed selector starts with; downloads are switched off after.
        super().__init__(
            None, auto_download=True, font_cache_dir=cache_dir or ".font_cache", verify_shaping=verify_shaping
        )
        self.auto_download = handwriting
        self.downloader.enabled = handwriting
        font_dir = Path(font_dir)
        manifest = json.loads((font_dir / "fonts.json").read_text(encoding="utf-8"))
        self.paths = [str(font_dir / name) for name in manifest["fonts"]]
        index = {name: i for i, name in enumerate(manifest["fonts"])}
        self._char_mask: dict[str, int] = {}
        self._script_mask: dict[str, int] = dict.fromkeys(COMPLEX_SCRIPT_TAGS, 0)
        for name, info in manifest["fonts"].items():
            bit = 1 << index[name]
            for c in info["chars"]:
                self._char_mask[c] = self._char_mask.get(c, 0) | bit
            for s in info["complex"]:
                self._script_mask[s] |= bit
        self._lang_mask = {
            lang: sum(1 << index[n] for n in names if n in index) for lang, names in manifest["languages"].items()
        }
        self._path_bit = {p: 1 << i for i, p in enumerate(self.paths)}
        self.language: str | None = None

    @property
    def languages(self) -> list[str]:
        """Languages the manifest has faces for."""
        return [lang for lang, m in self._lang_mask.items() if m]

    def _mask(self, text: str) -> int:
        mask = -1
        for c in set(text):
            if c.isspace():
                continue
            mask &= self._char_mask.get(c, 0)
            if not mask:
                return 0
            script = script_of(c)
            if script in self._script_mask:
                mask &= self._script_mask[script]
        return 0 if mask == -1 else mask

    def _paths(self, mask: int) -> list[str]:
        out = []
        while mask:
            low = mask & -mask
            out.append(self.paths[low.bit_length() - 1])
            mask ^= low
        return out

    def candidates(self, text: str, language: str | None = None) -> list[str]:
        """Faces that ink every character of ``text`` (the language's own faces first)."""
        mask = self._mask(text)
        lang = language if language is not None else self.language
        if mask and lang in self._lang_mask and mask & self._lang_mask[lang]:
            mask &= self._lang_mask[lang]
        return self._paths(mask)

    def get_font_for_text(self, text: str) -> str | None:
        """A face from :meth:`candidates` that shapes ``text`` cleanly, or ``None``."""
        found = self.candidates(text)
        return self._first_clean(text, found) if found else None

    def supports(self, text: str, font_path: str) -> bool:
        """Whether ``font_path`` inks every character of ``text`` and shapes it cleanly."""
        bit = self._path_bit.get(font_path)
        if bit is None:  # a downloaded handwriting face
            return super().supports(text, font_path)
        return bool(self._mask(text) & bit) and self.shaping.ok(text, font_path)

    def renders(self, text: str, language: str | None = None, rng: random.Random | None = None) -> bool:
        """Whether some face renders ``text`` cleanly (used to vet pool tokens)."""
        found = self.candidates(text, language)
        if not found:
            return False
        if not self.shaping.enabled:
            return True
        order = (rng or random).sample(found, min(len(found), self.MAX_SHAPING_TRIES))
        return any(self.shaping.ok(text, p) for p in order)
