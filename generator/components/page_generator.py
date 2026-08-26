# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import hashlib
import math
import multiprocessing as mp
import random
from dataclasses import dataclass, field
from queue import Empty

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .background_manager import BackgroundManager
from .capture import apply_capture, should_capture
from .config import GenerationConfig
from .font_selector import FontSelector
from .legibility import DegradationBudget
from .media import PageMedia, apply_delivery_resample, media_for_pinned_size, sample_media
from .text_renderer import TextRenderer, TextStyle
from .text_styling import (
    PagePalette,
    apply_final_degradations,
    build_final_augmentations,
    decide_text_style,
    recolor_coverage,
    sample_page_palette,
)
from .token_sampler import TokenSampler

__all__ = ["PageGenerator", "DetectionTask"]

Polygon = list[list[float]]


@dataclass
class DetectionTask:
    """A single page to render for the detection dataset.

    Attributes:
        words (list[str]): Candidate words to lay out (as many as fit are used).
        save_path (str): Where to write the page image.
        filename (str): Filename key for the labels file.
        split (str): "train" or "val".
    """

    words: list[str] = field(default_factory=list)
    save_path: str = ""
    filename: str = ""
    split: str = "train"


class PageGenerator:
    """Generates document-like pages with per-word polygons.

    Args:
        config (GenerationConfig): Configuration (see the ``det_*`` fields).
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.font_selector = FontSelector(
            config.resources.font_dir,
            auto_download=config.resources.auto_download_fonts,
            font_cache_dir=config.resources.font_cache_dir,
            download_timeout=config.resources.font_download_timeout,
        )
        self.text_renderer = TextRenderer(config)
        self.background_manager = BackgroundManager(
            config.resources.bg_image_dir,
            cache_size=config.resources.bg_cache_size,
            max_dimension=config.resources.bg_max_dimension,
        )
        self.final_augs = build_final_augmentations(config)
        # Per-page style context, reset by every ``generate_page`` call.
        self._page_fonts: dict[str, str] = {}
        self._palette: PagePalette | None = None
        self._rtl_page = False
        self._min_text_px = float("inf")
        self._media: PageMedia | None = None

    # -- layout -----------------------------------------------------------

    def generate_page(self, words: list[str]) -> tuple[Image.Image, list[Polygon]]:
        """Lay out ``words`` on a page and return the page image + word polygons.

        The layout (paragraph, dense newspaper columns, a label/value form, an
        ID-card with fields or a fully vertical page) is chosen per
        :attr:`DetectionConfig.layout`, so a single run can mimic the variety of
        real documents. Horizontal layouts additionally carry vertical text
        regions with probability :attr:`DetectionConfig.vertical_prob`.
        """
        cfg = self.config
        self._begin_page()
        layout = self._choose_layout()
        width, height = self._page_size(layout)

        # Detection ground truth only contains the words we place, so any text in
        # a background photo becomes an unlabelled false negative. Forms/ID-cards
        # always use clean generated paper; others mix in textures.
        if layout in ("form", "id_card") or random.random() < cfg.detection.plain_background_prob:
            page = self._paper_background((width, height)).convert("RGBA")
        else:
            page = self.background_manager.get_page_background((width, height)).convert("RGBA")
            page = self._calm_background(page)

        if not words:
            return page.convert("RGB"), []

        take_word = self._word_supplier(words)
        rtl = self._is_rtl(words)
        self._rtl_page = rtl
        margin = max(4, int(min(width, height) * cfg.detection.margin_ratio))
        area = (margin, margin, width - margin, height - margin)
        polygons: list[Polygon] = []

        if random.random() < cfg.detection.bleed_through_prob:
            self._add_bleed_through(page, area, take_word, rtl)

        # Carve vertical strips out of the content area *before* the body is laid
        # out, so vertical and horizontal text can never overlap.
        area, vertical_regions = self._plan_vertical_regions(area, layout)

        if layout == "table":
            self._layout_table(page, area, take_word, rtl, polygons)
        elif layout == "receipt":
            self._layout_receipt(page, area, take_word, rtl, polygons)
        elif layout == "newspaper":
            self._layout_newspaper(page, area, take_word, rtl, polygons)
        elif layout == "form":
            self._layout_form(page, area, take_word, rtl, polygons)
        elif layout == "id_card":
            self._layout_id_card(page, area, take_word, rtl, polygons)
        elif layout == "vertical":
            self._layout_vertical(page, area, take_word, rtl, polygons)
        else:
            self._layout_paragraph(page, area, take_word, rtl, polygons)

        for region in vertical_regions:
            self._draw_vertical_region(page, region, take_word, polygons)

        if layout not in ("id_card", "receipt") and random.random() < cfg.detection.furniture_prob:
            self._add_furniture(page, (margin, margin, width - margin, height - margin), take_word, rtl, polygons)

        page = page.convert("RGB")
        # A word running off the edge is legitimate, but its *box* must stop at
        # the page; anything fully outside is dropped.
        polygons = self._clip_polygons(polygons, width, height)

        # Pinning the page size pins the *output* size: capture composites the
        # sheet into a larger frame and the delivery resample resizes it, so
        # both are skipped rather than silently returning other dimensions.
        fixed_size = cfg.pins_page_size()
        if polygons and not fixed_size and should_capture(cfg):
            # A photographed capture already carries rotation in its homography,
            # so the flat page rotation is skipped rather than applied twice.
            page, polygons = apply_capture(cfg, page, polygons, DegradationBudget.for_text_height(self._min_text_px))
        elif polygons and not fixed_size and random.random() < cfg.detection.rotation_prob:
            # ``expand=True`` grows the canvas, so this is skipped too when the
            # caller pinned the output size.
            angle = random.uniform(*cfg.detection.rotation_range)
            page, polygons = self._rotate(page, polygons, angle)

        delivered = self._min_text_px
        softened = False
        if self._media is not None and polygons and not fixed_size:
            page, polygons, scale = apply_delivery_resample(cfg, page, polygons, self._media, self._min_text_px)
            delivered *= scale
            softened = scale < 0.95
            polygons = self._clip_polygons(polygons, page.width, page.height)

        # Degradations are budgeted against the *delivered* glyph height, since
        # that is the resolution the model will actually be trained on.
        page = apply_final_degradations(
            cfg, page, self.final_augs, DegradationBudget.for_text_height(delivered, already_softened=softened)
        )
        return page, polygons

    # -- layout building blocks ------------------------------------------

    def _choose_layout(self) -> str:
        cfg = self.config
        if cfg.detection.layout != "mixed":
            return cfg.detection.layout
        weights = cfg.detection.layout_weights or {
            "paragraph": 0.24,
            "newspaper": 0.18,
            "form": 0.14,
            "id_card": 0.12,
            "vertical": 0.10,
            "table": 0.14,
            "receipt": 0.08,
        }
        names = list(weights)
        return random.choices(names, weights=[weights[n] for n in names], k=1)[0]

    def _word_supplier(self, words: list[str]):
        """Return a ``take_word()`` drawing tokens with real line statistics.

        Words are recycled indefinitely so any region fills fully, and each draw
        goes through :class:`TokenSampler` for short-word clustering and attached
        punctuation. With ``function_word_ratio=0`` and ``punctuation_prob=0``
        this degrades to the previous uniform recycling.
        """
        cfg = self.config.detection
        sampler = TokenSampler(
            words,
            function_word_ratio=cfg.function_word_ratio,
            function_word_max_len=cfg.function_word_max_len,
            punctuation_prob=cfg.punctuation_prob,
        )
        return sampler.take

    def _begin_page(self) -> None:
        """Reset the per-page style context (fonts and ink)."""
        self._page_fonts = {}
        self._min_text_px = float("inf")
        self._media = None
        self._palette = sample_page_palette(self.config) if random.random() < 0.995 else None
        if random.random() >= self.config.detection.page_font_coherence:
            self._page_fonts = {"__free__": ""}  # sentinel: resolve per word

    def _font_for(self, text: str, role: str = "body") -> str | None:
        """Resolve a font, preferring the face already pinned for ``role``.

        Picking at random per word - the old behaviour - turns a page with a
        large font set into a ransom note. A page pins one face per role and
        falls back to per-word resolution only for text the pinned face cannot
        render (a different script, a missing symbol).
        """
        if "__free__" in self._page_fonts:
            return self.font_selector.get_font_for_text(text)

        if role == "heading" and "heading" not in self._page_fonts:
            # Headings usually reuse the body face; sometimes they get their own.
            if random.random() >= self.config.detection.heading_font_prob and "body" in self._page_fonts:
                self._page_fonts["heading"] = self._page_fonts["body"]

        pinned = self._page_fonts.get(role)
        if pinned:
            supported = self.font_selector.font_support_table.get(pinned)
            if supported is not None and all(c in supported for c in text if not c.isspace()):
                return pinned

        resolved = self.font_selector.get_font_for_text(text)
        if resolved and role not in self._page_fonts:
            self._page_fonts[role] = resolved
        return resolved

    def _style_at(
        self,
        page: Image.Image,
        x: float,
        y: float,
        w: float,
        h: float,
        bold_width: int,
        extent: float | None = None,
    ):
        """Choose ink for a block, laying a scrim first if the background varies too much.

        One ink is chosen per block from the *mean* of the area under it. That is
        fine on paper, but a photo background can run bright to dark inside a
        single block, and then the one ink that suits the average is invisible
        over the extremes - which is how text ends up unreadable on a texture
        even though its contrast looks correct on paper.

        Above ``background_scrim_std`` the block gets a translucent paper panel,
        the same thing a designer does when text has to sit over an image.
        """
        x0, y0 = int(max(0, x)), int(max(0, y))
        # ``h`` is a line or two, but the block goes on filling downwards. Ink
        # chosen from the first line alone suits the background there and can be
        # invisible further down a texture that shades from light to dark, which
        # is how a page ends up with a quarter of its words swallowed while every
        # individual style decision looks correct. Judge the whole extent.
        x1 = int(min(page.width, x + max(20, w)))
        y1 = int(min(page.height, y + max(6, extent if extent is not None else h)))
        if x1 <= x0 or y1 <= y0:
            x1, y1 = x0 + 1, y0 + 1

        sample = page.crop((x0, y0, x1, y1)).convert("RGB")
        threshold = self.config.detection.background_scrim_std
        if threshold > 0 and np.asarray(sample.convert("L"), dtype=np.float32).std() > threshold:
            pad = max(2, int((y1 - y0) * 0.12))
            box = (
                max(0, x0 - pad),
                max(0, y0 - pad),
                min(page.width, x1 + pad),
                min(page.height, y1 + pad),
            )
            tone = random.randint(226, 250) if random.random() < 0.75 else random.randint(18, 46)
            alpha = int(255 * random.uniform(0.62, 0.9))
            scrim = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (tone, tone, tone, alpha))
            page.alpha_composite(scrim, (box[0], box[1]))
            sample = page.crop((x0, y0, x1, y1)).convert("RGB")

        return decide_text_style(self.config, sample, bold_width=bold_width, outline_width=0, palette=self._palette)

    def _render_word(self, word: str, font_size: int, bold_width: int, role: str = "body"):
        font_path = self._font_for(word, role)
        if not font_path:
            return None
        return self.text_renderer.render_coverage(word, font_path, font_size, bold_width)

    def _emit(self, page, coverage, style, paste_x: float, y: float, polygons: list[Polygon]) -> bool:
        px, py = int(paste_x), int(y)
        page.alpha_composite(recolor_coverage(coverage, style), (px, py))
        bbox = coverage.getchannel("A").getbbox()
        if bbox:
            left, top, right, bottom = bbox
            polygons.append([
                [px + left, py + top],
                [px + right, py + top],
                [px + right, py + bottom],
                [px + left, py + bottom],
            ])
            # The smallest glyph on the page sets the degradation budget.
            self._min_text_px = min(self._min_text_px, float(bottom - top))
            return True
        return False

    def _place_token(
        self, page, token, x, y, font_size, style, bold_width, polygons, align_right=False, role="body", max_width=None
    ):
        """Place a single pre-built token (e.g. a field label) at (x, y).

        Returns 0 without drawing when the token is wider than ``max_width`` -
        a right-aligned cell that overflows would silently spill into the
        neighbouring column and overprint it.
        """
        coverage = self._render_word(token, font_size, bold_width, role)
        if coverage is None:
            return 0
        if max_width is not None and coverage.width > max_width:
            return 0
        paste_x = x - coverage.width if align_right else x
        self._emit(page, coverage, style, paste_x, y, polygons)
        return coverage.width

    def _fill_box(
        self,
        page,
        box,
        take_word,
        polygons,
        font_size,
        style,
        bold_width,
        rtl,
        max_lines=10**9,
        first_indent=0,
        line_spacing=(1.15, 1.4),
        role="body",
    ) -> int:
        """Fill a rectangular box with wrapped text; return the y reached."""
        bx0, by0, bx1, by1 = box
        line_height = max(font_size + 1, int(font_size * random.uniform(*line_spacing)))
        space = max(2, int(font_size * 0.33))
        y, lines = by0, 0
        while y + line_height <= by1 and lines < max_lines:
            indent = first_indent if lines == 0 else 0
            cursor = (bx1 - indent) if rtl else (bx0 + indent)
            line_has_word = False
            for _ in range(400):  # bounded attempts per line
                coverage = self._render_word(take_word(), font_size, bold_width, role)
                if coverage is None:
                    continue
                ww = coverage.width
                paste_x = cursor - ww if rtl else cursor
                overflow = (paste_x < bx0) if rtl else (paste_x + ww > bx1)
                if overflow:
                    if line_has_word:
                        break
                    continue  # single token wider than the box: try another
                if self._emit(page, coverage, style, paste_x, y, polygons):
                    line_has_word = True
                cursor = (paste_x - space) if rtl else (paste_x + ww + space)
            if not line_has_word:
                break  # nothing fits this box width
            y += line_height
            lines += 1
        return y

    def _add_bleed_through(self, page, area, take_word, rtl) -> None:
        """Composite mirrored, blurred text as if printed on the reverse side.

        Instantly recognisable on any duplex-printed or thin-stock document, and
        a useful hard negative: the detector must learn *not* to box it. Rendered
        into a scratch layer so none of it reaches the ground truth.
        """
        cfg = self.config.detection
        layer = Image.new("RGBA", page.size, (0, 0, 0, 0))
        ignored: list[Polygon] = []
        font_size = random.randint(*cfg.font_size_range)
        style = TextStyle(fill_color=(25, 25, 35), opacity=255, bold_width=0)
        self._fill_box(layer, area, take_word, ignored, font_size, style, 0, rtl)

        layer = layer.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        layer = layer.filter(ImageFilter.GaussianBlur(random.uniform(*cfg.bleed_through_blur_range)))
        alpha = np.asarray(layer.getchannel("A"), dtype=np.float32) * random.uniform(*cfg.bleed_through_alpha_range)
        layer.putalpha(Image.fromarray(alpha.astype(np.uint8), mode="L"))
        page.alpha_composite(layer)

    # -- page furniture ---------------------------------------------------

    def _add_furniture(self, page, area, take_word, rtl, polygons) -> None:
        """Add the things that sit *around* the body: headers, stamps, marks.

        The label split matters here. A stamp and a page number are **text** and
        must be boxed. A logo, a barcode and a redaction bar are **not** text and
        are deliberately left unlabelled - they are the hard negatives that teach
        a detector to ignore high-contrast non-text structure, which is a classic
        false-positive source on real documents.
        """
        cfg = self.config.detection
        bx0, by0, bx1, by1 = area
        draw = ImageDraw.Draw(page)

        if random.random() < 0.6:  # running header / footer with a page number
            fs = self._min_font()
            for edge_y, live in ((by0 - int(fs * 2.2), True), (by1 + int(fs * 0.8), True)):
                if not live or edge_y < 2 or edge_y + fs * 1.4 > page.height - 2:
                    continue
                if random.random() < 0.5:
                    continue
                style = self._style_at(page, bx0, edge_y, bx1 - bx0, fs * 2, 0)
                self._fill_box(
                    page,
                    (bx0, edge_y, int(bx1 - (bx1 - bx0) * 0.25), edge_y + fs * 2),
                    take_word,
                    polygons,
                    fs,
                    style,
                    0,
                    rtl,
                    max_lines=1,
                )
                page_no = random.choice([
                    str(random.randint(1, 99)),
                    f"- {random.randint(1, 99)} -",
                    f"{random.randint(1, 9)}/{random.randint(9, 99)}",
                ])
                self._place_token(page, page_no, bx1, edge_y, fs, style, 0, polygons, align_right=True)

        if random.random() < cfg.logo_prob:  # logo block or barcode: NOT text, NOT labelled
            self._draw_non_text_mark(page, area, draw)

        if random.random() < cfg.stamp_prob:  # a stamp IS text and IS labelled
            self._draw_stamp(page, area, take_word, polygons)

        if random.random() < cfg.redaction_prob:  # redaction bar: covers text, unlabelled
            fs = self._body_font()
            rx0 = random.randint(bx0, max(bx0, bx1 - int((bx1 - bx0) * 0.3)))
            rw = random.randint(int((bx1 - bx0) * 0.12), int((bx1 - bx0) * 0.4))
            ry = random.randint(by0, max(by0, by1 - fs * 2))
            bar = (rx0, ry, min(bx1, rx0 + rw), ry + int(fs * 1.3))
            draw.rectangle(bar, fill=(20, 20, 24))
            polygons[:] = [p for p in polygons if not self._mostly_inside(p, bar)]

        if random.random() < cfg.signature_prob:
            self._draw_signature(draw, area)

    @staticmethod
    def _clip_polygons(polygons: list[Polygon], width: int, height: int) -> list[Polygon]:
        """Clamp boxes to the page and drop any that fall entirely outside it."""
        kept: list[Polygon] = []
        for poly in polygons:
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            if max(xs) <= 0 or min(xs) >= width or max(ys) <= 0 or min(ys) >= height:
                continue
            clipped = [[min(max(x, 0), width), min(max(y, 0), height)] for x, y in poly]
            cxs = [pt[0] for pt in clipped]
            cys = [pt[1] for pt in clipped]
            if max(cxs) - min(cxs) < 2 or max(cys) - min(cys) < 2:
                continue  # a sliver of a glyph is not a usable label
            kept.append(clipped)
        return kept

    @staticmethod
    def _mostly_inside(poly: Polygon, box) -> bool:
        """True when a word polygon is largely covered by ``box``.

        A redaction bar hides the ink, so its labels have to go with it -
        otherwise the ground truth points at a solid black rectangle.
        """
        rx0, ry0, rx1, ry1 = box
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        px0, px1, py0, py1 = min(xs), max(xs), min(ys), max(ys)
        ox = max(0.0, min(px1, rx1) - max(px0, rx0))
        oy = max(0.0, min(py1, ry1) - max(py0, ry0))
        area = max(1e-6, (px1 - px0) * (py1 - py0))
        return (ox * oy) / area > 0.5

    def _draw_non_text_mark(self, page, area, draw) -> None:
        """A logo block or a barcode - high-contrast, structured and *not* text."""
        bx0, by0, bx1, by1 = area
        width = bx1 - bx0
        if random.random() < 0.5:  # barcode
            bw = int(width * random.uniform(0.25, 0.45))
            bh = int(bw * random.uniform(0.25, 0.4))
            x = random.randint(bx0, max(bx0, bx1 - bw))
            y = random.randint(by0, max(by0, by1 - bh))
            draw.rectangle((x - 4, y - 4, x + bw + 4, y + bh + 4), fill=(252, 252, 252))
            cursor = x
            while cursor < x + bw:
                bar_w = random.randint(1, max(2, bw // 40))
                if random.random() < 0.55:
                    draw.rectangle((cursor, y, cursor + bar_w, y + bh), fill=(15, 15, 18))
                cursor += bar_w + random.randint(1, 3)
        else:  # abstract logo mark
            size = int(width * random.uniform(0.08, 0.16))
            x = random.choice([bx0, bx1 - size])
            y = by0 - size // 2 if by0 - size // 2 > 2 else by0
            color = random.choice([(38, 62, 120), (28, 90, 70), (120, 40, 48), (40, 40, 48)])
            shape = random.random()
            if shape < 0.4:
                draw.ellipse((x, y, x + size, y + size), fill=color)
            elif shape < 0.7:
                draw.rectangle((x, y, x + size, y + size), fill=color)
            else:
                draw.polygon([(x + size / 2, y), (x + size, y + size), (x, y + size)], fill=color)

    def _draw_stamp(self, page, area, take_word, polygons) -> None:
        """A rotated stamp over the body text - text, so labelled.

        Stamps land *on top of* existing words, which is exactly the overlap a
        real detector has to survive; nothing is removed from the ground truth.
        """
        bx0, by0, bx1, by1 = area
        word = take_word().upper()
        font_size = int(random.randint(*self.config.detection.font_size_range) * random.uniform(1.3, 2.0))
        bold = self._heading_bold(font_size)
        coverage = self._render_word(word, font_size, bold, "heading")
        if coverage is None:
            return
        angle = random.uniform(-35, 35)
        coverage = coverage.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        if coverage.width >= bx1 - bx0 or coverage.height >= by1 - by0:
            return
        ink = random.choice([(170, 40, 45), (35, 60, 140), (40, 110, 70), (60, 60, 66)])
        style = TextStyle(fill_color=ink, opacity=random.randint(150, 225), bold_width=0)
        x = random.randint(bx0, bx1 - coverage.width)
        y = random.randint(by0, by1 - coverage.height)
        self._emit(page, coverage, style, x, y, polygons)
        if random.random() < 0.5:  # the box a rubber stamp leaves around the word
            pad = max(4, font_size // 3)
            ImageDraw.Draw(page).rectangle(
                (x - pad, y - pad // 2, x + coverage.width + pad, y + coverage.height + pad // 2),
                outline=(*ink, 210),
                width=max(2, font_size // 10),
            )

    @staticmethod
    def _draw_signature(draw, area) -> None:
        """A looping ink squiggle - handwriting-shaped, but not a word."""
        bx0, by0, bx1, by1 = area
        width = int((bx1 - bx0) * random.uniform(0.18, 0.35))
        height = int(width * random.uniform(0.2, 0.35))
        x = random.randint(bx0, max(bx0, bx1 - width))
        y = random.randint(by0, max(by0, by1 - height))
        ink = random.choice([(25, 35, 90), (20, 20, 25)])
        points = []
        for i in range(random.randint(18, 34)):
            t = i / 24.0
            points.append((
                x + width * (i / 30.0) + random.uniform(-3, 3),
                y + height / 2 + math.sin(t * random.uniform(5, 9)) * height / 2 + random.uniform(-2, 2),
            ))
        draw.line(points, fill=ink, width=max(1, height // 14), joint="curve")

    # -- vertical text ----------------------------------------------------

    def _choose_vertical_mode(self) -> str:
        """Pick how a vertical run is drawn: ``"cw"``, ``"ccw"`` or ``"stacked"``."""
        cfg = self.config.detection
        if random.random() < cfg.vertical_stacked_prob:
            return "stacked"
        return "ccw" if random.random() < cfg.vertical_ccw_prob else "cw"

    @staticmethod
    def _is_stackable(text: str) -> bool:
        """Whether upright glyph-by-glyph stacking is right for this text.

        Stacking is a CJK convention: the glyphs are square, so a column reads
        naturally. Latin, Cyrillic and Arabic are not - stacking them spells a
        word letter-under-letter, which outside of a shop sign no document does.
        Short numerals and single marks are fine either way, since they occur
        stacked on real signage and labels.
        """
        chars = [c for c in text if not c.isspace()]
        if not chars:
            return False
        if len(chars) <= 2 and all(c.isdigit() or not c.isalpha() for c in chars):
            return True  # "12", "%", "-" read fine upright
        for char in chars:
            code = ord(char)
            cjk = (
                0x3000 <= code <= 0x30FF  # CJK punctuation, kana
                or 0x3400 <= code <= 0x4DBF  # extension A
                or 0x4E00 <= code <= 0x9FFF  # unified ideographs
                or 0xAC00 <= code <= 0xD7AF  # hangul
                or 0xF900 <= code <= 0xFAFF  # compatibility ideographs
                or 0xFF00 <= code <= 0xFFEF  # fullwidth forms
            )
            if not cjk:
                return False
        return True

    def _render_stacked_word(self, word: str, font_size: int, bold_width: int, role: str = "body"):
        """Render a word as upright glyphs stacked top-to-bottom (signage / CJK).

        Each glyph is centred in a fixed-height cell so the column keeps an even
        rhythm even though per-glyph bounding boxes differ ("a" vs "A").
        """
        chars = [c for c in word if not c.isspace()]
        if not chars or len(chars) > self.config.detection.vertical_max_stacked_chars:
            return None
        glyphs = []
        for char in chars:
            glyph = self._render_word(char, font_size, bold_width, role)
            if glyph is None:
                return None
            glyphs.append(glyph)

        width = max(g.width for g in glyphs)
        cell = max(int(font_size * 1.05), max(g.height for g in glyphs))
        column = Image.new("RGBA", (width, cell * len(glyphs)), (0, 0, 0, 0))
        for i, glyph in enumerate(glyphs):
            column.alpha_composite(glyph, ((width - glyph.width) // 2, i * cell + (cell - glyph.height) // 2))
        return column

    def _render_vertical_word(
        self, word: str, font_size: int, bold_width: int, mode: str, role: str = "body", fallback: str = "ccw"
    ):
        """Render one word for a vertical run (rotated 90 degrees, or stacked).

        ``"ccw"`` reads bottom-to-top (the usual left-margin annotation),
        ``"cw"`` reads top-to-bottom (book spines, right-hand tabs) and
        ``"stacked"`` keeps the glyphs upright.
        """
        if mode == "stacked":
            if self._is_stackable(word):
                return self._render_stacked_word(word, font_size, bold_width, role)
            # A column drawing mixed scripts keeps one rotation for the whole
            # column, so the fallback is decided once by the caller.
            mode = fallback
        coverage = self._render_word(word, font_size, bold_width, role)
        if coverage is None:
            return None
        rotation = Image.Transpose.ROTATE_90 if mode == "ccw" else Image.Transpose.ROTATE_270
        return coverage.transpose(rotation)

    def _fill_column(
        self,
        page,
        box,
        take_word,
        polygons,
        font_size,
        style,
        bold_width,
        mode,
        max_columns=10**9,
        columns_rtl=True,
        rule_color=None,
        col_width=None,
        role="body",
    ) -> int:
        """Fill a box with vertical columns of text; return the x boundary reached.

        The transpose of :meth:`_fill_box`: words advance along y inside a column
        and columns advance along x. Columns run right-to-left by default, as in
        traditional vertical typesetting. ``col_width`` defaults to the font size
        scaled by ``vertical_line_spacing_range``; pass it explicitly to make the
        columns tile a region exactly.
        """
        bx0, by0, bx1, by1 = box
        if col_width is None:
            spacing = random.uniform(*self.config.detection.vertical_line_spacing_range)
            col_width = max(font_size + 2, int(font_size * spacing))
        col_width = max(1, int(col_width))
        fallback = "ccw" if random.random() < self.config.detection.vertical_ccw_prob else "cw"
        # Rotated words need more air between them than stacked glyphs do: a
        # stacked CJK column is meant to be tight, whereas rotated Latin words
        # run into each other and read as one long token.
        gap_scale = 0.33 if mode == "stacked" else random.uniform(*self.config.detection.vertical_word_gap_range)
        gap = max(2, int(font_size * gap_scale))
        upward = mode == "ccw"  # bottom-to-top reading order
        cols = 0
        while cols < max_columns:
            cx = (bx1 - (cols + 1) * col_width) if columns_rtl else (bx0 + cols * col_width)
            if cx < bx0 or cx + col_width > bx1:
                break
            cursor = by1 if upward else by0
            col_has_word = False
            for _ in range(400):  # bounded attempts per column
                coverage = self._render_vertical_word(take_word(), font_size, bold_width, mode, role, fallback)
                if coverage is None or coverage.width > bx1 - bx0:
                    continue
                wh = coverage.height
                paste_y = cursor - wh if upward else cursor
                overflow = (paste_y < by0) if upward else (paste_y + wh > by1)
                if overflow:
                    if col_has_word:
                        break
                    continue  # single token taller than the box: try another
                paste_x = cx + (col_width - coverage.width) // 2
                paste_x = max(bx0, min(paste_x, bx1 - coverage.width))
                if self._emit(page, coverage, style, paste_x, paste_y, polygons):
                    col_has_word = True
                cursor = (paste_y - gap) if upward else (paste_y + wh + gap)
            if not col_has_word:
                break
            cols += 1
            if rule_color is not None and cols < max_columns:
                rx = cx if columns_rtl else cx + col_width
                ImageDraw.Draw(page).line((rx, by0, rx, by1), fill=rule_color, width=1)
        return (bx1 - cols * col_width) if columns_rtl else (bx0 + cols * col_width)

    def _plan_vertical_regions(self, area, layout: str):
        """Reserve vertical strips at the edges of ``area``.

        Returns the (possibly narrowed) content area plus the reserved strips.
        Reserving up front is what keeps vertical and horizontal text from ever
        colliding - the body layout simply never sees that space.
        """
        cfg = self.config.detection
        if layout in ("vertical", "id_card", "receipt") or cfg.vertical_prob <= 0:
            return area, []
        if random.random() >= cfg.vertical_prob:
            return area, []

        bx0, by0, bx1, by1 = area
        regions: list[tuple[int, int, int, int]] = []
        wanted = 1 if random.random() < 0.8 else 2
        for _ in range(max(1, min(wanted, cfg.vertical_max_regions))):
            width = int((bx1 - bx0) * random.uniform(*cfg.vertical_region_width_range))
            # Leave enough width for a legible strip and a usable body area.
            if width < 16 or (bx1 - bx0) - width < 160:
                break
            gutter = max(4, int(width * 0.3))
            # Vertical accents rarely span the whole page: shrink from the ends.
            top = by0 + int((by1 - by0) * random.uniform(0.0, 0.15))
            bottom = by1 - int((by1 - by0) * random.uniform(0.0, 0.15))
            if random.random() < 0.5:
                regions.append((bx0, top, bx0 + width, bottom))
                bx0 += width + gutter
            else:
                regions.append((bx1 - width, top, bx1, bottom))
                bx1 -= width + gutter
        return (bx0, by0, bx1, by1), regions

    def _draw_vertical_region(self, page, region, take_word, polygons) -> None:
        """Render one reserved strip as a margin note or a solid colour banner."""
        cfg = self.config.detection
        rx0, ry0, rx1, ry1 = region
        width = rx1 - rx0
        # A rotated word occupies ~1.45x the font size across the column, so a
        # strip narrower than that cannot hold legible type: skip it entirely.
        if width < self._min_font() * 1.45 or ry1 - ry0 < 40:
            return

        if random.random() < cfg.vertical_banner_prob:
            band = random.choice([(38, 62, 120), (28, 90, 70), (120, 40, 48), (60, 55, 80), (35, 35, 42)])
            ImageDraw.Draw(page).rectangle((rx0, ry0, rx1, ry1), fill=band)

        # A rotated word occupies ~1.3x the font size across the column.
        font_size = max(self._min_font(), min(int(width / 1.45), cfg.font_size_range[1]))
        mode = self._choose_vertical_mode()
        bold_width = self._heading_bold(font_size) if random.random() < 0.35 else self._maybe_bold(font_size)
        style = self._style_at(page, rx0, ry0, width, ry1 - ry0, bold_width)
        self._fill_column(
            page,
            region,
            take_word,
            polygons,
            font_size,
            style,
            bold_width,
            mode,
            max_columns=1,
            columns_rtl=False,
            col_width=width,
        )

    def _layout_vertical(self, page, area, take_word, rtl, polygons) -> None:
        """A fully vertical page: columns of vertical text, right-to-left.

        Mirrors traditional CJK typesetting and rotated posters/covers: an
        optional horizontal masthead, an oversized title column on the reading
        side, then body columns with thin rules between them.
        """
        cfg = self.config
        bx0, by0, bx1, by1 = area
        draw = ImageDraw.Draw(page)
        mode = self._choose_vertical_mode()
        y = by0

        if random.random() < 0.5:  # horizontal masthead above the columns
            hfs = int(random.randint(*cfg.detection.font_size_range) * random.uniform(1.5, 2.4))
            hbw = self._heading_bold(hfs)
            hstyle = self._style_at(page, bx0, y, bx1 - bx0, hfs * 2, hbw)
            y = self._fill_box(
                page, (bx0, y, bx1, by1), take_word, polygons, hfs, hstyle, hbw, rtl, max_lines=1, role="heading"
            )
            y += int(hfs * 0.3)
            draw.line((bx0, y, bx1, y), fill=(120, 120, 130), width=1)
            y += int(hfs * 0.4)

        x = bx1
        if random.random() < 0.55:  # oversized title column on the first-read side
            tfs = int(random.randint(*cfg.detection.font_size_range) * random.uniform(1.5, 2.2))
            tbw = self._heading_bold(tfs)
            tstyle = self._style_at(page, bx1 - tfs * 2, y, tfs * 2, (by1 - y) // 2, tbw)
            x = self._fill_column(
                page, (bx0, y, x, by1), take_word, polygons, tfs, tstyle, tbw, mode, max_columns=1, role="heading"
            )
            x -= int(tfs * 0.4)

        # Body columns tile the remaining width exactly, so the page always fills
        # regardless of how many columns were drawn.
        lo, hi = cfg.detection.vertical_columns_range
        spacing = random.uniform(*cfg.detection.vertical_line_spacing_range)
        ncols = random.randint(lo, hi)
        col_width = max(14, (x - bx0) // max(1, ncols))
        font_size = max(self._min_font(), min(int(col_width / spacing), cfg.detection.font_size_range[1]))
        col_width = min(col_width, int(font_size * spacing) + 4)  # no oversized gutters
        ncols = max(1, (x - bx0) // col_width)

        bold_width = self._maybe_bold(font_size)
        style = self._style_at(page, bx0, y, x - bx0, by1 - y, bold_width)
        rule = (170, 170, 180) if random.random() < 0.3 else None
        self._fill_column(
            page,
            (bx0, y, x, by1),
            take_word,
            polygons,
            font_size,
            style,
            bold_width,
            mode,
            max_columns=ncols,
            rule_color=rule,
            col_width=col_width,
        )

    def _page_size(self, layout: str) -> tuple[int, int]:
        """Page dimensions for ``layout``.

        With the physical media model the page is a real sheet at a real scan
        resolution (see :mod:`.media`); ``self._media`` then drives type sizes
        too. Without it, pages are pixel rectangles as before, and receipts get
        their own geometry because a thermal roll is far narrower and longer
        than any value ``page_*_range`` can produce.
        """
        cfg = self.config.detection
        if self.config.media.enabled and not self.config.pins_page_size():
            self._media = sample_media(self.config, layout)
            return self._media.width_px, self._media.height_px

        if layout == "receipt" and not self.config.pins_page_size():
            width = random.randint(*cfg.receipt_width_range)
            height = random.randint(*cfg.receipt_height_range)
        else:
            width = random.randint(*cfg.page_width_range)
            height = random.randint(*cfg.page_height_range)

        # A pinned size still gets the media model, at the DPI those pixels imply,
        # so points and fine print keep working without breaking the contract.
        self._media = media_for_pinned_size(self.config, width, height) if self.config.media.enabled else None
        return width, height

    # -- handwriting ------------------------------------------------------

    def _handwriting_font(self, text: str) -> str | None:
        """Resolve (and pin for the page) a handwriting face for filled-in values."""
        if "handwriting" in self._page_fonts:
            pinned = self._page_fonts["handwriting"]
            if not pinned:
                return None
            supported = self.font_selector.font_support_table.get(pinned)
            if supported is not None and all(c in supported for c in text if not c.isspace()):
                return pinned
        resolved = self.font_selector.get_handwriting_font(text)
        self._page_fonts.setdefault("handwriting", resolved or "")
        return resolved

    def _write_by_hand(self, page, box, take_word, polygons, font_size, style) -> bool:
        """Fill ``box`` with a hand-written value; False if no handwriting face fits.

        Handwriting sits slightly above the printed baseline and wanders, so each
        word is nudged - a perfectly aligned "hand-written" row reads as printed.
        """
        bx0, by0, bx1, by1 = box
        cursor = bx1 if self._rtl_page else bx0
        placed = False
        for _ in range(24):
            word = take_word()
            font_path = self._handwriting_font(word)
            if font_path is None:
                return placed
            coverage = self.text_renderer.render_coverage(word, font_path, font_size, 0)
            width = coverage.width
            paste_x = cursor - width if self._rtl_page else cursor
            if (paste_x < bx0) if self._rtl_page else (paste_x + width > bx1):
                break
            wobble = random.randint(-max(1, font_size // 8), max(1, font_size // 8))
            if self._emit(page, coverage, style, paste_x, by0 + wobble, polygons):
                placed = True
            step = width + max(3, int(font_size * 0.35))
            cursor = (paste_x - max(3, int(font_size * 0.35))) if self._rtl_page else (paste_x + step)
        return placed

    def _min_font(self) -> int:
        """The smallest font size any layout may use.

        ``font_size_range[0]`` is a promise to the caller, not a hint. Layouts
        that cannot fit type at this size must change their geometry - fewer
        columns, a wider strip, or skip the element - because silently shrinking
        below the configured floor produces text that no amount of careful
        augmentation can keep readable.
        """
        if self._media is not None:
            # What must stay legible is the *delivered* glyph, so a page that
            # will be downscaled has to render its smallest type larger.
            return self._media.min_render_px_for_delivery(self.config.media.min_delivery_text_px)
        return max(6, self.config.detection.font_size_range[0])

    def _body_font(self, fine_print: bool = False) -> int:
        """Body type size in render pixels.

        Under the media model the size is chosen in *points* and converted at
        the page's scan resolution, which is what lets a 6pt footnote exist as
        genuinely small text instead of an illegible 8px smudge.
        """
        cfg = self.config.detection
        if self._media is not None:
            points = random.uniform(*(cfg.fine_print_point_range if fine_print else cfg.body_point_range))
            return max(self._min_font(), self._media.points_to_px(points))
        lo, hi = cfg.font_size_range
        if fine_print:
            return max(self._min_font(), int(lo * random.uniform(0.72, 0.92)))
        return random.randint(lo, max(lo, (lo + hi) // 2))  # smaller end -> denser

    def _layout_paragraph(self, page, area, take_word, rtl, polygons) -> None:
        cfg = self.config
        bx0, by0, bx1, by1 = area
        if random.random() < cfg.detection.edge_truncation_prob:
            # Let the block run past the margin so words are clipped by the page
            # edge. Every word used to sit fully inside the margins, so the model
            # never saw a partial glyph - common on any hand-held capture.
            overflow = int((bx1 - bx0) * random.uniform(0.04, 0.12))
            if random.random() < 0.5:
                bx0 -= overflow
            else:
                bx1 += overflow
        y, blocks = by0, 0
        while y < by1 and blocks < cfg.detection.max_blocks:
            blocks += 1
            base = random.randint(*cfg.detection.font_size_range)
            heading = random.random() < cfg.detection.heading_prob
            if heading:
                font_size = int(base * random.uniform(*cfg.detection.heading_point_scale))
            elif random.random() < cfg.detection.fine_print_prob:
                font_size = self._body_font(fine_print=True)  # footnote / legal small print
            else:
                font_size = base
            bold_width = self._heading_bold(font_size) if heading else self._maybe_bold(font_size)
            indent = random.randint(0, int((bx1 - bx0) * 0.08)) if (not heading and random.random() < 0.25) else 0
            max_lines = 2 if heading else random.randint(2, 8)
            # Judge the ink over every line the block will occupy, not just the first.
            style = self._style_at(
                page, bx0, y, bx1 - bx0, font_size * 2, bold_width, extent=font_size * 1.6 * max_lines
            )
            y = self._fill_box(
                page,
                (bx0, y, bx1, by1),
                take_word,
                polygons,
                font_size,
                style,
                bold_width,
                rtl,
                max_lines,
                indent,
                role="heading" if heading else "body",
            )
            y += int(font_size * random.uniform(*cfg.detection.block_gap_range))

    def _layout_newspaper(self, page, area, take_word, rtl, polygons) -> None:
        cfg = self.config
        bx0, by0, bx1, by1 = area
        draw = ImageDraw.Draw(page)
        rule = (55, 55, 65)
        y = by0
        # Masthead: big paper name, a double rule, then a small dateline.
        if random.random() < 0.92:
            hfs = int(random.randint(*cfg.detection.font_size_range) * random.uniform(2.2, 3.4))
            hbw = self._heading_bold(hfs)
            hstyle = self._style_at(page, bx0, y, bx1 - bx0, hfs * 2, hbw)
            y = self._fill_box(
                page,
                (bx0, y, bx1, by1),
                take_word,
                polygons,
                hfs,
                hstyle,
                hbw,
                rtl,
                max_lines=1,
                line_spacing=(1.05, 1.2),
                role="heading",
            )
            y += int(hfs * 0.18)
            draw.line((bx0, y, bx1, y), fill=rule, width=max(2, int(hfs * 0.05)))
            y += max(3, int(hfs * 0.12))
            draw.line((bx0, y, bx1, y), fill=rule, width=1)
            y += int(hfs * 0.2)
            dfs = max(cfg.detection.newspaper_font_size_range[1], int(hfs * 0.24))
            dstyle = self._style_at(page, bx0, y, bx1 - bx0, dfs * 2, 0)
            y = self._fill_box(page, (bx0, y, bx1, by1), take_word, polygons, dfs, dstyle, 0, rtl, max_lines=1)
            y += int(dfs * 0.5)
            draw.line((bx0, y, bx1, y), fill=(150, 150, 160), width=1)
            y += int(dfs * 0.5)

        # Dense columns with vertical column rules between them.
        lo, hi = cfg.detection.newspaper_columns_range
        gutter = max(6, int((bx1 - bx0) * 0.02))
        max_cols = max(2, int((bx1 - bx0 + gutter) / (90 + gutter)))  # keep columns legible
        ncols = max(2, min(random.randint(lo, hi), max_cols))
        col_w = ((bx1 - bx0) - (ncols - 1) * gutter) / ncols
        spacing = cfg.detection.newspaper_line_spacing_range
        col_top = y
        for c in range(1, ncols):  # column separators sit in the gutters
            sx = int(bx0 + c * (col_w + gutter) - gutter / 2)
            draw.line((sx, col_top, sx, by1), fill=(165, 165, 175), width=1)

        for c in range(ncols):
            cx0 = bx0 + c * (col_w + gutter)
            cx1 = cx0 + col_w
            cy, cblocks = col_top, 0
            while cy < by1 and cblocks < 80:
                cblocks += 1
                if random.random() < 0.2:  # article headline (+ optional byline)
                    hfs = int(random.randint(*cfg.detection.newspaper_font_size_range) * random.uniform(1.4, 1.9))
                    hbw = self._heading_bold(hfs)
                    hstyle = self._style_at(page, cx0, cy, col_w, hfs * 2, hbw)
                    cy = self._fill_box(
                        page,
                        (cx0, cy, cx1, by1),
                        take_word,
                        polygons,
                        hfs,
                        hstyle,
                        hbw,
                        rtl,
                        max_lines=random.randint(1, 2),
                        line_spacing=(1.1, 1.25),
                        role="heading",
                    )
                    if random.random() < 0.45:
                        bfs = max(cfg.detection.newspaper_font_size_range[0], int(hfs * 0.55))
                        bstyle = self._style_at(page, cx0, cy, col_w, bfs * 2, 0)
                        cy = self._fill_box(
                            page,
                            (cx0, cy, cx1, by1),
                            take_word,
                            polygons,
                            bfs,
                            bstyle,
                            0,
                            rtl,
                            max_lines=1,
                            line_spacing=(1.1, 1.2),
                        )
                    gap = hfs
                else:  # body paragraph
                    font_size = random.randint(*cfg.detection.newspaper_font_size_range)
                    bold_width = self._maybe_bold(font_size)
                    body_lines = random.randint(5, 16)
                    style = self._style_at(
                        page, cx0, cy, col_w, font_size * 2, bold_width, extent=font_size * 1.4 * body_lines
                    )
                    cy = self._fill_box(
                        page,
                        (cx0, cy, cx1, by1),
                        take_word,
                        polygons,
                        font_size,
                        style,
                        bold_width,
                        rtl,
                        max_lines=body_lines,
                        line_spacing=spacing,
                    )
                    gap = font_size
                cy += int(gap * random.uniform(0.2, 0.5))

    def _layout_table(self, page, area, take_word, rtl, polygons) -> None:
        """A ruled or unruled data table: header row, text and numeric columns.

        Invoices, statements and reports are largely this shape, and the
        label/value ``form`` layout does not cover it: a table has a header row,
        aligned columns and *right-aligned* numeric cells, which puts word boxes
        in a very different place on the line.
        """
        cfg = self.config.detection
        bx0, by0, bx1, by1 = area
        y = by0
        # A real statement or invoice is table *blocks* separated by notes, not
        # one table and two thirds of blank paper.
        for block in range(6):
            if by1 - y < self.config.detection.font_size_range[0] * 6:
                break
            if block == 0 or random.random() < 0.7:  # caption above the block
                tfs = int(random.randint(*cfg.font_size_range) * random.uniform(1.2, 1.7))
                tbw = self._heading_bold(tfs)
                tstyle = self._style_at(page, bx0, y, bx1 - bx0, tfs * 2, tbw)
                y = self._fill_box(
                    page, (bx0, y, bx1, by1), take_word, polygons, tfs, tstyle, tbw, rtl, max_lines=1, role="heading"
                )
                y += int(tfs * 0.5)
            y = self._draw_table_block(page, (bx0, y, bx1, by1), take_word, rtl, polygons)
            y += int(cfg.font_size_range[0] * random.uniform(1.0, 2.2))
            if random.random() < 0.45 and by1 - y > cfg.font_size_range[1] * 4:  # terms / notes
                nfs = self._body_font()
                nstyle = self._style_at(page, bx0, y, bx1 - bx0, nfs * 2, 0)
                y = self._fill_box(
                    page, (bx0, y, bx1, by1), take_word, polygons, nfs, nstyle, 0, rtl, max_lines=random.randint(2, 5)
                )
                y += int(nfs * random.uniform(0.8, 1.8))

    def _draw_table_block(self, page, box, take_word, rtl, polygons) -> int:
        """Draw one ruled/zebra table and return the y it reached."""
        cfg = self.config.detection
        bx0, by0, bx1, by1 = box
        draw = ImageDraw.Draw(page)
        rule = (120, 122, 132)
        y = by0

        font_size = self._body_font(fine_print=random.random() < self.config.detection.fine_print_prob * 0.6)
        row_h = int(font_size * random.uniform(1.8, 2.4))
        # Pick a column count the page can actually carry: a numeric cell needs
        # roughly 6.5 glyph widths, so drop columns until the narrowest one fits
        # type at the configured floor. Shrinking the font instead is what made
        # dense tables unreadable.
        lo, hi = cfg.table_columns_range
        min_font = self._min_font()
        ncols = random.randint(lo, hi)
        while ncols > 2:
            weights = [2.3] + [1.05] * (ncols - 1)
            narrow = (bx1 - bx0) * min(weights) / sum(weights)
            if narrow >= min_font * 6.5:
                break
            ncols -= 1

        # The first column carries labels and is wider; the rest are numeric.
        weights = [random.uniform(1.8, 2.8)] + [random.uniform(0.8, 1.3) for _ in range(ncols - 1)]
        total = sum(weights)
        edges, acc = [bx0], bx0
        for weight in weights:
            acc += (bx1 - bx0) * weight / total
            edges.append(acc)

        # Columns are laid out first, then the type is sized to the narrowest of
        # them: a table whose font is chosen up front produces unreadable cells
        # as soon as the column count goes up.
        narrowest = min(edges[i + 1] - edges[i] for i in range(1, ncols)) if ncols > 1 else (bx1 - bx0)
        font_size = max(min_font, min(font_size, int(narrowest / 6.5)))
        row_h = max(int(font_size * 1.8), min(row_h, int(font_size * 2.6)))

        ruled = random.random() < cfg.table_prob_ruled
        zebra = random.random() < cfg.table_zebra_prob
        nrows = random.randint(*cfg.table_rows_range)
        header_y = y

        for row in range(nrows + 1):
            if y + row_h > by1:
                break
            is_header = row == 0
            if is_header:
                draw.rectangle((bx0, y, bx1, y + row_h), fill=(226, 229, 236))
            elif zebra and row % 2 == 0:
                draw.rectangle((bx0, y, bx1, y + row_h), fill=(240, 241, 246))

            bold = self._heading_bold(font_size) if is_header else self._maybe_bold(font_size)
            pad = max(3, int(font_size * 0.35))
            for col in range(ncols):
                cx0, cx1 = edges[col] + pad, edges[col + 1] - pad
                if cx1 - cx0 < font_size:
                    continue
                style = self._style_at(page, cx0, y, cx1 - cx0, font_size * 2, bold)
                cell_y = y + int((row_h - font_size * 1.2) / 2)
                if col > 0 and not is_header:
                    # Numeric cells are right-aligned, like every real ledger.
                    # Several candidates are tried so a long value (a formatted
                    # currency amount) never overflows into the column beside it.
                    for _ in range(4):
                        value = self._numeric_cell()
                        if self._place_token(
                            page,
                            value,
                            cx1,
                            cell_y,
                            font_size,
                            style,
                            bold,
                            polygons,
                            align_right=True,
                            max_width=cx1 - cx0,
                        ):
                            break
                else:
                    self._fill_box(
                        page,
                        (cx0, cell_y, cx1, y + row_h),
                        take_word,
                        polygons,
                        font_size,
                        style,
                        bold,
                        rtl,
                        max_lines=1,
                        role="heading" if is_header else "body",
                    )
            if ruled:
                draw.line((bx0, y + row_h, bx1, y + row_h), fill=(190, 192, 200), width=1)
            y += row_h

        if ruled:
            draw.rectangle((bx0, header_y, bx1, y), outline=rule, width=1)
            for edge in edges[1:-1]:
                draw.line((edge, header_y, edge, y), fill=(190, 192, 200), width=1)
            draw.line((bx0, header_y + row_h, bx1, header_y + row_h), fill=rule, width=2)
        return y

    @staticmethod
    def _numeric_cell() -> str:
        """A plausible numeric cell: amount, percentage, count, date or code."""
        kind = random.random()
        if kind < 0.45:
            value = f"{random.uniform(0.5, 99999):,.2f}"
            symbol = random.choice(["", "", "", "$", "EUR ", "GBP "])
            return f"{symbol}{value}" if symbol else value
        if kind < 0.6:
            return f"{random.uniform(0, 100):.1f}%"
        if kind < 0.75:
            return str(random.randint(1, 9999))
        if kind < 0.9:
            return f"{random.randint(1, 28):02d}.{random.randint(1, 12):02d}.{random.randint(2015, 2026)}"
        return f"{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{random.randint(1000, 99999)}"

    def _layout_receipt(self, page, area, take_word, rtl, polygons) -> None:
        """A thermal receipt: centred header, item/price rows, dotted separators.

        Rendered on the narrow page geometry from :meth:`_page_size`, since the
        aspect ratio is as recognisable as the content.
        """
        cfg = self.config.detection
        bx0, by0, bx1, by1 = area
        draw = ImageDraw.Draw(page)
        width = bx1 - bx0
        font_size = max(self._min_font(), min(int(width / 22), cfg.font_size_range[1]))
        y = by0

        def separator(yy: int) -> int:
            dash, gap = max(3, font_size // 2), max(2, font_size // 3)
            x = bx0
            while x < bx1:
                draw.line((x, yy, min(x + dash, bx1), yy), fill=(120, 120, 128), width=1)
                x += dash + gap
            return yy + int(font_size * 0.9)

        # Centred store header, progressively smaller.
        for scale, weight in ((random.uniform(1.8, 2.4), True), (1.0, False), (1.0, False)):
            fs = max(self._min_font(), int(font_size * scale))
            bold = self._heading_bold(fs) if weight else 0
            token = take_word().upper() if weight else take_word()
            style = self._style_at(page, bx0, y, width, fs * 2, bold)
            coverage = self._render_word(token, fs, bold, "heading" if weight else "body")
            if coverage is not None:
                self._emit(page, coverage, style, bx0 + (width - coverage.width) / 2, y, polygons)
            y += int(fs * 1.5)
        y = separator(y)

        # Item rows: description left, price right.
        style = self._style_at(page, bx0, y, width, by1 - y, 0)
        row_h = int(font_size * 1.6)
        while y + row_h < by1 - font_size * 8:
            if random.random() < 0.08:
                y = separator(y)
                continue
            price = f"{random.uniform(0.5, 249):.2f}"
            self._place_token(page, price, bx1, y, font_size, style, 0, polygons, align_right=True)
            self._fill_box(
                page,
                (bx0, y, bx1 - int(width * 0.3), y + row_h),
                take_word,
                polygons,
                font_size,
                style,
                0,
                rtl,
                max_lines=1,
            )
            y += row_h

        y = separator(y)
        # Totals, then a hand-signed line on card receipts.
        for label_scale in (1.25, 1.0):
            fs = max(self._min_font(), int(font_size * label_scale))
            bold = self._heading_bold(fs) if label_scale > 1 else 0
            tstyle = self._style_at(page, bx0, y, width, fs * 2, bold)
            self._place_token(page, take_word().upper(), bx0, y, fs, tstyle, bold, polygons)
            self._place_token(
                page, f"{random.uniform(5, 999):.2f}", bx1, y, fs, tstyle, bold, polygons, align_right=True
            )
            y += int(fs * 1.7)

        if random.random() < cfg.handwriting_prob and y + font_size * 3 < by1:
            hstyle = self._style_at(page, bx0, y, width, font_size * 2, 0)
            self._write_by_hand(
                page, (bx0, y + font_size, bx1, y + font_size * 3), take_word, polygons, font_size, hstyle
            )

    def _layout_form(self, page, area, take_word, rtl, polygons) -> None:
        bx0, by0, bx1, by1 = area
        draw = ImageDraw.Draw(page)
        ink = (90, 90, 100)
        y = by0
        # Title with a header rule underneath.
        tfs = int(random.randint(*self.config.detection.font_size_range) * random.uniform(1.6, 2.2))
        tbw = self._heading_bold(tfs)
        tstyle = self._style_at(page, bx0, y, bx1 - bx0, tfs * 2, tbw)
        y = self._fill_box(
            page, (bx0, y, bx1, by1), take_word, polygons, tfs, tstyle, tbw, rtl, max_lines=1, role="heading"
        )
        y += int(tfs * 0.35)
        draw.line((bx0, y, bx1, y), fill=ink, width=2)
        y += int(tfs * 0.5)

        label_col = int((bx1 - bx0) * random.uniform(0.3, 0.42))
        boxed = random.random() < 0.5  # this form draws boxed fields rather than underlines
        while y < by1:
            font_size = self._body_font()
            row_h = int(font_size * 2.1)
            if y + row_h > by1:
                break
            r = random.random()
            if r < 0.12:  # section header on a shaded bar
                sfs = int(font_size * 1.25)
                bar_h = int(sfs * 1.7)
                if y + bar_h > by1:
                    break
                draw.rectangle((bx0, y, bx1, y + bar_h), fill=(225, 227, 234))
                sbw = self._heading_bold(sfs)
                sstyle = self._style_at(page, bx0, y, bx1 - bx0, sfs * 2, sbw)
                self._fill_box(
                    page,
                    (bx0 + int(font_size * 0.4), y + int(sfs * 0.2), bx1, y + bar_h),
                    take_word,
                    polygons,
                    sfs,
                    sstyle,
                    sbw,
                    rtl,
                    max_lines=1,
                    role="heading",
                )
                y += bar_h + int(font_size * 0.4)
                continue
            if r < 0.24:  # checkbox row
                box = int(font_size)
                ticked = random.random() < 0.5
                if rtl:
                    draw.rectangle((bx1 - box, y, bx1, y + box), outline=ink, width=1)
                    if ticked:
                        draw.line((bx1 - box, y, bx1, y + box), fill=ink, width=1)
                        draw.line((bx1 - box, y + box, bx1, y), fill=ink, width=1)
                    cbox = (bx0, y, bx1 - box - int(font_size * 0.5), y + row_h)
                else:
                    draw.rectangle((bx0, y, bx0 + box, y + box), outline=ink, width=1)
                    if ticked:
                        draw.line((bx0, y, bx0 + box, y + box), fill=ink, width=1)
                        draw.line((bx0, y + box, bx0 + box, y), fill=ink, width=1)
                    cbox = (bx0 + box + int(font_size * 0.5), y, bx1, y + row_h)
                cstyle = self._style_at(page, cbox[0], y, cbox[2] - cbox[0], font_size * 2, 0)
                self._fill_box(page, cbox, take_word, polygons, font_size, cstyle, 0, rtl, max_lines=1)
                y += row_h
                continue
            # "Label:" + value, mirrored for RTL.
            label = take_word().capitalize() + ":"
            lstyle = self._style_at(page, bx0, y, label_col, font_size * 2, 0)
            if rtl:
                self._place_token(page, label, bx1, y, font_size, lstyle, 0, polygons, align_right=True)
                vbox = (bx0, y, bx1 - label_col, y + row_h)
                fx0, fx1 = bx0, bx1 - label_col
            else:
                self._place_token(page, label, bx0, y, font_size, lstyle, 0, polygons)
                vbox = (bx0 + label_col, y, bx1, y + row_h)
                fx0, fx1 = bx0 + label_col, bx1
            vstyle = self._style_at(page, vbox[0], y, vbox[2] - vbox[0], font_size * 2, 0)
            # Printed label, hand-written value - how a filled-in form actually
            # looks. Falls back to print when no handwriting face covers the text.
            written = False
            if random.random() < self.config.detection.handwriting_prob:
                written = self._write_by_hand(page, vbox, take_word, polygons, font_size, vstyle)
            if not written:
                self._fill_box(page, vbox, take_word, polygons, font_size, vstyle, 0, rtl, max_lines=1)
            if boxed:
                draw.rectangle(
                    (fx0, y - int(font_size * 0.2), fx1, y + int(font_size * 1.5)), outline=(150, 150, 165), width=1
                )
            elif random.random() < 0.6:
                ly = int(y + font_size * 1.35)
                draw.line((fx0, ly, fx1, ly), fill=(165, 165, 175), width=1)
            y += row_h

    def _layout_id_card(self, page, area, take_word, rtl, polygons) -> None:
        bx0, by0, bx1, by1 = area
        aw, ah = bx1 - bx0, by1 - by0
        draw = ImageDraw.Draw(page)

        card_w = int(aw * random.uniform(0.74, 0.96))
        card_h = int(min(ah * random.uniform(0.5, 0.72), card_w * 0.64))
        cx0 = bx0 + (aw - card_w) // 2
        cy0 = by0 + int(ah * random.uniform(0.05, 0.18))
        cx1, cy1 = cx0 + card_w, cy0 + card_h
        tone = random.randint(225, 248)
        radius = int(card_h * 0.06)
        draw.rounded_rectangle(
            (cx0, cy0, cx1, cy1), radius=radius, fill=(tone, tone, max(0, tone - 6)), outline=(140, 145, 160), width=2
        )
        pad = int(card_h * 0.07)

        # Optional vertical side band, as printed on many residence permits and
        # access badges. Reserved from the card first so nothing else uses it.
        side_band = None
        band_width = int(card_w * random.uniform(0.1, 0.16))
        if random.random() < self.config.detection.vertical_prob and band_width >= 14 and card_w - band_width > 170:
            if rtl:
                side_band = (cx0, cx0 + band_width)
                cx0 += band_width
            else:
                side_band = (cx1 - band_width, cx1)
                cx1 -= band_width
            card_w -= band_width

        # Header band (issuing authority): a coloured bar with an emblem and light
        # text (the ink is picked automatically from the dark band background).
        band_h = int(card_h * 0.2)
        band = random.choice([(38, 62, 120), (28, 90, 70), (120, 40, 48), (60, 55, 80)])
        draw.rounded_rectangle((cx0, cy0, cx1, cy0 + band_h + radius), radius=radius, fill=band)
        draw.rectangle((cx0, cy0 + radius, cx1, cy0 + band_h), fill=band)
        em = int(band_h * 0.6)
        ex = (cx1 - pad - em) if rtl else (cx0 + pad)
        ey = cy0 + (band_h - em) // 2
        draw.ellipse((ex, ey, ex + em, ey + em), fill=(tone, tone, tone), outline=(205, 205, 215))
        tfs = max(self._min_font(), int(band_h * 0.42))
        tbw = self._heading_bold(tfs)
        if rtl:
            tbox = (cx0 + pad, cy0 + (band_h - tfs) // 2, ex - pad, cy0 + band_h)
        else:
            tbox = (ex + em + pad, cy0 + (band_h - tfs) // 2, cx1 - pad, cy0 + band_h)
        tstyle = self._style_at(page, tbox[0], tbox[1], tbox[2] - tbox[0], tfs * 1.6, tbw)
        self._fill_box(page, tbox, take_word, polygons, tfs, tstyle, tbw, rtl, max_lines=1, role="heading")

        if side_band is not None:
            sx0, sx1 = side_band
            self._draw_vertical_region(page, (sx0, cy0 + radius, sx1, cy1 - radius), take_word, polygons)

        # Photo placeholder under the band.
        body_top = cy0 + band_h + pad
        photo_w = int(card_w * 0.24)
        photo_h = int((cy1 - body_top) * 0.6)
        px0 = (cx1 - pad - photo_w) if rtl else (cx0 + pad)
        draw.rectangle(
            (px0, body_top, px0 + photo_w, body_top + photo_h), fill=(200, 202, 210), outline=(150, 150, 162)
        )

        # Field rows beside the photo, reserving space for a signature + MRZ.
        if rtl:
            fx0, fx1 = cx0 + pad, px0 - pad
        else:
            fx0, fx1 = px0 + photo_w + pad, cx1 - pad
        fs = max(self._min_font(), int(card_h * 0.062))
        label_col = int((fx1 - fx0) * 0.42)
        mrz_reserve = int(fs * 2.6)
        sig_reserve = int(fs * 1.6)
        fy = body_top
        while fy + int(fs * 1.8) <= cy1 - pad - mrz_reserve - sig_reserve:
            label = take_word().capitalize() + ":"
            lbw = self._maybe_bold(fs)
            if rtl:
                lstyle = self._style_at(page, fx1 - label_col, fy, label_col, fs * 2, lbw)
                self._place_token(page, label, fx1, fy, fs, lstyle, lbw, polygons, align_right=True)
                vbox = (fx0, fy, fx1 - label_col, fy + int(fs * 1.7))
            else:
                lstyle = self._style_at(page, fx0, fy, label_col, fs * 2, lbw)
                self._place_token(page, label, fx0, fy, fs, lstyle, lbw, polygons)
                vbox = (fx0 + label_col, fy, fx1, fy + int(fs * 1.7))
            vstyle = self._style_at(page, vbox[0], fy, vbox[2] - vbox[0], fs * 2, 0)
            written = False
            if random.random() < self.config.detection.handwriting_prob * 0.5:
                written = self._write_by_hand(page, vbox, take_word, polygons, fs, vstyle)
            if not written:
                self._fill_box(page, vbox, take_word, polygons, fs, vstyle, 0, rtl, max_lines=1)
            fy += int(fs * 1.95)

        # Signature line above the MRZ.
        sy = cy1 - pad - mrz_reserve - int(fs * 0.6)
        draw.line((fx0, sy, fx0 + int((fx1 - fx0) * 0.5), sy), fill=(120, 120, 135), width=1)

        # MRZ block at the bottom, sized to fit the card width.
        mfs = max(self._min_font(), int(card_h * 0.07))
        avail = cx1 - cx0 - 2 * pad
        n_chars = max(12, int(avail / (mfs * 0.62)))
        my = cy1 - pad - int(mfs * 2.4)
        mstyle = self._style_at(page, cx0 + pad, my, avail, mfs * 2, 0)
        for _ in range(2):
            mrz = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<<<<") for _ in range(n_chars))
            self._place_token(page, mrz, cx0 + pad, my, mfs, mstyle, 0, polygons)
            my += int(mfs * 1.15)

    @staticmethod
    def _is_rtl(words: list[str]) -> bool:
        """Decide whether the page should be laid out right-to-left.

        True when the majority of the candidate words use a right-to-left script
        (Arabic, Hebrew, Syriac, ...), so Arabic/Hebrew pages read naturally.
        """

        def char_is_rtl(cp: int) -> bool:
            return (
                0x0590 <= cp <= 0x05FF  # Hebrew
                or 0x0600 <= cp <= 0x06FF  # Arabic
                or 0x0700 <= cp <= 0x074F  # Syriac
                or 0x0750 <= cp <= 0x077F  # Arabic Supplement
                or 0x08A0 <= cp <= 0x08FF  # Arabic Extended-A
                or 0xFB1D <= cp <= 0xFEFC  # Hebrew/Arabic presentation forms
            )

        sample = words[:200]
        if not sample:
            return False
        rtl = sum(1 for w in sample if any(char_is_rtl(ord(c)) for c in w))
        return rtl > len(sample) / 2

    def _calm_background(self, page: Image.Image) -> Image.Image:
        """Tame a texture photo down to something text can sit on.

        A document is ink on paper: the paper may be textured, but the texture is
        low-amplitude next to the ink. A photo of crumpled fabric or wood grain
        has structure at the same scale and amplitude as the glyphs, and swallows
        them - contrast alone cannot rescue text competing with detail its own
        size.

        The large-scale look (colour, shading, vignetting) is kept; only the fine
        detail is compressed, to a residual standard deviation of
        ``detection.background_texture_std``.
        """
        target = self.config.detection.background_texture_std
        if target <= 0:
            return page
        rgb = page.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32)
        # Everything the glyphs have to compete with lives above this scale.
        radius = max(2.0, min(page.width, page.height) / 90.0)
        smooth = np.asarray(rgb.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
        residual = arr - smooth
        std = float(residual.std())
        if std <= target:
            return page
        calmed = smooth + residual * (target / std)
        out = Image.fromarray(np.clip(calmed, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")
        out.putalpha(page.getchannel("A"))
        return out

    @staticmethod
    def _paper_background(size: tuple[int, int]) -> Image.Image:
        """A clean, subtly-shaded paper background."""
        width, height = size
        tone = random.randint(236, 252)
        arr = np.full((height, width, 3), float(tone), dtype=np.float32)
        # Gentle low-frequency shading across the page + faint grain.
        gx = np.linspace(random.uniform(-7, 7), random.uniform(-7, 7), width)
        gy = np.linspace(random.uniform(-7, 7), random.uniform(-7, 7), height)
        arr += gx[None, :, None] + gy[:, None, None]
        arr += np.random.normal(0.0, 2.0, (height, width, 3))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

    def _maybe_bold(self, font_size: int) -> int:
        if random.random() < self.config.realism.bold_prob:
            frac = random.uniform(*self.config.realism.bold_width_frac_range)
            return max(1, round(font_size * self.text_renderer.supersample * frac))
        return 0

    def _heading_bold(self, font_size: int) -> int:
        frac = self.config.realism.bold_width_frac_range[1]
        return max(1, round(font_size * self.text_renderer.supersample * frac))

    @staticmethod
    def _rotate(page: Image.Image, polygons: list[Polygon], angle: float) -> tuple[Image.Image, list[Polygon]]:
        """Rotate the page and its polygons together about the page centre."""
        width, height = page.size
        rotated = page.rotate(
            angle,
            expand=True,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(255, 255, 255),
        )
        new_w, new_h = rotated.size
        cx, cy = width / 2.0, height / 2.0
        ncx, ncy = new_w / 2.0, new_h / 2.0
        # PIL rotates the image counter-clockwise; with the y-axis pointing down
        # this maps a source point (x, y) to the destination below.
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        def transform(px: float, py: float) -> list[float]:
            dx, dy = px - cx, py - cy
            return [dx * cos_a + dy * sin_a + ncx, -dx * sin_a + dy * cos_a + ncy]

        new_polys = [[transform(px, py) for px, py in poly] for poly in polygons]
        return rotated, new_polys

    # -- multiprocessing worker ------------------------------------------

    @staticmethod
    def detection_worker_process(
        task_queue: mp.Queue, result_queue: mp.Queue, config: GenerationConfig, worker_id: int
    ):
        """Worker process: render pages and return their polygon annotations."""
        print(f"Detection worker {worker_id} starting...")
        save_format = "JPEG" if config.core.output_jpeg else "PNG"
        save_kwargs = {"quality": config.core.output_jpeg_quality} if config.core.output_jpeg else {}

        try:
            generator = PageGenerator(config)
            processed = 0
            while True:
                try:
                    task = task_queue.get()
                    if task is None:
                        break
                    success: bool = False
                    dims: list[int] | None = None
                    polygons: list[Polygon] = []
                    try:
                        page, polygons = generator.generate_page(task.words)
                        if polygons:
                            page.save(task.save_path, save_format, **save_kwargs)
                            with open(task.save_path, "rb") as fh:
                                img_hash = hashlib.sha256(fh.read()).hexdigest()
                            dims = [page.height, page.width]
                            success = True
                            processed += 1
                            if processed % 200 == 0:
                                print(f"Detection worker {worker_id}: {processed} pages")
                        else:
                            img_hash = ""
                    except Exception as e:
                        img_hash = ""
                        print(f"Detection worker {worker_id}: error on '{task.filename}': {e}")

                    result_queue.put((task.filename, task.split, dims, img_hash, polygons, success))
                except Empty:
                    continue
                except Exception as e:
                    print(f"Detection worker {worker_id}: unexpected error: {e}")
                    break
        except Exception as e:
            print(f"Detection worker {worker_id}: failed to initialize: {e}")
        print(f"Detection worker {worker_id} finished. Rendered {processed} pages.")
