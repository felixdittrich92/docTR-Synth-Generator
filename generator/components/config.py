# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

__all__ = [
    "GenerationConfig",
    "CoreConfig",
    "ResourceConfig",
    "CorpusConfig",
    "BalanceConfig",
    "CoverageConfig",
    "RecognitionConfig",
    "RealismConfig",
    "DetectionConfig",
    "CaptureConfig",
    "MediaConfig",
]


@dataclass
class CoreConfig:
    """What to produce and where.

    Attributes:
        task: ``"recognition"`` (word/line crops) or ``"detection"`` (pages).
        output_dir: directory the dataset is written to.
        num_images: total images (split into train/val by ``val_percent``).
        languages: ISO 639-1 codes whose corpora are downloaded when no
            ``resources.wordlist_path`` is given. Defaults to ``["en"]``.
        val_percent: fraction of images held out for validation.
        num_workers: worker processes for parallel generation.
        max_attempts: max attempts to render visible text for a sample.
        queue_maxsize: bound on the multiprocessing work queue.
        output_jpeg: save samples as JPEG instead of PNG.
        output_jpeg_quality: JPEG quality used when ``output_jpeg`` is True.
    """

    task: str = "recognition"
    output_dir: str = "output_dataset"
    num_images: int = 1000
    languages: list[str] | None = None
    val_percent: float = 0.2
    num_workers: int = 4
    max_attempts: int = 5
    queue_maxsize: int = 100
    output_jpeg: bool = False
    output_jpeg_quality: int = 90


@dataclass
class ResourceConfig:
    """Where words, fonts and backgrounds come from (local or downloaded).

    Local paths take precedence over the matching automatic download.

    Attributes:
        wordlist_path: path to a wordlist file (else corpora are downloaded).
        font_dir: directory with TTF/OTF fonts (else fonts are downloaded).
        bg_image_dir: directory with background images (else downloaded/blank).
        auto_download_fonts: fetch a covering font when no local font matches.
        auto_download_backgrounds: fetch a curated background set if none given.
        font_cache_dir / corpus_cache_dir / background_cache_dir: cache dirs.
        font_download_timeout: per-request font download timeout (seconds).
        background_manifest_url: optional newline-separated list of background
            filenames/URLs to use instead of the default set.
        bg_cache_size: number of decoded backgrounds kept in memory.
        bg_max_dimension: backgrounds larger than this are downscaled.
    """

    wordlist_path: str | None = None
    font_dir: str | None = None
    bg_image_dir: str | None = None
    auto_download_fonts: bool = True
    auto_download_backgrounds: bool = True
    font_cache_dir: str | None = None
    corpus_cache_dir: str | None = None
    background_cache_dir: str | None = None
    font_download_timeout: int = 30
    background_manifest_url: str | None = None
    bg_cache_size: int = 50
    bg_max_dimension: int = 2000


@dataclass
class CorpusConfig:
    """Cleaning and enrichment of downloaded word corpora.

    Attributes:
        words_per_language: max words kept per language (most frequent first).
        min_word_length / max_word_length: length bounds (characters).
        filter_by_script: drop words whose script doesn't match the language.
        casing_variant_prob: probability of adding a Title/UPPER variant per word.
        numeric_token_ratio: fraction of the vocab filled with numbers, prices,
            dates, codes, units, operators and standalone symbols. Frequency word
            lists contain none of these, yet they are everywhere on invoices,
            receipts, forms and IDs - 0.05 left currency symbols at roughly 0.5%
            of the dataset. Note a restrictive ``charset`` filters them back out.
        seed: optional RNG seed for reproducible vocabularies.
    """

    words_per_language: int = 50000
    min_word_length: int = 1
    max_word_length: int = 24
    filter_by_script: bool = True
    casing_variant_prob: float = 0.3
    numeric_token_ratio: float = 0.18
    seed: int | None = None


@dataclass
class BalanceConfig:
    """How the image budget is split across languages.

    Attributes:
        language_balance: ``"balanced"`` (equal) or ``"proportional"`` (by word
            count). Ignored when ``language_weights`` is set.
        language_weights: explicit per-language weights (need not sum to 1).
        min_char_coverage: if > 0, ensure each character appears >= N times.
        print_balance_report: print a summary of the resulting distribution.
    """

    language_balance: str = "balanced"
    language_weights: dict[str, float] | None = None
    min_char_coverage: int = 0
    print_balance_report: bool = True


@dataclass
class CoverageConfig:
    """Recognition vocab coverage and restriction.

    Attributes:
        ensure_vocab_coverage: synthesise tokens so every character of the target
            vocab appears (a no-op for scripts with no fixed small vocab, e.g. CJK).
        target_vocab: vocab to cover *and* restrict labels to - a ``VOCABS`` key,
            a literal charset, or a list of those (their union).
        restrict_to_vocab: drop any word with a character outside ``target_vocab``
            so a docTR model trained on that vocab never sees an un-encodable
            label. Takes effect only when ``target_vocab`` is set.
        vocab_coverage_min_count: minimum samples each vocab character must appear in.
    """

    ensure_vocab_coverage: bool = True
    target_vocab: str | list[str] | None = None
    restrict_to_vocab: bool = True
    vocab_coverage_min_count: int = 3


@dataclass
class RecognitionConfig:
    """Recognition crop geometry.

    Attributes:
        font_size_range: range of font sizes for recognition crops.
        padding: padding (pixels) around the text in a crop.
    """

    font_size_range: tuple[int, int] = (12, 40)
    padding: int = 2


@dataclass
class RealismConfig:
    """Rendering realism: ink, contrast, glyph augmentation and degradations.

    Attributes:
        supersample: render at this integer scale then downsample (1 disables).
        text_opacity_range: glyph alpha range (0-255).
        ink_color_jitter: per-channel std-dev of ink colour jitter.
        colored_ink_prob: probability of colourful (vs near-neutral) ink.
        outline_prob / outline_width_frac_range: contrasting glyph outline.
        page_contrast_bias: where in the contrast range a page's pinned ink is
            drawn from (0 = anywhere, 1 = always max). Contrast is page-wide, so
            a low draw fades the whole document rather than one block.
        crop_texture_std: how much fine detail a background photo may keep behind
            text. Structure at glyph scale camouflages strokes no matter what ink
            is chosen; 0 disables the compression.
        min_ink_separation: guaranteed luminance separation between ink and paper
            *after* opacity is applied. Contrast, hue scaling, jitter and opacity
            each look reasonable alone and compound into invisible text.
        min_contrast / max_contrast: ink-vs-background contrast bounds [0, 1].
        invert_prob: probability of inverting polarity on mid-tone backgrounds.
        bold_prob / bold_width_frac_range: faux-bold probability and stroke width.
        rotation_prob / rotation_range: glyph rotation.
        blur_prob / blur_radius_range: glyph blur.
        perspective_prob / perspective_margin: glyph perspective distortion.
        pixel_dropout_prob / pixel_dropout_range: ink-erosion dropout.
        final_blur_prob / final_blur_radius_range: whole-crop blur.
        noise_prob / noise_std_range: Gaussian sensor noise.
        jpeg_prob / jpeg_quality_range: JPEG compression artifacts.
        brightness_jitter / contrast_jitter: max relative brightness/contrast change.
    """

    supersample: int = 3
    text_opacity_range: tuple[int, int] = (200, 255)
    ink_color_jitter: float = 12.0
    colored_ink_prob: float = 0.25
    outline_prob: float = 0.05
    outline_width_frac_range: tuple[float, float] = (0.02, 0.045)
    min_contrast: float = 0.45
    page_contrast_bias: float = 0.45
    min_ink_separation: float = 62.0
    crop_texture_std: float = 10.0
    max_contrast: float = 0.95
    invert_prob: float = 0.15
    bold_prob: float = 0.3
    bold_width_frac_range: tuple[float, float] = (0.03, 0.06)
    rotation_prob: float = 0.15
    blur_prob: float = 0.2
    perspective_prob: float = 0.2
    pixel_dropout_prob: float = 0.05
    rotation_range: tuple[float, float] = (-2, 2)
    blur_radius_range: tuple[float, float] = (0.3, 1.0)
    perspective_margin: int = 2
    pixel_dropout_range: tuple[float, float] = (0.1, 0.2)
    final_blur_prob: float = 0.15
    final_blur_radius_range: tuple[float, float] = (0.3, 1.2)
    noise_prob: float = 0.2
    noise_std_range: tuple[float, float] = (2.0, 12.0)
    jpeg_prob: float = 0.3
    jpeg_quality_range: tuple[int, int] = (35, 92)
    brightness_jitter: float = 0.12
    contrast_jitter: float = 0.12


@dataclass
class DetectionConfig:
    """Detection page layout (``task="detection"``).

    Attributes:
        page_width_range / page_height_range: page size range in pixels.
        font_size_range: body font size range on a page.
        max_words_per_page: candidate words per page (the page fills top-to-bottom).
        margin_ratio: page margin as a fraction of min(width, height).
        block_gap_range: gap between paragraph blocks (fraction of line height).
        layout: ``"mixed"`` (default), ``"paragraph"``, ``"newspaper"``,
            ``"form"``, ``"id_card"``, ``"vertical"``, ``"table"`` or ``"receipt"``.
        layout_weights: weights for the ``"mixed"`` blend.
        vertical_prob: probability that a *horizontal* page additionally carries
            a vertical text region (margin note, side banner, spine title). Set
            to 0 to reproduce the old horizontal-only behaviour.
        vertical_ccw_prob: share of rotated runs turned counter-clockwise
            (reading bottom-to-top) instead of clockwise (top-to-bottom).
        vertical_stacked_prob: share of vertical runs drawn as upright stacked
            glyphs (CJK / signage style) instead of rotated whole words.
        vertical_columns_range: number of columns in the ``"vertical"`` layout.
        vertical_line_spacing_range: column width as a multiple of the font size.
        vertical_max_regions: max vertical regions carved out of a horizontal page.
        vertical_region_width_range: width of such a region, as a fraction of the
            content width.
        vertical_banner_prob: probability a vertical region is drawn as a solid
            coloured banner (light ink on dark) rather than plain margin text.
        vertical_max_stacked_chars: words longer than this are skipped in stacked
            mode (a 20-glyph column rarely fits and never looks real).
        vertical_word_gap_range: space between rotated words in a vertical column,
            as a multiple of the font size. Stacked CJK columns stay tight.
        newspaper_columns_range / newspaper_font_size_range /
            newspaper_line_spacing_range: newspaper density controls.
        max_blocks: safety cap on paragraph blocks per page.
        heading_prob: probability a block starts as a larger heading.
        plain_background_prob: probability of a clean generated paper background
            (texture photos may carry their own unlabelled text).
        rotation_prob / rotation_range: small global page rotation.
        page_font_coherence: probability that a page pins one font per role
            instead of re-picking a face for every word (0 = old behaviour).
        heading_font_prob: probability headings get their own pinned face rather
            than reusing the body face.
        ink_deviation_prob: probability a block departs from the page ink - real
            documents are near-monochrome with the occasional accent.
        bleed_through_prob: probability that mirrored text from the reverse side
            of the sheet shows through.
        bleed_through_alpha_range / bleed_through_blur_range: strength and
            softness of that show-through.
        body_point_range: body type size in points, used when the physical media
            model is enabled (see :class:`MediaConfig`).
        heading_point_scale: heading size as a multiple of body type.
        fine_print_prob: probability a block is set as fine print - footnotes,
            legal small print, table footers.
        fine_print_point_range: type size for those blocks, in points.
        function_word_ratio: share of tokens drawn from the short-word bucket, so
            lines alternate function and content words (0 = uniform sampling).
        function_word_max_len: length bound of that bucket.
        punctuation_prob: probability a token carries attached punctuation.
        table_*: ruled/zebra table layout controls.
        receipt_*: thermal-receipt layout controls (its page geometry overrides
            ``page_*_range``, since a receipt is far narrower than any document).
        handwriting_prob: probability a form/receipt value is written by hand.
        furniture_prob: probability a page carries furniture - header/footer with
            a page number, a stamp, a signature, a redaction bar, a logo.
        stamp_prob / redaction_prob / signature_prob / logo_prob: the mix of
            furniture, conditional on ``furniture_prob``.
        edge_truncation_prob: probability a block bleeds past the margin so words
            are clipped by the page edge.
        background_texture_std: how much fine detail a texture background may
            keep, as a residual standard deviation. Photo textures carry
            structure at glyph scale and amplitude, which no amount of ink
            contrast can overcome; 0 disables the compression.
        background_scrim_std: above this luminance spread under a block, a
            translucent panel is laid down first. One ink per block cannot suit a
            background that runs bright to dark inside that block; 0 disables it.
    """

    page_width_range: tuple[int, int] = (700, 1100)
    page_height_range: tuple[int, int] = (900, 1500)
    font_size_range: tuple[int, int] = (12, 32)
    max_words_per_page: int = 600
    margin_ratio: float = 0.06
    block_gap_range: tuple[float, float] = (0.5, 1.5)
    layout: str = "mixed"
    layout_weights: dict[str, float] | None = None
    newspaper_columns_range: tuple[int, int] = (3, 6)
    newspaper_font_size_range: tuple[int, int] = (9, 15)
    newspaper_line_spacing_range: tuple[float, float] = (1.05, 1.2)
    max_blocks: int = 48
    heading_prob: float = 0.3
    plain_background_prob: float = 0.4
    rotation_prob: float = 0.15
    rotation_range: tuple[float, float] = (-2.5, 2.5)
    vertical_prob: float = 0.3
    vertical_ccw_prob: float = 0.6
    vertical_stacked_prob: float = 0.2
    vertical_columns_range: tuple[int, int] = (4, 12)
    vertical_line_spacing_range: tuple[float, float] = (1.25, 1.7)
    vertical_max_regions: int = 2
    vertical_region_width_range: tuple[float, float] = (0.06, 0.16)
    vertical_banner_prob: float = 0.35
    vertical_max_stacked_chars: int = 12
    vertical_word_gap_range: tuple[float, float] = (0.4, 0.75)
    page_font_coherence: float = 0.9
    heading_font_prob: float = 0.5
    ink_deviation_prob: float = 0.12
    bleed_through_prob: float = 0.15
    bleed_through_alpha_range: tuple[float, float] = (0.04, 0.13)
    bleed_through_blur_range: tuple[float, float] = (0.8, 2.2)
    body_point_range: tuple[float, float] = (8.5, 12.5)
    heading_point_scale: tuple[float, float] = (1.35, 2.4)
    fine_print_prob: float = 0.22
    fine_print_point_range: tuple[float, float] = (5.0, 7.0)
    function_word_ratio: float = 0.45
    function_word_max_len: int = 4
    punctuation_prob: float = 0.18
    table_prob_ruled: float = 0.6
    table_zebra_prob: float = 0.3
    table_columns_range: tuple[int, int] = (3, 6)
    table_rows_range: tuple[int, int] = (6, 22)
    receipt_width_range: tuple[int, int] = (300, 460)
    receipt_height_range: tuple[int, int] = (900, 1800)
    handwriting_prob: float = 0.35
    furniture_prob: float = 0.3
    stamp_prob: float = 0.15
    redaction_prob: float = 0.12
    signature_prob: float = 0.2
    logo_prob: float = 0.3
    edge_truncation_prob: float = 0.15
    background_texture_std: float = 14.0
    background_scrim_std: float = 24.0


@dataclass
class CaptureConfig:
    """Camera-capture simulation for detection pages (``task="detection"``).

    Without this the page *is* the image: axis-aligned, edge to edge - a flatbed
    scan. Real captures are a sheet photographed as an object in a scene, so the
    page is warped, placed on a surface, lit unevenly and slightly out of focus.

    Attributes:
        prob: probability a page is turned into a photographed capture.
        page_scale_range: fraction of the frame the page occupies.
        perspective: corner jitter as a fraction of the short page side.
        rotation_range: in-plane rotation of the sheet (degrees).
        shadow_prob / shadow_offset_frac / shadow_blur_frac: drop shadow cast by
            the sheet onto the surface.
        illumination_prob / illumination_strength: low-frequency lighting falloff.
        vignette_prob / vignette_strength: darkening toward the frame corners.
        glare_prob / glare_strength: a specular highlight blob.
        motion_blur_prob / motion_blur_length_range: directional camera shake.
        min_contrast_factor: the least local contrast lighting may leave, as a
            fraction of the original. Falloff and glare each scale contrast, and
            the product is what erases text, so the pair is bounded together.
        surface_tone_range: brightness range of the generated surface the sheet
            rests on (a *generated* surface, never a photo, so no unlabelled
            text can leak into the frame).
    """

    prob: float = 0.35
    page_scale_range: tuple[float, float] = (0.72, 0.95)
    perspective: float = 0.035
    rotation_range: tuple[float, float] = (-4.0, 4.0)
    shadow_prob: float = 0.8
    shadow_offset_frac: float = 0.012
    shadow_blur_frac: float = 0.01
    illumination_prob: float = 0.7
    illumination_strength: float = 0.28
    vignette_prob: float = 0.5
    vignette_strength: float = 0.35
    glare_prob: float = 0.2
    glare_strength: float = 0.35
    motion_blur_prob: float = 0.15
    motion_blur_length_range: tuple[int, int] = (3, 9)
    surface_tone_range: tuple[int, int] = (55, 205)
    min_contrast_factor: float = 0.68


@dataclass
class MediaConfig:
    """Physical page model and delivery resample for detection pages.

    With ``enabled`` the page stops being an arbitrary pixel rectangle: it is a
    sheet of a real size, scanned at a real resolution, with type specified in
    points. That is what makes dense small text possible - a 6pt footnote at
    300 DPI is 25px, small relative to the page and still sharp - and it is why
    ``detection.font_size_range`` becomes a safety clamp rather than the primary
    control. Set ``enabled=False`` to size pages and type directly in pixels, as
    before.

    Attributes:
        enabled: use the physical model (page size, DPI, points).
        dpi_range: capture resolution the page is rendered at.
        landscape_prob: probability a document format is rotated to landscape.
        receipt_length_range: receipt roll length, in inches.
        max_render_megapixels: cap on the rendered page; DPI is reduced rather
            than the page size when a format would exceed it.
        delivery_long_edge_range: long edge of the delivered image. Real images
            are stored smaller than they were captured, and that downscale is
            where they pick up their softening and aliasing.
        resample_prob: probability the delivery resample is applied at all.
        upscale_after_prob: probability the downscaled image is then blown back
            up - a second generation, as anything that has been through a chat
            app has had.
        min_delivery_text_px: the smallest glyph height that must survive *in
            the delivered image*. Replaces an absolute pixel floor: a page that
            will be halved has to render its smallest type twice as large.
    """

    enabled: bool = True
    dpi_range: tuple[float, float] = (150.0, 300.0)
    landscape_prob: float = 0.14
    receipt_length_range: tuple[float, float] = (5.0, 16.0)
    max_render_megapixels: float = 3.6
    delivery_long_edge_range: tuple[float, float] = (900.0, 2200.0)
    resample_prob: float = 0.85
    upscale_after_prob: float = 0.12
    min_delivery_text_px: float = 7.0


# Sub-config attribute name -> dataclass type.
_GROUP_TYPES = {
    "core": CoreConfig,
    "resources": ResourceConfig,
    "corpus": CorpusConfig,
    "balance": BalanceConfig,
    "coverage": CoverageConfig,
    "recognition": RecognitionConfig,
    "realism": RealismConfig,
    "detection": DetectionConfig,
    "capture": CaptureConfig,
    "media": MediaConfig,
}

# Groups whose flat keyword names carry a prefix to keep them unambiguous.
_GROUP_PREFIXES = {"detection": "det_", "capture": "capture_", "media": "media_"}

# Flat keyword name -> (sub-config attr, field name). Auto-built from the
# dataclasses so it stays in sync; detection fields get a ``det_`` prefix and a
# couple of corpus fields keep their historical, fully-qualified names.
_FLAT_RENAMES = {
    ("corpus", "filter_by_script"): "corpus_filter_by_script",
    ("corpus", "seed"): "corpus_seed",
}
_FLAT_MAP: dict[str, tuple[str, str]] = {}
for _group, _type in _GROUP_TYPES.items():
    for _f in dataclasses.fields(_type):
        _prefix = _GROUP_PREFIXES.get(_group)
        _flat = f"{_prefix}{_f.name}" if _prefix else _FLAT_RENAMES.get((_group, _f.name), _f.name)
        _FLAT_MAP[_flat] = (_group, _f.name)


@dataclass
class GenerationConfig:
    """Top-level configuration, composed of focused sub-configs.

    See the sub-config classes for the individual options. Build it nested, or
    use :meth:`flat` to pass flat keyword names.
    """

    core: CoreConfig = field(default_factory=CoreConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    balance: BalanceConfig = field(default_factory=BalanceConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    realism: RealismConfig = field(default_factory=RealismConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    # Flat option names the caller passed to :meth:`flat`; empty when the config
    # was built directly from sub-configs.
    explicit_options: frozenset[str] = field(default_factory=frozenset)

    def pins_page_size(self) -> bool:
        """Whether the caller pinned the detection page dimensions.

        The media model derives page size from a physical sheet and a DPI, but a
        caller who set ``det_page_width_range``/``det_page_height_range`` wants
        those pixels - for a fixed-size training batch, for instance. Silently
        overriding them is the same class of bug as ignoring the font floor.
        """
        if {"det_receipt_width_range", "det_receipt_height_range"} & self.explicit_options:
            return False  # a pinned receipt roll is its own, more specific pin
        return bool({"det_page_width_range", "det_page_height_range"} & self.explicit_options)

    def __post_init__(self):
        # With neither a wordlist nor explicit languages, default to English so a
        # bare config "just works" without any local resources.
        if self.resources.wordlist_path is None and self.core.languages is None:
            self.core.languages = ["en"]

        # ``val_percent`` is a fraction despite the name. Passing 15 for "15%"
        # silently produced a 77% validation split rather than 15%, so it is
        # rejected outright instead of quietly ruining the dataset.
        val = self.core.val_percent
        if not 0.0 <= val < 1.0:
            raise ValueError(
                f"val_percent must be a fraction in [0, 1), got {val!r}. For a 15% validation split pass 0.15, not 15."
            )

    @classmethod
    def flat(cls, **kwargs) -> "GenerationConfig":
        """Build a config from flat keyword names, routed into the sub-configs.

        Example::

            GenerationConfig.flat(num_images=1000, task="detection", det_layout="form")

        Raises:
            TypeError: if a keyword is not a known configuration option.
        """
        grouped: dict[str, dict] = {g: {} for g in _GROUP_TYPES}
        for key, value in kwargs.items():
            mapping = _FLAT_MAP.get(key)
            if mapping is None:
                raise TypeError(f"Unknown configuration option: {key!r}")
            group, field_name = mapping
            grouped[group][field_name] = value
        config = cls(**{g: _GROUP_TYPES[g](**vals) for g, vals in grouped.items() if vals})
        # Remember what was set explicitly: a pinned page size is a contract the
        # physical media model has to honour rather than quietly override.
        config.explicit_options = frozenset(kwargs)
        return config
