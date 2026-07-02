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
        numeric_token_ratio: fraction of the vocab filled with numbers/dates/codes.
        seed: optional RNG seed for reproducible vocabularies.
    """

    words_per_language: int = 50000
    min_word_length: int = 1
    max_word_length: int = 24
    filter_by_script: bool = True
    casing_variant_prob: float = 0.3
    numeric_token_ratio: float = 0.05
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
            ``"form"`` or ``"id_card"``.
        layout_weights: weights for the ``"mixed"`` blend.
        newspaper_columns_range / newspaper_font_size_range /
            newspaper_line_spacing_range: newspaper density controls.
        max_blocks: safety cap on paragraph blocks per page.
        heading_prob: probability a block starts as a larger heading.
        plain_background_prob: probability of a clean generated paper background
            (texture photos may carry their own unlabelled text).
        rotation_prob / rotation_range: small global page rotation.
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
    max_blocks: int = 60
    heading_prob: float = 0.3
    plain_background_prob: float = 0.4
    rotation_prob: float = 0.15
    rotation_range: tuple[float, float] = (-2.5, 2.5)


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
}

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
        if _group == "detection":
            _flat = f"det_{_f.name}"
        else:
            _flat = _FLAT_RENAMES.get((_group, _f.name), _f.name)
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

    def __post_init__(self):
        # With neither a wordlist nor explicit languages, default to English so a
        # bare config "just works" without any local resources.
        if self.resources.wordlist_path is None and self.core.languages is None:
            self.core.languages = ["en"]

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
        return cls(**{g: _GROUP_TYPES[g](**vals) for g, vals in grouped.items() if vals})
