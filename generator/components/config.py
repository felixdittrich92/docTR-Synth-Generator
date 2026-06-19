# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from dataclasses import dataclass

__all__ = ["GenerationConfig"]


@dataclass
class GenerationConfig:
    """Configuration for dataset generation.

    Both the source text and the fonts can be supplied locally *or* downloaded
    automatically, so a minimal run needs nothing but an output directory:

        >>> GenerationConfig(output_dir="ds", num_images=1000)          # en words + auto fonts
        >>> GenerationConfig(output_dir="ds", num_images=1000, languages=["de", "ru"])

    Attributes:
        wordlist_path (str | None): Path to a wordlist file. If ``None`` (the
            default), real words are downloaded for ``languages`` instead.
        languages (list[str] | None): ISO 639-1 codes whose real word corpora are
            downloaded when no ``wordlist_path`` is given. Defaults to ``["en"]``.
        font_dir (str | None): Directory with TTF/OTF fonts. May be ``None`` when
            relying on ``auto_download_fonts`` (which defaults to True).
        output_dir (str): Directory to save generated images.
        num_images (int): Total number of images to generate.
        bg_image_dir (str | None): Directory containing background images. If
            ``None`` and ``auto_download_backgrounds`` is set, a curated set is
            downloaded automatically (otherwise blank backgrounds are used).
        auto_download_backgrounds (bool): Download a curated background set when
            no ``bg_image_dir`` is given. Default True.
        background_cache_dir (str | None): Where to cache downloaded backgrounds.
        background_manifest_url (str | None): Optional URL of a newline-separated
            list of background filenames/URLs to use instead of the default set.
        val_percent (float): Percentage of images for the validation set.
        num_workers (int): Number of worker processes for parallel processing.
        font_size_range (tuple[int, int]): Range of font sizes to use.
        padding (int): Padding around the text in the image.
        max_attempts (int): Maximum attempts to render visible text.
        queue_maxsize (int): Limit on the size of the processing queue.

        # --- Automatic corpus (word) downloading ---
        words_per_language (int): Max words taken per language (most frequent kept).
        corpus_cache_dir (str | None): Where to cache downloaded corpora.
        min_word_length (int): Minimum word length (characters) to keep.
        max_word_length (int): Maximum word length (characters) to keep.
        corpus_filter_by_script (bool): Drop words whose script does not match the
            requested language (removes foreign-script contamination).
        casing_variant_prob (float): Probability of adding a Title/UPPER variant
            per word, so capital glyphs are represented (cased scripts only).
        numeric_token_ratio (float): Fraction of the vocabulary to additionally
            fill with realistic numeric/date/price/code tokens (0 disables).
        corpus_seed (int | None): Optional RNG seed for reproducible vocabularies.

        # --- Dataset balancing ---
        language_balance (str): How to split the image budget across languages:
            ``"balanced"`` (equal per language, default) or ``"proportional"``
            (by available word count). Ignored if ``language_weights`` is set.
        language_weights (dict[str, float] | None): Explicit per-language weights,
            e.g. ``{"en": 0.5, "de": 0.3, "ru": 0.2}`` (need not sum to 1).
        min_char_coverage (int): If > 0, ensure each character appears at least
            this many times across the dataset (best-effort, bounded).
        print_balance_report (bool): Print a summary of the resulting distribution.

        # --- Automatic font downloading ---
        auto_download_fonts (bool): Download a matching open-source font when no
            local font covers a word (instead of skipping it). Default True.
        font_cache_dir (str | None): Where to cache downloaded fonts.
        font_download_timeout (int): Per-request download timeout in seconds.

        # --- Glyph rendering realism ---
        supersample (int): Render at this integer scale then downsample for clean
            anti-aliasing. 1 disables.
        text_opacity_range (tuple[int, int]): Range of glyph alpha (0-255).
        ink_color_jitter (float): Per-channel std-dev of ink colour jitter.
        colored_ink_prob (float): Probability of colourful (vs near-neutral) ink.
        outline_prob (float): Probability of a contrasting glyph outline.
        outline_width_frac_range (tuple[float, float]): Outline stroke width as a
            fraction of the font size (proportional, like bold).

        # --- Text/background contrast ---
        min_contrast (float): Lower bound of ink-vs-background contrast [0, 1].
        max_contrast (float): Upper bound of ink-vs-background contrast.
        invert_prob (float): Probability of inverting polarity on mid-tone bgs.

        # --- Glyph-space augmentation probabilities ---
        bold_prob (float): Probability of faux-bold glyphs.
        bold_width_frac_range (tuple[float, float]): Faux-bold stroke width as a
            fraction of the font size (kept proportional so small text stays
            readable instead of blobbing).
        rotation_prob (float): Probability of applying rotation.
        blur_prob (float): Probability of a glyph blur.
        perspective_prob (float): Probability of perspective distortion.
        pixel_dropout_prob (float): Probability of ink-erosion pixel dropout.

        # --- Glyph-space augmentation parameters ---
        rotation_range (tuple[float, float]): Range of rotation angles.
        blur_radius_range (tuple[float, float]): Range of glyph blur radius.
        perspective_margin (int): Margin for perspective distortion.
        pixel_dropout_range (tuple[float, float]): Fraction-of-ink dropout range.

        # --- Image-space (post-composite) degradations ---
        final_blur_prob (float): Probability of blurring the whole composited crop.
        final_blur_radius_range (tuple[float, float]): Radius range for that blur.
        noise_prob (float): Probability of adding Gaussian sensor noise.
        noise_std_range (tuple[float, float]): Std-dev range of the sensor noise.
        jpeg_prob (float): Probability of JPEG compression artifacts.
        jpeg_quality_range (tuple[int, int]): JPEG quality factor range (1-100).
        brightness_jitter (float): Max relative brightness change.
        contrast_jitter (float): Max relative contrast change.
        output_jpeg (bool): Save samples as JPEG instead of PNG.
        output_jpeg_quality (int): Quality used when ``output_jpeg`` is True.

        # --- Detection dataset (task="detection") ---
        task (str): ``"recognition"`` (word/line crops, default) or
            ``"detection"`` (document-like pages with per-word polygons, in the
            docTR detection labels.json format).
        det_page_width_range (tuple[int, int]): Page width range in pixels.
        det_page_height_range (tuple[int, int]): Page height range in pixels.
        det_font_size_range (tuple[int, int]): Body font size range on a page.
        det_max_words_per_page (int): Candidate words supplied per page. The page
            is filled top-to-bottom with as many as fit, so this should be
            generous enough to fill a full page at small font sizes.
        det_margin_ratio (float): Page margin as a fraction of min(width, height).
        det_block_gap_range (tuple[float, float]): Gap between paragraph blocks as
            a fraction of the line height.
        det_max_blocks (int): Safety cap on paragraph blocks per page (the real
            limit is the available vertical space).
        det_heading_prob (float): Probability a block starts as a larger heading.
        det_plain_background_prob (float): Probability of using a clean generated
            paper background instead of a texture image. Texture photos that
            contain their own text would add unlabelled (false-negative) words,
            so for detection prefer text-free backgrounds or generated paper.
        det_rotation_prob (float): Probability of a small global page rotation.
        det_rotation_range (tuple[float, float]): Page rotation angle range (deg).
    """

    # Text source (wordlist OR downloaded corpus). All optional => no wordlist required.
    wordlist_path: str | None = None
    languages: list[str] | None = None

    # Fonts (local dir OR downloaded). Optional => no font dir required.
    font_dir: str | None = None

    output_dir: str = "output_dataset"
    num_images: int = 1000
    bg_image_dir: str | None = None
    bg_cache_size: int = 16
    bg_max_dimension: int = 2000
    auto_download_backgrounds: bool = True
    background_cache_dir: str | None = None
    background_manifest_url: str | None = None
    val_percent: float = 0.2
    num_workers: int = 4
    font_size_range: tuple[int, int] = (15, 40)
    padding: int = 4
    max_attempts: int = 5
    queue_maxsize: int = 100

    # Automatic corpus downloading
    words_per_language: int = 50000
    corpus_cache_dir: str | None = None
    min_word_length: int = 1
    max_word_length: int = 24
    corpus_filter_by_script: bool = True
    casing_variant_prob: float = 0.3
    numeric_token_ratio: float = 0.05
    corpus_seed: int | None = None

    # Dataset balancing
    language_balance: str = "balanced"
    language_weights: dict[str, float] | None = None
    min_char_coverage: int = 0
    print_balance_report: bool = True

    # Automatic font downloading
    auto_download_fonts: bool = True
    font_cache_dir: str | None = None
    font_download_timeout: int = 30

    # Glyph rendering realism
    supersample: int = 3
    text_opacity_range: tuple[int, int] = (200, 255)
    ink_color_jitter: float = 12.0
    colored_ink_prob: float = 0.25
    outline_prob: float = 0.05
    outline_width_frac_range: tuple[float, float] = (0.02, 0.045)

    # Text/background contrast
    min_contrast: float = 0.45
    max_contrast: float = 0.95
    invert_prob: float = 0.15

    # Glyph-space augmentation probabilities
    bold_prob: float = 0.3
    bold_width_frac_range: tuple[float, float] = (0.03, 0.06)
    rotation_prob: float = 0.6
    blur_prob: float = 0.3
    perspective_prob: float = 0.5
    pixel_dropout_prob: float = 0.2

    # Glyph-space augmentation parameters
    rotation_range: tuple[float, float] = (-2, 2)
    blur_radius_range: tuple[float, float] = (0.3, 1.0)
    perspective_margin: int = 2
    pixel_dropout_range: tuple[float, float] = (0.1, 0.2)

    # Image-space (post-composite) degradations
    final_blur_prob: float = 0.25
    final_blur_radius_range: tuple[float, float] = (0.3, 1.2)
    noise_prob: float = 0.5
    noise_std_range: tuple[float, float] = (2.0, 12.0)
    jpeg_prob: float = 0.6
    jpeg_quality_range: tuple[int, int] = (35, 92)
    brightness_jitter: float = 0.12
    contrast_jitter: float = 0.12
    output_jpeg: bool = False
    output_jpeg_quality: int = 90

    # Detection dataset (task="detection")
    task: str = "recognition"
    det_page_width_range: tuple[int, int] = (700, 1100)
    det_page_height_range: tuple[int, int] = (900, 1500)
    det_font_size_range: tuple[int, int] = (14, 32)
    det_max_words_per_page: int = 600
    det_margin_ratio: float = 0.06
    det_block_gap_range: tuple[float, float] = (0.5, 1.5)
    det_max_blocks: int = 60
    det_heading_prob: float = 0.3
    det_plain_background_prob: float = 0.4
    det_rotation_prob: float = 0.3
    det_rotation_range: tuple[float, float] = (-2.5, 2.5)

    def __post_init__(self):
        # When neither a wordlist nor explicit languages are given, default to
        # English so a bare config "just works" without any local resources.
        if self.wordlist_path is None and self.languages is None:
            self.languages = ["en"]
