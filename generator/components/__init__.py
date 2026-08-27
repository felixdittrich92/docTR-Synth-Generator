from .vocabs import VOCABS
from .vocab_coverage import LANGUAGE_TO_VOCAB, augment_words_for_coverage, resolve_target_vocab
from .config import (
    GenerationConfig, CoreConfig, ResourceConfig, CorpusConfig, BalanceConfig,
    CoverageConfig, RecognitionConfig, RealismConfig, DetectionConfig, CaptureConfig,
)
from .generator import TextImageGenerator, GenerationTask
from .font_selector import FontSelector
from .font_downloader import FontDownloader
from .corpus_downloader import CorpusDownloader, apply_casing_variants, generate_numeric_tokens
from .text_renderer import TextRenderer, TextStyle
from .background_manager import BackgroundManager
from .background_downloader import BackgroundDownloader
from .dataset_splitter import DatasetSplitter
from .page_generator import PageGenerator, DetectionTask
from .text_styling import decide_text_style, recolor_coverage, sample_page_palette
from .capture import apply_capture, should_capture
from .legibility import DegradationBudget
from .media import PageMedia, apply_delivery_resample, sample_media
from .token_sampler import TokenSampler
from .dataset_balancer import DatasetBalancer, BalanceResult

__all__ = [
    "VOCABS",
    "LANGUAGE_TO_VOCAB",
    "resolve_target_vocab",
    "augment_words_for_coverage",
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
    "TextImageGenerator",
    "GenerationTask",
    "FontSelector",
    "FontDownloader",
    "CorpusDownloader",
    "apply_casing_variants",
    "generate_numeric_tokens",
    "TextRenderer",
    "TextStyle",
    "BackgroundManager",
    "BackgroundDownloader",
    "DatasetSplitter",
    "DatasetBalancer",
    "BalanceResult",
    "PageGenerator",
    "DetectionTask",
    "decide_text_style",
    "recolor_coverage",
    "sample_page_palette",
    "apply_capture",
    "should_capture",
    "TokenSampler",
    "DegradationBudget",
    "PageMedia",
    "sample_media",
    "apply_delivery_resample",
]
