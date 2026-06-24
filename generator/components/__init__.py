from .vocabs import VOCABS
from .vocab_coverage import LANGUAGE_TO_VOCAB, augment_words_for_coverage, resolve_target_vocab
from .config import (
    GenerationConfig, CoreConfig, ResourceConfig, CorpusConfig, BalanceConfig,
    CoverageConfig, RecognitionConfig, RealismConfig, DetectionConfig,
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
from .text_styling import decide_text_style, recolor_coverage
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
]
