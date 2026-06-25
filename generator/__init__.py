from .dataset_generator import SyntheticDatasetGenerator
from .components.config import (
    GenerationConfig, CoreConfig, ResourceConfig, CorpusConfig, BalanceConfig,
    CoverageConfig, RecognitionConfig, RealismConfig, DetectionConfig,
)
from .api import generate_dataset
from .doctr_dataset import (
    build_recognition_datasets,
    build_detection_datasets,
    build_latex_recognition_datasets,
    synth_worker_init_fn,
)
from .version import __version__
