# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

import os
import random
import time
import warnings
from dataclasses import replace
from typing import Any, Callable

import numpy as np
from PIL import Image

from .components.config import GenerationConfig
from .components.generator import TextImageGenerator
from .components.page_generator import PageGenerator
from .components.vocab_coverage import VOCAB_TO_LANGUAGE, resolve_vocab_charset
from .dataset_generator import SyntheticDatasetGenerator

try:  # docTR's single-class name ("words"); fall back to the same literal if absent.
    from doctr.file_utils import CLASS_NAME
    from doctr.utils import Sample
except Exception:  # pragma: no cover - docTR not installed
    from dataclasses import dataclass

    CLASS_NAME = "words"

    @dataclass
    class Sample:  # type: ignore[no-redef]
        """Canonical data container for all transforms."""

        image: Any
        mask: Any | None = None
        target: np.ndarray | dict[str, np.ndarray] | None = None

        def replace(self, **kwargs) -> "Sample":
            return Sample(
                image=kwargs.get("image", self.image),
                mask=kwargs.get("mask", self.mask),
                target=kwargs.get("target", self.target),
            )


__all__ = [
    "CLASS_NAME",
    "Sample",
    "polygons_to_target",
    "render_recognition_sample",
    "render_detection_sample",
    "SyntheticRecognitionDataset",
    "SyntheticDetectionDataset",
    "build_recognition_datasets",
    "build_detection_datasets",
    "synth_worker_init_fn",
]


# --------------------------------------------------------------------------- #
# Framework-agnostic rendering helpers (no torch required - unit-testable).
# --------------------------------------------------------------------------- #
def polygons_to_target(polygons: list, use_polygons: bool) -> dict[str, np.ndarray]:
    """Convert absolute-pixel polygons into docTR's detection target dict.

    Mirrors :meth:`doctr.datasets.DetectionDataset.format_polygons` (no
    normalisation - coordinates stay in pixels, exactly as docTR stores them):

    * ``use_polygons=True``  -> ``{CLASS_NAME: (N, 4, 2)}`` rotated quads
    * ``use_polygons=False`` -> ``{CLASS_NAME: (N, 4)}`` straight boxes
      ``[xmin, ymin, xmax, ymax]`` from the polygon's min/max corners.
    """
    geoms = np.asarray(polygons, dtype=np.float32)
    if geoms.size == 0:
        geoms = np.zeros((0, 4, 2), dtype=np.float32)
    if not use_polygons:
        if geoms.shape[0]:
            geoms = np.concatenate((geoms.min(axis=1), geoms.max(axis=1)), axis=1)
        else:
            geoms = np.zeros((0, 4), dtype=np.float32)
    return {CLASS_NAME: geoms}


def render_recognition_sample(
    generator: TextImageGenerator, pool: list[str], *, max_attempts: int = 12
) -> tuple[Image.Image, str]:
    """Render one recognition crop: ``(RGB image, label)``.

    Words that fail to render visibly (missing glyph, invisible style) are
    skipped; after ``max_attempts`` a blank fallback keeps training unblocked.
    """
    word = random.choice(pool)
    for _ in range(max_attempts):
        word = random.choice(pool)
        img = generator.generate_image(word)
        if img is not None:
            return img.convert("RGB"), word
    return Image.new("RGB", (32, 32), (255, 255, 255)), word


def render_detection_sample(
    page_generator: PageGenerator,
    pool: list[str],
    words_per_page: int,
    use_polygons: bool,
    *,
    max_attempts: int = 4,
) -> tuple[Image.Image, dict[str, np.ndarray]]:
    """Render one detection page: ``(RGB image, {CLASS_NAME: geoms})``.

    Retries a few times to avoid the rare empty page; the candidate words are
    sampled with replacement just like the offline generator.
    """
    page: Image.Image | None = None
    polygons: list = []
    for _ in range(max_attempts):
        words = [random.choice(pool) for _ in range(words_per_page)]
        page, polygons = page_generator.generate_page(words)
        if polygons:
            break
    assert page is not None
    return page.convert("RGB"), polygons_to_target(polygons, use_polygons)


# --------------------------------------------------------------------------- #
# Torch glue (imported lazily).
# --------------------------------------------------------------------------- #
def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "PyTorch is required for the on-the-fly docTR datasets. Install it in your "
            "training environment, e.g. `pip install python-doctr`."
        ) from exc
    return torch


def _pil_to_tensor(img: Image.Image):
    """PIL RGB -> ``CxHxW`` float32 tensor in ``[0, 1]`` (matches docTR's reader)."""
    torch = _torch()
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div_(255.0)


def synth_worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker an independent RNG stream.

    PyTorch does not re-seed NumPy per worker (a well-known footgun), so without
    this two workers can emit identical "random" samples. Pass it as
    ``DataLoader(..., worker_init_fn=synth_worker_init_fn)`` for training sets.
    """
    torch = _torch()
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class _BaseSynthDataset:
    """Shared plumbing: virtual epoch length, per-sample seeding, transforms, collate."""

    def __init__(
        self,
        num_samples: int,
        *,
        img_transforms: Callable | None = None,
        sample_transforms: Callable | None = None,
        seed: int | None = None,
    ) -> None:
        _torch()  # fail fast with a clear message if torch is unavailable
        if num_samples <= 0:
            raise ValueError("num_samples (the virtual epoch length) must be > 0")
        self.num_samples = int(num_samples)
        self.img_transforms = img_transforms
        self.sample_transforms = sample_transforms
        self.seed = seed
        self._salt = random.randrange(2**31)

    def __len__(self) -> int:
        return self.num_samples

    def _seed_sample(self, index: int) -> None:
        # seed=None  -> fresh sample every access (training: maximum variety)
        # seed=int   -> deterministic per index (validation: a fixed virtual set)
        if self.seed is not None:
            s = (self.seed + index) & 0xFFFFFFFF
        else:
            s = hash((os.getpid(), index, time.perf_counter_ns(), self._salt)) & 0xFFFFFFFF
        random.seed(s)
        np.random.seed(s)

    def _render(self, index: int) -> tuple[Image.Image, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def __getitem__(self, index: int):
        self._seed_sample(index)
        img, target = self._render(index)
        tensor = _pil_to_tensor(img)
        sample = Sample(image=tensor, target=target)
        if self.img_transforms is not None:
            sample = self.img_transforms(sample)
        if self.sample_transforms is not None:
            sample = self.sample_transforms(sample)
        # Keep compatibility with the existing collate_fn
        return sample.image, sample.target

    @staticmethod
    def collate_fn(samples):
        torch = _torch()
        images, targets = zip(*samples)
        return torch.stack(list(images), dim=0), list(targets)


class SyntheticRecognitionDataset(_BaseSynthDataset):
    """On-the-fly recognition dataset (single-word crops) for docTR.

    Args:
        pool: words to sample labels from - usually from
            ``SyntheticDatasetGenerator(config).build_recognition_pools()``.
        config: the generation config controlling rendering/realism.
        num_samples: virtual epoch length (``len(dataset)``). Samples are
            generated fresh, so this only sets iterations per epoch.
        img_transforms / sample_transforms: passed through exactly as docTR's
            ``RecognitionDataset`` uses them.
        seed: ``None`` -> a fresh crop every access (training); an int -> a
            reproducible virtual set indexed deterministically (validation).
    """

    def __init__(
        self,
        pool: list[str],
        config: GenerationConfig,
        num_samples: int,
        *,
        vocab: str | list[str] | None = None,
        img_transforms: Callable | None = None,
        sample_transforms: Callable | None = None,
        seed: int | None = None,
        max_attempts: int = 12,
    ) -> None:
        super().__init__(num_samples, img_transforms=img_transforms, sample_transforms=sample_transforms, seed=seed)
        if not pool:
            raise ValueError("recognition word pool is empty")
        # Restrict labels to the model's vocab so docTR can encode every one of
        # them (a single out-of-vocab character crashes training). The vocab is
        # taken from an explicit ``vocab`` arg, else ``config.coverage.target_vocab``.
        spec = (
            vocab
            if vocab is not None
            else (config.coverage.target_vocab if config.coverage.restrict_to_vocab else None)
        )
        self.charset = resolve_vocab_charset(spec)
        if self.charset:
            pool = [w for w in pool if w and set(w) <= self.charset]
            if not pool:
                raise ValueError("recognition word pool is empty after vocab filtering")
        self.pool = list(pool)
        self.config = config
        self.max_attempts = max_attempts
        self._generator = TextImageGenerator(config)

    def _render(self, index: int) -> tuple[Image.Image, str]:
        return render_recognition_sample(self._generator, self.pool, max_attempts=self.max_attempts)


class SyntheticDetectionDataset(_BaseSynthDataset):
    """On-the-fly detection dataset (full synthetic pages) for docTR.

    Yields ``(image_tensor, {CLASS_NAME: geoms})`` with absolute-pixel geometry,
    matching docTR's ``DetectionDataset`` so its resize/normalisation transforms
    behave identically.

    Args:
        pool: detection word pool, e.g. from
            ``SyntheticDatasetGenerator(config).build_detection_pool()``.
        config: the generation config.
        num_samples: virtual epoch length.
        use_polygons: ``True`` keeps rotated quads ``(N, 4, 2)``; ``False`` emits
            straight boxes ``(N, 4)`` - matches the training script's
            ``--rotation`` flag.
        seed: ``None`` for fresh pages (training), an int for a fixed virtual set.
        words_per_page: candidate words per page (defaults to
            ``config.detection.max_words_per_page``).
    """

    def __init__(
        self,
        pool: list[str],
        config: GenerationConfig,
        num_samples: int,
        *,
        use_polygons: bool = False,
        img_transforms: Callable | None = None,
        sample_transforms: Callable | None = None,
        seed: int | None = None,
        words_per_page: int | None = None,
        max_attempts: int = 4,
    ) -> None:
        super().__init__(num_samples, img_transforms=img_transforms, sample_transforms=sample_transforms, seed=seed)
        if not pool:
            raise ValueError("detection word pool is empty")
        self.pool = list(pool)
        self.config = config
        self.use_polygons = use_polygons
        self.words_per_page = int(words_per_page or config.detection.max_words_per_page)
        self.max_attempts = max_attempts
        self._page_generator = PageGenerator(config)

    def _render(self, index: int) -> tuple[Image.Image, dict[str, np.ndarray]]:
        return render_detection_sample(
            self._page_generator,
            self.pool,
            self.words_per_page,
            self.use_polygons,
            max_attempts=self.max_attempts,
        )


# --------------------------------------------------------------------------- #
# Convenience factories: build matched (train, val) datasets in one call.
# --------------------------------------------------------------------------- #
def build_recognition_datasets(
    config: GenerationConfig,
    *,
    train_samples: int,
    val_samples: int,
    vocab: str | list[str] | None = None,
    img_transforms: Callable | None = None,
    sample_transforms: Callable | None = None,
    val_seed: int = 1234,
) -> tuple[SyntheticRecognitionDataset, SyntheticRecognitionDataset]:
    """Build ``(train, val)`` on-the-fly recognition datasets sharing one pipeline.

    The recognition pools (with vocab coverage) are built once. The train set
    draws fresh crops every access; the val set is a reproducible virtual set.
    ``config.core.num_images`` sets the *pool* size (variety); ``train_samples`` /
    ``val_samples`` set the per-epoch iteration counts.

    Pass ``vocab`` to pin the docTR vocab to train against - a ``VOCABS`` key, a
    literal charset, or a list such as ``["german", "urdu", "odia"]`` (their
    union). Every generated label is then guaranteed to fall within that
    character set, so a docTR recognition model trained on the matching
    ``--vocab`` never hits a label it cannot encode. When ``config.core.languages`` is
    empty the corpus languages are derived from the vocab keys (``german`` ->
    ``de``); keys with no corpus mapping are still covered via synthesis.
    """
    if vocab is not None:
        keys = [vocab] if isinstance(vocab, str) else list(vocab)
        derived = [VOCAB_TO_LANGUAGE[k] for k in keys if k in VOCAB_TO_LANGUAGE]
        unmapped = [k for k in keys if k not in VOCAB_TO_LANGUAGE]
        # Use the vocab-derived corpus languages unless the user picked their own
        # (i.e. set config.core.languages to something other than the "en" default).
        user_chose = bool(config.core.languages) and list(config.core.languages) != ["en"]  # type: ignore[arg-type]
        languages = list(config.core.languages) if user_chose else (derived or config.core.languages)  # type: ignore[arg-type]
        if unmapped and not user_chose:
            warnings.warn(
                f"No corpus language mapping for vocab key(s) {unmapped}; their characters "
                "will still be covered via synthesis, but pass an explicit `languages=[...]` "
                "on the config to also pull real words for them.",
                stacklevel=2,
            )
        config = replace(
            config,
            core=replace(config.core, languages=languages),
            coverage=replace(config.coverage, target_vocab=vocab, restrict_to_vocab=True),
        )

    pipeline = SyntheticDatasetGenerator(config)
    train_pool, val_pool = pipeline.build_recognition_pools()
    train_set = SyntheticRecognitionDataset(
        train_pool,
        config,
        train_samples,
        vocab=vocab,
        img_transforms=img_transforms,
        sample_transforms=sample_transforms,
        seed=None,
    )
    val_set = SyntheticRecognitionDataset(
        val_pool,
        config,
        val_samples,
        vocab=vocab,
        img_transforms=img_transforms,
        sample_transforms=sample_transforms,
        seed=val_seed,
    )
    return train_set, val_set


def build_detection_datasets(
    config: GenerationConfig,
    *,
    train_samples: int,
    val_samples: int,
    use_polygons: bool = False,
    img_transforms: Callable | None = None,
    sample_transforms: Callable | None = None,
    val_seed: int = 1234,
) -> tuple[SyntheticDetectionDataset, SyntheticDetectionDataset]:
    """Build ``(train, val)`` on-the-fly detection datasets sharing one pipeline.

    Backgrounds and the word pool are resolved once in the parent process; the
    train set draws fresh pages every access while the val set is reproducible.
    """
    pipeline = SyntheticDatasetGenerator(config)
    pool = pipeline.build_detection_pool()
    train_set = SyntheticDetectionDataset(
        pool,
        config,
        train_samples,
        use_polygons=use_polygons,
        img_transforms=img_transforms,
        sample_transforms=sample_transforms,
        seed=None,
    )
    val_set = SyntheticDetectionDataset(
        pool,
        config,
        val_samples,
        use_polygons=use_polygons,
        img_transforms=img_transforms,
        sample_transforms=sample_transforms,
        seed=val_seed,
    )
    return train_set, val_set
