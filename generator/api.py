# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

from .components.config import GenerationConfig
from .dataset_generator import SyntheticDatasetGenerator

__all__ = ["generate_dataset"]


def generate_dataset(
    output_dir: str,
    num_images: int = 1000,
    *,
    task: str = "recognition",
    languages: list[str] | None = None,
    vocab: str | list[str] | None = None,
    **overrides,
) -> GenerationConfig:
    """Generate a synthetic OCR dataset on disk in a single call.

    Writes ``train/`` and ``val/`` folders (images + ``labels.json``) under
    ``output_dir``, downloading the words, fonts and backgrounds it needs.

    Args:
        output_dir: where the dataset is written.
        num_images: total samples, split into train/val by ``val_percent`` (0.2).
        task: ``"recognition"`` (word/line crops) or ``"detection"`` (full pages
            with per-word polygons).
        languages: ISO 639-1 codes, e.g. ``["en", "de", "ar"]`` (defaults to
            English). Words, fonts and complex-script shaping are resolved per code.
        vocab: recognition only - restrict every label to this docTR vocab so a
            model trained on the matching ``--vocab`` never sees an un-encodable
            character. A ``VOCABS`` key, a literal charset, or a list such as
            ``["german", "urdu", "odia"]`` (their union).
        **overrides: any other ``GenerationConfig`` field, e.g.
            ``det_layout="newspaper"``, ``num_workers=8``, ``output_jpeg=True``.

    Returns:
        The ``GenerationConfig`` that was used (handy for logging/repro).
    """
    params: dict = {"task": task, "output_dir": output_dir, "num_images": num_images}
    if languages is not None:
        params["languages"] = languages
    if vocab is not None:
        params["target_vocab"] = vocab
    params.update(overrides)
    config = GenerationConfig.flat(**params)
    SyntheticDatasetGenerator(config).generate_dataset()
    return config
