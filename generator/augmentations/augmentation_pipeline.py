# Copyright (C) 2021-2025, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Callable

from PIL import Image

__all__ = ["AugmentationPipeline"]


class AugmentationPipeline:
    """A class to handle the augmentation pipeline for text image generation.

    Args:
        augmentations (list[Callable]): List of augmentation callables to apply.
    """

    def __init__(self, augmentations: list[Callable]):
        self.augmentations = augmentations

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply all augmentations to the given image."""
        for augmentation in self.augmentations:
            image = augmentation(image)
        return image

    def __repr__(self):
        return f"AugmentationPipeline(augmentations={self.augmentations})"
