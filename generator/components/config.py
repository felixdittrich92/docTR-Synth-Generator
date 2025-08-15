# Copyright (C) 2021-2025, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from dataclasses import dataclass

__all__ = ["GenerationConfig"]


@dataclass
class GenerationConfig:
    """Configuration for dataset generation

    Attributes:
        wordlist_path (str): Path to the wordlist file
        font_dir (str): Directory containing TTF/OTF font files
        output_dir (str): Directory to save generated images
        num_images (int): Total number of images to generate
        bg_image_dir (str): Directory containing background images
        val_percent (float): Percentage of images for validation set
        num_workers (int): Number of worker threads for parallel processing
        font_size_range (tuple[int, int]): Range of font sizes to use
        padding (int): Padding around text in the image
        max_attempts (int): Maximum attempts to find a suitable background crop
        queue_maxsize (int): Limit on the size of the processing queue
        # Augmentation probabilities and parameters
        bold_prob (float): Probability of applying bold effect
        rotation_prob (float): Probability of applying rotation augmentation
        blur_prob (float): Probability of applying blur augmentation
        perspective_prob (float): Probability of applying perspective distortion
        rotation_range (tuple[float, float]): Range of angles for rotation
        blur_radius_range (tuple[float, float]): Range of blur radius
        perspective_margin (int): Margin for perspective distortion
    """

    wordlist_path: str
    font_dir: str
    output_dir: str
    num_images: int
    bg_image_dir: str | None = None
    val_percent: float = 0.2
    num_workers: int = 4
    font_size_range: tuple[int, int] = (18, 35)
    padding: int = 4
    max_attempts: int = 5
    queue_maxsize: int = 100  # Limit queue size to control memory usage
    # Augmentation probabilities
    bold_prob: float = 0.5
    rotation_prob: float = 0.6
    blur_prob: float = 0.3
    perspective_prob: float = 0.5
    # Augmentation parameters
    rotation_range: tuple[float, float] = (-2, 2)
    blur_radius_range: tuple[float, float] = (0.3, 1.0)
    perspective_margin: int = 2
