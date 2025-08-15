# Copyright (C) 2021-2025, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
import random

__all__ = ["DatasetSplitter"]


class DatasetSplitter:
    """Handles dataset splitting logic"""

    @staticmethod
    def load_vocabulary(wordlist_path: str) -> list[str]:
        """Load vocabulary from wordlist file

        Args:
            wordlist_path (str): Path to the wordlist file

        Returns:
            list[str]: List of words from the wordlist
        """
        with open(wordlist_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def prepare_splits(words: list[str], num_images: int, val_percent: float) -> tuple[list[str], list[str]]:
        """Prepare train/validation splits ensuring vocabulary coverage

        Args:
            words (list[str]): List of words from the wordlist
            num_images (int): Total number of images to generate
            val_percent (float): Percentage of images for validation set

        Returns:
            tuple[list[str], list[str]]: Train and validation word lists
        """
        vocab = set(words)

        if num_images < len(vocab):
            print(
                f"Warning: num_images ({num_images}) is less than vocabulary size ({len(vocab)}). "
                "Some words won't be included."
            )

        # Calculate split sizes
        num_val = math.ceil(num_images * val_percent)
        num_train = num_images - num_val

        # Extend wordlist if needed
        extended_wordlist = words.copy()
        while len(extended_wordlist) < num_images:
            extended_wordlist.extend(words)
        extended_wordlist = extended_wordlist[:num_images]

        # Shuffle for randomization
        random.shuffle(extended_wordlist)

        # Split into train and val
        train_words = extended_wordlist[:num_train]
        val_words = extended_wordlist[num_train : num_train + num_val]

        # Ensure vocabulary coverage in both splits
        def ensure_vocab_coverage(word_subset):
            subset_vocab = set(word_subset)
            missing = vocab - subset_vocab
            if missing:
                print(f"Adding {len(missing)} missing vocab words to subset for coverage.")
                word_subset.extend(missing)
            return word_subset

        train_words = ensure_vocab_coverage(train_words)
        val_words = ensure_vocab_coverage(val_words)

        # Remove duplicates while preserving order
        def unique_preserve_order(seq):
            seen = set()
            return [x for x in seq if x not in seen]

        train_words = unique_preserve_order(train_words)
        val_words = unique_preserve_order(val_words)

        return train_words, val_words
