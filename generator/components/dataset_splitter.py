# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
import random

__all__ = ["DatasetSplitter"]


class DatasetSplitter:
    """Handles dataset splitting logic."""

    @staticmethod
    def load_vocabulary(wordlist_path: str) -> list[str]:
        """Load vocabulary from a wordlist file.

        Args:
            wordlist_path (str): Path to the wordlist file

        Returns:
            list[str]: List of words from the wordlist
        """
        with open(wordlist_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def prepare_splits(
        words: list[str],
        num_images: int,
        val_percent: float,
        ensure_coverage: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Prepare train/validation splits, honouring ``num_images`` as a hard cap.

        Behaviour:
          * ``num_images`` is always respected as the *total* number of samples.
          * If the vocabulary is larger than ``num_images`` a uniform random
            subset is drawn (full coverage is impossible by definition).
          * If the vocabulary is smaller, every unique word is included at least
            once (when ``ensure_coverage`` is set) and the remainder is filled by
            repeating words, so frequent words recur - as in real text.

        Args:
            words (list[str]): Source vocabulary (may contain duplicates).
            num_images (int): Total number of images to generate.
            val_percent (float): Fraction of images for validation.
            ensure_coverage (bool): Guarantee each unique word appears at least
                once across the combined dataset when it fits within num_images.

        Returns:
            tuple[list[str], list[str]]: Train and validation word lists.
        """
        # Unique vocabulary preserving (frequency) order.
        seen: set[str] = set()
        vocab = [w for w in words if not (w in seen or seen.add(w))]  # type: ignore[func-returns-value]

        if not vocab:
            return [], []

        num_val = math.ceil(num_images * val_percent)
        num_train = max(0, num_images - num_val)

        if num_images <= len(vocab):
            selected = random.sample(vocab, num_images)
        else:
            pool = list(vocab) if ensure_coverage else []
            while len(pool) < num_images:
                # Sample with replacement so frequent words naturally recur.
                pool.append(random.choice(vocab))
            random.shuffle(pool)
            selected = pool[:num_images]

        random.shuffle(selected)
        train_words = selected[:num_train]
        val_words = selected[num_train : num_train + num_val]
        return train_words, val_words
