# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Turn a flat word pool into tokens with realistic line statistics.

Pages used to be filled with independent uniform draws from the vocabulary,
which produces word salad: every token is equally likely, so lines have none of
the run-length structure of real text and no punctuation at all. Two things a
detector keys on are distorted by that.

* **Run length.** Real lines alternate short function words with longer content
  words. Uniform draws from a frequency-filtered vocabulary skew long, giving
  every line a near-constant token width.
* **Box shape.** Attached punctuation changes the extent of a box - a comma
  hangs below the baseline, quotes sit above the x-height, a closing bracket
  adds a narrow tail.

Real n-grams are not available: the corpora behind :class:`CorpusDownloader` are
frequency lists, not sentences. So this reproduces the *statistics* rather than
real language - short-word clustering and punctuation - which is what changes
the geometry the detector sees.
"""

import random

__all__ = ["TokenSampler"]

# Attached to a token, with the weight of each shape. Chosen for their effect on
# the bounding box: descenders, ascenders and narrow tails.
_TRAILING = [(",", 5), (".", 5), (";", 1), (":", 2), ("!", 1), ("?", 1), (")", 2), ('"', 2), ("'", 1)]
_LEADING = [("(", 2), ('"', 2), ("'", 1), ("-", 1)]


class TokenSampler:
    """Draw page tokens from a word pool with line-like statistics.

    Args:
        words (list[str]): The candidate vocabulary.
        function_word_ratio (float): Share of tokens drawn from the short-word
            bucket. 0 restores uniform sampling.
        function_word_max_len (int): Length bound of that bucket.
        punctuation_prob (float): Probability a token carries punctuation.
    """

    def __init__(
        self,
        words: list[str],
        function_word_ratio: float = 0.45,
        function_word_max_len: int = 4,
        punctuation_prob: float = 0.18,
    ):
        self.words = list(words) or [""]
        self.function_word_ratio = function_word_ratio
        self.punctuation_prob = punctuation_prob
        # Length is the usable proxy for "function word" here: the pool arrives
        # shuffled, so frequency rank is already gone, and short tokens are
        # overwhelmingly articles, prepositions and conjunctions.
        self.short = [w for w in self.words if len(w) <= function_word_max_len]
        self.long = [w for w in self.words if len(w) > function_word_max_len]
        if not self.short:
            self.short = self.words
        if not self.long:
            self.long = self.words

    def _base_word(self) -> str:
        if random.random() < self.function_word_ratio:
            return random.choice(self.short)
        return random.choice(self.long)

    def punctuate(self, word: str) -> str:
        """Attach leading/trailing punctuation to ``word`` with configured odds."""
        if not word or random.random() >= self.punctuation_prob:
            return word
        if random.random() < 0.25:
            marks, weights = zip(*_LEADING)
            word = random.choices(marks, weights=weights, k=1)[0] + word
        else:
            marks, weights = zip(*_TRAILING)
            word += random.choices(marks, weights=weights, k=1)[0]
        return word

    def take(self) -> str:
        """Return the next token: a word from a length-aware bucket, punctuated."""
        return self.punctuate(self._base_word())
