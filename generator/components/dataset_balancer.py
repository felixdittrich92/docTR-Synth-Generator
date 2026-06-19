# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import random
from collections import Counter

__all__ = ["DatasetBalancer", "BalanceResult"]


class BalanceResult:
    """Container for the balanced train/val word lists plus a summary."""

    def __init__(self, train: list[str], val: list[str], summary: dict):
        self.train = train
        self.val = val
        self.summary = summary


class DatasetBalancer:
    """Allocates a vocabulary across languages and splits it train/val.

    Args:
        seed (int | None): RNG seed for reproducible datasets.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    # -- allocation -------------------------------------------------------

    def _quotas(
        self,
        languages: list[str],
        pools: dict[str, list[str]],
        budget: int,
        strategy: str,
        weights: dict[str, float] | None,
    ) -> dict[str, int]:
        """Split ``budget`` samples across ``languages`` per the chosen strategy."""
        if budget <= 0 or not languages:
            return dict.fromkeys(languages, 0)

        if weights:
            raw = {lang: max(0.0, float(weights.get(lang, 0.0))) for lang in languages}
            total = sum(raw.values()) or 1.0
            quotas = {lang: int(budget * raw[lang] / total) for lang in languages}
        elif strategy == "proportional":
            sizes = {lang: len(set(pools.get(lang, []))) for lang in languages}
            total = sum(sizes.values()) or 1
            quotas = {lang: int(budget * sizes[lang] / total) for lang in languages}
        else:  # "balanced" (default): as equal as possible
            base = budget // len(languages)
            quotas = dict.fromkeys(languages, base)

        # Distribute any rounding remainder deterministically over the languages.
        assigned = sum(quotas.values())
        i = 0
        while assigned < budget:
            quotas[languages[i % len(languages)]] += 1
            assigned += 1
            i += 1
        return quotas

    def _select(self, pool: list[str], quota: int) -> list[str]:
        """Pick ``quota`` words from a frequency-ordered pool.

        Unique words are sampled uniformly (good character/word coverage). If the
        quota exceeds the unique count, the surplus repeats with a Zipf-like
        weighting so frequent words recur, as in real text.
        """
        if quota <= 0:
            return []
        uniq = list(dict.fromkeys(pool))
        if not uniq:
            return []
        if quota <= len(uniq):
            return self.rng.sample(uniq, quota)
        out = list(uniq)
        weights = [1.0 / (i + 1) for i in range(len(uniq))]
        out.extend(self.rng.choices(uniq, weights=weights, k=quota - len(uniq)))
        return out

    def _split_bucket(self, items: list[str], val_percent: float) -> tuple[list[str], list[str]]:
        """Stratified split of one bucket; unique words go to val first (no leak)."""
        items = list(items)
        self.rng.shuffle(items)
        n_val = round(len(items) * val_percent)
        # Prefer putting words into val that don't also appear in the train part,
        # to avoid train/val leakage where the bucket has duplicates.
        val: list[str] = []
        train: list[str] = []
        seen_val: set[str] = set()
        for w in items:
            if len(val) < n_val and w not in seen_val:
                val.append(w)
                seen_val.add(w)
            else:
                train.append(w)
        return train, val

    # -- character coverage ----------------------------------------------

    def _enforce_char_coverage(
        self,
        selected: dict[str, list[str]],
        pools: dict[str, list[str]],
        min_count: int,
        max_extra_ratio: float,
    ) -> int:
        """Top up under-represented characters by adding words that contain them.

        Returns the number of extra words added. Bounded by ``max_extra_ratio`` of
        the current total so coverage enforcement can never blow up the dataset.
        """
        if min_count <= 0:
            return 0

        total = sum(len(v) for v in selected.values())
        budget_extra = int(total * max_extra_ratio)
        added = 0

        char_counts: Counter = Counter()
        for words in selected.values():
            for w in words:
                char_counts.update(w)

        # Index: char -> list of (language, word) that contain it.
        char_index: dict[str, list[tuple[str, str]]] = {}
        for lang, pool in pools.items():
            for w in dict.fromkeys(pool):
                for ch in set(w):
                    char_index.setdefault(ch, []).append((lang, w))

        for ch, idx in char_index.items():
            if added >= budget_extra:
                break
            while char_counts[ch] < min_count and idx and added < budget_extra:
                lang, w = idx[self.rng.randrange(len(idx))]
                selected[lang].append(w)
                char_counts.update(w)
                added += 1
        return added

    # -- public API -------------------------------------------------------

    def allocate_and_split(
        self,
        language_pools: dict[str, list[str]],
        num_images: int,
        val_percent: float = 0.2,
        strategy: str = "balanced",
        weights: dict[str, float] | None = None,
        numeric_tokens: list[str] | None = None,
        numeric_ratio: float = 0.0,
        min_char_coverage: int = 0,
        report: bool = True,
    ) -> BalanceResult:
        """Build balanced, stratified train/val word lists.

        Args:
            language_pools (dict[str, list[str]]): lang -> frequency-ordered words
                (casing variants should already be folded in per language).
            num_images (int): Total samples (hard cap).
            val_percent (float): Fraction of each bucket assigned to validation.
            strategy (str): ``"balanced"`` or ``"proportional"`` (ignored if
                ``weights`` is given).
            weights (dict[str, float] | None): Optional explicit per-language
                weights (need not sum to 1).
            numeric_tokens (list[str] | None): Pool of numeric/date/price tokens.
            numeric_ratio (float): Fraction of ``num_images`` filled with numerics.
            min_char_coverage (int): If > 0, ensure each character appears at least
                this many times (best-effort, bounded).
            report (bool): Print a balance summary.

        Returns:
            BalanceResult: train list, val list, and a summary dict.
        """
        languages = [lang for lang, pool in language_pools.items() if pool]
        numeric_tokens = numeric_tokens or []
        numeric_count = round(num_images * numeric_ratio) if (numeric_tokens and numeric_ratio > 0) else 0
        numeric_count = min(numeric_count, num_images)
        lang_budget = num_images - numeric_count

        quotas = self._quotas(languages, language_pools, lang_budget, strategy, weights)

        selected: dict[str, list[str]] = {}
        for lang in languages:
            selected[lang] = self._select(language_pools[lang], quotas[lang])
        if numeric_count:
            selected["_numeric"] = [self.rng.choice(numeric_tokens) for _ in range(numeric_count)]

        extra_added = self._enforce_char_coverage(selected, language_pools, min_char_coverage, max_extra_ratio=0.25)

        train: list[str] = []
        val: list[str] = []
        per_bucket: dict[str, tuple[int, int]] = {}
        for bucket, words in selected.items():
            tr, va = self._split_bucket(words, val_percent)
            train.extend(tr)
            val.extend(va)
            per_bucket[bucket] = (len(tr), len(va))

        self.rng.shuffle(train)
        self.rng.shuffle(val)

        summary = self._summarize(train, val, per_bucket, extra_added)
        if report:
            self._print_report(summary)
        return BalanceResult(train, val, summary)

    # -- reporting --------------------------------------------------------

    @staticmethod
    def _summarize(train, val, per_bucket, extra_added) -> dict:
        all_words = train + val
        char_counts: Counter = Counter()
        for w in all_words:
            char_counts.update(w)
        lengths = [len(w) for w in all_words] or [0]
        return {
            "total": len(all_words),
            "train": len(train),
            "val": len(val),
            "per_bucket": per_bucket,
            "unique_words": len(set(all_words)),
            "train_val_overlap": len(set(train) & set(val)),
            "distinct_chars": len(char_counts),
            "rare_chars": sum(1 for c in char_counts.values() if c < 5),
            "min_char_count": min(char_counts.values()) if char_counts else 0,
            "length_min": min(lengths),
            "length_mean": round(sum(lengths) / len(lengths), 2),
            "length_max": max(lengths),
            "coverage_extra_added": extra_added,
        }

    @staticmethod
    def _print_report(s: dict) -> None:
        print("=" * 56)
        print("Dataset balance report")
        print("-" * 56)
        print(f"  total: {s['total']}  (train {s['train']} / val {s['val']})")
        print("  per bucket (train/val):")
        for bucket, (tr, va) in s["per_bucket"].items():
            name = "numeric" if bucket == "_numeric" else bucket
            print(f"    {name:>10s}: {tr:>7d} / {va:<7d}")
        print(f"  unique words: {s['unique_words']}  | train/val word overlap: {s['train_val_overlap']}")
        print(
            f"  distinct chars: {s['distinct_chars']}  | chars seen <5x: {s['rare_chars']}  "
            f"| min char count: {s['min_char_count']}"
        )
        print(f"  word length: min {s['length_min']} / mean {s['length_mean']} / max {s['length_max']}")
        if s["coverage_extra_added"]:
            print(f"  char-coverage top-up added: {s['coverage_extra_added']} words")
        print("=" * 56)
