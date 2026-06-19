# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import json
import multiprocessing as mp
import time
from pathlib import Path
from queue import Empty
from typing import Dict, List, Tuple

from .components import (
    CorpusDownloader,
    DatasetBalancer,
    DatasetSplitter,
    GenerationConfig,
    GenerationTask,
    TextImageGenerator,
    apply_casing_variants,
    generate_numeric_tokens,
)

__all__ = ["SyntheticDatasetGenerator", "GenerationConfig"]


class SyntheticDatasetGenerator:
    """Main orchestrator class for dataset generation

    Attributes:
        config (GenerationConfig): Configuration for dataset generation
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

    def _prepare_train_val(self) -> tuple[list[str], list[str]]:
        """Resolve the text source and return balanced (train, val) word lists.

        Precedence: an explicit ``wordlist_path`` wins and uses the simple
        splitter (no language strata). Otherwise real per-language corpora are
        downloaded and handed to the :class:`DatasetBalancer` for controlled
        language balancing and a stratified split.
        """
        cfg = self.config

        if cfg.wordlist_path is not None:
            words = DatasetSplitter.load_vocabulary(cfg.wordlist_path)
            print(f"Loaded {len(words)} words from wordlist '{cfg.wordlist_path}'.")
            return DatasetSplitter.prepare_splits(words, cfg.num_images, cfg.val_percent, ensure_coverage=True)

        languages = cfg.languages or ["en"]
        cache_dir = cfg.corpus_cache_dir or ".corpus_cache"
        print(f"No wordlist given - downloading real words for languages: {languages}")
        downloader = CorpusDownloader(
            cache_dir=cache_dir,
            timeout=cfg.font_download_timeout,
            min_word_length=cfg.min_word_length,
            max_word_length=cfg.max_word_length,
            filter_by_script=cfg.corpus_filter_by_script,
        )

        # Per-language pools (casing variants folded in per language so they are
        # balanced alongside their own language rather than globally).
        language_pools: dict[str, list[str]] = {}
        for lang in languages:
            pool = downloader.fetch(lang)[: cfg.words_per_language]
            if not pool:
                print(f"  warning: no words downloaded for '{lang}', skipping it.")
                continue
            if cfg.casing_variant_prob > 0:
                pool = apply_casing_variants(pool, prob=cfg.casing_variant_prob, seed=cfg.corpus_seed)
            language_pools[lang] = pool

        if not language_pools:
            raise ValueError(
                f"Could not download any words for languages {languages}. "
                "Check connectivity or provide a wordlist_path."
            )

        numeric_tokens = None
        if cfg.numeric_token_ratio > 0:
            n_pool = max(256, int(cfg.num_images * cfg.numeric_token_ratio) * 2)
            numeric_tokens = generate_numeric_tokens(n_pool, seed=cfg.corpus_seed)

        balancer = DatasetBalancer(seed=cfg.corpus_seed)
        result = balancer.allocate_and_split(
            language_pools=language_pools,
            num_images=cfg.num_images,
            val_percent=cfg.val_percent,
            strategy=cfg.language_balance,
            weights=cfg.language_weights,
            numeric_tokens=numeric_tokens,
            numeric_ratio=cfg.numeric_token_ratio,
            min_char_coverage=cfg.min_char_coverage,
            report=cfg.print_balance_report,
        )
        return result.train, result.val

    def generate_dataset(self):
        """Generate the complete dataset with queue-based multiprocessing"""
        print(f"Generating dataset with {self.config.num_workers} workers...")

        ext = "jpg" if getattr(self.config, "output_jpeg", False) else "png"

        # Resolve the text source and build balanced, stratified splits.
        train_words, val_words = self._prepare_train_val()

        print(f"Generating {len(train_words)} training images and {len(val_words)} validation images.")

        # Create output directories
        train_dir = Path(self.config.output_dir) / "train"
        val_dir = Path(self.config.output_dir) / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        train_images_dir = train_dir / "images"
        val_images_dir = val_dir / "images"
        train_images_dir.mkdir(parents=True, exist_ok=True)
        val_images_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directories created: {train_dir}, {val_dir}")

        # Generate training images
        print("Generating training images...")
        train_tasks = [
            GenerationTask(text=text, save_path=str(train_images_dir / f"{idx:05d}.{ext}"), filename=f"{idx:05d}.{ext}")
            for idx, text in enumerate(train_words)
        ]
        train_success, train_labels = self._generate_images_with_queue(train_tasks)

        # Save training labels
        train_labels_path = train_dir / "labels.json"
        with open(train_labels_path, "w", encoding="utf-8") as f:
            json.dump(train_labels, f, ensure_ascii=False, indent=2)
        print(f"Training labels saved to {train_labels_path}")

        # Generate validation images
        print("Generating validation images...")
        val_tasks = [
            GenerationTask(text=text, save_path=str(val_images_dir / f"{idx:05d}.{ext}"), filename=f"{idx:05d}.{ext}")
            for idx, text in enumerate(val_words)
        ]
        val_success, val_labels = self._generate_images_with_queue(val_tasks)

        # Save validation labels
        val_labels_path = val_dir / "labels.json"
        with open(val_labels_path, "w", encoding="utf-8") as f:
            json.dump(val_labels, f, ensure_ascii=False, indent=2)
        print(f"Validation labels saved to {val_labels_path}")

        print("Dataset generation completed!")
        print(f"Training: {train_success}/{len(train_tasks)} images generated successfully")
        print(f"Validation: {val_success}/{len(val_tasks)} images generated successfully")

    def _generate_images_with_queue(self, tasks: List[GenerationTask]) -> Tuple[int, Dict[str, str]]:
        """Generate images using queue-based multiprocessing"""
        # Create queues
        task_queue: mp.Queue = mp.Queue(maxsize=self.config.queue_maxsize)
        result_queue: mp.Queue = mp.Queue()

        # Start worker processes
        workers = []
        for worker_id in range(self.config.num_workers):
            worker = mp.Process(
                target=TextImageGenerator.worker_process, args=(task_queue, result_queue, self.config, worker_id)
            )
            worker.start()
            workers.append(worker)

        # Add tasks to queue
        for task in tasks:
            task_queue.put(task)

        # Monitor progress and collect labels
        completed = 0
        successful = 0
        labels = {}
        start_time = time.time()

        while completed < len(tasks):
            try:
                text, filename, success = result_queue.get(timeout=10)
                completed += 1
                if success:
                    successful += 1
                    labels[filename] = text

                if completed % 50 == 0 or completed == len(tasks):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(tasks) - completed) / rate if rate > 0 else 0
                    print(
                        f"Progress: {completed}/{len(tasks)} ({successful} successful) "
                        f"Rate: {rate:.1f} img/s ETA: {eta:.1f}s"
                    )

            except Empty:
                print("Waiting for results...")
                continue

        # Send poison pills to stop workers
        for _ in range(self.config.num_workers):
            task_queue.put(None)

        # Wait for all workers to finish
        for worker in workers:
            worker.join()

        return successful, labels
