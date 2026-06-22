# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import json
import math
import multiprocessing as mp
import random
import threading
import time
from pathlib import Path
from queue import Empty
from typing import Dict, List, Tuple

from .components import (
    BackgroundDownloader,
    CorpusDownloader,
    DatasetBalancer,
    DatasetSplitter,
    DetectionTask,
    GenerationConfig,
    GenerationTask,
    PageGenerator,
    TextImageGenerator,
    apply_casing_variants,
    generate_numeric_tokens,
)
from .components.vocab_coverage import augment_words_for_coverage, resolve_target_vocab

__all__ = ["SyntheticDatasetGenerator", "GenerationConfig"]


class SyntheticDatasetGenerator:
    """Main orchestrator class for dataset generation

    Attributes:
        config (GenerationConfig): Configuration for dataset generation
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

    def _resolve_background_dir(self) -> None:
        """Resolve the effective background directory once (main process).

        Precedence: an explicit ``bg_image_dir`` wins (backward compatible).
        Otherwise, when ``auto_download_backgrounds`` is set, a curated set is
        downloaded + cached and that directory is used. Resolving here - before
        workers spawn - means the download happens once rather than per worker.
        """
        cfg = self.config
        if cfg.bg_image_dir:
            return
        if not cfg.auto_download_backgrounds:
            return
        cache_dir = cfg.background_cache_dir or ".background_cache"
        print("No bg_image_dir given - downloading curated background images...")
        downloader = BackgroundDownloader(
            cache_dir=cache_dir,
            manifest_url=cfg.background_manifest_url,
            timeout=cfg.font_download_timeout,
            enabled=True,
        )
        paths = downloader.download_all()
        if paths:
            cfg.bg_image_dir = cache_dir
            print(f"Using {len(paths)} downloaded backgrounds from {cache_dir}")
        else:
            print("No backgrounds available - falling back to blank/generated backgrounds.")

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
            train, val = DatasetSplitter.prepare_splits(words, cfg.num_images, cfg.val_percent, ensure_coverage=True)
            target = self._coverage_target([])  # only an explicit target_vocab applies here
            if target:
                train, val = self._inject_coverage(train, val, target)
            return train, val

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
        train, val = result.train, result.val
        # Hard per-split coverage guarantee: append synthesised tokens so train
        # and val each contain every renderable vocab character. Done after the
        # split (not via the balancer's stratified top-up) so a rare character
        # can never end up in only one split.
        target = self._coverage_target(languages)
        if target:
            train, val = self._inject_coverage(train, val, target)
        return train, val

    def build_recognition_pools(self) -> tuple[list[str], list[str]]:
        """Build the ``(train, val)`` recognition word pools (with vocab coverage).

        Public entry point for the on-the-fly datasets in
        :mod:`generator.doctr_dataset`. These are the very pools the offline
        recognition generator samples from; sampling from them at train time
        reproduces the same balance and per-split character coverage without
        ever touching disk. ``config.num_images`` controls the pool size.
        """
        return self._prepare_train_val()

    def build_detection_pool(self) -> list[str]:
        """Resolve backgrounds (once) and return the detection word pool.

        Call this in the parent process before constructing an on-the-fly
        detection dataset so background images are downloaded/resolved a single
        time rather than inside every DataLoader worker.
        """
        self._resolve_background_dir()
        return self._resolve_word_pool()

    def _coverage_target(self, languages: list[str]) -> set[str] | None:
        """Union of vocab characters to guarantee, or None when disabled/unknown."""
        cfg = self.config
        if not cfg.ensure_vocab_coverage:
            return None
        chars: set[str] = set()
        if cfg.target_vocab:
            explicit = resolve_target_vocab(None, cfg.target_vocab)
            if explicit:
                chars |= explicit
        else:
            for lang in languages:
                resolved = resolve_target_vocab(lang, None)
                if resolved:
                    chars |= resolved
        return chars or None

    def _inject_coverage(self, train: list[str], val: list[str], target: set[str]) -> tuple[list[str], list[str]]:
        """Ensure both splits cover ``target`` by appending synthesised tokens.

        Characters that no available font can render (e.g. ``฿`` for a Latin-script
        language) will still be skipped at render time - that is a font limitation,
        not a logic gap.
        """
        cfg = self.config
        seed = cfg.corpus_seed
        train, n_train, miss_train = augment_words_for_coverage(
            train, target, min_count=cfg.vocab_coverage_min_count, seed=seed
        )
        # Val only needs presence (min_count 1) so it stays small.
        val, n_val, _ = augment_words_for_coverage(
            val, target, min_count=1, seed=(seed + 1 if seed is not None else None)
        )
        if n_train or n_val:
            print(
                f"Vocab coverage: {miss_train} char(s) under-represented in train -> "
                f"added {n_train} train / {n_val} val coverage tokens."
            )
        return train, val

    def generate_dataset(self):
        """Generate the complete dataset with queue-based multiprocessing"""
        if getattr(self.config, "task", "recognition") == "detection":
            return self._generate_detection_dataset()

        print(f"Generating dataset with {self.config.num_workers} workers...")
        self._resolve_background_dir()

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

    # ------------------------------------------------------------------
    # Detection dataset generation
    # ------------------------------------------------------------------

    def _resolve_word_pool(self) -> List[str]:
        """Build a shuffled pool of words to populate detection pages with."""
        cfg = self.config
        if cfg.wordlist_path is not None:
            words = DatasetSplitter.load_vocabulary(cfg.wordlist_path)
        else:
            languages = cfg.languages or ["en"]
            print(f"No wordlist given - downloading real words for languages: {languages}")
            downloader = CorpusDownloader(
                cache_dir=cfg.corpus_cache_dir or ".corpus_cache",
                timeout=cfg.font_download_timeout,
                min_word_length=cfg.min_word_length,
                max_word_length=cfg.max_word_length,
                filter_by_script=cfg.corpus_filter_by_script,
            )
            words = downloader.build_vocabulary(languages, words_per_language=cfg.words_per_language)
            if cfg.casing_variant_prob > 0:
                words = apply_casing_variants(words, prob=cfg.casing_variant_prob, seed=cfg.corpus_seed)
            if cfg.numeric_token_ratio > 0:
                words = words + generate_numeric_tokens(int(len(words) * cfg.numeric_token_ratio), seed=cfg.corpus_seed)
        if not words:
            raise ValueError("No words available for detection page generation. Check connectivity or wordlist.")
        random.Random(cfg.corpus_seed).shuffle(words)
        return words

    def _generate_detection_dataset(self):
        """Generate document-like pages with per-word polygons (docTR format)."""
        cfg = self.config
        print(f"Generating DETECTION dataset with {cfg.num_workers} workers...")
        self._resolve_background_dir()
        ext = "jpg" if cfg.output_jpeg else "png"

        pool = self._resolve_word_pool()
        rng = random.Random(cfg.corpus_seed)

        num_val = math.ceil(cfg.num_images * cfg.val_percent)
        num_train = max(0, cfg.num_images - num_val)
        print(f"Generating {num_train} training pages and {num_val} validation pages.")

        train_dir = Path(cfg.output_dir) / "train"
        val_dir = Path(cfg.output_dir) / "val"
        train_images_dir = train_dir / "images"
        val_images_dir = val_dir / "images"
        for d in (train_images_dir, val_images_dir):
            d.mkdir(parents=True, exist_ok=True)
        print(f"Output directories created: {train_dir}, {val_dir}")

        def make_tasks(count: int, images_dir: Path, split: str):
            # A generator so the (large) per-page candidate word lists are created
            # just-in-time as they are fed to the queue, instead of all at once.
            for idx in range(count):
                page_words = [rng.choice(pool) for _ in range(cfg.det_max_words_per_page)]
                yield DetectionTask(
                    words=page_words,
                    save_path=str(images_dir / f"{idx:05d}.{ext}"),
                    filename=f"{idx:05d}.{ext}",
                    split=split,
                )

        print("Generating training pages...")
        train_success, train_labels = self._generate_pages_with_queue(
            make_tasks(num_train, train_images_dir, "train"), num_train
        )
        with open(train_dir / "labels.json", "w", encoding="utf-8") as f:
            json.dump(train_labels, f, ensure_ascii=False, indent=2)
        print(f"Training labels saved to {train_dir / 'labels.json'}")

        print("Generating validation pages...")
        val_success, val_labels = self._generate_pages_with_queue(make_tasks(num_val, val_images_dir, "val"), num_val)
        with open(val_dir / "labels.json", "w", encoding="utf-8") as f:
            json.dump(val_labels, f, ensure_ascii=False, indent=2)
        print(f"Validation labels saved to {val_dir / 'labels.json'}")

        print("Detection dataset generation completed!")
        print(f"Training: {train_success}/{num_train} pages generated successfully")
        print(f"Validation: {val_success}/{num_val} pages generated successfully")

    def _generate_pages_with_queue(self, task_iter, total: int) -> Tuple[int, Dict[str, dict]]:
        """Render detection pages with multiprocessing and collect polygon labels.

        Tasks are fed from a background thread while results are collected
        concurrently. Detection results are large (hundreds of polygons each), so
        feeding everything up front before collecting could fill the result pipe
        and dead-lock the workers; the feeder thread avoids that.
        """
        task_queue: mp.Queue = mp.Queue(maxsize=self.config.queue_maxsize)
        result_queue: mp.Queue = mp.Queue()

        workers = []
        for worker_id in range(self.config.num_workers):
            worker = mp.Process(
                target=PageGenerator.detection_worker_process,
                args=(task_queue, result_queue, self.config, worker_id),
            )
            worker.start()
            workers.append(worker)

        def feed():
            for task in task_iter:
                task_queue.put(task)
            for _ in range(self.config.num_workers):
                task_queue.put(None)  # poison pills

        feeder = threading.Thread(target=feed, daemon=True)
        feeder.start()

        completed = 0
        successful = 0
        labels: Dict[str, dict] = {}
        start_time = time.time()

        while completed < total:
            try:
                filename, _split, dims, img_hash, polygons, success = result_queue.get(timeout=60)
                completed += 1
                if success:
                    successful += 1
                    labels[filename] = {
                        "img_dimensions": dims,
                        "img_hash": img_hash,
                        "polygons": polygons,
                    }
                if completed % 25 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    print(
                        f"Progress: {completed}/{total} ({successful} successful) "
                        f"Rate: {rate:.1f} pages/s ETA: {eta:.1f}s"
                    )
            except Empty:
                print("Waiting for results...")
                continue

        feeder.join()
        for worker in workers:
            worker.join()

        return successful, labels
