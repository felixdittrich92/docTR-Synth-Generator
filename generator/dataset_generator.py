# Copyright (C) 2021-2025, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import json
import multiprocessing as mp
import time
from pathlib import Path
from queue import Empty
from typing import Dict, List, Tuple

from .components import DatasetSplitter, GenerationConfig, GenerationTask, TextImageGenerator

__all__ = ["SyntheticDatasetGenerator", "GenerationConfig"]


class SyntheticDatasetGenerator:
    """Main orchestrator class for dataset generation

    Attributes:
        config (GenerationConfig): Configuration for dataset generation
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

    def generate_dataset(self):
        """Generate the complete dataset with queue-based multiprocessing"""
        print(f"Generating dataset with {self.config.num_workers} workers...")

        # Load vocabulary and prepare splits
        words = DatasetSplitter.load_vocabulary(self.config.wordlist_path)
        train_words, val_words = DatasetSplitter.prepare_splits(words, self.config.num_images, self.config.val_percent)

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
            GenerationTask(text=text, save_path=str(train_images_dir / f"{idx:05d}.png"), filename=f"{idx:05d}.png")
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
            GenerationTask(text=text, save_path=str(val_images_dir / f"{idx:05d}.png"), filename=f"{idx:05d}.png")
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
