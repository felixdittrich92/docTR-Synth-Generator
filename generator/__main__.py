# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from __future__ import annotations

import argparse

from .api import generate_dataset


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and generate a dataset."""
    parser = argparse.ArgumentParser(
        prog="doctr-synth",
        description="Generate a synthetic OCR dataset for docTR.",
    )
    parser.add_argument("output_dir", help="where to write the dataset (train/ and val/)")
    parser.add_argument("-n", "--num-images", type=int, default=1000, help="total samples (default: 1000)")
    parser.add_argument(
        "-t",
        "--task",
        choices=["recognition", "detection"],
        default="recognition",
        help="recognition crops or detection pages (default: recognition)",
    )
    parser.add_argument(
        "-l",
        "--languages",
        nargs="+",
        metavar="ISO",
        help="ISO 639-1 codes, e.g. -l en de ar (default: en)",
    )
    parser.add_argument(
        "--vocab",
        nargs="+",
        metavar="KEY",
        help="recognition: restrict labels to these VOCABS key(s), e.g. --vocab german urdu",
    )
    parser.add_argument(
        "--layout",
        metavar="NAME",
        help="detection layout: mixed | paragraph | newspaper | form | id_card",
    )
    parser.add_argument("-w", "--workers", type=int, help="worker processes")
    parser.add_argument("--val-percent", type=float, help="validation fraction (default: 0.2)")
    parser.add_argument("--jpeg", action="store_true", help="write JPEG instead of PNG")
    args = parser.parse_args(argv)

    overrides: dict = {}
    if args.layout:
        overrides["det_layout"] = args.layout
    if args.workers is not None:
        overrides["num_workers"] = args.workers
    if args.val_percent is not None:
        overrides["val_percent"] = args.val_percent
    if args.jpeg:
        overrides["output_jpeg"] = True
    vocab = args.vocab[0] if args.vocab and len(args.vocab) == 1 else args.vocab

    generate_dataset(
        args.output_dir,
        num_images=args.num_images,
        task=args.task,
        languages=args.languages,
        vocab=vocab,
        **overrides,
    )


if __name__ == "__main__":
    main()
