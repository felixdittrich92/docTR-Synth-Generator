#!/usr/bin/env python3
# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

r"""Build a multilingual detection dataset: ~120,000 pages in 99 languages.

    python scripts/build_multilingual_detection.py --out multilingual_det
    python scripts/build_multilingual_detection.py --out multilingual_det --workers 32 --input-size 1536

114,000 train + 6,000 val pages (5%), every vocab language including Chinese,
Japanese (vertical pages) and Korean. The recipe defaults give the mix of real
collections: letters, reports, forms, tables, receipts, ID cards, newspapers and
magazine pages; scanned (colour, grayscale, black-and-white) or photographed on
tables and tablecloths; 55/30/15% low/medium/high text density; 5% pages
without any text (hard negatives).

Steps, all from :mod:`generator.recipes.detection`:

1. ``prepare`` - word pools and vetted fonts per language (skipped when ``meta/``
   already exists);
2. ``render`` - in parts, resumable: re-run the script after an interruption.

Any other recipe option is passed through (``--input-size``, ``--density``,
``--negative-share``, ``--offline``, ``--png``, ...; see
``python -m generator.recipes.detection --help``). Train with docTR, with
``--rotation`` (photographed pages carry rotated polygons) and the same input size:

    python references/detection/train.py db_resnet50 --rotation \\
        --train_path multilingual_det/train --val_path multilingual_det/val

Expect several hours with 24 workers (~7 pages/s) and roughly 40-60 GB of JPEGs.
"""

import argparse
import time
from pathlib import Path

from generator.recipes import detection

TOTAL = 120_000
VAL_SHARE = 0.05
LANGUAGES = "all"  # 99 languages, CJK included


def _step(step: str, out: Path, extra: list[str]) -> None:
    print(f"\n=== {step} ===", flush=True)
    detection.main([step, "--out", str(out), *extra])


def main() -> None:
    """Command line entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="multilingual_det", help="dataset directory")
    p.add_argument("--total", type=int, default=TOTAL, help="train + val pages")
    p.add_argument("--val-share", type=float, default=VAL_SHARE, help="fraction of --total used for validation")
    p.add_argument("--force-prepare", action="store_true", help="re-run prepare even if meta/ exists")
    args, extra = p.parse_known_args()

    out = Path(args.out)
    num_val = round(args.total * args.val_share)
    sizes = ["--languages", LANGUAGES, "--num-train", str(args.total - num_val), "--num-val", str(num_val)]
    langs = extra[extra.index("--languages") + 1] if "--languages" in extra else f"{LANGUAGES} languages"
    print(f"{args.total - num_val:,} train + {num_val:,} val pages ({langs}) -> {out}")

    start = time.time()
    if args.force_prepare or not (out / "meta" / "languages.json").exists():
        _step("prepare", out, sizes + extra)
    else:
        print("meta/ exists - skipping prepare (pass --force-prepare to redo it)")
    _step("render", out, sizes + extra)
    print(f"\nDone in {(time.time() - start) / 3600:.2f} h: {out}/train and {out}/val (images/ + labels.json)")


if __name__ == "__main__":
    main()
