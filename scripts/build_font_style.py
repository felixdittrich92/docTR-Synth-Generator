#!/usr/bin/env python3
# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

r"""Build a multilingual font style dataset: 300,000 word crops in 99 languages.

    python scripts/build_font_style.py --out font_style
    python scripts/build_font_style.py --out font_style --total 1000000 --languages european

285,000 train + 15,000 val crops (5%) for a font style classifier next to docTR's
orientation models (mindee/doctr#2025). Each crop is labelled with its font family
(the classes in ``meta/classes.json``, as in mixfont/lens), category, weight
(100-900), bold, italic, underline (none/single/double), strike-through, marker
highlight and ink colour - see :mod:`generator.recipes.font_style`.

Steps, all from :mod:`generator.recipes.font_style`:

1. ``prepare`` - word pools, vetted fonts, the family inventory and the crop plan
   (skipped when ``meta/`` already exists);
2. ``render`` - sharded and resumable: re-run the script after an interruption;
   ends with a report of the class balance and attribute shares;
3. ``preview`` - ``<out>/preview.jpg``, a sheet of crops with their labels.

Any other recipe option is passed through (``--languages``, ``--workers``,
``--offline``, ``--png``, ...; see ``python -m generator.recipes.font_style --help``).
"""

import argparse
import time
from pathlib import Path

from generator.recipes import font_style

TOTAL = 300_000
VAL_SHARE = 0.05
LANGUAGES = "all"  # 99 languages, CJK included


def _step(step: str, out: Path, extra: list[str]) -> None:
    print(f"\n=== {step} ===", flush=True)
    font_style.main([step, "--out", str(out), *extra])


def main() -> None:
    """Command line entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="font_style", help="dataset directory")
    p.add_argument("--total", type=int, default=TOTAL, help="train + val crops")
    p.add_argument("--val-share", type=float, default=VAL_SHARE, help="fraction of --total used for validation")
    p.add_argument("--force-prepare", action="store_true", help="re-run prepare even if meta/ exists")
    args, extra = p.parse_known_args()

    out = Path(args.out)
    num_val = round(args.total * args.val_share)
    sizes = ["--languages", LANGUAGES, "--num-train", str(args.total - num_val), "--num-val", str(num_val)]
    langs = extra[extra.index("--languages") + 1] if "--languages" in extra else f"{LANGUAGES} languages"
    print(f"{args.total - num_val:,} train + {num_val:,} val crops ({langs}) -> {out}")

    start = time.time()
    if args.force_prepare or not (out / "meta" / "train_plan.tsv").exists():
        _step("prepare", out, sizes + extra)
    else:
        print("meta/ exists - skipping prepare (pass --force-prepare to redo it)")
    _step("render", out, sizes + extra)
    _step("preview", out, extra)
    print(f"\nDone in {(time.time() - start) / 3600:.2f} h: {out}/train (shards), {out}/val, classes in {out}/meta")


if __name__ == "__main__":
    main()
