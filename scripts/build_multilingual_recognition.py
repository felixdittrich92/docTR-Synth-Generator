#!/usr/bin/env python3
# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

r"""Build a multilingual recognition dataset: 1.2 million word crops, 96 languages.

    python scripts/build_multilingual_recognition.py --out multilingual_rec
    python scripts/build_multilingual_recognition.py --out multilingual_rec --workers 32 --min-char-count 500

Every vocab language except Chinese, Japanese and Korean (the ``multilingual``
preset): Latin, Cyrillic, Greek, Arabic, Hebrew, Indic, South-East Asian,
Georgian, Armenian, Ethiopic ... 1,140,000 train + 60,000 val crops (5%).

Steps, all from :mod:`generator.recipes.recognition`:

1. ``prepare`` - word pools, vetted fonts, the model vocab and labels that put every
   vocab character into train *and* val (skipped when ``meta/`` already exists);
2. ``render`` - sharded and resumable: re-run the script after an interruption;
3. ``verify`` - every vocab character in the rendered labels of both splits;
   ``topup`` renders exactly what is missing, then ``verify`` runs again.

Any other recipe option is passed through (``--min-char-count``, ``--offline``,
``--handwriting-share``, ``--png``, ...; see
``python -m generator.recipes.recognition --help``). Train with docTR:

    python references/recognition/train.py crnn_vgg16_bn \\
        --train_path multilingual_rec/train --val_path multilingual_rec/val --vocab <name>

after adding ``multilingual_rec/meta/vocab.txt`` to docTR's ``VOCABS`` under ``<name>``.
"""

import argparse
import sys
import time
from pathlib import Path

from generator.recipes import recognition
from generator.recipes.languages import PRESETS

TOTAL = 1_200_000
VAL_SHARE = 0.05
LANGUAGES = "multilingual"  # PRESETS["all"] without zh, ja, ko


def _step(step: str, out: Path, extra: list[str]) -> bool:
    """Run one recipe step in-process; ``False`` when it exits with an error (failed verify)."""
    print(f"\n=== {step} ===", flush=True)
    try:
        recognition.main([step, "--out", str(out), *extra])
    except SystemExit as e:
        return e.code in (0, None)
    return True


def main() -> None:
    """Command line entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="multilingual_rec", help="dataset directory")
    p.add_argument("--total", type=int, default=TOTAL, help="train + val crops")
    p.add_argument("--val-share", type=float, default=VAL_SHARE, help="fraction of --total used for validation")
    p.add_argument("--force-prepare", action="store_true", help="re-run prepare even if meta/ exists")
    args, extra = p.parse_known_args()

    excluded = set(PRESETS["all"]) - set(PRESETS[LANGUAGES])
    assert excluded == {"zh", "ja", "ko"}, f"the {LANGUAGES!r} preset changed: it now leaves out {excluded}"
    out = Path(args.out)
    num_val = round(args.total * args.val_share)
    sizes = ["--languages", LANGUAGES, "--num-train", str(args.total - num_val), "--num-val", str(num_val)]
    langs = extra[extra.index("--languages") + 1] if "--languages" in extra else None  # an override wins
    langs = langs or f"{len(PRESETS[LANGUAGES])} languages (no {', '.join(sorted(excluded))})"
    print(f"{langs} | {args.total - num_val:,} train + {num_val:,} val crops -> {out}")

    start = time.time()
    if args.force_prepare or not (out / "meta" / "train_labels.tsv").exists():
        _step("prepare", out, sizes + extra)
    else:
        print("meta/ exists - skipping prepare (pass --force-prepare to redo it)")
    _step("render", out, sizes + extra)
    if not _step("verify", out, extra) and not _step("topup", out, extra):
        sys.exit("vocab coverage incomplete after topup - see the verify report above")
    print(f"\nDone in {(time.time() - start) / 3600:.2f} h: {out}/train (shards) and {out}/val, vocab in {out}/meta")


if __name__ == "__main__":
    main()
