# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import generator
import generator.__main__ as cli
import generator.dataset_generator as dg
from generator.api import generate_dataset


def test_generate_dataset_is_exported():
    assert generator.generate_dataset is generate_dataset


def test_generate_dataset_builds_config_and_runs(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        dg.SyntheticDatasetGenerator,
        "generate_dataset",
        lambda self: captured.setdefault("cfg", self.config),
    )
    cfg = generate_dataset(
        "/tmp/out",
        num_images=12,
        task="detection",
        languages=["en", "de"],
        vocab=["german"],
        det_layout="newspaper",
        num_workers=2,
    )
    assert cfg.core.task == "detection" and cfg.core.num_images == 12 and cfg.core.languages == ["en", "de"]
    assert cfg.core.output_dir == "/tmp/out"
    assert cfg.coverage.target_vocab == ["german"]  # vocab -> target_vocab
    assert cfg.detection.layout == "newspaper" and cfg.core.num_workers == 2  # passthrough overrides
    assert captured["cfg"] is cfg


def test_generate_dataset_defaults(monkeypatch):
    monkeypatch.setattr(dg.SyntheticDatasetGenerator, "generate_dataset", lambda self: None)
    cfg = generate_dataset("/tmp/out")
    assert cfg.core.task == "recognition" and cfg.core.num_images == 1000 and cfg.coverage.target_vocab is None


def test_cli_maps_arguments(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "generate_dataset", lambda *a, **k: seen.update(out=a[0], kw=k))
    cli.main([
        "/tmp/out",
        "-t",
        "detection",
        "-l",
        "en",
        "de",
        "--vocab",
        "german",
        "urdu",
        "--layout",
        "form",
        "-n",
        "7",
        "--jpeg",
        "-w",
        "3",
        "--val-percent",
        "0.1",
    ])
    assert seen["out"] == "/tmp/out"
    kw = seen["kw"]
    assert kw["task"] == "detection" and kw["languages"] == ["en", "de"]
    assert kw["vocab"] == ["german", "urdu"] and kw["num_images"] == 7
    assert kw["det_layout"] == "form" and kw["output_jpeg"] is True
    assert kw["num_workers"] == 3 and kw["val_percent"] == 0.1


def test_cli_single_vocab_collapses_to_str(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "generate_dataset", lambda *a, **k: seen.update(k))
    cli.main(["/tmp/out", "--vocab", "german"])
    assert seen["vocab"] == "german"  # a single key becomes a plain string
