"""Tests for scripts/make_variant_config.py (planned extras, DECISIONS.md D34)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_variant_config.py"
BASE = Path(__file__).resolve().parents[1] / "configs" / "e2_effb0_512_ce.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("make_variant_config", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _diff(a: dict, b: dict, prefix: str = "") -> set[str]:
    keys = set(a) | set(b)
    out: set[str] = set()
    for k in keys:
        if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
            out |= _diff(a[k], b[k], f"{prefix}{k}.")
        elif a.get(k) != b.get(k):
            out.add(f"{prefix}{k}")
    return out


def test_seed_variant_changes_only_seed_and_name(tmp_path: Path) -> None:
    out = tmp_path / "v.yaml"
    assert _module().main(["--base", str(BASE), "--run-name", "v", "--seed", "43", "--out", str(out)]) == 0
    assert _diff(yaml.safe_load(BASE.read_text()), yaml.safe_load(out.read_text())) == {"run_name", "seed"}


def test_size_variant_changes_only_size_and_name(tmp_path: Path) -> None:
    out = tmp_path / "v.yaml"
    assert _module().main(["--base", str(BASE), "--run-name", "v", "--input-size", "384", "--out", str(out)]) == 0
    assert _diff(yaml.safe_load(BASE.read_text()), yaml.safe_load(out.read_text())) == {"run_name", "model.input_size"}


def test_replicate_changes_only_name(tmp_path: Path) -> None:
    out = tmp_path / "v.yaml"
    assert _module().main(["--base", str(BASE), "--run-name", "v", "--replicate", "--out", str(out)]) == 0
    assert _diff(yaml.safe_load(BASE.read_text()), yaml.safe_load(out.read_text())) == {"run_name"}


def test_no_change_without_replicate_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _module().main(["--base", str(BASE), "--run-name", "v", "--out", str(tmp_path / "v.yaml")])


def test_replicate_with_a_change_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _module().main(["--base", str(BASE), "--run-name", "v", "--replicate", "--seed", "43", "--out", str(tmp_path / "v.yaml")])
