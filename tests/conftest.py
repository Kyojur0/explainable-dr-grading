"""Shared test setup: keep every test away from the real, committed provenance files.

scripts/predict.py and the analysis scripts read the test allowlist and append to the
test access log (src/drgrade/provenance.py). Each test gets its own temporary copies,
so the suite can never write to results/test_access_log.jsonl.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drgrade.provenance import ENV_ACCESS_LOG, ENV_ALLOWLIST

TEST_ALLOWED_RUNS = ("tiny_predict",)


@pytest.fixture(autouse=True)
def _isolated_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "_provenance"
    folder.mkdir()
    allowlist = folder / "test_allowlist.yaml"
    allowlist.write_text("allowed_runs:\n" + "".join(f"  - {r}\n" for r in TEST_ALLOWED_RUNS), encoding="utf-8")
    monkeypatch.setenv(ENV_ALLOWLIST, str(allowlist))
    monkeypatch.setenv(ENV_ACCESS_LOG, str(folder / "test_access_log.jsonl"))
