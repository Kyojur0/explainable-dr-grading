"""Write a run config that differs from a base config in declared fields only.

Used for the planned extras (DECISIONS.md D34): the stability run (another seed) and
the resolution ablation (another input size) copy the selected configuration
exactly and change one factor. The new file records which base it came from and
what changed, so the comparison stays auditable.

Usage:
    python scripts/make_variant_config.py --base configs/e2_effb0_512_ce.yaml \
        --run-name e5_effb0_512_ce_seed43 --seed 43 --out configs/e5_effb0_512_ce_seed43.yaml
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from drgrade.run_config import load_run_config
from drgrade.utils import setup_logging

logger = logging.getLogger(__name__)


def make_variant(
    base: dict[str, Any],
    run_name: str,
    seed: int | None,
    input_size: int | None,
    replicate: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Return a deep copy of `base` with the requested changes and a list describing them."""
    variant = copy.deepcopy(base)
    changes = [f"run_name: {base['run_name']} -> {run_name}"]
    variant["run_name"] = run_name
    if seed is not None and seed != base["seed"]:
        changes.append(f"seed: {base['seed']} -> {seed}")
        variant["seed"] = seed
    if input_size is not None and input_size != base["model"]["input_size"]:
        changes.append(f"model.input_size: {base['model']['input_size']} -> {input_size}")
        variant["model"]["input_size"] = input_size
    if len(changes) == 1 and not replicate:
        raise ValueError("the variant changes nothing but its name; pass --seed, --input-size or --replicate")
    if len(changes) > 1 and replicate:
        raise ValueError("--replicate must not change anything but the name")
    return variant, changes


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True, help="base run config (the selected model)")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument(
        "--replicate",
        action="store_true",
        help="identical settings under a new name (to rerun the base on other hardware)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    base = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    variant, changes = make_variant(base, args.run_name, args.seed, args.input_size, args.replicate)
    header = (
        f"# Variant of {args.base.as_posix()} (planned extras, DECISIONS.md D34).\n"
        "# Identical to the base except:\n" + "".join(f"#   {c}\n" for c in changes)
        + ("#   (replicate: same settings, rerun on different hardware)\n" if args.replicate else "")
    )
    args.out.write_text(header + yaml.safe_dump(variant, sort_keys=False), encoding="utf-8")
    load_run_config(args.out)  # validate with the same schema the trainer uses
    logger.info("wrote %s (%s)", args.out, "; ".join(changes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
