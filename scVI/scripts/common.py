#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DATA_ROOT = Path(os.environ.get("SCVI_DATA_ROOT", str(REPO_ROOT / "data")))
DEFAULT_DATASET_DIR = PROJECT_DATA_ROOT / "h5ad_objects"
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("SCVI_OUTPUT_ROOT", str(REPO_ROOT)))


def _env_path_list(name: str) -> tuple[Path, ...]:
    value = os.environ.get(name, "").strip()
    if not value:
        return ()
    return tuple(Path(part.strip()) for part in value.split(",") if part.strip())


DEFAULT_EMBEDDING_PATHS = _env_path_list("SCVI_PERT_EMBEDDING_PATHS")


def env_default(name: str, fallback: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return fallback
    return value


def default_cell_line(task: str) -> str:
    del task
    return "HepG2"


def default_split_column() -> str:
    return "split_calib"


def default_split_csv(cell_line: str) -> Path:
    return PROJECT_DATA_ROOT / "csv" / f"DE_DOC_{cell_line}_splitted.csv"


def default_output_dir() -> Path:
    return DEFAULT_OUTPUT_ROOT / "outputs"


def default_train_split_groups(mode: str) -> str:
    if mode != "cross":
        return "train"
    return "train,test"


def default_train_prediction_groups(mode: str) -> str:
    if mode != "cross":
        return "test"
    return "test,calibration"


def default_calib_groups(mode: str) -> str:
    if mode == "cross":
        return "calibration"
    return ""


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.replace(",", " ").split() if part.strip()]


def infer_calib_groups(test_groups: str | None, calib_groups: str | None) -> str | None:
    if calib_groups is not None:
        return calib_groups
    groups = {group.lower() for group in split_csv(test_groups or "")}
    if "calibration" in groups:
        return "calibration"
    return None


def bool_string(value: bool) -> str:
    return "true" if value else "false"


def hydra_string(value: str) -> str:
    escaped = value.replace("'", "\\'")
    return f"'{escaped}'"


def hydra_list(values: Iterable[str]) -> str:
    return "[" + ",".join(hydra_string(str(value)) for value in values) + "]"


def format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{REPO_ROOT}:{current_pythonpath}"
    else:
        env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    print(format_command(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=build_env(), check=True)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cell-line",
        default=env_default("CELL_LINE"),
        help="Target cell line. In cross mode this is the held-out cell line.",
    )
    parser.add_argument(
        "--mode",
        default=env_default("MODE", "cross"),
        choices=("cross", "within"),
        help="Evaluation setting: hold out a cell line (cross) or split perturbations within one cell line (within).",
    )
    parser.add_argument(
        "--python-bin",
        default=env_default("PYTHON_BIN", sys.executable),
        help="Python interpreter used to run scVI_src. Defaults to the current interpreter.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-dir",
        default=env_default("SCVI_DATASET_DIR"),
        help="Directory containing <CELL_LINE>_raw.h5ad files.",
    )
    parser.add_argument(
        "--split-csv",
        default=env_default("SCVI_SPLIT_CSV"),
        help="Perturbation split CSV. Defaults to data/csv/DE_DOC_<CELL_LINE>_splitted.csv.",
    )
    parser.add_argument(
        "--split-column",
        default=env_default("SCVI_SPLIT_COLUMN"),
        help="Split column in the CSV. Defaults to split_calib; QA-style splits typically use split.",
    )
    parser.add_argument(
        "--use-split-gene-space",
        choices=("true", "false"),
        default=None,
        help="Use the split CSV gene column as the runtime/export gene space.",
    )
    parser.add_argument(
        "--dropped-test-perts-json",
        default=env_default("DROPPED_TEST_PERTS_JSON"),
        help="Optional JSON mapping cell lines to test perturbations that should be excluded.",
    )


def resolve_common_paths(args, task: str) -> None:
    if args.cell_line is None:
        args.cell_line = default_cell_line(task)
    if args.dataset_dir is None:
        args.dataset_dir = str(DEFAULT_DATASET_DIR)
    if args.split_csv is None:
        args.split_csv = str(default_split_csv(args.cell_line))
    if args.split_column is None:
        args.split_column = default_split_column()


def resolve_output_dir(args) -> str:
    if getattr(args, "output_dir", None):
        return args.output_dir
    return str(default_output_dir())


def resolve_run_dir(args) -> str:
    if getattr(args, "run_dir", None):
        return args.run_dir
    output_dir = resolve_output_dir(args)
    run_name = getattr(args, "run_name", None) or f"{args.cell_line}_{args.mode}"
    return str(Path(output_dir) / run_name)
