#!/usr/bin/env python

from __future__ import annotations

import argparse

from common import (
    add_common_args,
    add_data_args,
    infer_calib_groups,
    resolve_common_paths,
    resolve_run_dir,
    run_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export scVI test predictions from a trained run.",
        epilog=(
            "Examples:\n"
            "  python scripts/predict.py --cell-line HepG2 --prediction-modes mean\n"
            "  python scripts/predict.py --cell-line HepG2 --prediction-modes both\n"
            "  python scripts/predict.py --mode within --cell-line Jurkat --run-dir /path/to/Jurkat_within\n"
            "  python scripts/predict.py --cell-line RPE1 --split-csv data/perturbqa_datasets/rpe1-de.csv --split-column split --use-split-gene-space true\n"
            "\n"
            "For cross mode, --test-split-groups test targets the held-out cell line. "
            "If the groups include calibration, --calib-split-groups is inferred as calibration unless provided."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser)
    add_data_args(parser)
    parser.add_argument("--output-dir", default=None, help="Root output directory containing <cell>_<mode>.")
    parser.add_argument("--run-dir", default=None, help="Concrete trained run directory.")
    parser.add_argument("--run-name", default=None, help="Run name under --output-dir. Defaults to <cell>_<mode>.")
    parser.add_argument("--checkpoint", default="best", help="Checkpoint name/path or 'best'.")
    parser.add_argument("--sample-size", default="1", help="Prediction sample multiplier per perturbation.")
    parser.add_argument("--test-split-groups", default="test", help="Comma-separated split labels exported as test predictions.")
    parser.add_argument("--calib-split-groups", default=None, help="Comma-separated calibration labels for cross prediction filtering.")
    parser.add_argument(
        "--prediction-modes",
        choices=("mean", "sampling", "both"),
        default="mean",
        help="Run deterministic mean prediction, sampling prediction, or both.",
    )
    parser.add_argument("--toml-name", default=None, help="Prediction Cell Load TOML name/path written under the run directory.")
    return parser.parse_args()


def prediction_mode_values(mode: str) -> list[str]:
    if mode == "both":
        return ["false", "true"]
    if mode == "sampling":
        return ["true"]
    return ["false"]


def main() -> None:
    args = parse_args()
    resolve_common_paths(args, task="predict")
    run_dir = resolve_run_dir(args)
    calib_groups = infer_calib_groups(args.test_split_groups, args.calib_split_groups)
    use_split_gene_space = args.use_split_gene_space

    base_cmd = [
        args.python_bin,
        "-u",
        "scVI_src/train/get_predictions.py",
        "--output_dir",
        run_dir,
        "--checkpoint",
        args.checkpoint,
        "--prediction_sample_multiplier",
        args.sample_size,
        "--split_csv_path",
        args.split_csv,
        "--split_column",
        args.split_column,
        "--test_split_groups",
        args.test_split_groups,
    ]
    if calib_groups is not None:
        base_cmd.extend(["--calib_split_groups", calib_groups])
    if use_split_gene_space is not None:
        base_cmd.extend(["--use_split_gene_space", use_split_gene_space])
    if args.dropped_test_perts_json:
        base_cmd.extend(["--dropped_test_perts_json", args.dropped_test_perts_json])
    if args.toml_name:
        base_cmd.extend(["--toml_name", args.toml_name])

    print(
        "scVI predict interface: "
        f"cell_line={args.cell_line}, mode={args.mode}, "
        f"test_groups={args.test_split_groups}, calib_groups={calib_groups or ''}, "
        f"prediction_modes={args.prediction_modes}"
    )
    print(f"run_dir={run_dir}")
    for prediction_use_sampling in prediction_mode_values(args.prediction_modes):
        cmd = [*base_cmd, "--prediction_use_sampling", prediction_use_sampling]
        run_command(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
