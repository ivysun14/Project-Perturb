#!/usr/bin/env python

from __future__ import annotations

import argparse

from common import add_common_args, add_data_args, resolve_common_paths, resolve_run_dir, run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export cross-cell-line predict_calib predictions from a trained scVI run.",
        epilog=(
            "Example:\n"
            "  python scripts/predict_calib.py --cell-line HepG2 --prediction-modes sampling\n"
            "  python scripts/predict_calib.py --cell-line HepG2 --prediction-modes both\n"
            "\n"
            "This is not probability calibration or fine-tuning. It reuses a trained cross-cell-line run "
            "and exports predictions on non-held-out cell lines' calibration split perturbations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser)
    add_data_args(parser)
    parser.add_argument("--output-dir", default=None, help="Root output directory containing <cell>_<mode>.")
    parser.add_argument("--run-dir", default=None, help="Concrete trained run directory.")
    parser.add_argument("--run-name", default=None, help="Run name under --output-dir. Defaults to <cell>_<mode>.")
    parser.add_argument("--checkpoint", default="best", help="Checkpoint name/path or 'best'.")
    parser.add_argument("--sample-size", default="1", help="Prediction sample multiplier per calibration perturbation.")
    parser.add_argument("--calib-split-groups", default="calibration", help="Split labels used as calibration targets.")
    parser.add_argument("--toml-name", default="cell_prediction_calibration.toml", help="Prediction TOML name/path.")
    parser.add_argument(
        "--prediction-modes",
        choices=("mean", "sampling", "both"),
        default="sampling",
        help="Run deterministic mean prediction, sampling prediction, or both.",
    )
    return parser.parse_args()


def prediction_mode_values(mode: str) -> list[str]:
    if mode == "both":
        return ["false", "true"]
    if mode == "sampling":
        return ["true"]
    return ["false"]


def main() -> None:
    args = parse_args()
    resolve_common_paths(args, task="predict_calib")
    if args.mode != "cross":
        raise ValueError("predict_calib export is only defined for cross mode.")
    run_dir = resolve_run_dir(args)

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
        "--calib_split_groups",
        args.calib_split_groups,
        "--toml_name",
        args.toml_name,
    ]
    if args.use_split_gene_space is not None:
        base_cmd.extend(["--use_split_gene_space", args.use_split_gene_space])
    if args.dropped_test_perts_json:
        base_cmd.extend(["--dropped_test_perts_json", args.dropped_test_perts_json])

    print(
        "scVI predict_calib interface: "
        f"holdout_cell_line={args.cell_line}, "
        f"calib_groups={args.calib_split_groups}, prediction_modes={args.prediction_modes}"
    )
    print(f"run_dir={run_dir}")
    for prediction_use_sampling in prediction_mode_values(args.prediction_modes):
        cmd = [*base_cmd, "--prediction_use_sampling", prediction_use_sampling]
        run_command(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
