#!/usr/bin/env python
"""
CLI wrapper for launching scVI training with project defaults.
"""

from __future__ import annotations

import argparse

from common import (
    DEFAULT_EMBEDDING_PATHS,
    add_common_args,
    add_data_args,
    bool_string,
    default_calib_groups,
    default_train_prediction_groups,
    default_train_split_groups,
    hydra_list,
    hydra_string,
    resolve_common_paths,
    resolve_output_dir,
    run_command,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for a training run.

    Returns
    -------
    argparse.Namespace
        Training wrapper options before path/default resolution.
    """
    parser = argparse.ArgumentParser(
        description="Launch scVI training with stable data-interface defaults.",
        epilog=(
            "Examples:\n"
            "  python scripts/train.py --cell-line HepG2\n"
            "  python scripts/train.py --mode within --cell-line Jurkat --train-split-groups train --test-split-groups test\n"
            "  python scripts/train.py --cell-line RPE1 --split-csv data/perturbqa_datasets/rpe1-de.csv --split-column split --test-split-groups test --calib-split-groups ''\n"
            "\n"
            "The wrapper forwards Hydra overrides to scVI_src/train/__main__.py. "
            "Use --extra-override for less common config fields."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser)
    add_data_args(parser)
    parser.add_argument("--output-dir", default=None, help="Root directory where <run-name>/ will be written.")
    parser.add_argument("--run-name", default=None, help="Run name under --output-dir. Defaults to <cell-line>_<mode>.")
    parser.add_argument("--dataset-name", default=None, help="Dataset basename without .h5ad. Defaults to <cell-line>_raw.")
    parser.add_argument("--cell-type-name", default=None, help="Cell type label to write/use when a dataset lacks obs['cell_type'].")
    parser.add_argument("--train-split-groups", default=None, help="Comma-separated split labels used as the training perturbation pool.")
    parser.add_argument("--test-split-groups", default=None, help="Comma-separated split labels exported as prediction targets.")
    parser.add_argument(
        "--calib-split-groups",
        default=None,
        help="Comma-separated split labels used to build non-held-out calibration datasets in cross mode.",
    )
    parser.add_argument(
        "--pert-embedding-path",
        action="append",
        default=None,
        help="Perturbation embedding pickle path. Repeat to merge multiple files, or set SCVI_PERT_EMBEDDING_PATHS as a comma-separated list.",
    )
    parser.add_argument("--no-pert-embeddings", action="store_true", help="Do not pass perturbation embedding paths from the environment.")
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override passed through to scVI_src/train/__main__.py.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Build Hydra overrides and run the underlying training entry point.
    """
    args = parse_args()
    resolve_common_paths(args, task="train")

    output_dir = resolve_output_dir(args)
    run_name = args.run_name or f"{args.cell_line}_{args.mode}"
    dataset_name = args.dataset_name or f"{args.cell_line}_raw"
    cell_type_name = args.cell_type_name or args.cell_line
    train_groups = args.train_split_groups or default_train_split_groups(args.mode)
    test_groups = args.test_split_groups or default_train_prediction_groups(args.mode)
    calib_groups = args.calib_split_groups
    if calib_groups is None:
        calib_groups = default_calib_groups(args.mode)
    use_split_gene_space = args.use_split_gene_space or "false"

    overrides = [
        f"output_dir={output_dir}",
        f"name={run_name}",
        f"data.kwargs.dataset_dir={args.dataset_dir}",
        f"data.kwargs.split_csv_path={args.split_csv}",
        f"data.kwargs.split_column={args.split_column}",
        f"data.kwargs.train_split_groups={hydra_string(train_groups)}",
        f"data.kwargs.test_split_groups={hydra_string(test_groups)}",
        f"data.kwargs.calib_split_groups={hydra_string(calib_groups)}",
        f"data.kwargs.use_split_gene_space={use_split_gene_space}",
        f"data.kwargs.dataset_name={dataset_name}",
        f"data.kwargs.cell_type_name={cell_type_name}",
        f"model.kwargs.cell_line={args.cell_line}",
        f"model.kwargs.mode={args.mode}",
    ]

    if args.dropped_test_perts_json:
        overrides.append(f"data.kwargs.dropped_test_perts_json={args.dropped_test_perts_json}")

    embedding_paths = args.pert_embedding_path
    if embedding_paths is None and not args.no_pert_embeddings:
        embedding_paths = [str(path) for path in DEFAULT_EMBEDDING_PATHS]
    if embedding_paths:
        overrides.append(f"model.kwargs.pert_embedding_paths={hydra_list(embedding_paths)}")

    overrides.extend(args.extra_override)

    cmd = [
        args.python_bin,
        "-u",
        "scVI_src/train/__main__.py",
        *overrides,
    ]
    print(
        "scVI train interface: "
        f"cell_line={args.cell_line}, mode={args.mode}, "
        f"train_groups={train_groups}, test_groups={test_groups}, calib_groups={calib_groups}"
    )
    print(f"output_dir={output_dir}, run_name={run_name}")
    print(f"use_split_gene_space={bool_string(use_split_gene_space == 'true')}")
    run_command(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
