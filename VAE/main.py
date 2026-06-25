"""
main.py
=======

Entry script for training and evaluating the perturbation VAE model.
This script supports two experimental settings:

1. Cross-Cell-Line Generalization (mode="cross")
   - Hold out one entire cell line.
   - Train VAE on remaining cell lines.
   - Transfer perturbation effect to held-out cell line.

2. Within-Cell-Line Unseen Perturbation Prediction (mode="within")
   - Split perturbations within a cell line into train/test sets.
   - Train VAE on training perturbations.
   - Use embedding similarity to find nearest perturbations.
   - Transfer weighted latent delta to unseen perturbations.

"""

import anndata
import os
import argparse
import scanpy as sc
from utils import (
    data_read_split,
    get_embeddings,
    train_valid_split,
    train_test_split,
    similarity_emb,
    log_setting,
    resolve_split_csv_path,
    _read_split_genes,
    _subset_gene_space,
)
from train_test import (
    train_cell_line,
    reconstruct_cross_cell_line,
    reconstruct_cross_cell_line_calib,
    reconstruct_within_cell_line,
)

def main(
    z_dim,
    alpha,
    cell_line,
    mode,
    n_epochs,
    batch_size,
    dropout_rate,
    learning_rate,
    train_continue=False,
    seed=42,
    log="train.log",
    stage="train",
    split_set_train="train",
    split_set_test="test",
    split_set_calibration="calibration",
    dataset_dir="data/h5ad_objects",
    split_csv_path="data/csv/DE_DOC_{cell}_splitted.csv",
    split_column="split_calib",
    embedding_path="data/embedding/GenePT_gene_protein_embedding_model_3_text.pickle.",
    missing_embedding_path="data/embedding/additional_genePT_embeddings.pkl",
    output_root="results",
    use_split_gene_space=False,
    dropped_test_perts_json=None,
):
    """
    Main experiment pipeline for perturbation VAE training and evaluation.

    Important args:
    alpha : float
        Weight of KL divergence term in VAE loss, Final loss = recon_loss + alpha * KL_loss.
    cell_line : str
        Target cell line for experiment.
    mode : str
        Experiment mode:"cross" or "within"
    train_continue : bool, default=False
        Whether to resume training from saved checkpoint.
    """
    cell_lines = ["K562", "RPE1", "Jurkat", "HepG2"]
    data_path_dict = {
        cell: os.path.join(dataset_dir, f"{cell}_raw.h5ad")
        for cell in cell_lines
    }
    emb_path = embedding_path
    emb_path_miss = missing_embedding_path
    run_dir = os.path.join(output_root, f"{cell_line}_{mode}")
    os.makedirs(run_dir, exist_ok=True)
    log = log_setting(log_dir=run_dir, log_name=log)
    use_split_gene_space = parse_bool(use_split_gene_space)

    if mode == "cross":
        log.info("Extracting data...")
        adata_non_holdout, adata_holdout_ctrl, adata_holdout_pert, adata_calibration = data_read_split(data_path_dict, cell_line, target_col="gene", 
                                                                                                    control_name="non-targeting", cell_line_key="cell_line",
                                                                                                    split_csv_path=split_csv_path, split_column=split_column,
                                                                                                    split_set_non_holdout=split_set_train,
                                                                                                    split_set_holdout=split_set_test,
                                                                                                    split_set_calibration=split_set_calibration,
                                                                                                    use_split_gene_space=use_split_gene_space,
                                                                                                    dropped_test_perts_json=dropped_test_perts_json)
        adata_train, adata_valid = train_valid_split(adata_non_holdout,
                                                    0.1, seed, "gene", "cell_line", min_count_per_group = 40)                                     
        pert_set_train = set(adata_non_holdout.obs["gene"])
        pert_set_test = set(adata_holdout_pert.obs["gene"])
        pert_set_calibration = set(adata_calibration.obs["gene"])
        pert_union = pert_set_test | pert_set_train | pert_set_calibration
        embeddings_train = get_embeddings(emb_path, emb_path_miss, pert_set_train)
        embeddings_test = get_embeddings(emb_path, emb_path_miss, pert_set_test)
        embeddings_calibration = get_embeddings(emb_path, emb_path_miss, pert_set_calibration)
        embeddings = get_embeddings(emb_path, emb_path_miss, pert_union)
        similarity_df = similarity_emb(embeddings_train, embeddings_test, n=1)
        similarity_df_calib = similarity_emb(embeddings_train, embeddings_calibration, n=1)

        if stage == "train":
            log.info("Start training...")
            train_cell_line(adata_train, adata_valid, log,
                            z_dim, alpha, n_epochs, batch_size, dropout_rate, learning_rate,
                            cell_line, embeddings_train, mode, train_continue, run_dir=run_dir)

        if stage == "test":
            log.info("Start Reconstructing for cross_cell_line data...")
            reconstruct_cross_cell_line(
                adata_non_holdout,
                adata_holdout_ctrl,
                adata_holdout_pert,
                log,
                similarity_df,
                z_dim,
                cell_line,
                embeddings,
                seed,
                output_tag=split_set_test,
                run_dir=run_dir,
            )
        
        if stage == "calibration":
            log.info("Start Calibration for cross_cell_line data...")
            reconstruct_cross_cell_line_calib(
                adata_non_holdout,
                adata_calibration,
                log,
                similarity_df_calib,
                z_dim,
                cell_line,
                embeddings,
                seed,
                output_tag=split_set_calibration,
                run_dir=run_dir,
            )

    elif mode == "within":
        log.info("Constructing dataset and model...")
        adata = anndata.read_h5ad(data_path_dict[cell_line])
        adata.obs["cell_line"] = cell_line
        csv_path = resolve_split_csv_path(split_csv_path, cell_line)
        if use_split_gene_space:
            split_genes = _read_split_genes(csv_path, "gene")
            adata = _subset_gene_space(adata, split_genes)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        if "gene_name" not in adata.var.columns:
            adata.var["gene_name"] = adata.var_names.astype(str)
        full_control_adata = adata[adata.obs["gene"] == "non-targeting", :].copy()
        adata_train, adata_test = train_test_split(
            adata,
            csv_path,
            "gene",
            "non-targeting",
            split_set_train=split_set_train,
            split_set_test=split_set_test,
            split_column=split_column,
        )
        adata_train, adata_valid = train_valid_split(adata_train, 0.1, seed, "gene", "cell_line", min_count_per_group = 40)

        pert_set_train = set(adata_train.obs["gene"])
        pert_set_test = set(adata_test.obs["gene"])
        pert_union = set(pert_set_train) | set(pert_set_test)

        embeddings_train = get_embeddings(emb_path, emb_path_miss, pert_set_train)
        embeddings_test = get_embeddings(emb_path, emb_path_miss, pert_set_test)
        embeddings = get_embeddings(emb_path, emb_path_miss, pert_union)

        similarity_df = similarity_emb(embeddings_train, embeddings_test, n=1)

        if stage == "train":
            log.info("Start training...")
            train_cell_line(adata_train, adata_valid, log,
                            z_dim, alpha, n_epochs, batch_size, dropout_rate, learning_rate,
                            cell_line, embeddings_train, mode, train_continue, run_dir=run_dir)

        if stage == "test":
            log.info(f"Start Reconstructing for within_cell_line {split_set_test} data...")
            reconstruct_within_cell_line(
                adata_train,
                adata_test,
                log,
                similarity_df,
                z_dim,
                cell_line,
                embeddings,
                seed,
                output_tag=split_set_test,
                full_control_adata=full_control_adata,
                run_dir=run_dir,
            )

    log.info("All done!")

def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--cell_line", type=str, default="K562")
    parser.add_argument("--mode", type=str, choices=["cross", "within"], default="cross")
    parser.add_argument("--train_continue", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dropout_rate", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", type=str, default="train.log")
    parser.add_argument("--stage", type=str, choices=["train", "test", "calibration"], default="train")
    parser.add_argument("--split_set_train", type=str, default="train")
    parser.add_argument("--split_set_test", type=str, default="test")
    parser.add_argument("--split_set_calibration", type=str, default="calibration")
    parser.add_argument("--dataset_dir", type=str, default="data/h5ad_objects")
    parser.add_argument(
        "--split_csv_path",
        type=str,
        default="data/csv/DE_DOC_{cell}_splitted.csv",
    )
    parser.add_argument("--split_column", type=str, default="split_calib")
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="data/embedding/GenePT_gene_protein_embedding_model_3_text.pickle.",
    )
    parser.add_argument(
        "--missing_embedding_path",
        type=str,
        default="data/embedding/additional_genePT_embeddings.pkl",
    )
    parser.add_argument("--output_root", type=str, default="results")
    parser.add_argument("--use_split_gene_space", type=parse_bool, default=False)
    parser.add_argument(
        "--dropped_test_perts_json",
        type=str,
        default=None,
        help="JSON of holdout test perturbations to drop in cross-cell-line mode.",
    )
    args = parser.parse_args()

    main(
        z_dim=args.z_dim,
        alpha=args.alpha,
        cell_line=args.cell_line,
        mode=args.mode,
        train_continue=args.train_continue,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        seed=args.seed,
        log=args.log,
        stage=args.stage,
        split_set_train=args.split_set_train,
        split_set_test=args.split_set_test,
        split_set_calibration=args.split_set_calibration,
        dataset_dir=args.dataset_dir,
        split_csv_path=args.split_csv_path,
        split_column=args.split_column,
        embedding_path=args.embedding_path,
        missing_embedding_path=args.missing_embedding_path,
        output_root=args.output_root,
        use_split_gene_space=args.use_split_gene_space,
        dropped_test_perts_json=args.dropped_test_perts_json,
    )
    # Example usage:
    # python main.py --cell_line K562 --mode cross --stage train
