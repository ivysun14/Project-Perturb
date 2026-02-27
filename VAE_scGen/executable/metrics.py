import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse

def compute_3class_prf(truth_csv, pred_csv):
    """
    Compute precision, recall, and F1 score for 3-class classification (non-DE, up, down).
    """
    df_true = pd.read_csv(truth_csv)
    df_pred = pd.read_csv(pred_csv)
    df_true = df_true[["pert", "gene", "label"]]
    df_pred = df_pred[["pert", "gene", "label"]]
    print("Truth size:", len(df_true))
    print("Pred size:", len(df_pred))
    df_pred = df_pred.rename(columns={"label": "pred_label"})
    df = df_true.merge(df_pred, on=["pert", "gene"], how="inner")
    print("Merged size:", len(df))

    print("Dropped from truth:", len(df_true) - len(df))
    print("Dropped from pred :", len(df_pred) - len(df))

    y_true = df["label"].values
    y_pred = df["pred_label"].values
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    results = pd.DataFrame({
        "Class": ["Non-DE (0)", "Up (1)", "Down (2)"],
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute precision, recall, and F1 score for 3-class classification.")
    parser.add_argument("--gt", type=str, required=True, help="Path to the ground truth CSV file.")
    parser.add_argument("--pred", type=str, required=True, help="Path to the predictions CSV file.")
    args = parser.parse_args()

    metrics_df = compute_3class_prf(args.gt, args.pred)
    print(metrics_df)
    # Example usage
    # python metrics.py --gt /path/to/ground_truth.csv --pred /path/to/predictions.csv