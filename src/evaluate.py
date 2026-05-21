import argparse
import os

import pandas as pd
from sklearn.metrics import average_precision_score, classification_report

from dataset import PROJECT_ROOT

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DEFAULT_THRESHOLDS = [0.49,]


def evaluate(csv_path: str, thresholds):
    df = pd.read_csv(csv_path)
    y_true = df["label"].astype(int).values
    y_proba = df["prob"].astype(float).values

    auprc = average_precision_score(y_true, y_proba)
    print(f"File: {csv_path}")
    print(f"AUPRC: {auprc}")

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        print(f"\n--- threshold = {thr} ---")
        print(classification_report(y_true, y_pred, digits=4))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a prediction CSV at one or more thresholds.")
    parser.add_argument(
        "--csv",
        type=str,
        default="rf_pred_0.556894.csv",
        help="Prediction CSV (relative paths resolve against results/).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help=f"One or more probability thresholds (default: {DEFAULT_THRESHOLDS}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.csv if os.path.isabs(args.csv) or os.path.exists(args.csv) else os.path.join(RESULTS_DIR, args.csv)
    evaluate(path, args.thresholds)


if __name__ == "__main__":
    main()
