#!/usr/bin/env python3
"""
Score network connection records with the trained hybrid detector.

Input: a CSV in NSL-KDD format (41 features; label/difficulty columns optional).
Output: one row per connection with the supervised attack probability, the anomaly
score, and the final layered verdict.

Usage:  python src/predict.py data/KDDTest+.txt --limit 20
Note: models/*.joblib are created by train.py. Never load .joblib/.pickle files
from sources you don't trust - unpickling can execute arbitrary code.
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import COLUMNS, features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--models", default="models")
    ap.add_argument("--limit", type=int, default=0, help="only score the first N rows")
    ap.add_argument("--out", help="write scored rows to this CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, header=None)
    df.columns = COLUMNS[: df.shape[1]]
    if args.limit:
        df = df.head(args.limit)

    sup = joblib.load(Path(args.models) / "binary_fusion_model.joblib")
    ano = joblib.load(Path(args.models) / "anomaly_model.joblib")
    X = features(df)
    p_attack = sup.predict_proba(X)[:, 1]
    a_score = -ano["iso"].score_samples(ano["prep"].transform(X))

    out = df[["protocol_type", "service", "flag", "src_bytes", "dst_bytes"]].copy()
    out["p_attack"] = p_attack.round(3)
    out["anomaly_score"] = a_score.round(3)
    out["supervised_alert"] = p_attack >= 0.5
    out["anomaly_alert"] = a_score > ano["threshold"]
    out["verdict"] = (out["supervised_alert"] | out["anomaly_alert"]).map({True: "ALERT", False: "ok"})
    if "label" in df:
        out["true_label"] = df["label"]

    print(out.to_string(index=False))
    print(f"\n{(out['verdict'] == 'ALERT').sum()} of {len(out)} connections flagged")
    if args.out:
        out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
