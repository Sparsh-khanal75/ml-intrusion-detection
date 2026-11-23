#!/usr/bin/env python3
"""
Train and evaluate intrusion detection models on NSL-KDD.

Experiments
  1. Binary detection (normal vs attack): Logistic Regression, Random Forest,
     Histogram Gradient Boosting, and an entropy-weighted fusion of all three.
  2. Multi-class (Normal / DoS / Probe / R2L / U2R) with the same models.
  3. Unsupervised anomaly detection: Isolation Forest trained on normal traffic only.

Everything is evaluated on KDDTest+, which contains 17 attack types never seen in
training - a test of generalisation, not memorisation.

Usage:  python src/train.py --data data --out results
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from features import CLASSES, MITRE, features, load, make_preprocessor  # noqa: E402
from fusion import EntropyWeightedEnsemble, entropy_weighted_fusion  # noqa: E402

SEED = 42
OUT_DIR = Path("results")
MODEL_DIR = Path("models")


def models(multiclass: bool) -> dict:
    cw = "balanced" if multiclass else None
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight=cw),
        "RandomForest": RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=SEED,
                                               class_weight=cw),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED, class_weight=cw),
    }


def pipe(est) -> Pipeline:
    return Pipeline([("prep", make_preprocessor()), ("model", est)])


def binary_metrics(y, pred, score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, score), "false_positive_rate": fp / (fp + tn),
    }


def run_binary(Xtr, ytr, Xte, yte, test_df, unseen_mask, out: Path) -> dict:
    results, fitted, probas = {}, {}, {}
    for name, est in models(False).items():
        t = time.time()
        fitted[name] = pipe(est).fit(Xtr, ytr)
        probas[name] = fitted[name].predict_proba(Xte)
        pred = probas[name].argmax(1)
        results[name] = binary_metrics(yte, pred, probas[name][:, 1])
        results[name]["train_seconds"] = round(time.time() - t, 1)
        print(f"  [binary] {name:<22} F1={results[name]['f1']:.4f}  ({results[name]['train_seconds']}s)")

    avg = np.mean(list(probas.values()), axis=0)
    results["SoftVoting (mean)"] = binary_metrics(yte, avg.argmax(1), avg[:, 1])
    fused, weights = entropy_weighted_fusion(list(probas.values()))
    results["EntropyWeightedFusion"] = binary_metrics(yte, fused.argmax(1), fused[:, 1])
    print(f"  [binary] {'SoftVoting (mean)':<22} F1={results['SoftVoting (mean)']['f1']:.4f}")
    print(f"  [binary] {'EntropyWeightedFusion':<22} F1={results['EntropyWeightedFusion']['f1']:.4f}")

    # Detection rate per attack category and on attack types never seen in training
    pred = fused.argmax(1)
    per_cat = {c: float(pred[(test_df["category"] == c).values].mean())
               for c in CLASSES[1:]}
    unseen = float(pred[unseen_mask].mean())
    mean_w = dict(zip(probas.keys(), weights.mean(0).round(3).tolist()))

    ens = EntropyWeightedEnsemble.from_fitted(fitted)
    joblib.dump(ens, MODEL_DIR / "binary_fusion_model.joblib", compress=3)
    plot_feature_importance(fitted["RandomForest"], out / "feature_importance.png")
    return {"models": results, "fusion_detection_rate_by_category": per_cat,
            "fusion_detection_rate_unseen_attacks": unseen, "fusion_mean_weights": mean_w}, fused


def run_multiclass(Xtr, ytr, Xte, yte, out: Path) -> dict:
    results, probas = {}, {}
    for name, est in models(True).items():
        m = pipe(est).fit(Xtr, ytr)
        probas[name] = m.predict_proba(Xte)
        classes = m.classes_
        pred = classes[probas[name].argmax(1)]
        results[name] = {"accuracy": accuracy_score(yte, pred),
                         "macro_f1": f1_score(yte, pred, average="macro"),
                         "weighted_f1": f1_score(yte, pred, average="weighted")}
        print(f"  [multi]  {name:<22} macro-F1={results[name]['macro_f1']:.4f}")
    fused, _ = entropy_weighted_fusion(list(probas.values()))
    pred = classes[fused.argmax(1)]
    results["EntropyWeightedFusion"] = {"accuracy": accuracy_score(yte, pred),
                                        "macro_f1": f1_score(yte, pred, average="macro"),
                                        "weighted_f1": f1_score(yte, pred, average="weighted")}
    print(f"  [multi]  {'EntropyWeightedFusion':<22} macro-F1={results['EntropyWeightedFusion']['macro_f1']:.4f}")
    report = classification_report(yte, pred, labels=CLASSES, output_dict=True, zero_division=0)
    plot_confusion(yte, pred, out / "confusion_matrix_multiclass.png")
    return {"models": results, "fusion_per_class": {c: report[c] for c in CLASSES}}


def run_anomaly(train_df, Xtr, Xte, yte, unseen_mask) -> dict:
    normal = train_df["is_attack"].values == 0
    prep = make_preprocessor().fit(Xtr[normal])
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=SEED, n_jobs=-1)
    iso.fit(prep.transform(Xtr[normal]))
    score = -iso.score_samples(prep.transform(Xte))
    # Threshold = 95th percentile of normal *training* scores (no test-label leakage)
    thr = np.percentile(-iso.score_samples(prep.transform(Xtr[normal])), 95)
    pred = (score > thr).astype(int)
    m = binary_metrics(yte, pred, score)
    m["detection_rate_unseen_attacks"] = float(pred[unseen_mask].mean())
    print(f"  [anomaly] IsolationForest        F1={m['f1']:.4f}  unseen-attack recall={m['detection_rate_unseen_attacks']:.3f}")
    joblib.dump({"prep": prep, "iso": iso, "threshold": thr}, MODEL_DIR / "anomaly_model.joblib", compress=3)
    return m, pred, score


def run_hybrid(fused, anomaly_pred, anomaly_score, yte, test_df, unseen_mask) -> dict:
    """Layered detection: alert if the supervised fusion OR the anomaly detector fires."""
    pred = ((fused.argmax(1) == 1) | (anomaly_pred == 1)).astype(int)
    # Continuous score for AUC: max of the two normalised scores
    rank = (anomaly_score.argsort().argsort() / len(anomaly_score))
    m = binary_metrics(yte, pred, np.maximum(fused[:, 1], rank))
    m["detection_rate_by_category"] = {c: float(pred[(test_df["category"] == c).values].mean())
                                       for c in CLASSES[1:]}
    m["detection_rate_unseen_attacks"] = float(pred[unseen_mask].mean())
    print(f"  [hybrid] Fusion OR IsolationForest F1={m['f1']:.4f}  unseen-attack recall={m['detection_rate_unseen_attacks']:.3f}  FPR={m['false_positive_rate']:.3f}")
    return m


def plot_confusion(y, pred, path):
    cm = confusion_matrix(y, pred, labels=CLASSES, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted", ylabel="Actual", title="Entropy-weighted fusion - KDDTest+ (row-normalised)")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(rf_pipe: Pipeline, path, top=15):
    names = rf_pipe.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    imp = rf_pipe.named_steps["model"].feature_importances_
    order = np.argsort(imp)[-top:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([names[i] for i in order], imp[order], color="#2b6cb0")
    ax.set(title="Top 15 features - Random Forest (binary)", xlabel="Mean decrease in impurity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(binary: dict, path):
    names = list(binary["models"].keys())
    f1 = [binary["models"][n]["f1"] for n in names]
    fpr = [binary["models"][n]["false_positive_rate"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(x - 0.2, f1, 0.4, label="F1 (attack)", color="#2b6cb0")
    ax.bar(x + 0.2, fpr, 0.4, label="False positive rate", color="#c05621")
    ax.set_xticks(x, [n.replace("Weighted", "Weighted\n").replace(" (", "\n(") for n in names], fontsize=8)
    ax.set(ylim=(0, 1), title="Binary detection on KDDTest+ (higher F1, lower FPR is better)")
    ax.legend()
    for i, v in enumerate(f1):
        ax.text(i - 0.2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.2, fpr[i] + 0.01, f"{fpr[i]:.1%}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_layered(binary, anomaly, hybrid, path):
    groups = CLASSES[1:] + ["Unseen attack\ntypes"]
    series = {
        "Supervised fusion": list(binary["fusion_detection_rate_by_category"].values())
        + [binary["fusion_detection_rate_unseen_attacks"]],
        "Isolation Forest (anomaly)": list(anomaly["detection_rate_by_category"].values())
        + [anomaly["detection_rate_unseen_attacks"]],
        "Hybrid (either fires)": list(hybrid["detection_rate_by_category"].values())
        + [hybrid["detection_rate_unseen_attacks"]],
    }
    colors = ["#2b6cb0", "#c05621", "#2f855a"]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=name, color=colors[i])
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1) * 0.27, v + 0.01, f"{v:.0%}", ha="center", fontsize=7)
    ax.set_xticks(x, groups)
    ax.set(ylim=(0, 1.08), ylabel="Detection rate (recall)",
           title="Detection rate by attack category on KDDTest+")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_df = load(f"{args.data}/KDDTrain+.txt")
    test_df = load(f"{args.data}/KDDTest+.txt")
    Xtr, Xte = features(train_df), features(test_df)
    unseen_labels = sorted(set(test_df["label"]) - set(train_df["label"]))
    unseen_mask = test_df["label"].isin(unseen_labels).values
    print(f"Train {len(train_df):,} rows | Test {len(test_df):,} rows | "
          f"{len(unseen_labels)} attack types only in test ({unseen_mask.sum():,} rows)")

    global OUT_DIR
    OUT_DIR = out
    MODEL_DIR.mkdir(exist_ok=True)
    binary, fused = run_binary(Xtr, train_df["is_attack"].values, Xte, test_df["is_attack"].values,
                        test_df, unseen_mask, out)
    multi = run_multiclass(Xtr, train_df["category"].values, Xte, test_df["category"].values, out)
    anomaly, a_pred, a_score = run_anomaly(train_df, Xtr, Xte, test_df["is_attack"].values, unseen_mask)
    hybrid = run_hybrid(fused, a_pred, a_score, test_df["is_attack"].values, test_df, unseen_mask)
    anomaly["detection_rate_by_category"] = {c: float(a_pred[(test_df["category"] == c).values].mean())
                                             for c in CLASSES[1:]}
    plot_model_comparison({"models": {**binary["models"], "IsolationForest": anomaly,
                                      "Hybrid (Fusion OR IForest)": hybrid}},
                          out / "model_comparison.png")
    plot_layered(binary, anomaly, hybrid, out / "layered_detection.png")

    summary = {
        "dataset": {"train_rows": len(train_df), "test_rows": len(test_df),
                    "train_class_counts": train_df["category"].value_counts().to_dict(),
                    "test_class_counts": test_df["category"].value_counts().to_dict(),
                    "unseen_attack_types": unseen_labels, "unseen_attack_rows": int(unseen_mask.sum())},
        "binary": binary, "multiclass": multi, "anomaly_isolation_forest": anomaly,
        "hybrid_fusion_or_anomaly": hybrid,
        "mitre_mapping": MITRE,
    }
    with open(out / "metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"Saved metrics and figures to {out}/")


if __name__ == "__main__":
    main()
