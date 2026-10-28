"""Train, compare and save fraud models.

Run with:  python -m fraudguard.train

Pipeline:
1. Generate raw transactions and build behavioural features.
2. Split by *time* (train -> validation -> test), as a bank would in practice:
   a random split would let the model peek at the future.
3. Train a logistic regression baseline and a gradient-boosted tree model.
4. Pick the alert threshold on the validation set by minimising business cost
   (missed fraud costs the transaction amount, a false alarm costs a fixed review fee).
5. Report on the untouched test set and save the best model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraudguard import __version__
from fraudguard.data import generate_transactions
from fraudguard.features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_behavioural_features

ROOT = Path(__file__).resolve().parents[2]
REVIEW_COST = 10.0  # dollars per false alarm (analyst time + customer friction)


def build_models(seed: int = 42) -> dict[str, Pipeline]:
    onehot = OneHotEncoder(handle_unknown="ignore")
    logreg = Pipeline(
        [
            ("prep", ColumnTransformer([("num", StandardScaler(), NUMERIC_FEATURES), ("cat", onehot, CATEGORICAL_FEATURES)])),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    gbm = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", "passthrough", NUMERIC_FEATURES),
                        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
                    ]
                ),
            ),
            (
                "clf",
                # Settings chosen on the validation period. Re-weighting classes ("balanced")
                # made this model worse, so the imbalance is handled by the cost-based threshold instead.
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=300,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=seed,
                ),
            ),
        ]
    )
    return {"logistic_regression": logreg, "gradient_boosting": gbm}


def cost_optimal_threshold(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray) -> tuple[float, float]:
    """Return the threshold with the lowest total cost, and that cost."""
    best_t, best_cost = 0.5, float("inf")
    for t in np.unique(np.r_[np.linspace(0.01, 0.99, 99), np.linspace(0.99, 0.999, 10)]):
        flagged = scores >= t
        missed = amounts[(y == 1) & ~flagged].sum()
        false_alarms = ((y == 0) & flagged).sum() * REVIEW_COST
        cost = missed + false_alarms
        if cost < best_cost:
            best_t, best_cost = float(t), float(cost)
    return best_t, best_cost


def evaluate(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, threshold: float) -> dict:
    flagged = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, flagged, labels=[0, 1]).ravel()
    fraud_value = amounts[y == 1].sum()
    caught_value = amounts[(y == 1) & (flagged == 1)].sum()
    cost = amounts[(y == 1) & (flagged == 0)].sum() + fp * REVIEW_COST
    return {
        "pr_auc": round(float(average_precision_score(y, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
        "threshold": round(threshold, 3),
        "precision": round(float(precision_score(y, flagged, zero_division=0)), 4),
        "recall": round(float(recall_score(y, flagged)), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "fraud_dollars_caught_pct": round(float(caught_value / fraud_value), 4) if fraud_value else None,
        "total_cost": round(float(cost), 2),
        "cost_if_no_model": round(float(fraud_value), 2),
    }


def main(n_customers: int = 2000, n_days: int = 120, seed: int = 42, out_dir: Path = ROOT) -> dict:
    raw = generate_transactions(n_customers=n_customers, n_days=n_days, seed=seed)
    df = add_behavioural_features(raw)

    day = (df["timestamp"] - df["timestamp"].min()).dt.days
    train = df[day < int(n_days * 0.6)]
    val = df[(day >= int(n_days * 0.6)) & (day < int(n_days * 0.75))]
    test = df[day >= int(n_days * 0.75)]

    summary = {
        "version": __version__,
        "data": {
            "transactions": len(df),
            "customers": int(df.customer_id.nunique()),
            "fraud_rate": round(float(df.is_fraud.mean()), 5),
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
        },
        "review_cost_per_alert": REVIEW_COST,
        "models": {},
    }

    fitted = {}
    for name, model in build_models(seed).items():
        model.fit(train[ALL_FEATURES], train["is_fraud"])
        val_scores = model.predict_proba(val[ALL_FEATURES])[:, 1]
        threshold, _ = cost_optimal_threshold(val["is_fraud"].to_numpy(), val_scores, val["amount"].to_numpy())
        test_scores = model.predict_proba(test[ALL_FEATURES])[:, 1]
        summary["models"][name] = {
            "val_pr_auc": round(float(average_precision_score(val["is_fraud"], val_scores)), 4),
            **evaluate(test["is_fraud"].to_numpy(), test_scores, test["amount"].to_numpy(), threshold),
        }
        fitted[name] = (model, threshold, test_scores)
        print(f"{name}: {summary['models'][name]}")

    # Choose on validation, never on test, so the test numbers stay an honest estimate.
    best = max(summary["models"], key=lambda k: summary["models"][k]["val_pr_auc"])
    summary["best_model"] = best
    model, threshold, _ = fitted[best]

    # Save the model bundle used by the API and demo.
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "threshold": threshold, "features": ALL_FEATURES, "model_name": best, "version": __version__},
        out_dir / "models" / "fraud_model.joblib",
    )

    # Reports.
    fig_dir = out_dir / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    y_test = test["is_fraud"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (_, _, scores) in fitted.items():
        p, r, _ = precision_recall_curve(y_test, scores)
        ax.plot(r, p, label=f"{name.replace('_', ' ')} (PR-AUC {summary['models'][name]['pr_auc']:.3f})")
    ax.axhline(y_test.mean(), color="grey", ls="--", lw=1, label=f"random ({y_test.mean():.3f})")
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_title("Precision-recall on the held-out test period")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "pr_curve.png", dpi=150)
    plt.close(fig)

    sample = test.sample(n=min(20000, len(test)), random_state=seed)
    imp = permutation_importance(
        model, sample[ALL_FEATURES], sample["is_fraud"], scoring="average_precision", n_repeats=5, random_state=seed
    )
    importance = pd.Series(imp.importances_mean, index=ALL_FEATURES).sort_values()
    summary["feature_importance"] = {k: round(float(v), 4) for k, v in importance.sort_values(ascending=False).items()}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    importance.plot.barh(ax=ax, color="#3b6fb6")
    ax.set_xlabel("Drop in PR-AUC when feature is shuffled")
    ax.set_title(f"What the {best.replace('_', ' ')} model relies on")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    with open(out_dir / "reports" / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Best model: {best}. Saved to models/fraud_model.joblib and reports/.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument("--customers", type=int, default=2000)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(n_customers=args.customers, n_days=args.days, seed=args.seed)
