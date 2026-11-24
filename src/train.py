"""Train and evaluate Logistic Regression and Decision Tree classifiers.

Usage:  python -m src.train [--data PATH] [--test-size 0.2] [--no-balance]
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, classification_report)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from .data_utils import (CATEGORICAL, FEATURES, MODELS_DIR, NUMERIC, REPORTS_DIR, TARGET,
                         find_dataset, load_dataset)

sns.set_theme(style="whitegrid")


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),                                   # feature scaling
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),         # SEX / EDUCATION / MARRIAGE
    ])


def build_models(class_weight, seed):
    lr = GridSearchCV(
        Pipeline([("prep", build_preprocessor()),
                  ("clf", LogisticRegression(max_iter=3000, class_weight=class_weight, random_state=seed))]),
        {"clf__C": [0.01, 0.1, 1, 10]},
        scoring="f1", cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=-1,
    )
    dt = GridSearchCV(
        Pipeline([("prep", build_preprocessor()),
                  ("clf", DecisionTreeClassifier(class_weight=class_weight, random_state=seed))]),
        {"clf__max_depth": [3, 4, 5, 6, 8], "clf__min_samples_leaf": [20, 50, 100]},
        scoring="f1", cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=-1,
    )
    return {"Logistic Regression": lr, "Decision Tree": dt}


def evaluate(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return proba, pred, {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def feature_names(pipe):
    return pipe.named_steps["prep"].get_feature_names_out()


def save_plots(fitted, results, y_test, out):
    out.mkdir(parents=True, exist_ok=True)

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, r["pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No default", "Default"], yticklabels=["No default", "Default"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=130)
    plt.close(fig)

    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {r['metrics']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=130)
    plt.close(fig)

    # Metric comparison
    df = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T
    df = df[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    ax = df.T.plot(kind="bar", figsize=(8, 4.5), color=["#3D6EA8", "#C8553D"], rot=0)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison (test set)")
    plt.tight_layout()
    plt.savefig(out / "metric_comparison.png", dpi=130)
    plt.close()

    # Logistic regression coefficients
    lr = fitted["Logistic Regression"].best_estimator_
    coefs = pd.Series(lr.named_steps["clf"].coef_[0], index=feature_names(lr))
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index[:15]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index.str.replace("num__", "").str.replace("cat__", ""), top.values,
            color=["#D1495B" if v > 0 else "#4C9F70" for v in top.values])
    ax.set_title("Logistic Regression - top 15 coefficients\n(red = raises default risk)")
    fig.tight_layout()
    fig.savefig(out / "lr_coefficients.png", dpi=130)
    plt.close(fig)

    # Decision tree importances
    dt = fitted["Decision Tree"].best_estimator_
    imp = pd.Series(dt.named_steps["clf"].feature_importances_, index=feature_names(dt))
    imp = imp.sort_values(ascending=False).head(15).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index.str.replace("num__", "").str.replace("cat__", ""), imp.values, color="#3D6EA8")
    ax.set_title("Decision Tree - top 15 feature importances")
    fig.tight_layout()
    fig.savefig(out / "dt_feature_importance.png", dpi=130)
    plt.close(fig)

    # Tree structure (top 3 levels)
    fig, ax = plt.subplots(figsize=(20, 9))
    plot_tree(dt.named_steps["clf"], max_depth=3, feature_names=[n.split("__", 1)[1] for n in feature_names(dt)],
              class_names=["No default", "Default"], filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("Decision Tree - first 3 levels")
    fig.tight_layout()
    fig.savefig(out / "decision_tree.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Train credit default models")
    ap.add_argument("--data", default=None, help="Dataset path (default: auto-detect in data/)")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-balance", action="store_true", help="Disable class_weight='balanced'")
    args = ap.parse_args()

    df = load_dataset(args.data)
    source = str(args.data or find_dataset())
    X, y = df[FEATURES], df[TARGET]
    print(f"Loaded {len(df):,} rows from {source} | default rate {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed)

    class_weight = None if args.no_balance else "balanced"
    models = build_models(class_weight, args.seed)

    fitted, results = {}, {}
    for name, search in models.items():
        print(f"\nTraining {name} (5-fold CV grid search, scoring=F1) ...")
        search.fit(X_train, y_train)
        fitted[name] = search
        proba, pred, metrics = evaluate(search, X_test, y_test)
        results[name] = {"proba": proba, "pred": pred, "metrics": metrics}
        print(f"  best params: {search.best_params_}")
        print(classification_report(y_test, pred, target_names=["No default", "Default"], digits=3))
        print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted["Logistic Regression"].best_estimator_, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(fitted["Decision Tree"].best_estimator_, MODELS_DIR / "decision_tree.joblib")

    summary = {
        "data_source": source,
        "n_rows": int(len(df)),
        "default_rate": float(y.mean()),
        "class_weight": class_weight,
        "test_size": args.test_size,
        "models": {n: {"metrics": {k: round(float(v), 4) for k, v in r["metrics"].items()},
                       "best_params": {k: (v.item() if hasattr(v, "item") else v)
                                       for k, v in fitted[n].best_params_.items()}}
                   for n, r in results.items()},
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    save_plots(fitted, results, y_test, REPORTS_DIR)

    print("\n=== Test-set summary ===")
    print(pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T.round(3).to_string())
    print(f"\nModels saved to {MODELS_DIR}\nReports saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
