"""Exploratory data analysis: saves charts and summary tables to reports/eda/."""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .data_utils import (EDU_MAP, MARRIAGE_MAP, NUMERIC, PAY_COLS, REPORTS_DIR,
                         SEX_MAP, TARGET, load_dataset)

sns.set_theme(style="whitegrid")
OUT = REPORTS_DIR / "eda"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)


def run_eda(df):
    OUT.mkdir(parents=True, exist_ok=True)

    # Summary tables
    df.describe().T.round(2).to_csv(OUT / "summary_statistics.csv")
    print("Rows:", len(df), "| Default rate:", f"{df[TARGET].mean():.2%}",
          "| Missing values:", int(df.isna().sum().sum()))

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET].value_counts().sort_index()
    ax.bar(["No default (0)", "Default (1)"], counts.values, color=["#4C9F70", "#D1495B"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom")
    ax.set_title("Class distribution")
    ax.set_ylim(0, counts.max() * 1.18)
    _save(fig, "01_class_distribution.png")

    # 2. Default rate by demographic group
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, mapping) in zip(axes, [("SEX", SEX_MAP), ("EDUCATION", EDU_MAP), ("MARRIAGE", MARRIAGE_MAP)]):
        rate = df.groupby(col)[TARGET].mean().rename(index=mapping)
        sns.barplot(x=rate.index, y=rate.values, ax=ax, color="#3D6EA8")
        ax.set_title(f"Default rate by {col.title()}")
        ax.set_ylabel("Default rate")
        ax.tick_params(axis="x", rotation=20)
    _save(fig, "02_default_rate_by_group.png")

    # 3. Histograms of credit limit & age by outcome
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in [(axes[0], "LIMIT_BAL", "Credit limit"), (axes[1], "AGE", "Age")]:
        sns.histplot(data=df, x=col, hue=TARGET, bins=40, stat="density", common_norm=False,
                     element="step", kde=True, ax=ax, palette=["#4C9F70", "#D1495B"])
        ax.set_title(f"{title} distribution by default status")
    _save(fig, "03_limit_age_histograms.png")

    # 3b. Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, col in zip(axes, ["LIMIT_BAL", "AGE"]):
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax, palette=["#4C9F70", "#D1495B"], hue=TARGET, legend=False)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No default", "Default"])
        ax.set_title(f"{col} by default status")
        ax.set_xlabel("")
    _save(fig, "03b_boxplots.png")

    # 4. Repayment status vs default rate
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for ax, col in zip(axes.ravel(), PAY_COLS):
        rate = df.groupby(col)[TARGET].mean()
        sns.barplot(x=rate.index, y=rate.values, ax=ax, color="#C8553D")
        ax.set_title(f"Default rate by {col}")
        ax.set_xlabel("Repayment status (-2..8)")
    _save(fig, "04_repayment_status_vs_default.png")

    # 5. Correlation heatmap
    fig, ax = plt.subplots(figsize=(13, 10))
    corr = df[NUMERIC + [TARGET]].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation matrix")
    _save(fig, "05_correlation_heatmap.png")

    # 6. Correlation with target
    fig, ax = plt.subplots(figsize=(7, 8))
    tc = corr[TARGET].drop(TARGET).sort_values()
    ax.barh(tc.index, tc.values, color=["#D1495B" if v > 0 else "#4C9F70" for v in tc.values])
    ax.set_title("Correlation of each feature with default")
    _save(fig, "06_target_correlation.png")

    print(f"EDA charts saved to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run exploratory data analysis")
    ap.add_argument("--data", default=None, help="Path to dataset (default: auto-detect in data/)")
    run_eda(load_dataset(ap.parse_args().data))
