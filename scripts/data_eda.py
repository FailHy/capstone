"""EDA report generator for movement dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CLEAN_DATASET_PATH, FEATURE_COLUMNS, REPORT_DIR


def save_bar(series: pd.Series, title: str, output: Path, xlabel: str = "", ylabel: str = "count") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = series.fillna("missing").value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_hist_by_label(df: pd.DataFrame, feature: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in sorted(df["label"].dropna().unique()):
        values = pd.to_numeric(df.loc[df["label"] == label, feature], errors="coerce").dropna()
        ax.hist(values, bins=20, alpha=0.45, label=str(label))
    ax.set_title(f"Distribution: {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_correlation(df: pd.DataFrame, features, output: Path) -> None:
    if len(features) < 2:
        return
    corr = df[features].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(6, len(features) * 0.7), max(5, len(features) * 0.6)))
    image = ax.imshow(corr, vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_yticklabels(features)
    ax.set_title("Feature Correlation")
    for i in range(len(features)):
        for j in range(len(features)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run_eda(input_path: Path, output_dir: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = [col for col in FEATURE_COLUMNS if col in df.columns]
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "features_detected": features,
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
    }

    for col in ["label", "error_type", "subject_id", "session_id", "camera_position", "lighting_condition", "exercise_type"]:
        if col in df.columns:
            summary[f"distribution_{col}"] = df[col].fillna("missing").value_counts().to_dict()
            save_bar(df[col], f"Distribution: {col}", output_dir / f"eda_{col}_distribution.png", xlabel=col)

    for feature in features:
        if "label" in df.columns:
            save_hist_by_label(df, feature, output_dir / f"eda_{feature}_by_label.png")

    save_correlation(df, features, output_dir / "eda_feature_correlation.png")

    # Feature-label correlation for binary labels only.
    if "label" in df.columns and set(df["label"].dropna().unique()).issubset({"correct", "incorrect"}):
        temp = df.copy()
        temp["label_encoded"] = temp["label"].map({"correct": 0, "incorrect": 1})
        corr_to_label = temp[features + ["label_encoded"]].corr(numeric_only=True)["label_encoded"].drop("label_encoded")
        summary["feature_correlation_to_label"] = corr_to_label.sort_values(ascending=False).to_dict()

    with (output_dir / "eda_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"INFO: EDA files saved to {output_dir}")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA on clean movement dataset.")
    parser.add_argument("--input", type=Path, default=CLEAN_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_eda(args.input, args.output_dir)


if __name__ == "__main__":
    main()
