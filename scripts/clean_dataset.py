from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import (
    CLEAN_DATASET_PATH,
    DATASET_COLUMNS,
    FEATURE_COLUMNS,
    MAX_ABSURD_ANGLE_RANGE,
    MAX_ABSURD_ROM,
    MAX_ABSURD_STD,
    METADATA_COLUMNS,
    MIN_ABSURD_ROM,
    RAW_DATASET_PATH,
    REJECTED_SAMPLE_LOG,
    TARGET_COLUMNS,
    VALID_BINARY_LABELS,
    VALID_ERROR_TYPES,
    VISIBILITY_THRESHOLD,
)


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Backward compatibility with old uploaded script.
    if "exercise" in df.columns and "exercise_type" not in df.columns:
        df = df.rename(columns={"exercise": "exercise_type"})

    # Add missing metadata with safe defaults.
    defaults = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "subject_id": "unknown_subject",
        "session_id": "unknown_session",
        "camera_position": "unknown_camera",
        "lighting_condition": "unknown_lighting",
        "exercise_type": "biceps",
        "active_arm": "unknown_arm",
        "notes": "",
    }

    if "sample_id" not in df.columns:
        df.insert(0, "sample_id", range(1, len(df) + 1))

    for col in METADATA_COLUMNS + TARGET_COLUMNS:
        if col not in df.columns:
            if col == "label":
                df[col] = np.nan
            elif col == "error_type":
                df[col] = np.nan
            else:
                df[col] = defaults.get(col, "unknown")

    # Normalize labels.
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df.loc[~df["label"].isin(VALID_BINARY_LABELS), "label"] = np.nan

    # If error_type is missing, derive it safely from label.
    df["error_type"] = df["error_type"].astype(str).str.strip().str.lower()
    df.loc[df["error_type"].isin({"", "nan", "none"}), "error_type"] = np.nan
    df.loc[(df["label"] == "correct") & (df["error_type"].isna()), "error_type"] = "correct"
    df.loc[(df["label"] == "incorrect") & (df["error_type"].isna()), "error_type"] = "unknown_incorrect"
    df.loc[~df["error_type"].isin(VALID_ERROR_TYPES), "error_type"] = "unknown_incorrect"
    df.loc[df["label"] == "correct", "error_type"] = "correct"

    return df


def available_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in FEATURE_COLUMNS if col in df.columns]


def validate_row(row: pd.Series, features: List[str]) -> Tuple[bool, str]:
    if pd.isna(row.get("label")):
        return False, "missing_or_invalid_label"

    if not features:
        return False, "no_feature_columns"

    for col in features:
        value = row.get(col)
        if pd.isna(value) or not np.isfinite(float(value)):
            return False, f"invalid_feature:{col}"

    # Data-quality checks only. Do not use class-separating thresholds here.
    if "mean_visibility" in row.index:
        visibility = float(row["mean_visibility"])
        if visibility < VISIBILITY_THRESHOLD:
            return False, f"low_visibility:{visibility:.3f}"

    if "rom_elbow" in row.index:
        rom = float(row["rom_elbow"])
        if rom < MIN_ABSURD_ROM or rom > MAX_ABSURD_ROM:
            return False, f"absurd_rom:{rom:.3f}"

    for col in ["torso_sway_range", "shoulder_angle_range"]:
        if col in row.index:
            val = float(row[col])
            if val < 0 or val > MAX_ABSURD_ANGLE_RANGE:
                return False, f"absurd_{col}:{val:.3f}"

    if "upper_arm_angle_std" in row.index:
        val = float(row["upper_arm_angle_std"])
        if val < 0 or val > MAX_ABSURD_STD:
            return False, f"absurd_upper_arm_angle_std:{val:.3f}"

    return True, "ok"


def clean_dataset(input_path: Path, output_path: Path, rejected_log_path: Path, shuffle: bool = False) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"INFO: raw rows = {len(df)}")

    df = normalize_schema(df)
    features = available_feature_columns(df)
    if not features:
        raise ValueError("No usable feature columns were found. Check dataset header.")

    print(f"INFO: detected features = {features}")

    # Convert features to numeric.
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    kept_rows = []
    rejected_rows = []
    for idx, row in df.iterrows():
        is_valid, reason = validate_row(row, features)
        if is_valid:
            kept_rows.append(row)
        else:
            rejected = row.to_dict()
            rejected["row_index"] = idx
            rejected["reject_reason"] = reason
            rejected_rows.append(rejected)

    cleaned = pd.DataFrame(kept_rows)

    # Stable order: metadata, detected features, targets, then any extra columns.
    base_cols = [c for c in METADATA_COLUMNS if c in cleaned.columns]
    target_cols = [c for c in TARGET_COLUMNS if c in cleaned.columns]
    ordered = base_cols + features + target_cols
    extras = [c for c in cleaned.columns if c not in ordered]
    cleaned = cleaned[ordered + extras]

    if shuffle:
        cleaned = cleaned.sample(frac=1.0, random_state=42).reset_index(drop=True)
    else:
        cleaned = cleaned.reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    rejected_log_path.parent.mkdir(parents=True, exist_ok=True)
    if rejected_rows:
        pd.DataFrame(rejected_rows).to_csv(rejected_log_path, index=False)
    else:
        with rejected_log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["message"])
            writer.writerow(["no_rejected_rows"])

    print(f"INFO: kept rows     = {len(cleaned)}")
    print(f"INFO: rejected rows = {len(rejected_rows)}")
    print(f"INFO: clean dataset = {output_path}")
    print(f"INFO: rejected log  = {rejected_log_path}")

    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean dataset without threshold leakage.")
    parser.add_argument("--input", type=Path, default=RAW_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=CLEAN_DATASET_PATH)
    parser.add_argument("--rejected-log", type=Path, default=REJECTED_SAMPLE_LOG)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle output only if temporal split is not needed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_dataset(args.input, args.output, args.rejected_log, shuffle=args.shuffle)


if __name__ == "__main__":
    main()
