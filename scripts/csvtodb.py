"""Upload clean CSV dataset to SQL database.

Use environment variable DATABASE_URL, for example:
    export DATABASE_URL='postgresql+psycopg2://user:password@localhost:5432/db_capstone'
    python scripts/csvtodb.py --input dataset/data_biceps_clean.csv --table data_biceps --if-exists replace

Do not hardcode database passwords in source code.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from config import CLEAN_DATASET_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload CSV dataset to database.")
    parser.add_argument("--input", type=Path, default=CLEAN_DATASET_PATH)
    parser.add_argument("--table", default="data_biceps")
    parser.add_argument("--if-exists", choices=["fail", "replace", "append"], default="append")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.database_url:
        raise EnvironmentError("DATABASE_URL is not set. Do not hardcode credentials in the script.")
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    engine = create_engine(args.database_url)
    df.to_sql(args.table, engine, if_exists=args.if_exists, index=False)
    print(f"INFO: uploaded {len(df)} rows to table '{args.table}' with if_exists='{args.if_exists}'.")


if __name__ == "__main__":
    main()
