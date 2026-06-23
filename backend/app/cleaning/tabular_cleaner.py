from __future__ import annotations

import re

import pandas as pd


def normalize_column(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "columna"


def clean_tabular(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data.columns = [normalize_column(col) for col in data.columns]
    data = data.replace({"": pd.NA, "ND": pd.NA, "N/D": pd.NA, "-": pd.NA})
    data = data.drop_duplicates()
    for column in data.columns:
        if data[column].dtype == object:
            parsed = pd.to_numeric(data[column].astype(str).str.replace(",", ".", regex=False), errors="coerce")
            if parsed.notna().mean() > 0.7:
                data[column] = parsed
            else:
                data[column] = data[column].astype("string").str.strip()
    return data.reset_index(drop=True)


def summarize_quality(frame: pd.DataFrame) -> dict[str, object]:
    original_rows = len(frame)
    cleaned = clean_tabular(frame)
    return {
        "original_rows": original_rows,
        "cleaned_rows": len(cleaned.dropna(how="all")),
        "rows_removed": max(original_rows - len(cleaned.dropna(how="all")), 0),
        "duplicates": int(frame.duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "columns": list(cleaned.columns),
    }
