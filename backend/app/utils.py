from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        parsed = float(str(value).replace(",", "."))
        if math.isfinite(parsed):
            return parsed
    except Exception:
        pass
    return default


def infer_data_types(frame: pd.DataFrame) -> dict[str, str]:
    types: dict[str, str] = {}
    for column in frame.columns:
        series = frame[column].dropna()
        if series.empty:
            types[column] = "vacio"
        elif pd.api.types.is_numeric_dtype(series):
            types[column] = "numero"
        elif pd.api.types.is_datetime64_any_dtype(series):
            types[column] = "fecha"
        else:
            parsed = pd.to_datetime(series.head(50), errors="coerce")
            types[column] = "fecha" if parsed.notna().mean() > 0.8 else "texto"
    return types


def entropy(text: str) -> float:
    if not text:
        return 0.0
    values, counts = np.unique(list(text), return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())
