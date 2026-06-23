from __future__ import annotations

import pandas as pd


def clean_opsd(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    timestamp_col = "utc_timestamp" if "utc_timestamp" in data.columns else data.columns[0]
    data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce", utc=True)
    data = data.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    numeric_cols = [col for col in data.columns if col != timestamp_col]
    for column in numeric_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data[numeric_cols] = data[numeric_cols].interpolate(limit_direction="both").fillna(0)
    return data.rename(columns={timestamp_col: "timestamp"}).reset_index(drop=True)


def pick_target_column(frame: pd.DataFrame) -> str:
    preferred = [
        "DE_load_actual_entsoe_transparency",
        "load_actual_entsoe_transparency",
    ]
    for column in preferred:
        if column in frame.columns:
            return column
    numeric_cols = [col for col in frame.columns if col != "timestamp" and pd.api.types.is_numeric_dtype(frame[col])]
    if not numeric_cols:
        raise ValueError("No hay columnas numericas para serie temporal.")
    return numeric_cols[0]


def build_windows(frame: pd.DataFrame, target_col: str, window: int = 24):
    values = frame[target_col].astype(float).to_numpy()
    dates = frame["timestamp"].astype(str).to_numpy()
    x_rows = []
    y_rows = []
    y_dates = []
    for idx in range(window, len(values)):
        x_rows.append(values[idx - window:idx])
        y_rows.append(values[idx])
        y_dates.append(dates[idx])
    return x_rows, y_rows, y_dates
