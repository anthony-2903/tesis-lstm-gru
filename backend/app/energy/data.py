from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class EnergyValidationReport:
    rows: int
    timestamp_column: str
    target_column: str
    feature_columns: list[str]
    start_at: str
    end_at: str
    duplicate_timestamps: int
    missing_values_after_cleaning: int
    demo_mode: bool
    inferred_frequency: str
    non_hourly_intervals: int


@dataclass(frozen=True)
class SequencePartition:
    x: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    row_start: str
    row_end: str


@dataclass(frozen=True)
class PreparedEnergyData:
    train: SequencePartition
    validation: SequencePartition
    test: SequencePartition
    feature_columns: list[str]
    target_column: str
    target_feature_index: int
    feature_imputer: SimpleImputer
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    split_metadata: dict[str, Any]


@dataclass(frozen=True)
class PreparedEnergyFold:
    fold_id: int
    train: SequencePartition
    validation: SequencePartition
    feature_columns: list[str]
    target_column: str
    target_feature_index: int
    feature_imputer: SimpleImputer
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    metadata: dict[str, Any]


def validate_energy_frame(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    target_column: str,
    strict: bool = True,
    minimum_rows: int = 8760,
    expected_frequency: str = "hourly",
) -> tuple[pd.DataFrame, EnergyValidationReport]:
    if frame.empty:
        raise ValueError("El dataset energetico esta vacio.")
    missing_columns = [column for column in [timestamp_column, target_column] if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas energeticas obligatorias: {', '.join(missing_columns)}")

    data = frame.copy()
    data[timestamp_column] = pd.to_datetime(data[timestamp_column], errors="coerce", utc=True)
    data = data.dropna(subset=[timestamp_column]).sort_values(timestamp_column).reset_index(drop=True)
    duplicate_timestamps = int(data[timestamp_column].duplicated().sum())
    if duplicate_timestamps:
        raise ValueError("El primer pipeline energetico requiere una sola serie sin timestamps duplicados.")

    candidate_features = [column for column in data.columns if column != timestamp_column]
    for column in candidate_features:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    feature_columns = [column for column in candidate_features if data[column].notna().any()]
    if target_column not in feature_columns:
        raise ValueError(f"La variable objetivo {target_column} no contiene valores numericos.")
    if not feature_columns:
        raise ValueError("No existen variables numericas para preparar secuencias.")

    # The target cannot be imputed. Feature imputation is deliberately deferred
    # until after the chronological split so its statistics come from train only.
    data = data.dropna(subset=[target_column]).reset_index(drop=True)
    timestamp_deltas = data[timestamp_column].diff().dropna()
    non_hourly_intervals = int((timestamp_deltas != pd.Timedelta(hours=1)).sum())
    inferred_frequency = "hourly" if non_hourly_intervals == 0 else "irregular"
    if strict and expected_frequency == "hourly" and non_hourly_intervals:
        raise ValueError(
            f"El protocolo de tesis exige continuidad horaria; existen {non_hourly_intervals} intervalos irregulares."
        )
    if strict and len(data) < minimum_rows:
        raise ValueError(
            f"El protocolo de tesis exige al menos {minimum_rows} filas; se recibieron {len(data)}. "
            "Use strict=False solo para pruebas demo."
        )
    if len(data) < 64:
        raise ValueError("Se requieren al menos 64 filas para construir train, validation y test sin solapamiento.")

    report = EnergyValidationReport(
        rows=int(len(data)),
        timestamp_column=timestamp_column,
        target_column=target_column,
        feature_columns=feature_columns,
        start_at=str(data[timestamp_column].iloc[0]),
        end_at=str(data[timestamp_column].iloc[-1]),
        duplicate_timestamps=duplicate_timestamps,
        missing_values_after_cleaning=int(data[feature_columns].isna().sum().sum()),
        demo_mode=not strict,
        inferred_frequency=inferred_frequency,
        non_hourly_intervals=non_hourly_intervals,
    )
    return data[[timestamp_column, *feature_columns]], report


def prepare_energy_sequences(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    target_column: str,
    feature_columns: list[str],
    window: int,
    horizon: int,
    gap_steps: int = 24,
) -> PreparedEnergyData:
    if window < 2 or horizon < 1 or gap_steps < 0:
        raise ValueError("window, horizon y gap_steps tienen valores invalidos.")

    row_count = len(frame)
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    validation_start = train_end + gap_steps
    test_start = validation_end + gap_steps

    train_rows = frame.iloc[:train_end].copy()
    validation_rows = frame.iloc[validation_start:validation_end].copy()
    test_rows = frame.iloc[test_start:].copy()
    required_partition_rows = window + horizon
    for name, partition in [("train", train_rows), ("validation", validation_rows), ("test", test_rows)]:
        if len(partition) < required_partition_rows:
            raise ValueError(
                f"La particion {name} tiene {len(partition)} filas; requiere al menos {required_partition_rows}. "
                "Reduzca window/gap o use mas datos."
            )

    feature_imputer = SimpleImputer(strategy="median")
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    imputed_train = feature_imputer.fit_transform(train_rows[feature_columns])
    feature_scaler.fit(imputed_train)
    target_scaler.fit(train_rows[[target_column]])

    train = _build_partition(
        train_rows,
        timestamp_column=timestamp_column,
        target_column=target_column,
        feature_columns=feature_columns,
        feature_imputer=feature_imputer,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        window=window,
        horizon=horizon,
    )
    validation = _build_partition(
        validation_rows,
        timestamp_column=timestamp_column,
        target_column=target_column,
        feature_columns=feature_columns,
        feature_imputer=feature_imputer,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        window=window,
        horizon=horizon,
    )
    test = _build_partition(
        test_rows,
        timestamp_column=timestamp_column,
        target_column=target_column,
        feature_columns=feature_columns,
        feature_imputer=feature_imputer,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        window=window,
        horizon=horizon,
    )

    split_metadata = {
        "strategy": "chronological_holdout_with_gap",
        "window": window,
        "horizon": horizon,
        "gapSteps": gap_steps,
        "trainRows": int(len(train_rows)),
        "validationRows": int(len(validation_rows)),
        "testRows": int(len(test_rows)),
        "trainRange": [str(train_rows[timestamp_column].iloc[0]), str(train_rows[timestamp_column].iloc[-1])],
        "validationRange": [str(validation_rows[timestamp_column].iloc[0]), str(validation_rows[timestamp_column].iloc[-1])],
        "testRange": [str(test_rows[timestamp_column].iloc[0]), str(test_rows[timestamp_column].iloc[-1])],
    }
    return PreparedEnergyData(
        train=train,
        validation=validation,
        test=test,
        feature_columns=feature_columns,
        target_column=target_column,
        target_feature_index=feature_columns.index(target_column),
        feature_imputer=feature_imputer,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        split_metadata=split_metadata,
    )


def prepare_energy_oof_folds(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    target_column: str,
    feature_columns: list[str],
    window: int,
    horizon: int,
    n_splits: int = 5,
    gap_steps: int = 24,
    development_fraction: float = 0.85,
) -> list[PreparedEnergyFold]:
    if n_splits < 3:
        raise ValueError("Walk-forward requiere al menos tres folds.")
    if not 0.5 <= development_fraction < 1.0:
        raise ValueError("development_fraction debe estar entre 0.5 y 1.0.")

    development_end = int(len(frame) * development_fraction)
    development = frame.iloc[:development_end].copy()
    initial_train_rows = max(int(len(development) * 0.50), window + horizon + gap_steps)
    remaining_rows = len(development) - initial_train_rows - (gap_steps * n_splits)
    validation_size = remaining_rows // n_splits
    if validation_size < window + horizon:
        raise ValueError(
            "No hay suficientes filas para walk-forward. Reduzca window/gap/folds o utilice un dataset mayor."
        )

    folds: list[PreparedEnergyFold] = []
    for fold_id in range(n_splits):
        train_end = initial_train_rows + fold_id * (validation_size + gap_steps)
        validation_start = train_end + gap_steps
        validation_end = validation_start + validation_size
        train_rows = development.iloc[:train_end].copy()
        validation_rows = development.iloc[validation_start:validation_end].copy()
        folds.append(
            _prepare_fold(
                fold_id=fold_id,
                train_rows=train_rows,
                validation_rows=validation_rows,
                timestamp_column=timestamp_column,
                target_column=target_column,
                feature_columns=feature_columns,
                window=window,
                horizon=horizon,
                gap_steps=gap_steps,
            )
        )
    return folds


def _prepare_fold(
    *,
    fold_id: int,
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
    feature_columns: list[str],
    window: int,
    horizon: int,
    gap_steps: int,
) -> PreparedEnergyFold:
    feature_imputer = SimpleImputer(strategy="median")
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(feature_imputer.fit_transform(train_rows[feature_columns]))
    target_scaler.fit(train_rows[[target_column]])
    train = _build_partition(
        train_rows,
        timestamp_column=timestamp_column,
        target_column=target_column,
        feature_columns=feature_columns,
        feature_imputer=feature_imputer,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        window=window,
        horizon=horizon,
    )
    validation = _build_partition(
        validation_rows,
        timestamp_column=timestamp_column,
        target_column=target_column,
        feature_columns=feature_columns,
        feature_imputer=feature_imputer,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        window=window,
        horizon=horizon,
    )
    metadata = {
        "foldId": fold_id,
        "strategy": "expanding_window",
        "gapSteps": gap_steps,
        "trainRows": int(len(train_rows)),
        "validationRows": int(len(validation_rows)),
        "trainRange": [str(train_rows[timestamp_column].iloc[0]), str(train_rows[timestamp_column].iloc[-1])],
        "validationRange": [str(validation_rows[timestamp_column].iloc[0]), str(validation_rows[timestamp_column].iloc[-1])],
    }
    return PreparedEnergyFold(
        fold_id=fold_id,
        train=train,
        validation=validation,
        feature_columns=feature_columns,
        target_column=target_column,
        target_feature_index=feature_columns.index(target_column),
        feature_imputer=feature_imputer,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        metadata=metadata,
    )


def _build_partition(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    target_column: str,
    feature_columns: list[str],
    feature_imputer: SimpleImputer,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    window: int,
    horizon: int,
) -> SequencePartition:
    features = feature_scaler.transform(feature_imputer.transform(frame[feature_columns])).astype(np.float32)
    targets = target_scaler.transform(frame[[target_column]]).reshape(-1).astype(np.float32)
    timestamps = frame[timestamp_column].astype(str).to_numpy()
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    y_timestamps: list[str] = []
    for end in range(window, len(frame) - horizon + 1):
        target_index = end + horizon - 1
        x_rows.append(features[end - window:end])
        y_rows.append(float(targets[target_index]))
        y_timestamps.append(str(timestamps[target_index]))
    return SequencePartition(
        x=np.asarray(x_rows, dtype=np.float32),
        y=np.asarray(y_rows, dtype=np.float32),
        timestamps=np.asarray(y_timestamps),
        row_start=str(frame[timestamp_column].iloc[0]),
        row_end=str(frame[timestamp_column].iloc[-1]),
    )
