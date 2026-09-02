from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.phishing.ingestion import sha256_file
from app.phishing.models import hash_keras_artifact
from app.utils import read_json


class CheckpointCompatibilityError(ValueError):
    pass


class PhishingOOFCheckpointStore:
    """Atomic, hash-verified checkpoints for model/fold/seed OOF units."""

    def __init__(
        self,
        *,
        root: Path,
        fingerprint: str,
        run_id: str,
        expected_units: int,
        contract: dict[str, Any],
        preflight: dict[str, Any],
    ) -> None:
        self.root = root
        self.units_root = root / "checkpoints" / "units"
        self.manifest_path = root / "checkpoint_manifest.json"
        self.fingerprint = fingerprint
        self.run_id = run_id
        self.expected_units = expected_units
        self.invalid_units: list[dict[str, str]] = []
        self.root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            manifest = read_json(self.manifest_path)
            if manifest.get("fingerprint") != fingerprint:
                raise CheckpointCompatibilityError(
                    "El checkpoint existe, pero su dataset o configuración no coincide con la corrida solicitada."
                )
            if manifest.get("contract") != contract:
                raise CheckpointCompatibilityError("El contrato del checkpoint cambió; no es seguro reanudarlo.")
            self.manifest = manifest
            self.manifest["resumeCount"] = int(self.manifest.get("resumeCount", 0)) + 1
            self.manifest["lastResumedAt"] = _now()
            self.manifest["status"] = "in_progress"
        else:
            self.manifest = {
                "schemaVersion": "1.0.0",
                "runId": run_id,
                "status": "in_progress",
                "createdAt": _now(),
                "updatedAt": _now(),
                "fingerprint": fingerprint,
                "contract": contract,
                "expectedUnits": expected_units,
                "completedUnits": 0,
                "completedUnitKeys": [],
                "resumeCount": 0,
                "preflight": preflight,
                "testSetLocked": True,
                "testSetUsed": False,
            }
        self._persist_manifest()

    @property
    def resume_count(self) -> int:
        return int(self.manifest.get("resumeCount", 0))

    def load_unit(
        self,
        *,
        seed: int,
        fold: int,
        model_id: str,
        sample_ids: list[str],
        labels: np.ndarray,
        tokenizer_sha256: str,
    ) -> dict[str, Any] | None:
        unit_path = self._unit_manifest_path(seed, fold, model_id)
        if not unit_path.exists():
            return None
        key = _unit_key(seed, fold, model_id)
        try:
            unit = read_json(unit_path)
            if unit.get("fingerprint") != self.fingerprint or unit.get("unitKey") != key:
                raise ValueError("identidad o fingerprint incompatible")
            if unit.get("tokenizerSha256") != tokenizer_sha256:
                raise ValueError("tokenizador incompatible")
            predictions_path = self._prediction_path(seed, fold, model_id)
            model_path = self._model_path(seed, fold, model_id)
            prediction_hash, prediction_bytes = sha256_file(predictions_path)
            model_hash, model_bytes = hash_keras_artifact(model_path)
            artifacts = unit.get("artifacts", {})
            if artifacts.get("predictions", {}).get("sha256") != prediction_hash:
                raise ValueError("hash de predicciones inválido")
            if artifacts.get("predictions", {}).get("bytes") != prediction_bytes:
                raise ValueError("tamaño de predicciones inválido")
            if artifacts.get("model", {}).get("sha256") != model_hash:
                raise ValueError("hash del modelo inválido")
            if artifacts.get("model", {}).get("bytes") != model_bytes:
                raise ValueError("tamaño del modelo inválido")
            frame = pd.read_csv(predictions_path, dtype={"sample_id": str})
            if list(frame.columns) != ["sample_id", "is_phishing", "probability"]:
                raise ValueError("contrato de predicciones inválido")
            if frame["sample_id"].tolist() != sample_ids:
                raise ValueError("cobertura de muestras incompatible")
            if not np.array_equal(frame["is_phishing"].to_numpy(dtype=np.int32), labels):
                raise ValueError("etiquetas incompatibles")
            probabilities = frame["probability"].to_numpy(dtype=np.float64)
            if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
                raise ValueError("probabilidades fuera de contrato")
            return {"metrics": unit["metrics"], "probabilities": probabilities, "unit": unit}
        except (OSError, KeyError, ValueError, TypeError, pd.errors.ParserError) as exc:
            self.invalid_units.append({"unitKey": key, "reason": str(exc)})
            return None

    def save_unit(
        self,
        *,
        seed: int,
        fold: int,
        model_id: str,
        sample_ids: list[str],
        labels: np.ndarray,
        probabilities: np.ndarray,
        metrics: dict[str, Any],
        tokenizer_sha256: str,
        model_sha256: str,
        model_bytes: int,
    ) -> dict[str, Any]:
        key = _unit_key(seed, fold, model_id)
        predictions_path = self._prediction_path(seed, fold, model_id)
        prediction_frame = pd.DataFrame({
            "sample_id": sample_ids,
            "is_phishing": labels.astype(np.int32),
            "probability": probabilities.astype(np.float64),
        })
        _atomic_write_csv(predictions_path, prediction_frame)
        prediction_hash, prediction_bytes = sha256_file(predictions_path)
        model_path = self._model_path(seed, fold, model_id)
        actual_model_hash, actual_model_bytes = hash_keras_artifact(model_path)
        if actual_model_hash != model_sha256 or actual_model_bytes != model_bytes:
            raise ValueError("El modelo guardado no coincide con el hash reportado por el clasificador.")
        unit = {
            "schemaVersion": "1.0.0",
            "unitKey": key,
            "fingerprint": self.fingerprint,
            "completedAt": _now(),
            "seed": int(seed),
            "fold": int(fold),
            "modelId": model_id,
            "tokenizerSha256": tokenizer_sha256,
            "rows": len(sample_ids),
            "metrics": metrics,
            "artifacts": {
                "model": {
                    "path": str(model_path.resolve()),
                    "sha256": actual_model_hash,
                    "bytes": actual_model_bytes,
                },
                "predictions": {
                    "path": str(predictions_path.resolve()),
                    "sha256": prediction_hash,
                    "bytes": prediction_bytes,
                    "rows": len(prediction_frame),
                },
            },
        }
        _atomic_write_json(self._unit_manifest_path(seed, fold, model_id), unit)
        completed = set(self.manifest.get("completedUnitKeys", []))
        completed.add(key)
        self.manifest["completedUnitKeys"] = sorted(completed)
        self.manifest["completedUnits"] = len(completed)
        self._persist_manifest()
        return unit

    def model_path(self, seed: int, fold: int, model_id: str) -> Path:
        path = self._model_path(seed, fold, model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def mark_completed(self, *, final_manifest_path: Path, final_manifest_sha256: str) -> None:
        self.manifest["status"] = "completed"
        self.manifest["completedAt"] = _now()
        self.manifest["completedUnits"] = self.expected_units
        self.manifest["finalManifest"] = {
            "path": str(final_manifest_path.resolve()),
            "sha256": final_manifest_sha256,
        }
        self._persist_manifest()

    def _persist_manifest(self) -> None:
        self.manifest["updatedAt"] = _now()
        _atomic_write_json(self.manifest_path, self.manifest)

    def _unit_manifest_path(self, seed: int, fold: int, model_id: str) -> Path:
        return self.units_root / f"seed_{seed}" / f"fold_{fold}" / f"{model_id}.json"

    def _prediction_path(self, seed: int, fold: int, model_id: str) -> Path:
        return self.root / "checkpoint_predictions" / f"seed_{seed}" / f"fold_{fold}" / f"{model_id}.csv"

    def _model_path(self, seed: int, fold: int, model_id: str) -> Path:
        return self.root / "models" / f"seed_{seed}" / f"fold_{fold}" / f"{model_id}.keras"


def _unit_key(seed: int, fold: int, model_id: str) -> str:
    return f"seed={seed}|fold={fold}|model={model_id}"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
