from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from app.config import MODELS_DIR, RESULTS_DIR, SILVER_DIR, ensure_dirs
from app.phishing.ingestion import sha256_file
from app.utils import read_json, write_json


PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
PAD_ID = 0
UNKNOWN_ID = 1
REQUIRED_COLUMNS = {
    "canonical_url",
    "registrable_domain",
    "is_phishing",
    "split",
}
EXPECTED_OUTER_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class CharacterTokenizer:
    token_to_id: dict[str, int]
    max_length: int
    truncation: str = "head_tail"

    @classmethod
    def fit(
        cls,
        urls: Iterable[str],
        *,
        max_vocabulary: int = 256,
        length_percentile: float = 99.0,
        max_length_cap: int = 512,
    ) -> "CharacterTokenizer":
        values = [str(url) for url in urls]
        if not values:
            raise ValueError("No se puede ajustar el tokenizador sin URLs de entrenamiento.")
        if max_vocabulary < 4:
            raise ValueError("max_vocabulary debe reservar espacio para PAD, UNK y caracteres.")
        if not 50.0 <= length_percentile <= 100.0:
            raise ValueError("length_percentile debe estar entre 50 y 100.")
        if max_length_cap < 8:
            raise ValueError("max_length_cap debe ser al menos 8.")

        frequencies = Counter(character for url in values for character in url)
        ordered = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
        characters = [character for character, _ in ordered[: max_vocabulary - 2]]
        token_to_id = {PAD_TOKEN: PAD_ID, UNKNOWN_TOKEN: UNKNOWN_ID}
        token_to_id.update({character: index + 2 for index, character in enumerate(characters)})
        lengths = np.asarray([len(url) for url in values], dtype=np.int32)
        learned_length = int(math.ceil(float(np.percentile(lengths, length_percentile, method="higher"))))
        return cls(token_to_id=token_to_id, max_length=max(8, min(learned_length, max_length_cap)))

    def encode(self, urls: Iterable[str]) -> np.ndarray:
        values = [str(url) for url in urls]
        encoded = np.full((len(values), self.max_length), PAD_ID, dtype=np.int32)
        for row_index, url in enumerate(values):
            token_ids = [self.token_to_id.get(character, UNKNOWN_ID) for character in url]
            token_ids = self._truncate(token_ids)
            encoded[row_index, : len(token_ids)] = token_ids
        return encoded

    def statistics(self, urls: Iterable[str]) -> dict[str, Any]:
        values = [str(url) for url in urls]
        total_characters = sum(len(url) for url in values)
        unknown_characters = sum(
            1 for url in values for character in url if character not in self.token_to_id
        )
        truncated = sum(len(url) > self.max_length for url in values)
        return {
            "rows": len(values),
            "characters": int(total_characters),
            "unknownCharacters": int(unknown_characters),
            "unknownCharacterRate": float(unknown_characters / total_characters) if total_characters else 0.0,
            "truncatedRows": int(truncated),
            "truncatedRowRate": float(truncated / len(values)) if values else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "1.0.0",
            "type": "deterministic_character_tokenizer",
            "specialTokens": {"pad": PAD_ID, "unknown": UNKNOWN_ID},
            "tokenToId": self.token_to_id,
            "vocabularySize": len(self.token_to_id),
            "maxLength": self.max_length,
            "padding": "post",
            "truncation": self.truncation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CharacterTokenizer":
        return cls(
            token_to_id={str(token): int(identifier) for token, identifier in payload["tokenToId"].items()},
            max_length=int(payload["maxLength"]),
            truncation=str(payload.get("truncation", "head_tail")),
        )

    def _truncate(self, token_ids: list[int]) -> list[int]:
        if len(token_ids) <= self.max_length:
            return token_ids
        if self.truncation != "head_tail":
            return token_ids[: self.max_length]
        head_length = int(math.ceil(self.max_length * 0.60))
        tail_length = self.max_length - head_length
        return token_ids[:head_length] + token_ids[-tail_length:]


def prepare_phishing_sequence_protocol(
    *,
    silver_path: Path | None = None,
    metadata_path: Path | None = None,
    assignments_path: Path | None = None,
    tokenizer_directory: Path | None = None,
    audit_path: Path | None = None,
    manifest_path: Path | None = None,
    folds: int = 5,
    seed: int = 42,
    max_vocabulary: int = 256,
    length_percentile: float = 99.0,
    max_length_cap: int = 512,
) -> dict[str, Any]:
    if folds < 3:
        raise ValueError("OOF requiere al menos tres folds.")
    ensure_dirs()
    source = silver_path or (SILVER_DIR / "phishing.csv")
    source_metadata = metadata_path or (SILVER_DIR / "phishing.metadata.json")
    assignment_destination = assignments_path or (SILVER_DIR / "phishing_oof_assignments.csv")
    tokenizers = tokenizer_directory or (MODELS_DIR / "phishing" / "tokenizers")
    audit_destination = audit_path or (RESULTS_DIR / "phishing_sequence_audit.json")
    manifest_destination = manifest_path or (SILVER_DIR / "phishing_sequence_manifest.json")
    if not source.exists():
        raise FileNotFoundError("No existe el dataset silver de phishing.")

    source_hash, source_bytes = sha256_file(source)
    metadata = read_json(source_metadata) if source_metadata.exists() else {}
    expected_hash = metadata.get("silver", {}).get("sha256")
    if expected_hash and expected_hash != source_hash:
        raise ValueError("El hash del dataset silver no coincide con sus metadatos.")
    frame = pd.read_csv(source)
    validation = validate_phishing_sequence_frame(frame)
    train = frame[frame["split"] == "train"].copy().reset_index(drop=True)
    validation_frame = frame[frame["split"] == "validation"].copy().reset_index(drop=True)
    test = frame[frame["split"] == "test"].copy().reset_index(drop=True)

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignments = np.full(len(train), -1, dtype=np.int16)
    tokenizers.mkdir(parents=True, exist_ok=True)
    fold_audits: list[dict[str, Any]] = []
    for fold, (fit_indices, holdout_indices) in enumerate(
        splitter.split(train["canonical_url"], train["is_phishing"], train["registrable_domain"])
    ):
        if np.any(assignments[holdout_indices] != -1):
            raise ValueError("Una observación recibió más de un fold OOF.")
        assignments[holdout_indices] = fold
        fit_frame = train.iloc[fit_indices]
        holdout_frame = train.iloc[holdout_indices]
        overlap = set(fit_frame["registrable_domain"]) & set(holdout_frame["registrable_domain"])
        fold_tokenizer = CharacterTokenizer.fit(
            fit_frame["canonical_url"],
            max_vocabulary=max_vocabulary,
            length_percentile=length_percentile,
            max_length_cap=max_length_cap,
        )
        tokenizer_path = tokenizers / f"fold_{fold}.json"
        write_json(tokenizer_path, {
            **fold_tokenizer.to_dict(),
            "fitPolicy": "oof_fold_training_only",
            "fitFold": fold,
            "fitRows": int(len(fit_frame)),
        })
        tokenizer_hash, tokenizer_bytes = sha256_file(tokenizer_path)
        fold_audits.append({
            "fold": fold,
            "fitRows": int(len(fit_frame)),
            "holdoutRows": int(len(holdout_frame)),
            "fitGroups": int(fit_frame["registrable_domain"].nunique()),
            "holdoutGroups": int(holdout_frame["registrable_domain"].nunique()),
            "holdoutNegative": int((holdout_frame["is_phishing"] == 0).sum()),
            "holdoutPositive": int((holdout_frame["is_phishing"] == 1).sum()),
            "groupOverlap": int(len(overlap)),
            "tokenizer": {
                "path": str(tokenizer_path.resolve()),
                "sha256": tokenizer_hash,
                "bytes": tokenizer_bytes,
                "vocabularySize": len(fold_tokenizer.token_to_id),
                "maxLength": fold_tokenizer.max_length,
                "holdoutStatistics": fold_tokenizer.statistics(holdout_frame["canonical_url"]),
            },
        })

    if np.any(assignments < 0):
        raise ValueError("No todas las observaciones de entrenamiento recibieron un fold OOF.")
    train["oof_fold"] = assignments
    train["sample_id"] = train["canonical_url"].map(_sample_id)
    if train["sample_id"].duplicated().any():
        raise ValueError("Los sample_id OOF no son únicos.")
    assignment_destination.parent.mkdir(parents=True, exist_ok=True)
    train[["sample_id", "oof_fold"]].sort_values("sample_id").to_csv(assignment_destination, index=False)
    assignment_hash, assignment_bytes = sha256_file(assignment_destination)

    outer_tokenizer = CharacterTokenizer.fit(
        train["canonical_url"],
        max_vocabulary=max_vocabulary,
        length_percentile=length_percentile,
        max_length_cap=max_length_cap,
    )
    outer_tokenizer_path = tokenizers / "outer_train.json"
    write_json(outer_tokenizer_path, {
        **outer_tokenizer.to_dict(),
        "fitPolicy": "outer_train_only",
        "fitRows": int(len(train)),
        "forUseOn": ["validation"],
        "testPolicy": "locked; do not encode or evaluate until the final selected pipeline is frozen",
    })
    outer_tokenizer_hash, outer_tokenizer_bytes = sha256_file(outer_tokenizer_path)

    created_at = datetime.now(timezone.utc).isoformat()
    leakage_passed = all(item["groupOverlap"] == 0 for item in fold_audits)
    all_folds_binary = all(item["holdoutNegative"] > 0 and item["holdoutPositive"] > 0 for item in fold_audits)
    readiness_reasons: list[str] = []
    if not leakage_passed:
        readiness_reasons.append("Existe solapamiento de dominios entre fit y holdout en al menos un fold.")
    if not all_folds_binary:
        readiness_reasons.append("Al menos un fold no contiene ambas clases.")
    if len(set(assignments.tolist())) != folds:
        readiness_reasons.append("No se generaron todos los folds solicitados.")

    audit = {
        "schemaVersion": "1.0.0",
        "datasetId": metadata.get("datasetId", f"phishing-{source_hash[:12]}"),
        "createdAt": created_at,
        "sourceSilver": {"path": str(source.resolve()), "sha256": source_hash, "bytes": source_bytes},
        "outerSplit": validation,
        "testLock": {
            "locked": True,
            "rows": int(len(test)),
            "groups": int(test["registrable_domain"].nunique()),
            "usedForVocabulary": False,
            "usedForLengthSelection": False,
            "usedForOOF": False,
            "usedForThresholdSelection": False,
            "evaluated": False,
        },
        "oof": {
            "strategy": "StratifiedGroupKFold",
            "folds": folds,
            "seed": seed,
            "eligibleOuterSplit": "train",
            "coverageRows": int((assignments >= 0).sum()),
            "expectedRows": int(len(train)),
            "groupLeakagePassed": leakage_passed,
            "allHoldoutsContainBothClasses": all_folds_binary,
            "foldsAudit": fold_audits,
        },
        "tokenization": {
            "type": "character",
            "normalizationInput": "canonical_url from the audited silver dataset",
            "vocabularyFitSplit": "train only",
            "maxLengthFitSplit": "train only",
            "lengthPercentile": length_percentile,
            "maxLengthCap": max_length_cap,
            "outerTokenizer": {
                "path": str(outer_tokenizer_path.resolve()),
                "sha256": outer_tokenizer_hash,
                "bytes": outer_tokenizer_bytes,
                "vocabularySize": len(outer_tokenizer.token_to_id),
                "maxLength": outer_tokenizer.max_length,
            },
            "trainStatistics": outer_tokenizer.statistics(train["canonical_url"]),
            "validationStatistics": outer_tokenizer.statistics(validation_frame["canonical_url"]),
            "testStatisticsNotComputed": True,
        },
        "readiness": {
            "readyForBaseModelTraining": not readiness_reasons,
            "reasons": readiness_reasons,
            "testSetUsed": False,
        },
    }
    write_json(audit_destination, audit)
    audit_hash, audit_bytes = sha256_file(audit_destination)
    manifest = {
        "schemaVersion": "1.0.0",
        "datasetId": audit["datasetId"],
        "createdAt": created_at,
        "configuration": {
            "folds": folds,
            "seed": seed,
            "maxVocabulary": max_vocabulary,
            "lengthPercentile": length_percentile,
            "maxLengthCap": max_length_cap,
        },
        "artifacts": {
            "assignments": {
                "path": str(assignment_destination.resolve()),
                "sha256": assignment_hash,
                "bytes": assignment_bytes,
                "rows": int(len(train)),
            },
            "outerTokenizer": audit["tokenization"]["outerTokenizer"],
            "foldTokenizers": [item["tokenizer"] for item in fold_audits],
            "audit": {
                "path": str(audit_destination.resolve()),
                "sha256": audit_hash,
                "bytes": audit_bytes,
            },
        },
        "readyForBaseModelTraining": audit["readiness"]["readyForBaseModelTraining"],
    }
    write_json(manifest_destination, manifest)
    return {"manifest": manifest, "audit": audit}


def validate_phishing_sequence_frame(frame: pd.DataFrame) -> dict[str, Any]:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"El silver de phishing no contiene: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("El silver de phishing está vacío.")
    labels = set(pd.to_numeric(frame["is_phishing"], errors="raise").astype(int).unique())
    if labels != {0, 1}:
        raise ValueError("El contrato binario requiere ambas etiquetas: 0 y 1.")
    splits = set(frame["split"].astype(str).unique())
    if splits != EXPECTED_OUTER_SPLITS:
        raise ValueError("El contrato exige train, validation y test.")
    if frame["canonical_url"].isna().any() or frame["registrable_domain"].isna().any():
        raise ValueError("No se permiten URLs o dominios registrados faltantes.")
    if frame["canonical_url"].duplicated().any():
        raise ValueError("canonical_url debe ser única en el silver.")

    groups = {
        split: set(part["registrable_domain"].astype(str))
        for split, part in frame.groupby("split")
    }
    overlaps = {
        "trainValidation": len(groups["train"] & groups["validation"]),
        "trainTest": len(groups["train"] & groups["test"]),
        "validationTest": len(groups["validation"] & groups["test"]),
    }
    if any(overlaps.values()):
        raise ValueError("Existe fuga de dominios registrados entre particiones externas.")
    distribution = {
        split: {
            "rows": int(len(part)),
            "negative": int((part["is_phishing"] == 0).sum()),
            "positive": int((part["is_phishing"] == 1).sum()),
            "groups": int(part["registrable_domain"].nunique()),
        }
        for split, part in frame.groupby("split")
    }
    if any(item["negative"] == 0 or item["positive"] == 0 for item in distribution.values()):
        raise ValueError("Cada partición externa debe contener ambas clases.")
    return {
        "strategy": "preassigned_stratified_group_holdout",
        "groupKey": "registrable_domain",
        "distribution": distribution,
        "groupOverlapCounts": overlaps,
        "passed": not any(overlaps.values()),
    }


def get_phishing_sequence_status() -> dict[str, Any]:
    manifest_path = SILVER_DIR / "phishing_sequence_manifest.json"
    audit_path = RESULTS_DIR / "phishing_sequence_audit.json"
    if not manifest_path.exists() or not audit_path.exists():
        return {
            "available": False,
            "readyForBaseModelTraining": False,
            "message": "El protocolo de secuencias y folds OOF todavía no ha sido preparado.",
        }
    manifest = read_json(manifest_path)
    audit = read_json(audit_path)
    dataset_metadata_path = SILVER_DIR / "phishing.metadata.json"
    dataset_metadata = read_json(dataset_metadata_path) if dataset_metadata_path.exists() else {}
    artifacts_valid = True
    for artifact in _iter_artifacts(manifest.get("artifacts", {})):
        path = Path(artifact.get("path", ""))
        if not path.is_file():
            artifacts_valid = False
            break
        digest, size = sha256_file(path)
        if digest != artifact.get("sha256") or size != artifact.get("bytes"):
            artifacts_valid = False
            break
    readiness = audit.get("readiness", {})
    current_silver = dataset_metadata.get("silver", {})
    lineage_current = bool(dataset_metadata) and (
        manifest.get("datasetId") == dataset_metadata.get("datasetId")
        and audit.get("sourceSilver", {}).get("sha256") == current_silver.get("sha256")
    )
    return {
        "available": artifacts_valid,
        "integrityVerified": artifacts_valid,
        "readyForBaseModelTraining": artifacts_valid and lineage_current and bool(readiness.get("readyForBaseModelTraining")),
        "lineageCurrent": lineage_current,
        "datasetScientificReady": bool(dataset_metadata.get("readyForThesisTraining")),
        "message": None if lineage_current else "Las secuencias pertenecen a una versión anterior del dataset; deben regenerarse.",
        "datasetId": manifest.get("datasetId"),
        "createdAt": manifest.get("createdAt"),
        "configuration": manifest.get("configuration", {}),
        "outerSplit": audit.get("outerSplit", {}),
        "testLock": audit.get("testLock", {}),
        "oof": audit.get("oof", {}),
        "tokenization": audit.get("tokenization", {}),
        "readiness": readiness,
    }


def _iter_artifacts(artifacts: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ("assignments", "outerTokenizer", "audit"):
        artifact = artifacts.get(key)
        if isinstance(artifact, dict):
            yield artifact
    for artifact in artifacts.get("foldTokenizers", []):
        if isinstance(artifact, dict):
            yield artifact


def _sample_id(canonical_url: str) -> str:
    return hashlib.sha256(str(canonical_url).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara tokenización por caracteres y folds OOF de phishing.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-vocabulary", type=int, default=256)
    parser.add_argument("--length-percentile", type=float, default=99.0)
    parser.add_argument("--max-length-cap", type=int, default=512)
    args = parser.parse_args()
    result = prepare_phishing_sequence_protocol(
        folds=args.folds,
        seed=args.seed,
        max_vocabulary=args.max_vocabulary,
        length_percentile=args.length_percentile,
        max_length_cap=args.max_length_cap,
    )
    print(json.dumps({
        "datasetId": result["manifest"]["datasetId"],
        "readyForBaseModelTraining": result["manifest"]["readyForBaseModelTraining"],
        "folds": result["manifest"]["configuration"]["folds"],
        "maxLength": result["audit"]["tokenization"]["outerTokenizer"]["maxLength"],
        "testLocked": result["audit"]["testLock"]["locked"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
