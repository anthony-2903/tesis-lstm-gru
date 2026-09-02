from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from app.phishing.freeze import (
    FreezeGateError,
    audit_phishing_freeze_readiness,
    create_phishing_freeze_package,
)
from app.phishing.ingestion import sha256_file
from app.phishing.models import PHISHING_MODEL_IDS
from app.phishing.stacking import ENSEMBLE_IDS, META_MODEL_IDS
from app.utils import read_json, write_json


class PhishingFreezeGateTests(unittest.TestCase):
    def test_demo_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _qualified_lineage(Path(directory))
            base = read_json(paths["base"])
            base["status"] = "demo"
            base["protocol"] = "demo"
            base["configuration"]["seeds"] = [42]
            base["dataset"]["demoSubset"] = True
            write_json(paths["base"], base)

            audit = audit_phishing_freeze_readiness(
                validation_manifest_path=paths["validation"],
                diversity_manifest_path=paths["diversity"],
            )

            self.assertFalse(audit["ready"])
            failed_ids = {check["checkId"] for check in audit["failedChecks"]}
            self.assertIn("base_status", failed_ids)
            self.assertIn("base_protocol", failed_ids)
            self.assertIn("five_seeds", failed_ids)
            with self.assertRaises(FreezeGateError):
                create_phishing_freeze_package(
                    validation_manifest_path=paths["validation"],
                    diversity_manifest_path=paths["diversity"],
                    output_dir=Path(directory) / "frozen",
                )

    def test_qualified_lineage_creates_reusable_sealed_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _qualified_lineage(root)
            audit = audit_phishing_freeze_readiness(
                validation_manifest_path=paths["validation"],
                diversity_manifest_path=paths["diversity"],
                verify_artifacts=True,
            )
            self.assertTrue(audit["ready"])
            self.assertTrue(audit["artifactIntegrityVerified"])

            first = create_phishing_freeze_package(
                validation_manifest_path=paths["validation"],
                diversity_manifest_path=paths["diversity"],
                output_dir=root / "frozen",
            )
            second = create_phishing_freeze_package(
                validation_manifest_path=paths["validation"],
                diversity_manifest_path=paths["diversity"],
                output_dir=root / "frozen",
            )

            self.assertEqual(first["status"], "ready_for_single_test_evaluation")
            self.assertTrue(first["immutable"])
            self.assertTrue(first["sealVerified"])
            self.assertFalse(first["testAuthorization"]["granted"])
            self.assertFalse(first["testAuthorization"]["evaluated"])
            self.assertTrue(second["reused"])
            self.assertEqual(first["freezeId"], second["freezeId"])


def _qualified_lineage(root: Path) -> dict[str, Path]:
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    source = root / "phishing.csv"
    source.write_text("canonical_url,is_phishing,split\nhttps://example.test,0,train\n", encoding="utf-8")
    source_hash, source_bytes = sha256_file(source)
    metadata = source.with_suffix(".metadata.json")
    write_json(metadata, {
        "datasetId": "qualified-dataset",
        "readyForThesisTraining": True,
        "silver": {"sha256": source_hash, "bytes": source_bytes},
    })
    seeds = [42, 101, 202, 303, 404]
    folds = [0, 1, 2, 3, 4]
    model_ids = list(PHISHING_MODEL_IDS)

    oof_path = artifacts / "oof.csv"
    oof_path.write_text("fixture", encoding="utf-8")
    oof_hash, oof_bytes = sha256_file(oof_path)
    metrics_path = artifacts / "base_metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")
    metrics_hash, metrics_bytes = sha256_file(metrics_path)
    base_metrics = []
    fold_protocols = []
    for seed in seeds:
        for fold in folds:
            tokenizer_artifact = _file_artifact(artifacts / f"tokenizer_{seed}_{fold}.json", f"tokenizer-{seed}-{fold}")
            fold_protocols.append({"seed": seed, "fold": fold, "tokenizer": tokenizer_artifact})
            for model_id in model_ids:
                model_path = artifacts / f"oof_{seed}_{fold}_{model_id}.keras"
                model_path.write_text(f"{seed}-{fold}-{model_id}", encoding="utf-8")
                model_hash, model_bytes = _keras_hash(model_path)
                base_metrics.append({
                    "seed": seed,
                    "fold": fold,
                    "modelId": model_id,
                    "epochsCompleted": 3,
                    "modelPath": str(model_path.resolve()),
                    "modelSha256": model_hash,
                    "modelSizeBytes": model_bytes,
                })
    base_path = root / "oof_manifest.json"
    write_json(base_path, {
        "runId": "qualified-oof",
        "status": "thesis_base_models_candidate",
        "protocol": "thesis",
        "configuration": {"seeds": seeds},
        "dataset": {
            "sourcePath": str(source.resolve()),
            "sourceSha256": source_hash,
            "sourceBytes": source_bytes,
            "metadataPath": str(metadata.resolve()),
            "rowsUsed": 10,
            "outerTrainRowsAvailable": 10,
            "demoSubset": False,
        },
        "validation": {"oofFolds": folds, "foldProtocols": fold_protocols, "testSetLocked": True, "testSetUsed": False},
        "baseModels": model_ids,
        "baseFoldMetrics": base_metrics,
        "artifacts": {
            "oofProbabilities": {"path": str(oof_path.resolve()), "sha256": oof_hash, "bytes": oof_bytes, "rows": 50},
            "foldMetrics": {"path": str(metrics_path.resolve()), "sha256": metrics_hash, "bytes": metrics_bytes, "rows": 125},
        },
    })

    validation_objects = []
    training_records = []
    for seed in seeds:
        for model_id in model_ids:
            path = artifacts / f"final_{seed}_{model_id}.keras"
            path.write_text(f"final-{seed}-{model_id}", encoding="utf-8")
            digest, size = _keras_hash(path)
            validation_objects.append({"kind": "base_model", "candidateId": model_id, "seed": seed, "path": str(path.resolve()), "sha256": digest, "bytes": size})
            training_records.append({"seed": seed, "modelId": model_id, "selectedEpochsFromOof": 3})
        validation_objects.append({"kind": "ensemble_weights", "candidateId": "weighted_mean", "seed": seed, **_file_artifact(artifacts / f"weights_{seed}.json", f"weights-{seed}")})
        for meta_model_id in META_MODEL_IDS:
            validation_objects.append({"kind": "meta_model", "candidateId": meta_model_id, "seed": seed, **_file_artifact(artifacts / f"meta_{seed}_{meta_model_id}.joblib", f"meta-{seed}-{meta_model_id}")})
    validation_path = root / "external_validation_manifest.json"
    validation_artifacts = {
        "predictionsBySeed": _file_artifact(artifacts / "validation_by_seed.csv", "by-seed"),
        "meanPredictions": _file_artifact(artifacts / "validation_mean.csv", "mean"),
        "metrics": _file_artifact(artifacts / "validation_metrics.json", "metrics"),
        "fittedObjects": validation_objects,
    }
    write_json(validation_path, {
        "runId": "qualified-validation",
        "status": "validation_selection_candidate",
        "baseRun": {"runId": "qualified-oof", "manifestPath": str(base_path.resolve())},
        "dataset": {"sourcePath": str(source.resolve()), "validationRows": 4},
        "training": {"records": training_records},
        "metaFeatures": {"columns": [f"probability_{model_id}" for model_id in model_ids], "validationLabelsUsedForMetaFit": False},
        "selection": {"winnerCandidateId": "mean", "thresholdObjective": "mcc", "leadingStackingCandidateId": "stacking_ridge"},
        "comparison": [
            {"candidateId": candidate_id, "calibratedThreshold": 0.5}
            for candidate_id in [*model_ids, *ENSEMBLE_IDS]
        ],
        "validation": {"outerValidationUsed": True, "sameRowsForAllCandidates": True, "testSetLocked": True, "testFeaturesEncoded": False, "testSetUsed": False},
        "tokenizer": _file_artifact(artifacts / "outer_tokenizer.json", "tokenizer"),
        "artifacts": validation_artifacts,
    })

    ablation_ids = ["full", *(f"without_{model_id}" for model_id in model_ids)]
    diversity_objects = []
    for seed in seeds:
        for ablation_id in ablation_ids:
            for meta_model_id in META_MODEL_IDS:
                diversity_objects.append({
                    "seed": seed,
                    "ablationId": ablation_id,
                    "metaModelId": meta_model_id,
                    **_file_artifact(artifacts / f"ablation_{seed}_{ablation_id}_{meta_model_id}.joblib", f"{seed}-{ablation_id}-{meta_model_id}"),
                })
    diversity_path = root / "diversity_ablation_manifest.json"
    write_json(diversity_path, {
        "runId": "qualified-diversity",
        "status": "ablation_candidate",
        "validationRun": {"runId": "qualified-validation", "manifestPath": str(validation_path.resolve())},
        "ablation": {
            "configurations": [{"ablationId": ablation_id} for ablation_id in ablation_ids],
            "metrics": [{"ablationId": ablation_id, "metaModelId": meta_model_id} for ablation_id in ablation_ids for meta_model_id in META_MODEL_IDS],
        },
        "stability": {"seedCount": 5, "seedLevelInferenceAvailable": True, "bootstrapUnit": "registrable_domain", "bootstrapIterations": 300},
        "recommendation": {"referenceMetaModelId": "stacking_ridge", "recommendedBaseModels": model_ids, "recommendedAblationId": "full"},
        "validation": {"outerValidationUsed": True, "sameRowsForAllAblations": True, "testSetLocked": True, "testFeaturesEncoded": False, "testSetUsed": False},
        "artifacts": {
            "pairwiseDiversity": _file_artifact(artifacts / "pairs.csv", "pairs"),
            "ablationPredictions": _file_artifact(artifacts / "ablation_predictions.csv", "predictions"),
            "report": _file_artifact(artifacts / "ablation_report.json", "report"),
            "fittedObjects": diversity_objects,
        },
    })
    return {"base": base_path, "validation": validation_path, "diversity": diversity_path}


def _file_artifact(path: Path, content: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    digest, size = sha256_file(path)
    return {"path": str(path.resolve()), "sha256": digest, "bytes": size}


def _keras_hash(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    digest = hashlib.sha256(path.name.encode("utf-8") + payload).hexdigest()
    return digest, len(payload)


if __name__ == "__main__":
    unittest.main()
