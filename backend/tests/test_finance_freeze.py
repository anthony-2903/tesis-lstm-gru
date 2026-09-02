from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from app.finance.freeze import (
    FinanceFreezeGateError,
    audit_finance_freeze_readiness,
    create_finance_freeze_package,
)
from app.finance.models import FINANCE_MODEL_IDS
from app.finance.stacking import META_MODEL_IDS
from app.phishing.ingestion import sha256_file
from app.utils import write_json


SEEDS = [42, 101, 202, 303, 404]
FOLDS = [0, 1, 2, 3, 4]


class FinanceFreezeGateTests(unittest.TestCase):
    def test_demo_lineage_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._qualified_lineage(Path(directory))
            base = self._read(paths["base"])
            base["protocol"] = "demo"
            base["status"] = "demo"
            write_json(paths["base"], base)
            audit = audit_finance_freeze_readiness(
                validation_manifest_path=paths["validation"],
                diversity_manifest_path=paths["diversity"],
            )
            self.assertFalse(audit["ready"])
            self.assertIn("base_protocol", audit["blockingCheckIds"])
            self.assertIn("base_status", audit["blockingCheckIds"])
            with self.assertRaises(FinanceFreezeGateError):
                create_finance_freeze_package(
                    validation_manifest_path=paths["validation"],
                    diversity_manifest_path=paths["diversity"],
                    output_dir=Path(directory) / "freeze",
                )

    def test_qualified_lineage_creates_immutable_reusable_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._qualified_lineage(root)
            audit = audit_finance_freeze_readiness(
                validation_manifest_path=paths["validation"],
                diversity_manifest_path=paths["diversity"],
                verify_artifacts=True,
            )
            self.assertTrue(audit["ready"], audit["blockingCheckIds"])
            self.assertGreater(len(audit["artifactInventory"]), 5)

            first = create_finance_freeze_package(
                validation_manifest_path=paths["validation"],
                diversity_manifest_path=paths["diversity"],
                output_dir=root / "freeze",
            )
            second = create_finance_freeze_package(
                validation_manifest_path=paths["validation"],
                diversity_manifest_path=paths["diversity"],
                output_dir=root / "freeze",
            )
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(first["freezeId"], second["freezeId"])
            self.assertTrue(first["sealVerified"])
            self.assertFalse(first["testAuthorization"]["granted"])
            self.assertEqual(first["testAuthorization"]["maximumEvaluations"], 1)
            self.assertEqual(first["configuration"]["selectedCandidateId"], "tcn")

    @staticmethod
    def _read(path: Path) -> dict:
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    def _qualified_lineage(self, root: Path) -> dict[str, Path]:
        artifacts = root / "artifacts"
        artifacts.mkdir(parents=True)

        def artifact(name: str, content: bytes = b"verified-finance-artifact") -> dict:
            path = artifacts / name
            path.write_bytes(content)
            digest, size = sha256_file(path)
            return {"path": str(path.resolve()), "sha256": digest, "bytes": size}

        source = artifact("finance_transactions.csv", b"transaction_id,is_fraud\n1,0\n2,1\n")
        sequences = artifact("finance_sequences.npz")
        features = artifact("finance_causal_features.csv")
        assignments = artifact("finance_oof_assignments.csv")
        sequence_manifest_path = artifacts / "finance_sequence_manifest.json"
        write_json(sequence_manifest_path, {
            "schemaVersion": "1.0.0",
            "datasetId": "finance-real-fixture",
            "source": source,
            "readiness": {"readyForBaseModelPilot": True, "readyForThesisTraining": True},
            "testLock": {"locked": True, "encoded": False, "evaluated": False},
            "artifacts": {"features": features, "sequences": sequences, "assignments": assignments},
        })

        oof = artifact("oof.csv")
        fold_metrics = artifact("fold_metrics.json")
        base_path = root / "oof_manifest.json"
        base_metrics = [
            {"seed": seed, "fold": fold, "modelId": model_id}
            for seed in SEEDS
            for fold in FOLDS
            for model_id in FINANCE_MODEL_IDS
        ]
        write_json(base_path, {
            "runId": "finance-thesis-fixture",
            "status": "thesis_base_models_candidate",
            "protocol": "thesis",
            "configuration": {"seeds": SEEDS, "modelIds": list(FINANCE_MODEL_IDS), "demoMaxRowsPerFold": None},
            "dataset": {
                "datasetId": "finance-real-fixture",
                "sequencePath": str(sequences["path"]),
                "oofRowsAvailablePerSeed": 2,
                "oofRowsUsedPerSeed": 2,
                "demoSubset": False,
            },
            "validation": {"oofFolds": FOLDS, "testSetLocked": True, "testSetEncoded": False, "testSetUsed": False},
            "baseModels": list(FINANCE_MODEL_IDS),
            "baseFoldMetrics": base_metrics,
            "artifacts": {
                "oofProbabilities": {**oof, "rows": 10},
                "foldMetrics": {**fold_metrics, "rows": len(base_metrics)},
            },
        })

        stacking_predictions = artifact("stacking_predictions.csv")
        stacking_metrics = artifact("stacking_metrics.json")
        stacking_path = root / "stacking_manifest.json"
        write_json(stacking_path, {
            "runId": "finance-thesis-fixture-stacking",
            "status": "meta_oof_candidate",
            "baseRun": {"runId": "finance-thesis-fixture", "manifestPath": str(base_path.resolve())},
            "metaFeatures": {"baseProbabilityColumns": [f"probability_{model_id}" for model_id in FINANCE_MODEL_IDS]},
            "validation": {
                "strictTemporalCrossFit": True,
                "metaDependencyLeakageControlled": True,
                "futureFoldsUsed": False,
                "testSetLocked": True,
                "testSetEncoded": False,
                "testSetUsed": False,
            },
            "candidateAggregate": [{"candidateId": meta_model_id} for meta_model_id in META_MODEL_IDS],
            "artifacts": {"predictions": stacking_predictions, "metrics": stacking_metrics},
        })

        validation_predictions = artifact("validation_predictions.csv")
        validation_metrics = artifact("validation_metrics.json")
        frozen_selection = artifact("frozen_selection.json")
        validation_path = root / "temporal_validation_manifest.json"
        comparison = [
            {
                "candidateId": candidate_id,
                "calibratedThreshold": 0.25,
                "calibrationMethod": "isotonic",
            }
            for candidate_id in (*FINANCE_MODEL_IDS, "stacking")
        ]
        write_json(validation_path, {
            "runId": "finance-thesis-fixture-validation",
            "status": "validation_selection_candidate",
            "baseRun": {"runId": "finance-thesis-fixture", "manifestPath": str(base_path.resolve())},
            "stackingRun": {
                "manifestPath": str(stacking_path.resolve()),
                "sourceCandidateId": "stacking_gradient_boosting",
                "ablationId": "full",
            },
            "dataset": {
                "datasetId": "finance-real-fixture",
                "kind": "real_world_curated_financial_dataset",
                "trainRowsAvailable": 100,
                "trainRowsUsed": 100,
                "validationRowsAvailable": 40,
                "validationRowsUsed": 40,
                "demoTrainSubset": False,
                "demoValidationSubset": False,
            },
            "training": {
                "records": [
                    {"seed": seed, "modelId": model_id}
                    for seed in SEEDS
                    for model_id in FINANCE_MODEL_IDS
                ]
            },
            "calibration": {"selectionLabelsUsed": False},
            "selection": {"winnerCandidateId": "tcn"},
            "comparison": comparison,
            "validation": {
                "externalRealDatasetUsed": True,
                "chronologicalCalibrationBeforeSelection": True,
                "readyForFinalTestEvaluation": True,
                "testSetLocked": True,
                "testSetEncoded": False,
                "testSetUsed": False,
            },
            "freeze": {"eligibleForFinalTestEvaluation": True},
            "artifacts": {
                "predictionsBySeed": validation_predictions,
                "metrics": validation_metrics,
                "frozenSelection": frozen_selection,
            },
        })

        diversity_report = artifact("diversity_report.json")
        diversity_path = root / "diversity_ablation_manifest.json"
        ablation_ids = ["full", *(f"without_{model_id}" for model_id in FINANCE_MODEL_IDS)]
        write_json(diversity_path, {
            "runId": "finance-thesis-fixture-diversity",
            "status": "ablation_candidate",
            "validationRun": {"runId": "finance-thesis-fixture-validation"},
            "ablation": {
                "configurations": [{"ablationId": value} for value in ablation_ids],
                "metrics": [
                    {"ablationId": ablation_id, "metaModelId": meta_model_id}
                    for ablation_id in ablation_ids
                    for meta_model_id in META_MODEL_IDS
                ],
            },
            "stability": {
                "seedCount": 5,
                "seedLevelInferenceAvailable": True,
                "bootstrapUnit": "calendar_day_temporal_block",
                "bootstrapIterations": 500,
            },
            "recommendation": {"ablationAccepted": False, "recommendedAblationId": "full"},
            "validation": {"testSetLocked": True, "testSetEncoded": False, "testSetUsed": False},
            "artifacts": {"report": diversity_report},
        })
        revalidation_artifact = artifact("revalidation_comparison.json")
        revalidation_path = root / "optimized_stacking_revalidation_v1" / "optimized_stacking_revalidation_manifest.json"
        write_json(revalidation_path, {
            "runId": "finance-thesis-fixture-revalidation",
            "status": "optimized_stacking_revalidation_candidate",
            "diversityRun": {"runId": "finance-thesis-fixture-diversity"},
            "protocol": {"sameRowsForAllCandidates": True},
            "stackingSelection": {"selectedAblationId": "full", "selectedMetaModelId": "stacking_gradient_boosting"},
            "independentComparison": {
                "winnerCandidateId": "tcn",
                "metrics": [{"candidateId": value, "calibratedThreshold": 0.25, "calibrationMethod": "isotonic"} for value in (*FINANCE_MODEL_IDS, "stacking")],
                "seedMetrics": [
                    {"seed": seed, "candidateId": value}
                    for seed in (42, 101, 202, 303, 404)
                    for value in (*FINANCE_MODEL_IDS, "stacking")
                ],
            },
            "validation": {
                "independentComparisonLabelsUsedForCalibrationThresholdOrSelection": False,
                "testSetLocked": True,
                "testSetEncoded": False,
                "testSetUsed": False,
            },
            "artifacts": {"independentComparison": revalidation_artifact},
        })
        return {"base": base_path, "stacking": stacking_path, "validation": validation_path, "diversity": diversity_path}


if __name__ == "__main__":
    unittest.main()
