from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.energy.freeze import audit_energy_freeze_readiness, create_energy_freeze_package
from app.energy.models import THESIS_MODEL_IDS
from app.energy.revalidation import run_energy_optimized_stacking_revalidation


def _artifact(path: Path) -> dict:
    payload = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


class EnergyRevalidationFreezeTests(unittest.TestCase):
    def test_complete_lineage_can_be_frozen_without_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run" / "energy_oof_v1"
            root.mkdir(parents=True)
            base_path, data_status = self._fixture(root)
            result = run_energy_optimized_stacking_revalidation(base_manifest_path=base_path, bootstrap_iterations=300)
            revalidation_path = Path(result["manifest"]["path"])
            self.assertEqual(result["protocol"]["independentComparisonFold"], 4)
            self.assertNotIn(4, result["protocol"]["selectionFolds"])
            self.assertEqual(set(result["independentComparison"]["ranking"]), {*THESIS_MODEL_IDS, "optimized_stacking"})
            self.assertEqual(result["independentComparison"]["seedCount"], 5)
            self.assertEqual(len(result["independentComparison"]["seedMetrics"]), 30)
            self.assertFalse(result["protocol"]["testSetUsed"])
            self.assertTrue(result["xai"]["items"])
            freeze_root = Path(directory) / "energy_freezes"
            with patch("app.energy.freeze.get_energy_dataset_status", return_value=data_status), patch("app.energy.freeze.RESULTS_DIR", Path(directory)):
                audit = audit_energy_freeze_readiness(base_manifest_path=base_path, revalidation_manifest_path=revalidation_path, verify_artifacts=True)
                self.assertTrue(audit["ready"], audit["failedChecks"])
                frozen = create_energy_freeze_package(base_manifest_path=base_path, revalidation_manifest_path=revalidation_path)
                self.assertEqual(frozen["status"], "frozen_pre_test")
                self.assertFalse(frozen["testPolicy"]["testSetUsed"])
                self.assertTrue((freeze_root / frozen["freezeId"] / "freeze_seal.json").is_file())

    def _fixture(self, root: Path) -> tuple[Path, dict]:
        rng = np.random.default_rng(7)
        rows = []
        timestamps = pd.date_range("2024-01-01", periods=5 * 48, freq="h", tz="UTC")
        for seed in (42, 101, 202, 303, 404):
            for fold in range(5):
                for offset in range(48):
                    timestamp = timestamps[fold * 48 + offset]
                    actual = 100.0 + 10.0 * np.sin(offset / 8)
                    row = {"seed": seed, "fold": fold, "timestamp": str(timestamp), "actual": actual}
                    for index, model in enumerate(THESIS_MODEL_IDS):
                        row[model] = actual + rng.normal(0, 1.0 + index * 0.15)
                    rows.append(row)
        oof_path = root / "oof_predictions.csv"
        pd.DataFrame(rows).to_csv(oof_path, index=False)
        dataset_path = root / "opsd.csv"
        dataset_path.write_text("timestamp,target\n2024-01-01,100\n", encoding="utf-8")
        dataset_artifact = _artifact(dataset_path)
        metrics = [
            {"seed": seed, "fold": fold, "modelId": model, "rmse": 1.0, "mae": 0.8, "smape": 0.01, "r2": 0.9, "trainTimeSeconds": 1.0, "inferenceTimeMsPerSample": 0.1}
            for seed in (42, 101, 202, 303, 404) for fold in range(5) for model in THESIS_MODEL_IDS
        ]
        pairs = [{"seed": seed, "fold": fold} for seed in (42, 101, 202, 303, 404) for fold in range(1, 5)]
        manifest = {
            "runId": "energy-thesis-test",
            "status": "thesis_candidate",
            "dataset": {"sourceLineage": {"type": "real_versioned_silver", "verified": True, "datasetId": "opsd-test", **dataset_artifact}},
            "walkForward": {"seeds": [42, 101, 202, 303, 404], "folds": [{"foldId": fold} for fold in range(5)], "testSetUsed": False},
            "baseModels": list(THESIS_MODEL_IDS),
            "baseFoldMetrics": metrics,
            "ensembles": {"aggregate": [{"ensembleId": value} for value in ("mean", "weighted_mean", "stacking_gradient_boosting")]},
            "comparisonScope": {"fairComparison": True, "pairs": pairs},
            "anomalies": {"labelType": "estimated", "classificationMetricsAvailable": False},
            "aggregate": {"testSetEvaluatedOnce": False},
            "artifacts": {"oofPredictions": str(oof_path.resolve())},
            "artifactIntegrity": {"oofPredictions": _artifact(oof_path)},
        }
        base_path = root / "oof_manifest.json"
        base_path.write_text(json.dumps(manifest), encoding="utf-8")
        data_status = {"readyForThesisPilot": True, "datasetId": "opsd-test", "silver": {"sha256": dataset_artifact["sha256"]}}
        return base_path, data_status


if __name__ == "__main__":
    unittest.main()
