from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from app.finance.diversity import run_finance_diversity_ablation
from app.finance.models import FINANCE_MODEL_IDS
from app.finance.revalidation import run_finance_optimized_stacking_revalidation
from app.phishing.ingestion import sha256_file
from app.utils import write_json


class FinanceDiversityAblationTests(unittest.TestCase):
    def test_diversity_and_ablation_preserve_temporal_validation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed = 42

            oof_rows = []
            for index in range(160):
                label = int(index % 8 == 0)
                oof_rows.append({
                    "transaction_id": index + 1,
                    "seed": seed,
                    "is_fraud": label,
                    **self._probabilities(label, index),
                })
            oof_path = root / "oof.csv"
            pd.DataFrame(oof_rows).to_csv(oof_path, index=False)
            oof_hash, oof_bytes = sha256_file(oof_path)
            base_manifest_path = root / "oof_manifest.json"
            write_json(base_manifest_path, {
                "runId": "finance-diversity-fixture",
                "status": "demo",
                "baseModels": list(FINANCE_MODEL_IDS),
                "configuration": {"seeds": [seed]},
                "artifacts": {
                    "oofProbabilities": {
                        "path": str(oof_path.resolve()),
                        "sha256": oof_hash,
                        "bytes": oof_bytes,
                    }
                },
            })

            by_seed_rows = []
            mean_rows = []
            start_timestamp_ns = 1_700_000_000_000_000_000
            for index in range(80):
                label = int(index % 8 == 0)
                probabilities = self._probabilities(label, index + 1_000)
                partition = "calibration" if index < 40 else "selection"
                row = {
                    "transaction_id": 10_001 + index,
                    "timestamp_ns": start_timestamp_ns + index * 21_600_000_000_000,
                    "is_fraud": label,
                    **probabilities,
                }
                by_seed_rows.append({**row, "seed": seed})
                mean_rows.append({
                    **row,
                    "validation_partition": partition,
                    **{
                        f"calibrated_probability_{model_id}": probabilities[f"probability_{model_id}"]
                        for model_id in FINANCE_MODEL_IDS
                    },
                })
            by_seed_path = root / "validation_by_seed.csv"
            mean_path = root / "validation_mean.csv"
            pd.DataFrame(by_seed_rows).to_csv(by_seed_path, index=False)
            pd.DataFrame(mean_rows).to_csv(mean_path, index=False)
            by_seed_hash, by_seed_bytes = sha256_file(by_seed_path)
            mean_hash, mean_bytes = sha256_file(mean_path)
            validation_manifest_path = root / "temporal_validation_manifest.json"
            write_json(validation_manifest_path, {
                "runId": "finance-validation-fixture",
                "status": "demo_validation_selection",
                "baseRun": {"manifestPath": str(base_manifest_path.resolve())},
                "stackingRun": {"sourceCandidateId": "stacking_gradient_boosting"},
                "dataset": {"kind": "synthetic_benchmark"},
                "comparison": [
                    {"candidateId": model_id, "calibratedThreshold": 0.5}
                    for model_id in FINANCE_MODEL_IDS
                ],
                "validation": {
                    "chronologicalCalibrationBeforeSelection": True,
                    "testSetLocked": True,
                    "testSetEncoded": False,
                    "testSetUsed": False,
                },
                "artifacts": {
                    "predictionsBySeed": {
                        "path": str(by_seed_path.resolve()),
                        "sha256": by_seed_hash,
                        "bytes": by_seed_bytes,
                    },
                    "meanPredictions": {
                        "path": str(mean_path.resolve()),
                        "sha256": mean_hash,
                        "bytes": mean_bytes,
                    },
                },
            })

            manifest = run_finance_diversity_ablation(
                validation_manifest_path=validation_manifest_path,
                output_dir=root / "analysis",
                bootstrap_iterations=100,
            )

            raw = pd.read_csv(manifest["artifacts"]["rawAblationPredictions"]["path"])
            evaluated = pd.read_csv(manifest["artifacts"]["evaluatedSelectionPredictions"]["path"])
            self.assertEqual(len(manifest["diversity"]["pairs"]), 10)
            self.assertEqual(manifest["diversity"]["rows"], 40)
            self.assertEqual(len(manifest["ablation"]["configurations"]), 6)
            self.assertEqual(len(manifest["ablation"]["metrics"]), 12)
            self.assertEqual(len(manifest["ablation"]["contribution"]), 5)
            self.assertEqual(len(raw), 80 * 6 * 2)
            self.assertEqual(len(evaluated), 40 * 6 * 2)
            self.assertEqual(len(manifest["artifacts"]["fittedObjects"]), 6 * 2)
            self.assertEqual(len(manifest["ablation"]["calibrationRecords"]), 6 * 2)
            self.assertTrue(all(
                record["fitEndTimestampNs"] < record["selectionStartTimestampNs"]
                for record in manifest["ablation"]["calibrationRecords"]
            ))
            self.assertTrue(manifest["validation"]["sameRowsForAllAblations"])
            self.assertTrue(manifest["validation"]["metaFitUsesOofOnly"])
            self.assertTrue(manifest["validation"]["calibrationUsesFirstValidationHalfOnly"])
            self.assertFalse(manifest["validation"]["selectionLabelsUsedForCalibration"])
            self.assertFalse(manifest["validation"]["selectionLabelsUsedForThreshold"])
            self.assertTrue(manifest["validation"]["testSetLocked"])
            self.assertFalse(manifest["validation"]["testSetEncoded"])
            self.assertFalse(manifest["validation"]["testSetUsed"])

            for artifact_name in ("pairwiseDiversity", "rawAblationPredictions", "evaluatedSelectionPredictions", "report"):
                artifact = manifest["artifacts"][artifact_name]
                digest, size = sha256_file(Path(artifact["path"]))
                self.assertEqual(digest, artifact["sha256"])
                self.assertEqual(size, artifact["bytes"])

            revalidation = run_finance_optimized_stacking_revalidation(
                diversity_manifest_path=Path(manifest["manifest"]["path"]),
                output_dir=root / "revalidation",
                bootstrap_iterations=100,
            )
            self.assertEqual(revalidation["protocol"]["calibrationRows"], 26)
            self.assertEqual(revalidation["protocol"]["ablationSelectionRows"], 27)
            self.assertEqual(revalidation["protocol"]["independentComparisonRows"], 27)
            self.assertEqual(revalidation["stackingSelection"]["configurationsCompared"], 12)
            self.assertEqual(len(revalidation["independentComparison"]["metrics"]), 6)
            self.assertEqual({row["candidateId"] for row in revalidation["independentComparison"]["metrics"]}, {*FINANCE_MODEL_IDS, "stacking"})
            self.assertEqual(len(revalidation["independentComparison"]["seedMetrics"]), 6)
            self.assertEqual({row["candidateId"] for row in revalidation["independentComparison"]["seedMetrics"]}, {*FINANCE_MODEL_IDS, "stacking"})
            self.assertFalse(revalidation["validation"]["independentComparisonLabelsUsedForCalibrationThresholdOrSelection"])
            self.assertTrue(revalidation["validation"]["testSetLocked"])
            self.assertFalse(revalidation["validation"]["testSetEncoded"])
            self.assertFalse(revalidation["validation"]["testSetUsed"])

    @staticmethod
    def _probabilities(label: int, index: int) -> dict[str, float]:
        center = 0.76 if label else 0.10
        offsets = {
            "lstm": 0.03 * np.sin(index),
            "gru": 0.04 * np.cos(index / 2),
            "brnn": -0.10 if label and index % 5 == 0 else 0.02,
            "tcn": 0.12 if not label and index % 9 == 0 else -0.01,
            "transformer": 0.18 if index % 11 == 0 else -0.03,
        }
        return {
            f"probability_{model_id}": float(np.clip(center + offsets[model_id], 0.01, 0.99))
            for model_id in FINANCE_MODEL_IDS
        }


if __name__ == "__main__":
    unittest.main()
