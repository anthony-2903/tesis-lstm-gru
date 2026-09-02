from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from app.phishing.ingestion import sha256_file
from app.phishing.stacking import (
    ENSEMBLE_IDS,
    build_meta_features,
    fit_nonnegative_weights,
    run_phishing_stacking_experiment,
)
from app.utils import write_json


BASE_MODELS = ("lstm", "gru", "brnn", "tcn", "transformer")


class PhishingStackingTests(unittest.TestCase):
    def test_meta_features_include_probabilities_and_disagreement(self) -> None:
        probabilities = np.asarray([[0.1, 0.3], [0.8, 0.6]], dtype=np.float64)
        features, names = build_meta_features(
            probabilities,
            ["probability_lstm", "probability_gru"],
        )

        self.assertEqual(features.shape, (2, 9))
        self.assertEqual(names[:2], ["probability_lstm", "probability_gru"])
        self.assertIn("mean_absolute_disagreement", names)
        self.assertTrue(np.isfinite(features).all())

    def test_weight_optimizer_returns_a_probability_simplex(self) -> None:
        probabilities = np.asarray(
            [[0.1, 0.3], [0.2, 0.4], [0.8, 0.6], [0.9, 0.7]],
            dtype=np.float64,
        )
        weights = fit_nonnegative_weights(probabilities, np.asarray([0, 0, 1, 1]))

        self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)
        self.assertTrue((weights >= 0.0).all())
        self.assertTrue((weights <= 1.0).all())

    def test_crossfit_persists_complete_predictions_without_test_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oof_path = root / "base_oof.csv"
            rows = []
            for fold in range(5):
                for label in (0, 1):
                    for index in range(8):
                        row = {
                            "sample_id": f"sample-{fold}-{label}-{index}",
                            "seed": 42,
                            "fold": fold,
                            "is_phishing": label,
                        }
                        for model_index, model_id in enumerate(BASE_MODELS):
                            perturbation = ((index + model_index + fold) % 5 - 2) * 0.015
                            row[f"probability_{model_id}"] = float(
                                np.clip((0.18 if label == 0 else 0.82) + perturbation, 0.01, 0.99)
                            )
                        rows.append(row)
            pd.DataFrame(rows).to_csv(oof_path, index=False)
            oof_hash, oof_bytes = sha256_file(oof_path)
            base_manifest_path = root / "oof_manifest.json"
            base_aggregate = [
                {
                    "modelId": model_id,
                    "foldEvaluations": 5,
                    "prAucMean": 0.90 - index * 0.01,
                    "prAucStd": 0.01,
                    "rocAucMean": 0.90,
                    "f1Mean": 0.85,
                    "precisionMean": 0.85,
                    "recallMean": 0.85,
                    "mccMean": 0.70,
                    "balancedAccuracyMean": 0.85,
                    "falsePositiveRateMean": 0.10,
                }
                for index, model_id in enumerate(BASE_MODELS)
            ]
            write_json(base_manifest_path, {
                "schemaVersion": "1.0.0",
                "runId": "phishing-fixture",
                "status": "demo",
                "baseModels": list(BASE_MODELS),
                "aggregate": base_aggregate,
                "artifacts": {
                    "oofProbabilities": {
                        "path": str(oof_path.resolve()),
                        "sha256": oof_hash,
                        "bytes": oof_bytes,
                    }
                },
            })

            manifest = run_phishing_stacking_experiment(
                base_manifest_path=base_manifest_path,
                output_dir=root / "stacking",
            )

            predictions = pd.read_csv(manifest["artifacts"]["predictions"]["path"])
            probability_columns = [f"probability_{candidate_id}" for candidate_id in ENSEMBLE_IDS]
            self.assertEqual(len(predictions), len(rows))
            self.assertFalse(predictions[probability_columns].isna().any().any())
            self.assertEqual(set(manifest["candidates"]), set(ENSEMBLE_IDS))
            self.assertEqual(len(manifest["foldMetrics"]), 5 * len(ENSEMBLE_IDS))
            self.assertTrue(manifest["validation"]["directSampleOverlapPassed"])
            self.assertFalse(manifest["validation"]["strictNestedCrossFit"])
            self.assertFalse(manifest["validation"]["outerValidationUsed"])
            self.assertTrue(manifest["validation"]["testSetLocked"])
            self.assertFalse(manifest["validation"]["testSetUsed"])
            self.assertEqual(len(manifest["metaFeatures"]["columns"]), 12)
            for artifact in manifest["artifacts"]["fittedObjects"]:
                digest, size = sha256_file(Path(artifact["path"]))
                self.assertEqual(digest, artifact["sha256"])
                self.assertEqual(size, artifact["bytes"])


if __name__ == "__main__":
    unittest.main()
