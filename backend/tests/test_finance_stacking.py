from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from app.finance.models import FINANCE_MODEL_IDS
from app.finance.stacking import (
    build_finance_meta_features,
    fit_nonnegative_finance_weights,
    run_finance_stacking_experiment,
)
from app.phishing.ingestion import sha256_file
from app.utils import write_json


class FinanceStackingTests(unittest.TestCase):
    def test_meta_features_include_probabilities_and_disagreement(self) -> None:
        values = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5], [0.9, 0.8, 0.7, 0.6, 0.5]])
        columns = [f"probability_{model_id}" for model_id in FINANCE_MODEL_IDS]
        features, names = build_finance_meta_features(values, columns)
        self.assertEqual(features.shape, (2, 12))
        self.assertEqual(names[:5], columns)
        self.assertIn("probability_std", names)
        self.assertIn("mean_absolute_disagreement", names)
        self.assertNotIn("is_fraud", names)

    def test_weight_optimizer_returns_probability_simplex(self) -> None:
        values = np.asarray([[0.1, 0.2], [0.8, 0.6], [0.2, 0.3], [0.9, 0.7]])
        labels = np.asarray([0, 1, 0, 1])
        weights = fit_nonnegative_finance_weights(values, labels)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertTrue(np.all(weights <= 1.0))

    def test_temporal_cross_fit_persists_six_model_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_manifest = self._build_base_run(root)
            result = run_finance_stacking_experiment(
                base_manifest_path=base_manifest,
                output_dir=root / "stacking",
            )

            self.assertEqual(result["validation"]["evaluatedFolds"], [1, 2, 3, 4])
            self.assertEqual(result["validation"]["coverageRows"], 160)
            self.assertEqual(result["validation"]["warmupRowsExcluded"], 40)
            self.assertTrue(result["validation"]["strictTemporalCrossFit"])
            self.assertTrue(result["validation"]["metaDependencyLeakageControlled"])
            self.assertFalse(result["validation"]["futureFoldsUsed"])
            self.assertFalse(result["validation"]["externalValidationUsed"])
            self.assertTrue(result["validation"]["testSetLocked"])
            self.assertFalse(result["validation"]["testSetUsed"])
            self.assertEqual(len(result["sixModelComparison"]), 6)
            self.assertEqual({item["candidateId"] for item in result["baseAggregateOnCommonRows"]}, set(FINANCE_MODEL_IDS))
            self.assertIn(result["recommendation"]["leadingStackingCandidateId"], {"stacking_logistic", "stacking_gradient_boosting"})
            for protocol in result["validation"]["foldProtocols"]:
                self.assertTrue(all(fold < protocol["holdoutFold"] for fold in protocol["fitFolds"]))
                self.assertFalse(protocol["futureFoldsUsed"])
                self.assertEqual(protocol["transactionOverlap"], 0)
            for artifact_name in ("predictions", "metrics"):
                artifact = result["artifacts"][artifact_name]
                digest, size = sha256_file(Path(artifact["path"]))
                self.assertEqual(digest, artifact["sha256"])
                self.assertEqual(size, artifact["bytes"])

    def test_rejects_tampered_base_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_manifest = self._build_base_run(root)
            manifest = json.loads(base_manifest.read_text(encoding="utf-8"))
            oof_path = Path(manifest["artifacts"]["oofProbabilities"]["path"])
            oof_path.write_text(oof_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                run_finance_stacking_experiment(base_manifest_path=base_manifest, output_dir=root / "stacking")

    def _build_base_run(self, root: Path) -> Path:
        rng = np.random.default_rng(42)
        rows = []
        transaction_id = 1
        for fold in range(5):
            labels = np.asarray(([0] * 34) + ([1] * 6), dtype=np.int32)
            rng.shuffle(labels)
            for label in labels:
                probabilities = {}
                for position, model_id in enumerate(FINANCE_MODEL_IDS):
                    signal = 0.15 + 0.55 * label + 0.03 * position + rng.normal(0.0, 0.12)
                    probabilities[f"probability_{model_id}"] = float(np.clip(signal, 0.01, 0.99))
                rows.append(
                    {
                        "transaction_id": transaction_id,
                        "seed": 42,
                        "fold": fold,
                        "is_fraud": int(label),
                        **probabilities,
                    }
                )
                transaction_id += 1
        oof = pd.DataFrame(rows)
        oof_path = root / "oof.csv"
        oof.to_csv(oof_path, index=False)
        digest, size = sha256_file(oof_path)
        manifest = {
            "runId": "finance-base-test",
            "status": "demo",
            "baseModels": list(FINANCE_MODEL_IDS),
            "validation": {
                "testSetLocked": True,
                "testSetEncoded": False,
                "testSetUsed": False,
                "externalValidationUsed": False,
            },
            "artifacts": {
                "oofProbabilities": {"path": str(oof_path.resolve()), "sha256": digest, "bytes": size, "rows": len(oof)}
            },
        }
        manifest_path = root / "oof_manifest.json"
        write_json(manifest_path, manifest)
        return manifest_path


if __name__ == "__main__":
    unittest.main()
