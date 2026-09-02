from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from app.energy.service import build_energy_experiment_view, run_energy_experiment_for_api


class EnergyApiViewTests(unittest.TestCase):
    def test_demo_view_compares_base_models_and_stacking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "oof_predictions.csv"
            ensemble_path = root / "ensemble_oof_predictions.csv"
            pd.DataFrame([
                {"seed": 42, "fold": 1, "timestamp": "2026-01-01", "actual": 10.0, "lstm": 11.0, "gru": 9.0},
                {"seed": 42, "fold": 1, "timestamp": "2026-01-02", "actual": 12.0, "lstm": 13.0, "gru": 11.0},
            ]).to_csv(base_path, index=False)
            pd.DataFrame([
                {"seed": 42, "fold": 1, "timestamp": "2026-01-01", "actual": 10.0, "ensemble": "stacking_gradient_boosting", "prediction": 10.2, "residual": -0.2},
                {"seed": 42, "fold": 1, "timestamp": "2026-01-02", "actual": 12.0, "ensemble": "stacking_gradient_boosting", "prediction": 11.8, "residual": 0.2},
            ]).to_csv(ensemble_path, index=False)
            manifest = {
                "runId": "demo-test", "status": "demo", "createdAt": "2026-01-01T00:00:00Z", "protocolVersion": "1.0.0",
                "dataset": {"rows": 100, "features": ["load"], "target": "load"},
                "walkForward": {"folds": [{"fold": 0}, {"fold": 1}], "seeds": [42], "window": 12, "horizon": 1, "testSetUsed": False},
                "baseModels": ["lstm", "gru"],
                "baseFoldMetrics": [
                    {"seed": 42, "fold": 1, "modelId": "lstm", "rmse": 1.0, "mae": 1.0, "smape": 0.1, "r2": 0.7, "trainTimeSeconds": 1.0, "inferenceTimeMsPerSample": 0.2},
                    {"seed": 42, "fold": 1, "modelId": "gru", "rmse": 1.1, "mae": 1.0, "smape": 0.1, "r2": 0.6, "trainTimeSeconds": 0.8, "inferenceTimeMsPerSample": 0.2},
                ],
                "ensembles": {"winner": "stacking_gradient_boosting", "aggregate": [
                    {"ensembleId": "stacking_gradient_boosting", "foldEvaluations": 1, "rmseMean": 0.2, "rmseStd": 0.0, "maeMean": 0.2, "smapeMean": 0.02, "r2Mean": 0.95}
                ]},
                "anomalies": {"labelType": "estimated", "classificationMetricsAvailable": False, "recommendedMethod": "mad", "summaries": [
                    {"predictorFamily": "ensemble", "predictorId": "stacking_gradient_boosting", "method": "mad", "evaluatedSamples": 2, "estimatedAnomalies": 0, "estimatedAnomalyRate": 0.0, "meanScore": 0.2, "maxScore": 0.3}
                ]},
                "aggregate": {"stackingReady": False, "testSetEvaluatedOnce": False},
                "comparisonScope": {"fairComparison": True, "pairs": [{"seed": 42, "fold": 1}]},
                "artifacts": {"oofPredictions": str(base_path), "ensemblePredictions": str(ensemble_path)},
            }
            manifest_path = root / "oof_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            view = build_energy_experiment_view(manifest_path)

            self.assertTrue(view["available"])
            self.assertEqual(view["winner"]["overall"], "stacking_gradient_boosting")
            self.assertEqual(len(view["comparison"]), 3)
            self.assertEqual(len(view["timeline"]), 2)
            self.assertFalse(view["methodology"]["testSetUsed"])
            self.assertTrue(view["methodology"]["fairComparison"])
            self.assertFalse(view["anomalies"]["classificationMetricsAvailable"])

    def test_thesis_protocol_rejects_synthetic_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "muestra sintética"):
            run_energy_experiment_for_api(
                protocol="thesis", source="sample", window=12, horizon=1, gap_steps=6,
                n_splits=5, epochs=1, batch_size=16, seeds=(42,), model_ids=("lstm", "gru"),
            )


if __name__ == "__main__":
    unittest.main()
