from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from app.phishing.metrics import phishing_classification_metrics
from app.phishing.models import PHISHING_MODEL_IDS, build_phishing_keras_model
from app.phishing.pipeline import run_phishing_oof_experiment


class _FakeClassifier:
    fit_calls = 0

    def __init__(self, *, model_id: str, **kwargs) -> None:
        self.model_id = model_id

    def fit(self, x_train, y_train, x_validation, y_validation, *, class_weight=None):
        type(self).fit_calls += 1
        assert len(x_train) and len(x_validation)
        assert class_weight is not None
        return {"trainTimeSeconds": 0.01, "epochsCompleted": 1, "history": {}}

    def predict_proba(self, x):
        offset = {"lstm": 0.0, "gru": 0.01}.get(self.model_id, 0.02)
        values = np.linspace(0.1 + offset, 0.9 - offset, len(x), dtype=np.float64)
        return values, 0.001

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_id.encode("utf-8")
        path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest(), len(payload)


class PhishingModelTests(unittest.TestCase):
    def test_metrics_report_perfect_ranking(self) -> None:
        metrics = phishing_classification_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.05, 0.20, 0.80, 0.95]),
        )

        self.assertEqual(metrics["prAuc"], 1.0)
        self.assertEqual(metrics["mcc"], 1.0)
        self.assertEqual(metrics["falsePositiveRate"], 0.0)

    def test_all_neural_architectures_return_binary_probabilities(self) -> None:
        inputs = np.zeros((2, 24), dtype=np.int32)
        inputs[:, :5] = np.asarray([2, 3, 4, 5, 6])
        for model_id in PHISHING_MODEL_IDS:
            model = build_phishing_keras_model(
                model_id,
                vocabulary_size=32,
                sequence_length=24,
            )
            probabilities = model(inputs, training=False).numpy().reshape(-1)
            self.assertEqual(probabilities.shape, (2,))
            self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))

    def test_oof_pipeline_persists_complete_probabilities_without_test_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silver = root / "phishing.csv"
            assignments = root / "assignments.csv"
            rows = []
            assignment_rows = []
            for fold in range(5):
                for label in (0, 1):
                    for index in range(10):
                        url = f"https://f{fold}-l{label}-{index}.example-{fold}-{label}-{index}.com/path"
                        sample_id = hashlib.sha256(url.encode("utf-8")).hexdigest()
                        rows.append({
                            "canonical_url": url,
                            "registrable_domain": f"example-{fold}-{label}-{index}.com",
                            "is_phishing": label,
                            "split": "train",
                        })
                        assignment_rows.append({"sample_id": sample_id, "oof_fold": fold})
            pd.DataFrame(rows).to_csv(silver, index=False)
            pd.DataFrame(assignment_rows).to_csv(assignments, index=False)

            manifest = run_phishing_oof_experiment(
                output_dir=root / "run",
                silver_path=silver,
                assignments_path=assignments,
                protocol="demo",
                epochs=1,
                seeds=(42,),
                model_ids=("lstm", "gru"),
                demo_max_rows=None,
                classifier_factory=_FakeClassifier,
            )

            probabilities = pd.read_csv(manifest["artifacts"]["oofProbabilities"]["path"])
            self.assertEqual(len(probabilities), 100)
            self.assertFalse(probabilities[["probability_lstm", "probability_gru"]].isna().any().any())
            self.assertEqual(len(manifest["baseFoldMetrics"]), 10)
            self.assertTrue(manifest["validation"]["groupLeakagePassed"])
            self.assertFalse(manifest["validation"]["testSetUsed"])
            self.assertTrue(manifest["stacking"]["ready"])

    def test_oof_pipeline_resumes_only_hash_verified_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silver = root / "phishing.csv"
            assignments = root / "assignments.csv"
            rows = []
            assignment_rows = []
            for fold in range(5):
                for label in (0, 1):
                    for index in range(5):
                        url = f"https://resume-{fold}-{label}-{index}.example.com/path"
                        rows.append({
                            "canonical_url": url,
                            "registrable_domain": f"resume-{fold}-{label}-{index}.example.com",
                            "is_phishing": label,
                            "split": "train",
                        })
                        assignment_rows.append({
                            "sample_id": hashlib.sha256(url.encode("utf-8")).hexdigest(),
                            "oof_fold": fold,
                        })
            pd.DataFrame(rows).to_csv(silver, index=False)
            pd.DataFrame(assignment_rows).to_csv(assignments, index=False)
            _FakeClassifier.fit_calls = 0
            arguments = {
                "output_dir": root / "run",
                "silver_path": silver,
                "assignments_path": assignments,
                "execution_run_id": "resume-test",
                "protocol": "demo",
                "epochs": 1,
                "seeds": (42,),
                "model_ids": ("lstm", "gru"),
                "demo_max_rows": None,
                "classifier_factory": _FakeClassifier,
            }

            first = run_phishing_oof_experiment(**arguments)
            calls_after_first_run = _FakeClassifier.fit_calls
            resumed = run_phishing_oof_experiment(**arguments)

            self.assertEqual(calls_after_first_run, 10)
            self.assertEqual(_FakeClassifier.fit_calls, calls_after_first_run)
            self.assertEqual(resumed["execution"]["resumedUnits"], 10)
            self.assertEqual(resumed["execution"]["trainedUnitsThisInvocation"], 0)
            self.assertTrue(Path(first["execution"]["checkpointPath"]).is_file())


if __name__ == "__main__":
    unittest.main()
