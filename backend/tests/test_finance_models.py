from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from app.finance.benchmark import prepare_finance_benchmark
from app.finance.metrics import finance_classification_metrics
from app.finance.models import FINANCE_MODEL_IDS, build_finance_keras_model
from app.finance.pipeline import run_finance_oof_experiment
from app.finance.sequences import prepare_finance_sequence_protocol
from app.phishing.ingestion import sha256_file


class FakeFinanceClassifier:
    def __init__(self, *, model_id: str, **kwargs) -> None:
        self.model_id = model_id

    def fit(self, x_train, y_train, x_validation, y_validation, *, class_weight=None):
        if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_validation)) != {0, 1}:
            raise AssertionError("El fake recibió una partición monoclase.")
        return {"trainTimeSeconds": 0.001, "epochsCompleted": 1, "history": {"loss": [1.0]}}

    def predict_proba(self, x):
        offsets = {"lstm": -0.20, "gru": -0.10, "brnn": 0.0, "tcn": 0.10, "transformer": 0.20}
        logits = np.clip(x[:, -1, 0].astype(np.float64) + offsets[self.model_id], -8.0, 8.0)
        return 1.0 / (1.0 + np.exp(-logits)), 0.001

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fake-finance-model:{self.model_id}".encode("utf-8"))
        return sha256_file(path)


class FinanceModelTests(unittest.TestCase):
    def test_all_five_architectures_are_real_binary_keras_models(self) -> None:
        layer_expectations = {
            "lstm": tf.keras.layers.LSTM,
            "gru": tf.keras.layers.GRU,
            "brnn": tf.keras.layers.Bidirectional,
            "tcn": tf.keras.layers.Conv1D,
            "transformer": tf.keras.layers.MultiHeadAttention,
        }
        for model_id in FINANCE_MODEL_IDS:
            model = build_finance_keras_model(model_id, input_shape=(10, 11))
            self.assertEqual(model.output_shape, (None, 1))
            self.assertTrue(any(isinstance(layer, layer_expectations[model_id]) for layer in model.layers))

    def test_finance_metrics_include_ranking_threshold_and_calibration(self) -> None:
        result = finance_classification_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.01, 0.10, 0.90, 0.99]),
        )
        self.assertAlmostEqual(result["prAuc"], 1.0)
        self.assertAlmostEqual(result["rocAuc"], 1.0)
        self.assertAlmostEqual(result["f1"], 1.0)
        self.assertLess(result["brierScore"], 0.01)
        self.assertIn("logLoss", result)
        self.assertEqual(result["prAucBaseline"], 0.5)

    def test_oof_runner_covers_five_folds_and_resumes_verified_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silver = root / "finance.csv"
            metadata = root / "finance.metadata.json"
            audit = root / "finance.audit.json"
            sequences = root / "finance_sequences.npz"
            features = root / "finance_features.csv"
            assignments = root / "finance_oof.csv"
            sequence_manifest = root / "finance_sequence_manifest.json"
            output = root / "experiment"
            prepare_finance_benchmark(
                days=60,
                customers=100,
                terminals=40,
                seed=17,
                minimum_rows=1_000,
                silver_path=silver,
                metadata_path=metadata,
                audit_path=audit,
            )
            prepare_finance_sequence_protocol(
                window=6,
                silver_path=silver,
                metadata_path=metadata,
                audit_path=audit,
                sequences_path=sequences,
                features_path=features,
                assignments_path=assignments,
                manifest_path=sequence_manifest,
            )
            first = run_finance_oof_experiment(
                output_dir=output,
                sequences_path=sequences,
                assignments_path=assignments,
                sequence_manifest_path=sequence_manifest,
                protocol="demo",
                epochs=1,
                seeds=(42,),
                model_ids=FINANCE_MODEL_IDS,
                demo_max_rows_per_fold=100,
                execution_run_id="finance_test_run",
                classifier_factory=FakeFinanceClassifier,
            )
            second = run_finance_oof_experiment(
                output_dir=output,
                sequences_path=sequences,
                assignments_path=assignments,
                sequence_manifest_path=sequence_manifest,
                protocol="demo",
                epochs=1,
                seeds=(42,),
                model_ids=FINANCE_MODEL_IDS,
                demo_max_rows_per_fold=100,
                execution_run_id="finance_test_run",
                classifier_factory=FakeFinanceClassifier,
            )

            self.assertEqual(len(first["baseFoldMetrics"]), 25)
            self.assertEqual(len(first["aggregate"]), 5)
            self.assertEqual(first["dataset"]["oofRowsUsedPerSeed"], 500)
            self.assertEqual(first["validation"]["oofFolds"], [0, 1, 2, 3, 4])
            self.assertTrue(first["validation"]["futureLeakagePassed"])
            self.assertTrue(first["validation"]["testSetLocked"])
            self.assertFalse(first["validation"]["testSetUsed"])
            self.assertFalse(first["validation"]["externalValidationUsed"])
            self.assertTrue(first["stacking"]["ready"])
            self.assertEqual(second["execution"]["resumedUnits"], 25)
            self.assertEqual(second["execution"]["trainedUnitsThisInvocation"], 0)

    def test_thesis_protocol_rejects_synthetic_finance_benchmark(self) -> None:
        with self.assertRaisesRegex(ValueError, "cinco modelos"):
            run_finance_oof_experiment(protocol="thesis", model_ids=("lstm",), seeds=(42,))


if __name__ == "__main__":
    unittest.main()
