from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from app.cleaning.timeseries_cleaner import clean_opsd
from app.energy.anomalies import ANOMALY_METHODS, apply_anomaly_threshold, calibrate_anomaly_threshold, detect_walk_forward_anomalies
from app.energy.data import prepare_energy_oof_folds, prepare_energy_sequences, validate_energy_frame
from app.energy.metrics import energy_regression_metrics
from app.energy.models import LearnablePositionEmbedding, THESIS_MODEL_IDS, NaivePersistenceRegressor, build_energy_models
from app.energy.oof_pipeline import _build_common_anomaly_input
from app.energy.stacking import evaluate_energy_ensembles
from app.ingestion.samples import sample_opsd


TARGET = "DE_load_actual_entsoe_transparency"


class EnergyDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = clean_opsd(sample_opsd())

    def test_strict_protocol_rejects_demo_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "8760"):
            validate_energy_frame(
                self.frame,
                timestamp_column="timestamp",
                target_column=TARGET,
                strict=True,
            )

    def test_chronological_partitions_do_not_overlap(self) -> None:
        clean, report = validate_energy_frame(
            self.frame,
            timestamp_column="timestamp",
            target_column=TARGET,
            strict=False,
        )
        prepared = prepare_energy_sequences(
            clean,
            timestamp_column="timestamp",
            target_column=TARGET,
            feature_columns=report.feature_columns,
            window=12,
            horizon=1,
            gap_steps=6,
        )
        train_end = pd.Timestamp(prepared.split_metadata["trainRange"][1])
        validation_start = pd.Timestamp(prepared.split_metadata["validationRange"][0])
        validation_end = pd.Timestamp(prepared.split_metadata["validationRange"][1])
        test_start = pd.Timestamp(prepared.split_metadata["testRange"][0])
        self.assertLess(train_end, validation_start)
        self.assertLess(validation_end, test_start)
        self.assertAlmostEqual(float(prepared.train.x[:, :, 0].mean()), 0.0, delta=0.2)

    def test_naive_baseline_and_metrics_use_real_predictions(self) -> None:
        clean, report = validate_energy_frame(
            self.frame,
            timestamp_column="timestamp",
            target_column=TARGET,
            strict=False,
        )
        prepared = prepare_energy_sequences(
            clean,
            timestamp_column="timestamp",
            target_column=TARGET,
            feature_columns=report.feature_columns,
            window=12,
            horizon=1,
            gap_steps=6,
        )
        baseline = NaivePersistenceRegressor(prepared.target_feature_index)
        prediction_scaled = baseline.predict(prepared.validation.x)
        prediction = prepared.target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).reshape(-1)
        actual = prepared.target_scaler.inverse_transform(prepared.validation.y.reshape(-1, 1)).reshape(-1)
        metrics = energy_regression_metrics(actual, prediction)
        self.assertGreater(metrics["samples"], 0)
        self.assertGreaterEqual(metrics["rmse"], 0.0)
        self.assertTrue(np.isfinite(metrics["smape"]))

    def test_walk_forward_folds_are_expanding_and_gapped(self) -> None:
        clean, report = validate_energy_frame(
            self.frame,
            timestamp_column="timestamp",
            target_column=TARGET,
            strict=False,
        )
        folds = prepare_energy_oof_folds(
            clean,
            timestamp_column="timestamp",
            target_column=TARGET,
            feature_columns=report.feature_columns,
            window=12,
            horizon=1,
            n_splits=5,
            gap_steps=6,
        )
        self.assertEqual(len(folds), 5)
        train_sizes = [fold.metadata["trainRows"] for fold in folds]
        self.assertEqual(train_sizes, sorted(train_sizes))
        self.assertEqual(len(set(train_sizes)), 5)
        for fold in folds:
            train_end = pd.Timestamp(fold.metadata["trainRange"][1])
            validation_start = pd.Timestamp(fold.metadata["validationRange"][0])
            self.assertLess(train_end, validation_start)


class EnergyStackingTests(unittest.TestCase):
    def test_meta_models_only_train_on_previous_folds(self) -> None:
        rows = []
        for fold in range(4):
            for index in range(6):
                actual = float(10 + fold + index)
                rows.append(
                    {
                        "seed": 42,
                        "fold": fold,
                        "timestamp": f"{fold}-{index}",
                        "actual": actual,
                        "lstm": actual + 0.8,
                        "gru": actual - 0.4,
                    }
                )
        predictions, report = evaluate_energy_ensembles(pd.DataFrame(rows), base_model_ids=("lstm", "gru"))
        self.assertFalse(report["testSetUsed"])
        self.assertEqual(set(predictions["ensemble"]), {"mean", "weighted_mean", "stacking_gradient_boosting"})
        self.assertTrue(report["foldMetrics"])
        for metric in report["foldMetrics"]:
            self.assertLess(metric["metaTrainMaxFold"], metric["fold"])

    def test_anomaly_comparison_uses_common_ensemble_folds(self) -> None:
        base = pd.DataFrame([
            {"seed": 42, "fold": fold, "timestamp": f"{fold}-{index}", "actual": 10.0, "lstm": 10.5, "gru": 9.5}
            for fold in range(3) for index in range(3)
        ])
        ensembles = pd.DataFrame([
            {"seed": 42, "fold": fold, "timestamp": f"{fold}-{index}", "actual": 10.0, "ensemble": ensemble, "prediction": 10.1, "residual": -0.1}
            for fold in (1, 2) for index in range(3) for ensemble in ("mean", "weighted_mean", "stacking_gradient_boosting")
        ])
        common = _build_common_anomaly_input(base, ensembles, ("lstm", "gru"))
        base_folds = set(common[common["predictorFamily"] == "base"]["fold"])
        ensemble_folds = set(common[common["predictorFamily"] == "ensemble"]["fold"])
        self.assertEqual(base_folds, {1, 2})
        self.assertEqual(base_folds, ensemble_folds)


class EnergyAnomalyTests(unittest.TestCase):
    def test_all_threshold_methods_detect_a_large_residual(self) -> None:
        calibration = np.asarray([-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3], dtype=float)
        for method in ANOMALY_METHODS:
            threshold = calibrate_anomaly_threshold(calibration, method=method)
            scores, anomalies, severities = apply_anomaly_threshold([0.1, 12.0], threshold)
            self.assertFalse(bool(anomalies[0]), method)
            self.assertTrue(bool(anomalies[1]), method)
            self.assertGreater(scores[1], 1.0)
            self.assertIn(severities[1], {"low", "medium", "high"})

    def test_anomaly_thresholds_only_use_previous_folds(self) -> None:
        rows = []
        for fold in range(3):
            for index in range(8):
                residual = 0.1 * index if fold < 2 else (8.0 if index == 7 else 0.1 * index)
                actual = 50.0 + index
                rows.append(
                    {
                        "seed": 42,
                        "fold": fold,
                        "timestamp": f"{fold}-{index}",
                        "actual": actual,
                        "prediction": actual - residual,
                        "residual": residual,
                        "predictorFamily": "base",
                        "predictorId": "lstm",
                    }
                )
        predictions, report = detect_walk_forward_anomalies(pd.DataFrame(rows))
        self.assertFalse(report["classificationMetricsAvailable"])
        self.assertEqual(set(predictions["labelType"]), {"estimated_from_prior_validation_residuals"})
        self.assertTrue(predictions["isAnomaly"].any())
        for _, row in predictions.iterrows():
            self.assertLess(row["calibrationMaxFold"], row["fold"])


class EnergyArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = build_energy_models(
            input_shape=(12, 4),
            target_feature_index=0,
            epochs=1,
            batch_size=4,
            include_naive=False,
        )

    def test_factory_builds_all_five_real_models(self) -> None:
        self.assertEqual(tuple(model.model_id for model in self.models), THESIS_MODEL_IDS)
        sample = np.zeros((2, 12, 4), dtype=np.float32)
        for model in self.models:
            self.assertEqual(tuple(model.model(sample, training=False).shape), (2, 1))

    def test_architectures_contain_expected_keras_layers(self) -> None:
        import tensorflow as tf

        by_id = {model.model_id: model for model in self.models}
        self.assertTrue(any(isinstance(layer, tf.keras.layers.LSTM) for layer in by_id["lstm"].model.layers))
        self.assertTrue(any(isinstance(layer, tf.keras.layers.GRU) for layer in by_id["gru"].model.layers))
        self.assertTrue(any(isinstance(layer, tf.keras.layers.Bidirectional) for layer in by_id["brnn"].model.layers))
        tcn_convolutions = [layer for layer in by_id["tcn"].model.layers if isinstance(layer, tf.keras.layers.Conv1D)]
        self.assertTrue(any(layer.padding == "causal" and layer.dilation_rate[0] > 1 for layer in tcn_convolutions))
        self.assertTrue(any(isinstance(layer, tf.keras.layers.MultiHeadAttention) for layer in by_id["transformer"].model.layers))
        self.assertTrue(any(isinstance(layer, LearnablePositionEmbedding) for layer in by_id["transformer"].model.layers))

    def test_factory_rejects_unknown_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "no_soportado"):
            build_energy_models(
                input_shape=(12, 4),
                target_feature_index=0,
                model_ids=("no_soportado",),
                include_naive=False,
            )

    def test_transformer_with_custom_position_layer_is_serializable(self) -> None:
        import tensorflow as tf

        transformer = next(model for model in self.models if model.model_id == "transformer")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transformer.keras"
            transformer.save(path)
            restored = tf.keras.models.load_model(path, compile=False)
            output = restored(np.zeros((2, 12, 4), dtype=np.float32), training=False)
            self.assertEqual(tuple(output.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
