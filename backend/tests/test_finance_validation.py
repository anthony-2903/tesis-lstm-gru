from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.finance.benchmark import prepare_finance_benchmark
from app.finance.models import FINANCE_MODEL_IDS
from app.finance.pipeline import run_finance_oof_experiment
from app.finance.sequences import prepare_finance_sequence_protocol
from app.finance.stacking import run_finance_stacking_experiment
from app.finance.validation import (
    apply_temporal_calibrator,
    chronological_validation_split,
    fit_temporal_calibrator,
    paired_stratified_pr_auc_bootstrap,
    run_finance_temporal_validation,
    select_fbeta_threshold,
)
from app.phishing.ingestion import sha256_file


class FakeFinanceValidationClassifier:
    fit_calls = 0

    def __init__(self, *, model_id: str, **kwargs) -> None:
        self.model_id = model_id

    def fit(self, x_train, y_train, x_validation, y_validation, *, class_weight=None):
        return {"trainTimeSeconds": 0.001, "epochsCompleted": 1, "history": {"loss": [1.0]}}

    def fit_full(self, x_train, y_train, *, class_weight=None):
        type(self).fit_calls += 1
        if set(np.unique(y_train)) != {0, 1}:
            raise AssertionError("El refit recibio train monoclase.")
        return {"trainTimeSeconds": 0.001, "epochsCompleted": 1, "history": {"loss": [1.0]}}

    def predict_proba(self, x):
        offsets = {"lstm": -0.20, "gru": -0.10, "brnn": 0.0, "tcn": 0.10, "transformer": 0.20}
        logits = np.clip(x[:, -1, 0].astype(np.float64) + offsets[self.model_id], -8.0, 8.0)
        return 1.0 / (1.0 + np.exp(-logits)), 0.001

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fake-finance-validation:{self.model_id}".encode("utf-8"))
        return sha256_file(path)


class FinanceValidationTests(unittest.TestCase):
    def test_chronological_split_is_disjoint_and_ordered(self) -> None:
        labels = np.asarray(([0, 1] * 20), dtype=np.int32)
        calibration, selection = chronological_validation_split(labels)
        self.assertEqual(len(set(calibration) & set(selection)), 0)
        self.assertLess(calibration.max(), selection.min())
        self.assertEqual(len(calibration) + len(selection), len(labels))

    def test_calibration_and_threshold_ignore_selection_labels(self) -> None:
        calibration_labels = np.asarray(([0, 0, 0, 0, 1] * 20), dtype=np.int32)
        calibration_scores = np.linspace(0.01, 0.99, len(calibration_labels))
        first = fit_temporal_calibrator(calibration_labels, calibration_scores)
        second = fit_temporal_calibrator(calibration_labels, calibration_scores)
        first_scores = apply_temporal_calibrator(first, calibration_scores)
        second_scores = apply_temporal_calibrator(second, calibration_scores)
        self.assertEqual(first["method"], second["method"])
        self.assertAlmostEqual(select_fbeta_threshold(calibration_labels, first_scores), select_fbeta_threshold(calibration_labels, second_scores))

    def test_paired_bootstrap_is_deterministic(self) -> None:
        labels = np.asarray(([0] * 90) + ([1] * 10), dtype=np.int32)
        stacking = np.linspace(0.01, 0.99, 100)
        base = np.linspace(0.20, 0.80, 100)
        first = paired_stratified_pr_auc_bootstrap(labels, stacking, base, iterations=100, seed=17)
        second = paired_stratified_pr_auc_bootstrap(labels, stacking, base, iterations=100, seed=17)
        self.assertEqual(first, second)

    def test_full_validation_freezes_six_candidates_and_resumes_refits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silver = root / "finance.csv"
            metadata = root / "finance.metadata.json"
            audit = root / "finance.audit.json"
            sequences = root / "finance_sequences.npz"
            features = root / "finance_features.csv"
            assignments = root / "finance_oof.csv"
            sequence_manifest = root / "finance_sequence_manifest.json"
            base_output = root / "experiment"
            stacking_output = root / "stacking"
            validation_output = root / "validation"
            prepare_finance_benchmark(
                days=60,
                customers=100,
                terminals=40,
                seed=19,
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
            base = run_finance_oof_experiment(
                output_dir=base_output,
                sequences_path=sequences,
                assignments_path=assignments,
                sequence_manifest_path=sequence_manifest,
                protocol="demo",
                epochs=1,
                seeds=(42,),
                model_ids=FINANCE_MODEL_IDS,
                demo_max_rows_per_fold=100,
                execution_run_id="finance_validation_test",
                classifier_factory=FakeFinanceValidationClassifier,
            )
            base_manifest_path = base_output / "oof_manifest.json"
            stacking = run_finance_stacking_experiment(base_manifest_path=base_manifest_path, output_dir=stacking_output)
            stacking_manifest_path = Path(stacking["manifest"]["path"])
            FakeFinanceValidationClassifier.fit_calls = 0
            first = run_finance_temporal_validation(
                base_manifest_path=base_manifest_path,
                stacking_manifest_path=stacking_manifest_path,
                sequence_manifest_path=sequence_manifest,
                sequences_path=sequences,
                features_path=features,
                output_dir=validation_output,
                demo_max_train_rows=1_000,
                demo_max_validation_rows=1_000,
                bootstrap_iterations=100,
                classifier_factory=FakeFinanceValidationClassifier,
            )
            second = run_finance_temporal_validation(
                base_manifest_path=base_manifest_path,
                stacking_manifest_path=stacking_manifest_path,
                sequence_manifest_path=sequence_manifest,
                sequences_path=sequences,
                features_path=features,
                output_dir=validation_output,
                demo_max_train_rows=1_000,
                demo_max_validation_rows=1_000,
                bootstrap_iterations=100,
                classifier_factory=FakeFinanceValidationClassifier,
            )

            self.assertEqual(len(first["comparison"]), 6)
            self.assertEqual({row["candidateId"] for row in first["comparison"]}, set((*FINANCE_MODEL_IDS, "stacking")))
            self.assertTrue(first["validation"]["sameRowsForAllSixCandidates"])
            self.assertTrue(first["validation"]["chronologicalCalibrationBeforeSelection"])
            self.assertFalse(first["validation"]["testSetUsed"])
            self.assertFalse(first["validation"]["testSetEncoded"])
            self.assertFalse(first["validation"]["readyForFinalTestEvaluation"])
            self.assertFalse(first["freeze"]["eligibleForFinalThesisClaim"])
            self.assertLess(first["validation"]["calibrationEndTimestampNs"], first["validation"]["selectionStartTimestampNs"])
            self.assertEqual(FakeFinanceValidationClassifier.fit_calls, 5)
            self.assertEqual(second["training"]["resumedUnits"], 5)
            self.assertEqual(second["training"]["trainedUnitsThisInvocation"], 0)
            for artifact_name in ("predictionsBySeed", "meanPredictions", "metrics", "frozenSelection"):
                artifact = first["artifacts"][artifact_name]
                digest, size = sha256_file(Path(artifact["path"]))
                self.assertEqual(digest, artifact["sha256"])
                self.assertEqual(size, artifact["bytes"])


if __name__ == "__main__":
    unittest.main()
