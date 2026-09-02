from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from app.finance.benchmark import prepare_finance_benchmark
from app.finance.sequences import (
    FEATURE_COLUMNS,
    build_causal_finance_features,
    build_customer_sequences,
    build_finance_temporal_oof,
    get_finance_sequence_status,
    prepare_finance_sequence_protocol,
    scale_finance_sequences,
)


def _small_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4, 5],
            "transaction_time": pd.to_datetime(
                ["2024-01-01 08:00", "2024-01-01 08:05", "2024-01-01 09:00", "2024-01-02 08:00", "2024-01-03 08:00"]
            ),
            "customer_id": [10, 20, 10, 20, 10],
            "terminal_id": [100, 100, 101, 101, 100],
            "amount": [10.0, 50.0, 20.0, 60.0, 30.0],
            "is_fraud": [0, 0, 0, 1, 0],
            "split": ["train"] * 5,
        }
    )


class FinanceSequenceTests(unittest.TestCase):
    def test_causal_features_do_not_change_when_future_or_labels_change(self) -> None:
        original = _small_transactions()
        changed = original.copy()
        changed.loc[4, "amount"] = 999_999.0
        changed["is_fraud"] = 1 - changed["is_fraud"]

        before = build_causal_finance_features(original)
        after = build_causal_finance_features(changed)

        np.testing.assert_allclose(
            before.loc[:3, list(FEATURE_COLUMNS)].to_numpy(),
            after.loc[:3, list(FEATURE_COLUMNS)].to_numpy(),
        )
        self.assertNotIn("is_fraud", FEATURE_COLUMNS)

    def test_sequences_are_left_padded_and_isolated_by_customer(self) -> None:
        features = build_causal_finance_features(_small_transactions())
        x, _, transaction_ids, customer_ids, _, _ = build_customer_sequences(features, window=3)

        first_customer_10 = int(np.flatnonzero(transaction_ids == 1)[0])
        second_customer_10 = int(np.flatnonzero(transaction_ids == 3)[0])
        self.assertEqual(customer_ids[second_customer_10], 10)
        np.testing.assert_array_equal(x[first_customer_10, :, -1], [0.0, 0.0, 1.0])
        np.testing.assert_array_equal(x[second_customer_10, :, -1], [0.0, 1.0, 1.0])
        self.assertAlmostEqual(x[second_customer_10, 1, 0], np.log1p(10.0), places=6)
        self.assertAlmostEqual(x[second_customer_10, 2, 0], np.log1p(20.0), places=6)

    def test_oof_has_five_expanding_folds_and_train_only_scalers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silver = root / "finance.csv"
            metadata = root / "finance.metadata.json"
            audit = root / "finance.audit.json"
            prepare_finance_benchmark(
                days=40,
                customers=80,
                terminals=30,
                seed=5,
                minimum_rows=1_000,
                silver_path=silver,
                metadata_path=metadata,
                audit_path=audit,
            )
            raw = pd.read_csv(silver)
            features = build_causal_finance_features(raw[raw["split"].isin(["train", "validation"])])
            protocols, assignments = build_finance_temporal_oof(features, folds=5, purge_days=1)

            self.assertEqual(len(protocols), 5)
            self.assertFalse(assignments["transaction_id"].duplicated().any())
            train_ids = set(features.loc[features["split"] == "train", "transaction_id"])
            self.assertTrue(set(assignments["transaction_id"]).issubset(train_ids))
            for protocol in protocols:
                self.assertLess(pd.Timestamp(protocol["fitEndAt"]), pd.Timestamp(protocol["holdoutStartAt"]))
                self.assertFalse(protocol["futureRowsUsedForFit"])
                self.assertEqual(protocol["fitRows"], protocol["scaler"]["fitRows"])
                cutoff = pd.Timestamp(protocol["holdoutStartAt"]).normalize() - pd.Timedelta(days=1)
                expected_fit = features[
                    features["split"].eq("train") & (pd.to_datetime(features["transaction_time"]) < cutoff)
                ]
                np.testing.assert_allclose(
                    protocol["scaler"]["mean"],
                    expected_fit[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64).mean(axis=0),
                    rtol=1e-10,
                    atol=1e-10,
                )

    def test_preparation_persists_lineage_and_excludes_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "silver": root / "finance.csv",
                "metadata": root / "finance.metadata.json",
                "audit": root / "finance.audit.json",
                "sequences": root / "finance_sequences.npz",
                "features": root / "finance_features.csv",
                "assignments": root / "finance_oof.csv",
                "manifest": root / "finance_sequence_manifest.json",
            }
            prepare_finance_benchmark(
                days=40,
                customers=80,
                terminals=30,
                seed=11,
                minimum_rows=1_000,
                silver_path=paths["silver"],
                metadata_path=paths["metadata"],
                audit_path=paths["audit"],
            )
            manifest = prepare_finance_sequence_protocol(
                window=6,
                folds=5,
                purge_days=1,
                silver_path=paths["silver"],
                metadata_path=paths["metadata"],
                audit_path=paths["audit"],
                sequences_path=paths["sequences"],
                features_path=paths["features"],
                assignments_path=paths["assignments"],
                manifest_path=paths["manifest"],
            )
            raw = pd.read_csv(paths["silver"])
            persisted_features = pd.read_csv(paths["features"])
            with np.load(paths["sequences"]) as arrays:
                encoded_ids = set(arrays["transaction_id"].tolist())
                self.assertEqual(arrays["x"].shape[1:], (6, len(FEATURE_COLUMNS) + 1))
            test_ids = set(raw.loc[raw["split"] == "test", "transaction_id"])
            status = get_finance_sequence_status(manifest_path=paths["manifest"], metadata_path=paths["metadata"])

            self.assertTrue(test_ids.isdisjoint(encoded_ids))
            self.assertFalse((persisted_features["split"] == "test").any())
            self.assertEqual(manifest["sequences"]["testRowsEncoded"], 0)
            self.assertEqual(
                manifest["sequences"]["developmentRows"],
                manifest["sequences"]["trainingRows"] + manifest["sequences"]["externalValidationRows"],
            )
            self.assertTrue(manifest["testLock"]["locked"])
            self.assertFalse(manifest["testLock"]["evaluated"])
            self.assertTrue(status["artifactIntegrity"])
            self.assertTrue(status["lineageCurrent"])
            self.assertTrue(status["readyForBaseModelPilot"])
            self.assertFalse(status["readyForThesisTraining"])

    def test_scaling_preserves_padding_and_mask(self) -> None:
        x = np.zeros((1, 3, len(FEATURE_COLUMNS) + 1), dtype=np.float32)
        x[0, 2, :-1] = np.arange(len(FEATURE_COLUMNS), dtype=np.float32) + 2.0
        x[0, 2, -1] = 1.0
        scaler = {"mean": [1.0] * len(FEATURE_COLUMNS), "scale": [2.0] * len(FEATURE_COLUMNS)}
        scaled = scale_finance_sequences(x, scaler)

        np.testing.assert_array_equal(scaled[0, :2, :-1], 0.0)
        np.testing.assert_array_equal(scaled[0, :, -1], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(scaled[0, 2, :-1], (x[0, 2, :-1] - 1.0) / 2.0)


if __name__ == "__main__":
    unittest.main()
