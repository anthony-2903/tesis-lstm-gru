from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from app.finance.benchmark import (
    apply_chronological_finance_split,
    audit_finance_benchmark,
    generate_finance_transactions,
    get_finance_dataset_status,
    prepare_finance_benchmark,
)


class FinanceBenchmarkTests(unittest.TestCase):
    def test_generator_is_deterministic_and_has_real_event_contract(self) -> None:
        first = generate_finance_transactions(days=35, customers=40, terminals=20, seed=42)
        second = generate_finance_transactions(days=35, customers=40, terminals=20, seed=42)

        pd.testing.assert_frame_equal(first, second)
        self.assertGreater(int(first["is_fraud"].sum()), 0)
        self.assertEqual(first["transaction_id"].nunique(), len(first))
        self.assertTrue({0, 1}.issubset(set(first["is_fraud"])))

    def test_split_is_strictly_chronological_and_locks_test(self) -> None:
        generated = generate_finance_transactions(days=40, customers=80, terminals=30, seed=7)
        silver = apply_chronological_finance_split(generated, days=40)
        audit = audit_finance_benchmark(silver, minimum_rows=1_000)

        train_end = silver[silver["split"] == "train"]["transaction_time"].max()
        validation_start = silver[silver["split"] == "validation"]["transaction_time"].min()
        validation_end = silver[silver["split"] == "validation"]["transaction_time"].max()
        test_start = silver[silver["split"] == "test"]["transaction_time"].min()
        self.assertLess(train_end, validation_start)
        self.assertLess(validation_end, test_start)
        self.assertTrue(audit["temporalAudit"]["chronological"])
        self.assertTrue(audit["temporalAudit"]["testLocked"])
        self.assertFalse(audit["temporalAudit"]["testEvaluated"])

    def test_preparation_persists_hashes_and_never_promotes_synthetic_to_thesis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silver = root / "finance.csv"
            metadata = root / "finance.metadata.json"
            audit = root / "finance.audit.json"
            result = prepare_finance_benchmark(
                days=40,
                customers=80,
                terminals=30,
                seed=9,
                minimum_rows=1_000,
                silver_path=silver,
                metadata_path=metadata,
                audit_path=audit,
            )
            status = get_finance_dataset_status(metadata_path=metadata, audit_path=audit)

            self.assertTrue(status["available"])
            self.assertTrue(status["lineageCurrent"])
            self.assertTrue(status["readyForPipelinePilot"])
            self.assertFalse(status["readyForThesisTraining"])
            self.assertTrue(status["testLock"]["locked"])
            self.assertFalse(status["testLock"]["evaluated"])
            self.assertEqual(result["metadata"]["contract"]["target"], "is_fraud")

            silver.write_text(silver.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            stale = get_finance_dataset_status(metadata_path=metadata, audit_path=audit)
            self.assertFalse(stale["lineageCurrent"])
            self.assertFalse(stale["readyForPipelinePilot"])


if __name__ == "__main__":
    unittest.main()
