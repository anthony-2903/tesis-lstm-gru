from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.finance.jobs import FinanceExperimentJobManager
from app.finance.service import build_finance_experiment_view_from_manifest


class FinanceJobTests(unittest.TestCase):
    def test_background_job_persists_progress_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = FinanceExperimentJobManager(job_dir=Path(directory))
            config = {
                "protocol": "demo",
                "epochs": 1,
                "batchSize": 128,
                "patience": 1,
                "seeds": [42],
                "modelIds": ["lstm", "gru", "brnn", "tcn", "transformer"],
                "demoMaxRowsPerFold": 100,
            }

            def fake_run(**kwargs):
                kwargs["progress_callback"](
                    {
                        "stage": "training",
                        "event": "model_completed",
                        "completedUnits": 25,
                        "totalUnits": 25,
                        "message": "Piloto listo.",
                    }
                )
                return {"run": {"runId": "finance-run"}}

            try:
                with patch("app.finance.jobs.run_finance_experiment_for_api", side_effect=fake_run):
                    submitted = manager.submit(config)
                    manager._futures[submitted["jobId"]].result(timeout=5)
                completed = manager.get(submitted["jobId"])
                self.assertEqual(completed["status"], "completed")
                self.assertEqual(completed["resultRunId"], "finance-run")
                self.assertEqual(completed["progress"]["percent"], 100.0)
                self.assertTrue(list(Path(directory).glob("*.json")))
            finally:
                manager.shutdown()

    def test_view_ranks_pr_auc_and_preserves_test_lock(self) -> None:
        manifest = {
            "runId": "finance-demo",
            "status": "demo",
            "protocol": "demo",
            "aggregate": [
                {"modelId": "gru", "prAucMean": 0.30},
                {"modelId": "lstm", "prAucMean": 0.20},
            ],
            "validation": {"testSetLocked": True, "testSetUsed": False, "externalValidationUsed": False},
            "stacking": {"ready": True},
        }
        view = build_finance_experiment_view_from_manifest(manifest)
        self.assertEqual(view["winner"]["modelId"], "gru")
        self.assertEqual(view["winner"]["primaryMetric"], "prAuc")
        self.assertTrue(view["methodology"]["testSetLocked"])
        self.assertFalse(view["methodology"]["testSetUsed"])
        self.assertIn("Piloto", view["methodology"]["warning"])


if __name__ == "__main__":
    unittest.main()
