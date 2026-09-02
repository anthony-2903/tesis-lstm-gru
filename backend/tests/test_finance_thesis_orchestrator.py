from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from app.finance.thesis_orchestrator import FinanceThesisOrchestrator, FinanceThesisPaused


class FinanceThesisOrchestratorTests(unittest.TestCase):
    def test_observes_pause_requested_by_another_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = FinanceThesisOrchestrator(job_dir=root)
            manager._jobs["active"] = {"jobId": "active", "status": "running", "cancelRequested": False, "pauseRequested": False}
            manager._persist(manager._jobs["active"])
            path = root / "active.json"
            persisted = json.loads(path.read_text(encoding="utf-8"))
            persisted["pauseRequested"] = True
            path.write_text(json.dumps(persisted), encoding="utf-8")

            with self.assertRaises(FinanceThesisPaused):
                manager._check_control("active")
            self.assertTrue(manager.get("active")["pauseRequested"])
            self.assertFalse(manager.cancel("active")["cancelRequested"])
            manager.shutdown()

    def test_waits_for_private_real_dataset_without_starting_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "app.finance.thesis_orchestrator.get_real_finance_data_status",
            return_value={"readyForThesisTraining": False, "message": "Licencia pendiente", "manifestPath": "manifest.json"},
        ), patch("app.finance.thesis_orchestrator.run_finance_oof_experiment") as train:
            manager = FinanceThesisOrchestrator(job_dir=Path(directory))
            job = manager.submit({"epochs": 1})
            manager._futures[job["jobId"]].result(timeout=5)
            finished = manager.get(job["jobId"])
            self.assertEqual(finished["status"], "waiting_for_external_data")
            self.assertEqual(finished["blocker"]["code"], "ieee_cis_required")
            self.assertFalse(finished["testPolicy"]["testSetUsed"])
            train.assert_not_called()
            resumed = manager.resume(job["jobId"])
            self.assertNotEqual(resumed["jobId"], job["jobId"])
            self.assertEqual(resumed["executionRunId"], job["executionRunId"])
            self.assertEqual(resumed["resumedFromJobId"], job["jobId"])
            self.assertFalse(resumed["pauseRequested"])
            manager._futures[resumed["jobId"]].result(timeout=5)
            manager.shutdown()

    def test_runs_all_stages_and_seals_without_test_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stacking_path = root / "stacking.json"
            validation_path = root / "validation.json"
            diversity_path = root / "diversity.json"
            for path in (stacking_path, validation_path, diversity_path):
                path.write_text("{}", encoding="utf-8")
            patches = [
                patch("app.finance.thesis_orchestrator.get_real_finance_data_status", return_value={"readyForThesisTraining": True}),
                patch("app.finance.thesis_orchestrator.get_finance_sequence_status", return_value={"readyForThesisTraining": True}),
                patch("app.finance.thesis_orchestrator.run_finance_oof_experiment", return_value={"runId": "base"}),
                patch("app.finance.thesis_orchestrator.run_finance_stacking_experiment", return_value={"manifest": {"path": str(stacking_path)}}),
                patch("app.finance.thesis_orchestrator.run_finance_temporal_validation", return_value={"runId": "validation", "manifest": {"path": str(validation_path)}}),
                patch("app.finance.thesis_orchestrator.run_finance_diversity_ablation", return_value={"runId": "diversity", "manifest": {"path": str(diversity_path)}}),
                patch("app.finance.thesis_orchestrator.run_finance_optimized_stacking_revalidation", return_value={"runId": "revalidation"}),
                patch("app.finance.thesis_orchestrator.audit_finance_freeze_readiness", return_value={"ready": True, "blockingCheckIds": []}),
                patch("app.finance.thesis_orchestrator.create_finance_freeze_package", return_value={"freezeId": "finance-freeze-test"}),
            ]
            entered = [item.start() for item in patches]
            try:
                manager = FinanceThesisOrchestrator(job_dir=root / "jobs")
                job = manager.submit({"epochs": 1})
                manager._futures[job["jobId"]].result(timeout=5)
                finished = manager.get(job["jobId"])
                self.assertEqual(finished["status"], "completed")
                self.assertEqual(finished["result"]["freezeId"], "finance-freeze-test")
                self.assertEqual([item["stage"] for item in finished["stages"]], ["real_data_gate", "sequences", "base_models", "stacking", "validation", "diversity", "revalidation", "freeze"])
                self.assertFalse(finished["testPolicy"]["testEvaluationAuthorized"])
                manager.shutdown()
            finally:
                for item in reversed(patches):
                    item.stop()


if __name__ == "__main__":
    unittest.main()
