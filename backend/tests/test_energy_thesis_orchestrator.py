from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from app.energy.thesis_orchestrator import EnergyThesisOrchestrator, EnergyThesisPaused


class EnergyThesisOrchestratorTests(unittest.TestCase):
    def test_pause_endpoint_manager_refreshes_job_created_externally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = EnergyThesisOrchestrator(job_dir=root)
            external = {"jobId": "external", "status": "running", "cancelRequested": False, "pauseRequested": False, "progress": {"message": "running"}}
            (root / "external.json").write_text(json.dumps(external), encoding="utf-8")

            paused = controller.pause("external")

            self.assertIsNotNone(paused)
            self.assertTrue(paused["pauseRequested"])
            self.assertTrue(json.loads((root / "external.json").read_text(encoding="utf-8"))["pauseRequested"])
            controller.shutdown()

    def test_observes_pause_requested_by_another_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = EnergyThesisOrchestrator(job_dir=root)
            manager._jobs["active"] = {"jobId": "active", "status": "running", "cancelRequested": False, "pauseRequested": False}
            manager._persist(manager._jobs["active"])
            path = root / "active.json"
            persisted = json.loads(path.read_text(encoding="utf-8"))
            persisted["pauseRequested"] = True
            path.write_text(json.dumps(persisted), encoding="utf-8")

            with self.assertRaises(EnergyThesisPaused):
                manager._check_control("active")
            self.assertTrue(manager.get("active")["pauseRequested"])
            self.assertFalse(manager.cancel("active")["cancelRequested"])
            manager.shutdown()

    def test_data_gate_prevents_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch("app.energy.thesis_orchestrator.get_energy_dataset_status", return_value={"readyForThesisPilot": False, "message": "pending"}), patch("app.energy.thesis_orchestrator.run_energy_experiment_for_api") as train:
            manager = EnergyThesisOrchestrator(job_dir=Path(directory))
            job = manager.submit({"epochs": 1})
            manager._futures[job["jobId"]].result(timeout=5)
            result = manager.get(job["jobId"])
            self.assertEqual(result["status"], "waiting_for_scientific_data")
            self.assertFalse(result["testPolicy"]["testSetUsed"])
            train.assert_not_called()
            resumed = manager.resume(job["jobId"])
            self.assertNotEqual(resumed["jobId"], job["jobId"])
            self.assertEqual(resumed["executionRunId"], job["executionRunId"])
            self.assertEqual(resumed["resumedFromJobId"], job["jobId"])
            self.assertFalse(resumed["pauseRequested"])
            manager._futures[resumed["jobId"]].result(timeout=5)
            manager.shutdown()

    def test_complete_chain_seals_without_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revalidation_path = root / "revalidation.json"
            revalidation_path.write_text("{}", encoding="utf-8")
            patches = [
                patch("app.energy.thesis_orchestrator.get_energy_dataset_status", return_value={"readyForThesisPilot": True}),
                patch("app.energy.thesis_orchestrator.run_energy_experiment_for_api", return_value={"run": {"runId": "base"}}),
                patch("app.energy.thesis_orchestrator.run_energy_optimized_stacking_revalidation", return_value={"runId": "revalidation", "manifest": {"path": str(revalidation_path)}}),
                patch("app.energy.thesis_orchestrator.audit_energy_freeze_readiness", return_value={"ready": True, "blockingCheckIds": []}),
                patch("app.energy.thesis_orchestrator.create_energy_freeze_package", return_value={"freezeId": "energy-freeze-test"}),
            ]
            for item in patches:
                item.start()
            try:
                manager = EnergyThesisOrchestrator(job_dir=root / "jobs")
                job = manager.submit({"epochs": 1})
                manager._futures[job["jobId"]].result(timeout=5)
                result = manager.get(job["jobId"])
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["result"]["freezeId"], "energy-freeze-test")
                self.assertFalse(result["testPolicy"]["testEvaluationAuthorized"])
                manager.shutdown()
            finally:
                for item in reversed(patches):
                    item.stop()


if __name__ == "__main__":
    unittest.main()
