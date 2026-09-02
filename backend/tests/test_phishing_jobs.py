from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.phishing.jobs import PhishingExperimentJobManager
from app.phishing.service import build_phishing_experiment_view_from_manifest
from app.phishing.validation_jobs import PhishingValidationJobManager


class PhishingJobTests(unittest.TestCase):
    def test_job_persists_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PhishingExperimentJobManager(job_dir=Path(directory))
            fake_view = {"run": {"runId": "phishing-run"}}
            config = {
                "protocol": "demo",
                "epochs": 1,
                "batchSize": 64,
                "patience": 0,
                "seeds": [42],
                "modelIds": ["lstm", "gru"],
                "demoMaxRows": 500,
            }
            try:
                with patch("app.phishing.jobs.run_phishing_experiment_for_api", return_value=fake_view):
                    submitted = manager.submit(config)
                    manager._executor.shutdown(wait=True)
                completed = manager.get(submitted["jobId"])
                self.assertEqual(completed["status"], "completed")
                self.assertEqual(completed["resultRunId"], "phishing-run")
                self.assertTrue(list(Path(directory).glob("*.json")))
            finally:
                manager.shutdown()

    def test_view_ranks_by_primary_metric_and_keeps_methodology_warning(self) -> None:
        manifest = {
            "runId": "demo",
            "status": "demo",
            "protocol": "demo",
            "aggregate": [
                {"modelId": "gru", "prAucMean": 0.8},
                {"modelId": "lstm", "prAucMean": 0.7},
            ],
            "validation": {"testSetLocked": True, "testSetUsed": False},
            "stacking": {"ready": True},
        }

        view = build_phishing_experiment_view_from_manifest(manifest)

        self.assertEqual(view["winner"]["modelId"], "gru")
        self.assertFalse(view["methodology"]["testSetUsed"])
        self.assertIn("fuente", view["methodology"]["warning"])

    def test_failed_job_can_resume_same_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PhishingExperimentJobManager(job_dir=Path(directory))
            config = {
                "protocol": "demo",
                "epochs": 1,
                "batchSize": 64,
                "patience": 0,
                "seeds": [42],
                "modelIds": ["lstm"],
                "demoMaxRows": 500,
            }
            try:
                with patch("app.phishing.jobs.run_phishing_experiment_for_api", side_effect=RuntimeError("interrupted")):
                    first = manager.submit(config)
                    manager._futures[first["jobId"]].result(timeout=5)
                self.assertEqual(manager.get(first["jobId"])["status"], "failed")

                with patch(
                    "app.phishing.jobs.run_phishing_experiment_for_api",
                    return_value={"run": {"runId": first["executionRunId"]}},
                ):
                    resumed = manager.resume(first["jobId"])
                    manager._futures[resumed["jobId"]].result(timeout=5)

                completed = manager.get(resumed["jobId"])
                self.assertEqual(completed["status"], "completed")
                self.assertEqual(completed["executionRunId"], first["executionRunId"])
                self.assertEqual(completed["resumedFromJobId"], first["jobId"])
            finally:
                manager.shutdown()

    def test_external_validation_job_persists_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PhishingValidationJobManager(job_dir=Path(directory))

            def fake_validation(*, progress_callback):
                progress_callback({
                    "stage": "completed",
                    "event": "completed",
                    "completedUnits": 6,
                    "totalUnits": 6,
                    "message": "Validación lista.",
                })
                return {"runId": "phishing-validation-run"}

            try:
                with patch("app.phishing.validation_jobs.run_phishing_external_validation", side_effect=fake_validation):
                    submitted = manager.submit()
                    manager._executor.shutdown(wait=True)
                completed = manager.get(submitted["jobId"])
                self.assertEqual(completed["status"], "completed")
                self.assertEqual(completed["resultRunId"], "phishing-validation-run")
                self.assertEqual(completed["progress"]["percent"], 100.0)
                self.assertTrue(list(Path(directory).glob("*.json")))
            finally:
                manager.shutdown()


if __name__ == "__main__":
    unittest.main()
