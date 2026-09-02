from __future__ import annotations

import json
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

from app.energy.jobs import EnergyExperimentJobManager


CONFIG = {
    "protocol": "demo",
    "source": "sample",
    "window": 12,
    "horizon": 1,
    "gapSteps": 6,
    "folds": 3,
    "epochs": 1,
    "batchSize": 16,
    "seeds": [42],
    "modelIds": ["lstm", "gru"],
}


class EnergyExperimentJobTests(unittest.TestCase):
    def test_job_runs_in_background_and_persists_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = EnergyExperimentJobManager(job_dir=Path(directory))

            def fake_run(**kwargs):
                callback = kwargs["progress_callback"]
                callback({"stage": "training", "event": "model_completed", "completedUnits": 1, "totalUnits": 2, "modelId": "lstm", "fold": 0, "seed": 42, "message": "LSTM listo"})
                callback({"stage": "completed", "event": "completed", "completedUnits": 2, "totalUnits": 2, "message": "Listo"})
                return {"run": {"runId": "run-test"}}

            try:
                with patch("app.energy.jobs.run_energy_experiment_for_api", side_effect=fake_run):
                    submitted = manager.submit(CONFIG)
                    manager._futures[submitted["jobId"]].result(timeout=5)
                completed = manager.get(submitted["jobId"])
                self.assertEqual(completed["status"], "completed")
                self.assertEqual(completed["progress"]["percent"], 100.0)
                self.assertEqual(completed["resultRunId"], "run-test")
                stored = json.loads((Path(directory) / f"{submitted['jobId']}.json").read_text(encoding="utf-8"))
                self.assertEqual(stored["status"], "completed")
            finally:
                manager.shutdown()

    def test_running_job_from_previous_process_is_marked_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old-job.json"
            path.write_text(json.dumps({
                "jobId": "old-job", "status": "running", "createdAt": "2026-01-01T00:00:00+00:00",
                "startedAt": None, "finishedAt": None, "cancelRequested": False, "config": CONFIG,
                "progress": {"stage": "training", "message": "Entrenando"}, "resultRunId": None, "error": None,
            }), encoding="utf-8")
            manager = EnergyExperimentJobManager(job_dir=Path(directory))
            try:
                restored = manager.get("old-job")
                self.assertEqual(restored["status"], "interrupted")
                self.assertIn("reinició", restored["error"])
            finally:
                manager.shutdown()

    def test_queued_job_can_be_cancelled_without_starting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = EnergyExperimentJobManager(max_workers=1, job_dir=Path(directory))
            started = Event()
            release = Event()

            def slow_run(**kwargs):
                started.set()
                release.wait(timeout=5)
                return {"run": {"runId": "first-run"}}

            try:
                with patch("app.energy.jobs.run_energy_experiment_for_api", side_effect=slow_run):
                    first = manager.submit(CONFIG)
                    self.assertTrue(started.wait(timeout=2))
                    second = manager.submit(CONFIG)
                    cancelled = manager.cancel(second["jobId"])
                    self.assertEqual(cancelled["status"], "cancelled")
                    release.set()
                    manager._futures[first["jobId"]].result(timeout=5)
                self.assertEqual(manager.get(second["jobId"])["status"], "cancelled")
            finally:
                release.set()
                manager.shutdown()


if __name__ == "__main__":
    unittest.main()
