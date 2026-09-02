from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from app.thesis_status import get_thesis_status


class ThesisStatusTests(unittest.TestCase):
    def test_aggregates_disk_jobs_without_using_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_job(root, "phishing_thesis_jobs", "p1", "running", 18.0)
            self._write_job(root, "energy_thesis_jobs", "e1", "completed", 95.0)
            self._write_job(
                root,
                "finance_thesis_jobs",
                "f1",
                "waiting_for_external_data",
                0.0,
                blocker={"code": "ieee_cis_required", "message": "Falta IEEE-CIS."},
            )
            (root / "thesis_chain_state.json").write_text(
                json.dumps({"status": "running", "domain": "phishing", "ownerPid": 123}), encoding="utf-8"
            )

            result = get_thesis_status(root)

            self.assertEqual(result["overall"]["percent"], 39.33)
            self.assertEqual(result["overall"]["remainingPercent"], 60.67)
            self.assertEqual(result["overall"]["activeDomain"], "phishing")
            self.assertEqual(result["overall"]["completedDomains"], 1)
            self.assertEqual(result["domains"][1]["progress"]["percent"], 100.0)
            self.assertEqual(result["domains"][1]["progress"]["remainingPercent"], 0.0)
            self.assertEqual(result["blockers"][0]["code"], "ieee_cis_required")
            self.assertNotIn("manifestPath", result["blockers"][0])
            self.assertTrue(result["testPolicy"]["locked"])
            self.assertFalse(result["progressMethod"]["includesSoftwareImplementation"])

    def test_missing_and_malformed_jobs_are_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "phishing_thesis_jobs"
            malformed.mkdir()
            (malformed / "broken.json").write_text("{", encoding="utf-8")

            result = get_thesis_status(root)

            self.assertEqual(result["overall"]["percent"], 0.0)
            self.assertEqual(result["overall"]["remainingPercent"], 100.0)
            self.assertEqual([item["status"] for item in result["domains"]], ["pending", "pending", "pending"])

    def test_reports_safe_pause_as_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_job(root, "phishing_thesis_jobs", "p1", "paused", 22.8)

            result = get_thesis_status(root)

            self.assertEqual(result["overall"]["status"], "paused")
            self.assertIn("Reanudar", result["domains"][0]["nextAction"])
            self.assertTrue(result["testPolicy"]["locked"])

    @staticmethod
    def _write_job(
        root: Path,
        directory: str,
        job_id: str,
        status: str,
        percent: float,
        *,
        blocker: dict | None = None,
    ) -> None:
        target = root / directory
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{job_id}.json").write_text(
            json.dumps(
                {
                    "jobId": job_id,
                    "executionRunId": job_id,
                    "status": status,
                    "createdAt": "2026-01-01T00:00:00+00:00",
                    "progress": {"stage": "test_stage", "percent": percent, "message": "status"},
                    "blocker": blocker,
                    "testPolicy": {"testSetUsed": False, "testEvaluationAuthorized": False},
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
