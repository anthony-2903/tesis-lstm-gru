from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from scripts.run_thesis_chain import _run_or_resume_and_wait, _select_candidate


class ThesisChainSupervisorTests(unittest.TestCase):
    def test_newer_completed_job_wins_over_explicit_older_job(self) -> None:
        preferred = {"jobId": "old", "status": "paused", "createdAt": "2026-08-23T10:00:00+00:00"}
        latest = {"jobId": "new", "status": "completed", "createdAt": "2026-08-23T11:00:00+00:00"}

        self.assertIs(_select_candidate(latest, preferred), latest)

    def test_completed_domain_is_idempotent(self) -> None:
        completed = {"jobId": "done", "status": "completed", "createdAt": "2026-08-23T11:00:00+00:00"}
        manager = Mock()
        manager.latest.return_value = completed

        result = _run_or_resume_and_wait(manager, "energy")

        self.assertIs(result, completed)
        manager.resume.assert_not_called()
        manager.submit.assert_not_called()

    def test_paused_domain_resumes_and_waits_for_new_job(self) -> None:
        paused = {"jobId": "paused", "status": "paused", "createdAt": "2026-08-23T11:00:00+00:00"}
        resumed = {"jobId": "resumed", "status": "queued"}
        completed = {"jobId": "resumed", "status": "completed"}
        future = Mock()
        manager = Mock()
        manager.latest.return_value = paused
        manager.resume.return_value = resumed
        manager._futures = {"resumed": future}
        manager.get.return_value = completed

        with patch("scripts.run_thesis_chain._state"):
            result = _run_or_resume_and_wait(manager, "phishing")

        self.assertIs(result, completed)
        manager.resume.assert_called_once_with("paused")
        future.result.assert_called_once_with()
        manager.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
