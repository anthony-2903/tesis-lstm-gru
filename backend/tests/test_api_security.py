from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class ApiSecurityTests(unittest.TestCase):
    def test_local_loopback_origin_is_allowed(self) -> None:
        with TestClient(app) as client:
            response = client.get(
                "/api/health",
                headers={"origin": "http://127.0.0.1:5173"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "http://127.0.0.1:5173",
        )

    def test_renamed_cloudflare_worker_origin_is_allowed(self) -> None:
        origin = "https://tesis-recurrentes-staking.anthonyjanampacalderon10.workers.dev"
        with TestClient(app) as client:
            response = client.options(
                "/api/health",
                headers={
                    "origin": origin,
                    "access-control-request-method": "GET",
                    "access-control-request-headers": "content-type",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], origin)

    def test_unrelated_cloudflare_account_is_rejected(self) -> None:
        with TestClient(app) as client:
            response = client.options(
                "/api/health",
                headers={
                    "origin": "https://untrusted.other-account.workers.dev",
                    "access-control-request-method": "GET",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_write_token_blocks_mutation_but_not_health_read(self) -> None:
        with patch("app.main.API_WRITE_TOKEN", "correct-secret"):
            with TestClient(app) as client:
                denied = client.post("/api/pipeline/run")
                self.assertEqual(denied.status_code, 401)
                self.assertEqual(denied.headers["x-content-type-options"], "nosniff")
                self.assertEqual(denied.headers["referrer-policy"], "no-referrer")
                health = client.get("/api/health")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.headers["x-content-type-options"], "nosniff")

    def test_thesis_pause_routes_require_token_and_delegate_by_domain(self) -> None:
        cases = (
            ("/api/phishing/thesis/phishing-job/pause", "phishing-job", "app.main.phishing_thesis_orchestrator"),
            ("/api/energy/thesis/energy-job/pause", "energy-job", "app.main.energy_thesis_orchestrator"),
            ("/api/finance/thesis/finance-job/pause", "finance-job", "app.main.finance_thesis_orchestrator"),
        )
        with patch("app.main.API_WRITE_TOKEN", "correct-secret"), TestClient(app) as client:
            for route, job_id, manager_path in cases:
                with self.subTest(route=route):
                    denied = client.post(route)
                    self.assertEqual(denied.status_code, 401)
                    with patch(f"{manager_path}.pause", return_value={"jobId": "job", "status": "running", "pauseRequested": True}) as pause:
                        accepted = client.post(route, headers={"x-api-key": "correct-secret"})
                    self.assertEqual(accepted.status_code, 200)
                    self.assertTrue(accepted.json()["pauseRequested"])
                    pause.assert_called_once_with(job_id)


if __name__ == "__main__":
    unittest.main()
