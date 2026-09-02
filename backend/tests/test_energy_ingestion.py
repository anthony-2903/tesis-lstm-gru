from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from app.energy.ingestion import (
    TARGET_COLUMN,
    build_audited_energy_silver,
    download_versioned_energy_source,
    sha256_file,
)
from app.energy.data_jobs import EnergyDataPreparationManager
from unittest.mock import patch


class EnergyIngestionTests(unittest.TestCase):
    def test_selects_longest_hourly_segment_without_imputing_target(self) -> None:
        first = pd.date_range("2024-01-01", periods=60, freq="h", tz="UTC")
        second = pd.date_range("2024-01-05", periods=80, freq="h", tz="UTC")
        timestamps = first.append(second)
        frame = pd.DataFrame({
            "utc_timestamp": timestamps.astype(str),
            TARGET_COLUMN: np.arange(len(timestamps), dtype=float),
            "DE_wind_generation_actual": np.arange(len(timestamps), dtype=float),
        })
        frame.loc[70, "DE_wind_generation_actual"] = np.nan

        silver, audit = build_audited_energy_silver(frame, minimum_rows=64)

        self.assertEqual(len(silver), 80)
        self.assertTrue(audit["readiness"]["ready"])
        self.assertFalse(audit["readiness"]["targetImputed"])
        self.assertEqual(int(silver[TARGET_COLUMN].isna().sum()), 0)
        self.assertEqual(int(silver["DE_wind_generation_actual"].isna().sum()), 1)
        self.assertIn("hour_sin", silver.columns)

    def test_duplicate_timestamps_are_rejected(self) -> None:
        frame = pd.DataFrame({
            "utc_timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            TARGET_COLUMN: [10.0, 11.0],
        })
        with self.assertRaisesRegex(ValueError, "duplicados"):
            build_audited_energy_silver(frame, minimum_rows=1)

    def test_verified_raw_snapshot_is_reused_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "opsd_2020-10-06_test.csv"
            raw.write_text("utc_timestamp,value\n2024-01-01,1\n", encoding="utf-8")
            digest, size = sha256_file(raw)
            source_url = "https://example.test/2020-10-06/source.csv"
            metadata = {
                "sourceUrl": source_url,
                "sourceVersion": "2020-10-06",
                "retrievedAt": "2026-01-01T00:00:00Z",
                "rawPath": str(raw),
                "sha256": digest,
                "bytes": size,
                "http": {},
                "reused": False,
            }
            (root / "latest.json").write_text(json.dumps(metadata), encoding="utf-8")

            reused = download_versioned_energy_source(source_url, root)

            self.assertTrue(reused["reused"])
            self.assertEqual(reused["sha256"], digest)

    def test_data_preparation_job_persists_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "job.json"
            manager = EnergyDataPreparationManager(state_path=state_path)
            fake_result = {
                "metadata": {"datasetId": "opsd-test", "silver": {"rows": 9000}},
                "audit": {"readiness": {"ready": True}},
            }
            try:
                with patch("app.energy.data_jobs.prepare_real_energy_dataset", return_value=fake_result):
                    submitted = manager.submit()
                    manager._executor.shutdown(wait=True)
                completed = manager.latest()
                self.assertEqual(completed["jobId"], submitted["jobId"])
                self.assertEqual(completed["status"], "completed")
                self.assertEqual(completed["result"]["rows"], 9000)
                self.assertTrue(state_path.exists())
            finally:
                manager.shutdown()


if __name__ == "__main__":
    unittest.main()
