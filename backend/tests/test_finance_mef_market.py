from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from app.finance.mef_market_data import (
    TARGET_COLUMN,
    build_audited_mef_market_silver,
    get_mef_market_dataset_status,
    load_mef_sovereign_indices,
    prepare_mef_market_dataset,
)


class FinanceMefMarketDataTests(unittest.TestCase):
    def test_bom_columns_duplicates_and_zero_placeholders_are_audited(self) -> None:
        rows = []
        dates = pd.bdate_range("2018-01-01", periods=1_010)
        for index, timestamp in enumerate(dates):
            rows.append({
                "timestamp": timestamp,
                "nominal_index": 200.0 + index * 0.05,
                "nominal_annual_yield": 0.05,
                "real_index": 180.0 + index * 0.02,
                "real_annual_yield": 0.03,
            })
        rows[100]["real_index"] = 0.0
        rows[100]["real_annual_yield"] = 0.0
        rows.append({**rows[200], "nominal_annual_yield": 0.07})

        silver, audit = build_audited_mef_market_silver(pd.DataFrame(rows), minimum_rows=1_000)

        self.assertTrue(audit["readiness"]["ready"])
        self.assertEqual(audit["source"]["duplicateTimestamps"], 1)
        self.assertEqual(audit["source"]["conflictingDuplicateDates"], 1)
        self.assertEqual(audit["source"]["targetDuplicateConflicts"], 0)
        self.assertEqual(audit["source"]["realIndexZeroPlaceholdersConvertedToMissing"], 1)
        self.assertEqual(len(silver), 1_009)
        self.assertIn(TARGET_COLUMN, silver)
        self.assertTrue(audit["chronologicalSplit"]["testSetLocked"])
        self.assertFalse(audit["chronologicalSplit"]["testSetUsed"])

    def test_preparation_persists_hash_verified_lineage_without_using_test(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            dates = pd.bdate_range("2019-01-01", periods=1_010)
            frame = pd.DataFrame({
                "FECHA": dates.strftime("%Y-%m-%d"),
                "INDICE_NOMINAL": 200.0 + np.arange(len(dates)) * 0.03,
                "RENT_ANUAL_IN": np.full(len(dates), 0.05),
                "INDICE_REAL": 180.0 + np.arange(len(dates)) * 0.02,
                "RENT_ANUAL_IR": np.full(len(dates), 0.03),
            })
            frame.to_csv(source, index=False, encoding="utf-8-sig")
            raw = root / "raw"
            raw.mkdir()
            payload = source.read_bytes()
            import hashlib

            digest = hashlib.sha256(payload).hexdigest()
            snapshot = raw / f"indices_soberanos_{digest[:12]}.csv"
            snapshot.write_bytes(payload)
            (raw / "latest.json").write_text(json.dumps({
                "sourceUrl": "https://example.invalid/indices.csv",
                "sourceVersion": "fixture",
                "retrievedAt": "2026-01-01T00:00:00+00:00",
                "rawPath": str(snapshot.resolve()),
                "sha256": digest,
                "bytes": len(payload),
                "http": {},
            }), encoding="utf-8")
            silver = root / "silver.csv"
            audit = root / "audit.json"
            metadata = root / "metadata.json"
            result = prepare_mef_market_dataset(
                source_url="https://example.invalid/indices.csv",
                raw_directory=raw,
                silver_path=silver,
                audit_path=audit,
                metadata_path=metadata,
                minimum_rows=1_000,
            )
            status = get_mef_market_dataset_status(metadata_path=metadata, audit_path=audit)

            self.assertTrue(result["audit"]["readiness"]["ready"])
            self.assertTrue(status["readyForThesisTraining"])
            self.assertTrue(status["integrityVerified"])
            self.assertFalse(status["testPolicy"]["testSetUsed"])
            self.assertEqual(status["audit"]["chronologicalSplit"]["lockedTestRows"], 152)

    def test_loader_strips_utf8_bom(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mef.csv"
            path.write_text(
                "\ufeffFECHA,INDICE_NOMINAL,RENT_ANUAL_IN,INDICE_REAL,RENT_ANUAL_IR\n"
                "2020-01-02,200,0.05,180,0.03\n",
                encoding="utf-8",
            )
            loaded = load_mef_sovereign_indices(path)
            self.assertEqual(list(loaded.columns), ["timestamp", "nominal_index", "nominal_annual_yield", "real_index", "real_annual_yield"])


if __name__ == "__main__":
    unittest.main()
