from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from app.scientific_data_summary import get_scientific_data_summary


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ScientificDataSummaryTests(unittest.TestCase):
    def test_uses_audited_unique_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._assert_audited_summary(Path(directory))

    def _assert_audited_summary(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        experiments = tmp_path / "experiments"
        _write(
            results / "phishing_data_audit.json",
            {
                "datasetId": "phishing-test",
                "source": {"a": {"originalRows": 100}, "b": {"originalRows": 200}},
                "classDistribution": {"negative": 40, "positive": 40, "total": 80},
                "splitDistribution": {
                    "train": {"rows": 50},
                    "validation": {"rows": 15},
                    "test": {"rows": 15},
                },
                "readiness": {"testSetUsed": False},
            },
        )
        _write(
            results / "energy_data_audit.json",
            {
                "datasetId": "energy-test",
                "source": {"originalRows": 101},
                "selectedSegment": {"rows": 100, "startAt": "a", "endAt": "b"},
                "readiness": {"testSetUsed": False},
            },
        )
        _write(
            results / "finance_mef_data_audit.json",
            {
                "datasetId": "finance-test",
                "source": {"originalRows": 42, "usableRows": 40, "startAt": "a", "endAt": "b"},
                "chronologicalSplit": {"developmentRows": 30, "lockedTestRows": 10, "testSetLocked": True},
                "readiness": {"testSetUsed": False},
            },
        )

        temporal_folds = [
            {"trainRows": 30, "gapSteps": 2, "validationRows": 10},
            {"trainRows": 40, "gapSteps": 2, "validationRows": 10},
        ]
        _write(
            experiments / "phishing-run" / "phishing_oof_v1" / "oof_manifest.json",
            {
                "domain": "phishing",
                "status": "thesis_base_models_candidate",
                "dataset": {"rowsUsed": 50},
            },
        )
        _write(
            experiments / "energy-run" / "energy_oof_v1" / "oof_manifest.json",
            {
                "domain": "energia",
                "status": "thesis_candidate",
                "walkForward": {"window": 3, "horizon": 1, "folds": temporal_folds},
            },
        )
        _write(
            experiments / "finance-run" / "finance_market_oof_v1" / "oof_manifest.json",
            {
                "domain": "finanzas",
                "status": "thesis_candidate",
                "walkForward": {"window": 3, "horizon": 1, "folds": temporal_folds},
            },
        )

        summary = get_scientific_data_summary(results_dir=results, experiments_dir=experiments)
        by_domain = {item["id"]: item for item in summary["domains"]}

        self.assertEqual(summary["totalUsableObservations"], 220)
        self.assertTrue(summary["demoExcluded"])
        self.assertFalse(summary["finalTestUsed"])
        self.assertEqual(by_domain["phishing"]["originalRows"], 300)
        self.assertEqual(by_domain["phishing"]["developmentRows"], 65)
        self.assertEqual(by_domain["energia"]["developmentRows"], 52)
        self.assertEqual(by_domain["energia"]["lockedTestRows"], 48)
        self.assertEqual(by_domain["energia"]["oofUniqueRows"], 14)
        self.assertEqual(by_domain["finanzas"]["lockedTestRows"], 10)
        self.assertEqual(by_domain["finanzas"]["oofUniqueRows"], 14)

    def test_marks_missing_domains_without_inventing_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = get_scientific_data_summary(
                results_dir=root / "results",
                experiments_dir=root / "experiments",
            )

        self.assertFalse(summary["available"])
        self.assertEqual(summary["availableDomains"], 0)
        self.assertEqual(summary["totalUsableObservations"], 0)
        self.assertTrue(all(item["available"] is False for item in summary["domains"]))


if __name__ == "__main__":
    unittest.main()
