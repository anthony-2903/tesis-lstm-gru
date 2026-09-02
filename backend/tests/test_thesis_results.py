from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from app.thesis_results import get_thesis_results_summary


class ThesisResultsTests(unittest.TestCase):
    def test_builds_honest_six_candidate_table_without_claiming_final_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = root / "experiments"
            results = root / "results"
            target = experiments / "run" / "energy_oof_v1" / "energy_revalidation_v1"
            target.mkdir(parents=True)
            metrics = [
                {"predictorId": "lstm", "rmse": 2.0, "mae": 1.0},
                {"predictorId": "gru", "rmse": 1.9, "mae": 1.0},
                {"predictorId": "brnn", "rmse": 2.1, "mae": 1.0},
                {"predictorId": "tcn", "rmse": 1.8, "mae": 1.0},
                {"predictorId": "transformer", "rmse": 2.2, "mae": 1.0},
                {"predictorId": "optimized_stacking", "rmse": 1.7, "mae": 0.9},
            ]
            (target / "revalidation_manifest.json").write_text(
                json.dumps(
                    {
                        "runId": "energy-run",
                        "status": "thesis_optimized_candidate",
                        "protocol": {"independentComparisonFold": 5},
                        "independentComparison": {
                            "metrics": metrics,
                            "seedMetrics": [
                                {"seed": seed, "predictorId": row["predictorId"], "rmse": row["rmse"] + seed / 100_000}
                                for seed in (42, 101, 202, 303, 404)
                                for row in metrics
                            ],
                            "ranking": [row["predictorId"] for row in sorted(metrics, key=lambda row: row["rmse"])],
                            "pairedInference": {"metric": "rmse", "ciLower": -0.3, "ciUpper": -0.1},
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = get_thesis_results_summary(experiments, results)
            energy = summary["domains"][0]

            self.assertEqual(len(energy["comparison"]), 6)
            self.assertEqual(energy["sixCandidateWinner"], "stacking")
            self.assertEqual(energy["stackingRank"], 1)
            self.assertTrue(energy["stackingBeatsBestBase"])
            self.assertEqual(len(energy["seedDistributions"]), 6)
            self.assertTrue(all(item["seedCount"] == 5 for item in energy["seedDistributions"]))
            self.assertEqual(energy["evidenceStatus"], "scientific_candidate_pending_freeze")
            self.assertFalse(energy["eligibleForThesisConclusion"])
            self.assertFalse(summary["readiness"]["finalTestUsed"])

    def test_phishing_excludes_simple_ensemble_but_reports_all_candidate_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = root / "experiments"
            results = root / "results"
            target = experiments / "run" / "phishing_oof_v1" / "external_validation_v1"
            target.mkdir(parents=True)
            comparison = [{"candidateId": "mean", "family": "baseline", "prAuc": 0.99}]
            comparison.extend({"candidateId": model, "family": "base", "prAuc": 0.80 + index / 100} for index, model in enumerate(("lstm", "gru", "brnn", "tcn", "transformer")))
            comparison.append({"candidateId": "stacking_ridge", "family": "stacking", "prAuc": 0.98})
            for row in comparison:
                row["seedMetrics"] = [{"seed": 42, "prAuc": row["prAuc"]}]
            (target / "external_validation_manifest.json").write_text(
                json.dumps(
                    {
                        "runId": "p-validation",
                        "status": "validation_selection_candidate",
                        "baseRun": {"seeds": [42, 101, 202, 303, 404]},
                        "selection": {"leadingStackingCandidateId": "stacking_ridge"},
                        "comparison": comparison,
                    }
                ),
                encoding="utf-8",
            )

            phishing = get_thesis_results_summary(experiments, results)["domains"][1]

            self.assertEqual(len(phishing["comparison"]), 6)
            self.assertEqual(phishing["sixCandidateWinner"], "stacking")
            self.assertEqual(phishing["allEvaluatedCandidatesWinner"], "mean")
            self.assertEqual(len(phishing["seedDistributions"]), 6)
            self.assertFalse(phishing["eligibleForThesisConclusion"])


if __name__ == "__main__":
    unittest.main()
