from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

from app.phishing.thesis_orchestrator import PhishingThesisOrchestrator, PhishingThesisPaused, _keras_container_hash, _load_verified_diversity, _load_verified_validation


class PhishingThesisOrchestratorTests(unittest.TestCase):
    def test_observes_pause_requested_by_another_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = PhishingThesisOrchestrator(job_dir=root)
            manager._jobs["active"] = {"jobId": "active", "status": "running", "cancelRequested": False, "pauseRequested": False}
            manager._persist(manager._jobs["active"])
            path = root / "active.json"
            persisted = json.loads(path.read_text(encoding="utf-8"))
            persisted["pauseRequested"] = True
            path.write_text(json.dumps(persisted), encoding="utf-8")

            with self.assertRaises(PhishingThesisPaused):
                manager._check_control("active")
            self.assertTrue(manager.get("active")["pauseRequested"])
            self.assertFalse(manager.cancel("active")["cancelRequested"])
            manager.shutdown()

    def test_scientific_gate_prevents_training_when_dataset_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch("app.phishing.thesis_orchestrator.get_phishing_dataset_status", return_value={"readyForThesisTraining": False, "message": "curation pending"}), patch("app.phishing.thesis_orchestrator.run_phishing_oof_experiment") as train:
            manager = PhishingThesisOrchestrator(job_dir=Path(directory))
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

    def test_complete_chain_reaches_seal_without_test_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation_path, diversity_path = root / "validation.json", root / "diversity.json"
            validation_path.write_text("{}", encoding="utf-8")
            diversity_path.write_text("{}", encoding="utf-8")
            patches = [
                patch("app.phishing.thesis_orchestrator.get_phishing_dataset_status", return_value={"readyForThesisTraining": True}),
                patch("app.phishing.thesis_orchestrator.get_phishing_sequence_status", return_value={"readyForBaseModelTraining": True, "lineageCurrent": True}),
                patch("app.phishing.thesis_orchestrator.run_phishing_oof_experiment", return_value={"runId": "base"}),
                patch("app.phishing.thesis_orchestrator.run_phishing_stacking_experiment", return_value={"runId": "stacking"}),
                patch("app.phishing.thesis_orchestrator.run_phishing_external_validation", return_value={"runId": "validation", "manifest": {"path": str(validation_path)}}),
                patch("app.phishing.thesis_orchestrator.run_phishing_diversity_ablation", return_value={"runId": "diversity", "manifest": {"path": str(diversity_path)}}),
                patch("app.phishing.thesis_orchestrator.audit_phishing_freeze_readiness", return_value={"ready": True, "failedChecks": []}),
                patch("app.phishing.thesis_orchestrator.create_phishing_freeze_package", return_value={"freezeId": "phishing-freeze-test"}),
            ]
            for item in patches:
                item.start()
            try:
                manager = PhishingThesisOrchestrator(job_dir=root / "jobs")
                job = manager.submit({"epochs": 1})
                manager._futures[job["jobId"]].result(timeout=5)
                result = manager.get(job["jobId"])
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["result"]["freezeId"], "phishing-freeze-test")
                self.assertEqual([row["stage"] for row in result["stages"]], ["data_gate", "sequences", "base_models", "stacking", "validation", "diversity", "freeze"])
                self.assertFalse(result["testPolicy"]["testEvaluationAuthorized"])
                manager.shutdown()
            finally:
                for item in reversed(patches):
                    item.stop()

    def test_complete_chain_accepts_direct_validation_manifest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            execution = "phishing-direct-validation"
            validation_path = root / "experiments" / execution / "phishing_oof_v1" / "external_validation_v1" / "external_validation_manifest.json"
            diversity_path = root / "experiments" / execution / "phishing_oof_v1" / "diversity_ablation_v1" / "diversity_ablation_manifest.json"
            validation_path.parent.mkdir(parents=True)
            diversity_path.parent.mkdir(parents=True)
            validation_path.write_text("{}", encoding="utf-8")
            diversity_path.write_text("{}", encoding="utf-8")
            patches = [
                patch("app.phishing.thesis_orchestrator.EXPERIMENTS_DIR", root / "experiments"),
                patch("app.phishing.thesis_orchestrator.get_phishing_dataset_status", return_value={"readyForThesisTraining": True}),
                patch("app.phishing.thesis_orchestrator.get_phishing_sequence_status", return_value={"readyForBaseModelTraining": True, "lineageCurrent": True}),
                patch("app.phishing.thesis_orchestrator.run_phishing_oof_experiment", return_value={"runId": execution}),
                patch("app.phishing.thesis_orchestrator.run_phishing_stacking_experiment", return_value={"runId": "stacking"}),
                patch("app.phishing.thesis_orchestrator.run_phishing_external_validation", return_value={"runId": "validation"}),
                patch("app.phishing.thesis_orchestrator.run_phishing_diversity_ablation", return_value={"runId": "diversity"}),
                patch("app.phishing.thesis_orchestrator.audit_phishing_freeze_readiness", return_value={"ready": True, "failedChecks": []}),
                patch("app.phishing.thesis_orchestrator.create_phishing_freeze_package", return_value={"freezeId": "phishing-freeze-direct"}),
            ]
            for item in patches:
                item.start()
            try:
                manager = PhishingThesisOrchestrator(job_dir=root / "jobs")
                job = manager.submit({"epochs": 1}, execution_run_id=execution)
                manager._futures[job["jobId"]].result(timeout=5)
                result = manager.get(job["jobId"])
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["result"]["validationRunId"], "validation")
                manager.shutdown()
            finally:
                for item in reversed(patches):
                    item.stop()

    def test_reuses_only_integrity_verified_validation_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "oof_manifest.json"
            artifact_path = root / "validation.csv"
            validation_path = root / "external_validation_manifest.json"
            base_path.write_text("{}", encoding="utf-8")
            artifact_path.write_text("label,score\n1,0.9\n", encoding="utf-8")
            content = artifact_path.read_bytes()
            descriptor = {
                "path": str(artifact_path.resolve()),
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
            manifest = {
                "runId": "verified-validation",
                "status": "validation_selection_candidate",
                "baseRun": {"manifestPath": str(base_path.resolve())},
                "comparison": [{"candidateId": "stacking"}],
                "validation": {
                    "outerValidationUsed": True,
                    "testSetLocked": True,
                    "testFeaturesEncoded": False,
                    "testSetUsed": False,
                },
                "tokenizer": descriptor,
                "artifacts": {"predictions": descriptor},
            }
            validation_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(_load_verified_validation(validation_path, base_path=base_path)["runId"], "verified-validation")
            artifact_path.write_text("altered", encoding="utf-8")
            self.assertIsNone(_load_verified_validation(validation_path, base_path=base_path))

            keras_path = root / "model.keras"
            keras_path.write_bytes(b"keras-container")
            keras_hash, keras_bytes = _keras_container_hash(keras_path)
            keras_descriptor = {
                "kind": "base_model",
                "path": str(keras_path.resolve()),
                "sha256": keras_hash,
                "bytes": keras_bytes,
            }
            manifest["tokenizer"] = descriptor
            artifact_path.write_text("label,score\n1,0.9\n", encoding="utf-8")
            manifest["artifacts"] = {"baseModel": keras_descriptor}
            validation_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(_load_verified_validation(validation_path, base_path=base_path)["runId"], "verified-validation")

            diversity_artifact = root / "diversity.json"
            diversity_artifact.write_text("{}", encoding="utf-8")
            diversity_content = diversity_artifact.read_bytes()
            diversity_descriptor = {
                "path": str(diversity_artifact.resolve()),
                "sha256": hashlib.sha256(diversity_content).hexdigest(),
                "bytes": len(diversity_content),
            }
            diversity_manifest_path = root / "diversity_manifest.json"
            diversity_manifest = {
                "runId": "verified-diversity",
                "status": "ablation_candidate",
                "validationRun": {"manifestPath": str(validation_path.resolve())},
                "diversity": {"pairs": [{"modelA": "lstm", "modelB": "gru"}]},
                "ablation": {"configurations": [{"ablationId": "full"}]},
                "validation": {
                    "outerValidationUsed": True,
                    "testSetLocked": True,
                    "testFeaturesEncoded": False,
                    "testSetUsed": False,
                },
                "artifacts": {"report": diversity_descriptor},
            }
            diversity_manifest_path.write_text(json.dumps(diversity_manifest), encoding="utf-8")
            self.assertEqual(_load_verified_diversity(diversity_manifest_path, validation_path=validation_path)["runId"], "verified-diversity")
            diversity_artifact.write_text("tampered", encoding="utf-8")
            self.assertIsNone(_load_verified_diversity(diversity_manifest_path, validation_path=validation_path))


if __name__ == "__main__":
    unittest.main()
