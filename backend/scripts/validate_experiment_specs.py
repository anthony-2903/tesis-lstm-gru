from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "config" / "experiments.json"
RESULT_EXAMPLE_PATH = ROOT / "config" / "result-contract.example.json"
EXPECTED_DOMAINS = {"phishing", "energia", "finanzas"}
EXPECTED_MODELS = {"lstm", "gru", "brnn", "tcn", "transformer"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def validate_fraction(split: dict[str, Any], domain: str, failures: list[str]) -> None:
    fractions = [split.get("trainFraction"), split.get("validationFraction"), split.get("testFraction")]
    require(all(isinstance(value, (int, float)) for value in fractions), f"{domain}: split fractions must be numeric", failures)
    if all(isinstance(value, (int, float)) for value in fractions):
        require(abs(sum(fractions) - 1.0) < 1e-9, f"{domain}: split fractions must add up to 1", failures)


def validate_specs(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    common = payload.get("common", {})
    domains = payload.get("domains", {})

    require(payload.get("schemaVersion") == "1.0.0", "Unsupported experiment schemaVersion", failures)
    require(set(common.get("baseModels", [])) == EXPECTED_MODELS, "The five required base models must be declared", failures)
    require(common.get("minimumRepetitions", 0) >= 5, "At least five repetitions are required", failures)
    require(len(common.get("randomSeeds", [])) >= 5, "At least five random seeds are required", failures)
    require(set(domains) == EXPECTED_DOMAINS, "Exactly the three thesis domains must be configured", failures)

    for domain, spec in domains.items():
        require(bool(spec.get("researchQuestion")), f"{domain}: researchQuestion is required", failures)
        require(bool(spec.get("task")), f"{domain}: task is required", failures)
        require(bool(spec.get("unitOfAnalysis")), f"{domain}: unitOfAnalysis is required", failures)
        require(bool(spec.get("primaryMetric")), f"{domain}: primaryMetric is required", failures)
        require(bool(spec.get("secondaryMetrics")), f"{domain}: secondaryMetrics cannot be empty", failures)
        require(bool(spec.get("stackingFeatures")), f"{domain}: stackingFeatures cannot be empty", failures)
        require(spec.get("target", {}).get("required") is True, f"{domain}: a real target is mandatory", failures)
        require(spec.get("split", {}).get("innerFolds", 0) >= 5, f"{domain}: at least five inner folds are required", failures)
        validate_fraction(spec.get("split", {}), domain, failures)

    return failures


def validate_result_contract(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    require(payload.get("schemaVersion") == "1.0.0", "Unsupported result contract schemaVersion", failures)
    require(bool(payload.get("dataset")), "Result contract requires dataset provenance", failures)
    require(bool(payload.get("split")), "Result contract requires split metadata", failures)
    models = payload.get("models", [])
    require(any(model.get("family") == "base" for model in models), "Result contract requires a base model entry", failures)
    stacking = next((model for model in models if model.get("modelId") == "stacking"), None)
    require(stacking is not None, "Result contract requires a stacking entry", failures)
    if stacking:
        require(stacking.get("oofVerified") is True, "Stacking must explicitly verify out-of-fold features", failures)
        require(set(stacking.get("baseModels", [])) == EXPECTED_MODELS, "Stacking must declare all five base models", failures)
    return failures


def main() -> int:
    failures = validate_specs(load_json(SPEC_PATH))
    failures.extend(validate_result_contract(load_json(RESULT_EXAMPLE_PATH)))
    if failures:
        print("Experiment specification validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Experiment specifications and result contract are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
