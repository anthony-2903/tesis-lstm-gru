from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import EXPERIMENTS_DIR, RESULTS_DIR
from app.thesis_status import get_thesis_status


BASE_MODELS = ("lstm", "gru", "brnn", "tcn", "transformer")


def get_thesis_results_summary(
    experiments_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Return a presentation-safe comparison of five bases plus one Stacking.

    Development or validation metrics remain visible as candidate evidence, but
    they are never labeled as a thesis conclusion until the matching domain job is
    complete and its freeze seal is valid. Local artifact paths are not exposed.
    """

    experiments = experiments_dir or EXPERIMENTS_DIR
    results = results_dir or RESULTS_DIR
    execution = get_thesis_status(results)
    domain_execution = {item["id"]: item for item in execution["domains"]}

    domains = [
        _energy_summary(experiments, results, domain_execution["energia"]),
        _phishing_summary(experiments, domain_execution["phishing"]),
        _finance_summary(experiments, results, domain_execution["finanzas"]),
    ]
    eligible = sum(bool(item["eligibleForThesisConclusion"]) for item in domains)

    return {
        "schemaVersion": "1.0.0",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "title": "Comparación pre-test de cinco modelos base y Stacking",
        "candidateContract": {
            "baseModels": list(BASE_MODELS),
            "metaModelLabel": "stacking",
            "expectedCandidatesPerDomain": 6,
            "selectionSeparatedFromFinalTest": True,
        },
        "domains": domains,
        "readiness": {
            "eligibleDomains": eligible,
            "totalDomains": len(domains),
            "allDomainsFrozen": eligible == len(domains),
            "finalTestUsed": bool(execution["testPolicy"]["testSetUsed"]),
        },
        "interpretationPolicy": [
            "Una tabla de candidato o demostración no constituye el resultado final de tesis.",
            "Stacking solo se declara mejor si gana la métrica primaria en comparación independiente y la inferencia pareada respalda la mejora.",
            "Si Stacking no gana, se conserva el resultado sin cambiar datos, semillas, umbrales o particiones.",
        ],
    }


def _energy_summary(experiments: Path, results: Path, execution: dict[str, Any]) -> dict[str, Any]:
    path, manifest = _latest_json(experiments.glob("*/energy_oof_v1/energy_revalidation_v1/revalidation_manifest.json"))
    if manifest is None:
        return _unavailable("energia", "Energía eléctrica", "regression", "rmse", False, execution)
    independent = manifest.get("independentComparison", {})
    rows = [
        {
            "candidateId": str(row.get("predictorId")),
            "sourceCandidateId": str(row.get("predictorId")),
            "family": "stacking" if row.get("predictorId") == "optimized_stacking" else "base",
            "primaryValue": _number(row.get("rmse")),
            "secondary": {"mae": _number(row.get("mae")), "smape": _number(row.get("smape")), "r2": _number(row.get("r2"))},
        }
        for row in independent.get("metrics", [])
        if row.get("predictorId") in {*BASE_MODELS, "optimized_stacking"}
    ]
    for row in rows:
        if row["family"] == "stacking":
            row["candidateId"] = "stacking"
    seed_metrics = [
        {
            "seed": int(row.get("seed")),
            "candidateId": "stacking" if row.get("predictorId") == "optimized_stacking" else str(row.get("predictorId")),
            "primaryValue": _number(row.get("rmse")),
        }
        for row in independent.get("seedMetrics", [])
        if row.get("predictorId") in {*BASE_MODELS, "optimized_stacking"}
    ]
    freeze_valid = _matching_energy_freeze(results, path)
    scientific_candidate = str(manifest.get("status", "")).startswith("thesis_")
    return _finalize(
        domain_id="energia",
        label="Energía eléctrica",
        task_type="regression",
        metric="rmse",
        higher_is_better=False,
        rows=rows,
        seed_metrics=seed_metrics,
        all_candidate_winner=_source_id(independent.get("ranking", [None])[0] if independent.get("ranking") else None),
        inference=independent.get("pairedInference"),
        run_id=manifest.get("runId"),
        execution=execution,
        freeze_valid=freeze_valid,
        scientific_candidate=scientific_candidate,
        comparison_split=f"fold independiente {manifest.get('protocol', {}).get('independentComparisonFold', 'N/D')}",
    )


def _phishing_summary(experiments: Path, execution: dict[str, Any]) -> dict[str, Any]:
    path, manifest = _latest_json(experiments.glob("*/phishing_oof_v1/external_validation_v1/external_validation_manifest.json"))
    if manifest is None:
        return _unavailable("phishing", "Ciberseguridad / phishing", "classification", "prAuc", True, execution)
    comparison = manifest.get("comparison", [])
    selected_stacking = manifest.get("selection", {}).get("leadingStackingCandidateId")
    accepted = {*BASE_MODELS, selected_stacking}
    rows = []
    seed_metrics = []
    for row in comparison:
        source_id = row.get("candidateId")
        if source_id not in accepted:
            continue
        family = "stacking" if source_id == selected_stacking else "base"
        rows.append(
            {
                "candidateId": "stacking" if family == "stacking" else str(source_id),
                "sourceCandidateId": str(source_id),
                "family": family,
                "primaryValue": _number(row.get("prAuc")),
                "secondary": _classification_secondary(row),
            }
        )
        seed_metrics.extend(
            {
                "seed": int(seed_row.get("seed")),
                "candidateId": "stacking" if family == "stacking" else str(source_id),
                "primaryValue": _number(seed_row.get("prAuc")),
            }
            for seed_row in row.get("seedMetrics", [])
        )
    freeze_valid = _matching_freeze(
        experiments.glob("*/phishing_oof_v1/frozen_pipeline_v1/frozen_pipeline_manifest.json"),
        path,
        "validationManifestPath",
    )
    scientific_candidate = manifest.get("status") == "validation_selection_candidate" and len(manifest.get("baseRun", {}).get("seeds", [])) >= 5
    all_winner = comparison[0].get("candidateId") if comparison else None
    return _finalize(
        domain_id="phishing",
        label="Ciberseguridad / phishing",
        task_type="classification",
        metric="prAuc",
        higher_is_better=True,
        rows=rows,
        seed_metrics=seed_metrics,
        all_candidate_winner=_source_id(all_winner),
        inference=None,
        run_id=manifest.get("runId"),
        execution=execution,
        freeze_valid=freeze_valid,
        scientific_candidate=scientific_candidate,
        comparison_split="validación externa por dominio; test bloqueado",
    )


def _finance_summary(experiments: Path, results: Path, execution: dict[str, Any]) -> dict[str, Any]:
    market_path, market_manifest = _latest_json(
        experiments.glob("*/finance_market_oof_v1/finance_market_revalidation_v1/revalidation_manifest.json")
    )
    if market_manifest is not None:
        independent = market_manifest.get("independentComparison", {})
        rows = [
            {
                "candidateId": "stacking" if row.get("predictorId") == "optimized_stacking" else str(row.get("predictorId")),
                "sourceCandidateId": str(row.get("predictorId")),
                "family": "stacking" if row.get("predictorId") == "optimized_stacking" else "base",
                "primaryValue": _number(row.get("rmse")),
                "secondary": {"mae": _number(row.get("mae")), "smape": _number(row.get("smape")), "r2": _number(row.get("r2"))},
            }
            for row in independent.get("metrics", [])
            if row.get("predictorId") in {*BASE_MODELS, "optimized_stacking"}
        ]
        seed_metrics = [
            {
                "seed": int(row.get("seed")),
                "candidateId": "stacking" if row.get("predictorId") == "optimized_stacking" else str(row.get("predictorId")),
                "primaryValue": _number(row.get("rmse")),
            }
            for row in independent.get("seedMetrics", [])
            if row.get("predictorId") in {*BASE_MODELS, "optimized_stacking"}
        ]
        freeze_valid = _matching_named_freeze(results / "finance_market_freezes", market_path)
        return _finalize(
            domain_id="finanzas",
            label="Finanzas / Indices Soberanos MEF",
            task_type="regression",
            metric="rmse",
            higher_is_better=False,
            rows=rows,
            seed_metrics=seed_metrics,
            all_candidate_winner=_source_id(independent.get("ranking", [None])[0] if independent.get("ranking") else None),
            inference=independent.get("pairedInference"),
            run_id=market_manifest.get("runId"),
            execution=execution,
            freeze_valid=freeze_valid,
            scientific_candidate=str(market_manifest.get("status", "")).startswith("thesis_"),
            comparison_split=f"fold financiero independiente {market_manifest.get('protocol', {}).get('independentComparisonFold', 'N/D')}",
        )

    # The former IEEE-CIS/ULB classification manifests remain on disk as
    # historical pilots, but they are no longer evidence for the selected MEF
    # sovereign-index research question.
    return _unavailable("finanzas", "Finanzas / Indices Soberanos MEF", "regression", "rmse", False, execution)

    path, manifest = _latest_json(experiments.glob("*/finance_oof_v1/optimized_stacking_revalidation_v1/optimized_stacking_revalidation_manifest.json"))
    if manifest is None:
        return _unavailable("finanzas", "Finanzas / Indices Soberanos MEF", "regression", "rmse", False, execution)
    independent = manifest.get("independentComparison", {})
    rows = [
        {
            "candidateId": str(row.get("candidateId")),
            "sourceCandidateId": str(row.get("sourceCandidateId") or row.get("candidateId")),
            "family": str(row.get("family") or ("stacking" if row.get("candidateId") == "stacking" else "base")),
            "primaryValue": _number(row.get("prAuc")),
            "secondary": _classification_secondary(row),
        }
        for row in independent.get("metrics", [])
        if row.get("candidateId") in {*BASE_MODELS, "stacking"}
    ]
    seed_metrics = [
        {
            "seed": int(row.get("seed")),
            "candidateId": str(row.get("candidateId")),
            "primaryValue": _number(row.get("prAuc")),
        }
        for row in independent.get("seedMetrics", [])
        if row.get("candidateId") in {*BASE_MODELS, "stacking"}
    ]
    freeze_valid = _matching_freeze(
        experiments.glob("*/finance_oof_v1/frozen_pipeline_v1/frozen_pipeline_manifest.json"),
        path,
        "revalidationManifestPath",
    )
    scientific_candidate = manifest.get("validation", {}).get("eligibleForFreeze") is True
    return _finalize(
        domain_id="finanzas",
        label="Finanzas / fraude",
        task_type="classification",
        metric="prAuc",
        higher_is_better=True,
        rows=rows,
        seed_metrics=seed_metrics,
        all_candidate_winner=_source_id(independent.get("winnerCandidateId")),
        inference=independent.get("stackingVersusBestBase"),
        run_id=manifest.get("runId"),
        execution=execution,
        freeze_valid=freeze_valid,
        scientific_candidate=scientific_candidate,
        comparison_split="tercer bloque cronológico independiente; test bloqueado",
    )


def _finalize(
    *,
    domain_id: str,
    label: str,
    task_type: str,
    metric: str,
    higher_is_better: bool,
    rows: list[dict[str, Any]],
    seed_metrics: list[dict[str, Any]],
    all_candidate_winner: str | None,
    inference: Any,
    run_id: Any,
    execution: dict[str, Any],
    freeze_valid: bool,
    scientific_candidate: bool,
    comparison_split: str,
) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["primaryValue"] is not None]
    valid_rows.sort(key=lambda row: float(row["primaryValue"]), reverse=higher_is_better)
    for rank, row in enumerate(valid_rows, start=1):
        row["rank"] = rank
    stacking = next((row for row in valid_rows if row["candidateId"] == "stacking"), None)
    best_base = next((row for row in valid_rows if row["family"] == "base"), None)
    complete_contract = {row["candidateId"] for row in valid_rows} == {*BASE_MODELS, "stacking"}
    eligible = bool(
        freeze_valid
        and execution.get("status") == "completed"
        and complete_contract
        and scientific_candidate
        and not execution.get("testPolicy", {}).get("testSetUsed")
    )
    evidence_status = (
        "scientific_frozen_pre_test"
        if eligible
        else "scientific_candidate_pending_freeze"
        if scientific_candidate
        else "demonstration_only"
    )
    stacking_better = None
    if stacking is not None and best_base is not None:
        stacking_better = stacking["primaryValue"] > best_base["primaryValue"] if higher_is_better else stacking["primaryValue"] < best_base["primaryValue"]

    return {
        "id": domain_id,
        "label": label,
        "available": bool(valid_rows),
        "taskType": task_type,
        "primaryMetric": metric,
        "higherIsBetter": higher_is_better,
        "comparisonSplit": comparison_split,
        "runId": str(run_id) if run_id else None,
        "evidenceStatus": evidence_status,
        "eligibleForThesisConclusion": eligible,
        "freezeSealValid": freeze_valid,
        "sixCandidateContractComplete": complete_contract,
        "sixCandidateWinner": valid_rows[0]["candidateId"] if valid_rows else None,
        "allEvaluatedCandidatesWinner": all_candidate_winner,
        "stackingRank": stacking.get("rank") if stacking else None,
        "stackingBeatsBestBase": stacking_better,
        "bestBaseCandidateId": best_base["candidateId"] if best_base else None,
        "comparison": valid_rows,
        "seedDistributions": _seed_distributions(seed_metrics),
        "pairedInference": _public_inference(inference),
        "testSetUsed": bool(execution.get("testPolicy", {}).get("testSetUsed")),
        "interpretation": _interpretation(evidence_status, stacking_better, valid_rows[0]["candidateId"] if valid_rows else None),
    }


def _unavailable(domain_id: str, label: str, task_type: str, metric: str, higher: bool, execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": domain_id,
        "label": label,
        "available": False,
        "taskType": task_type,
        "primaryMetric": metric,
        "higherIsBetter": higher,
        "comparisonSplit": None,
        "runId": None,
        "evidenceStatus": "unavailable",
        "eligibleForThesisConclusion": False,
        "freezeSealValid": False,
        "sixCandidateContractComplete": False,
        "sixCandidateWinner": None,
        "allEvaluatedCandidatesWinner": None,
        "stackingRank": None,
        "stackingBeatsBestBase": None,
        "bestBaseCandidateId": None,
        "comparison": [],
        "seedDistributions": [],
        "pairedInference": None,
        "testSetUsed": bool(execution.get("testPolicy", {}).get("testSetUsed")),
        "interpretation": "La tabla aún no existe; el protocolo debe completar su comparación pre-test.",
    }


def _matching_energy_freeze(results: Path, revalidation_path: Path | None) -> bool:
    if revalidation_path is None:
        return False
    freeze_path, freeze = _latest_json(results.glob("energy_freezes/*/freeze_manifest.json"))
    if freeze is None or freeze_path is None or not _seal_valid(freeze_path):
        return False
    configuration = _read_json(freeze_path.parent / "frozen_configuration.json") or {}
    return _same_path(configuration.get("revalidationManifestPath"), revalidation_path)


def _matching_named_freeze(directory: Path, revalidation_path: Path | None) -> bool:
    if revalidation_path is None:
        return False
    freeze_path, freeze = _latest_json(directory.glob("*/freeze_manifest.json"))
    if freeze is None or freeze_path is None or not _seal_valid(freeze_path):
        return False
    configuration = _read_json(freeze_path.parent / "frozen_configuration.json") or {}
    return _same_path(configuration.get("revalidationManifestPath"), revalidation_path)


def _matching_freeze(paths: Any, source_path: Path | None, lineage_key: str) -> bool:
    if source_path is None:
        return False
    freeze_path, freeze = _latest_json(paths)
    if freeze is None or freeze_path is None or not _seal_valid(freeze_path):
        return False
    return _same_path(freeze.get("lineage", {}).get(lineage_key), source_path)


def _seal_valid(manifest_path: Path) -> bool:
    seal = _read_json(manifest_path.parent / "freeze_seal.json")
    if seal is None:
        return False
    content = manifest_path.read_bytes()
    return (
        seal.get("manifestSha256") == hashlib.sha256(content).hexdigest()
        and int(seal.get("manifestBytes", -1)) == len(content)
    )


def _latest_json(paths: Any) -> tuple[Path | None, dict[str, Any] | None]:
    existing = [path for path in paths if path.is_file()]
    for path in sorted(existing, key=lambda item: item.stat().st_mtime_ns, reverse=True):
        payload = _read_json(path)
        if payload is not None:
            return path, payload
    return None, None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _same_path(value: Any, path: Path) -> bool:
    if not value:
        return False
    try:
        return Path(str(value)).resolve() == path.resolve()
    except OSError:
        return False


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _classification_secondary(row: dict[str, Any]) -> dict[str, float | None]:
    return {key: _number(row.get(key)) for key in ("f1", "mcc", "recall", "falsePositiveRate")}


def _seed_distributions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for candidate_id in (*BASE_MODELS, "stacking"):
        candidate_rows = [row for row in rows if row.get("candidateId") == candidate_id and row.get("primaryValue") is not None]
        candidate_rows.sort(key=lambda row: int(row["seed"]))
        values = [float(row["primaryValue"]) for row in candidate_rows]
        if not values:
            continue
        ordered = sorted(values)
        result.append(
            {
                "candidateId": candidate_id,
                "seeds": [int(row["seed"]) for row in candidate_rows],
                "values": values,
                "seedCount": len(values),
                "minimum": ordered[0],
                "q1": _quantile(ordered, 0.25),
                "median": _quantile(ordered, 0.5),
                "q3": _quantile(ordered, 0.75),
                "maximum": ordered[-1],
            }
        )
    return result


def _quantile(ordered: list[float], probability: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _public_inference(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    allowed = {
        "metric",
        "method",
        "iterations",
        "observedDelta",
        "ciLower",
        "ciUpper",
        "probabilityStackingBetter",
        "statisticallyClearAt95Percent",
        "bestBaseCandidateId",
        "referenceId",
        "candidateId",
        "deltaDefinition",
        "observedDeltaRmse",
        "confidenceInterval95",
        "probabilityCandidateBetter",
        "calendarDays",
    }
    return {key: value[key] for key in allowed if key in value}


def _source_id(value: Any) -> str | None:
    if value is None:
        return None
    return "stacking" if str(value) == "optimized_stacking" else str(value)


def _interpretation(evidence_status: str, stacking_better: bool | None, winner: str | None) -> str:
    prefix = {
        "scientific_frozen_pre_test": "Evidencia científica sellada pre-test.",
        "scientific_candidate_pending_freeze": "Evidencia científica candidata; aún no está sellada.",
        "demonstration_only": "Resultado demostrativo; no sustenta la conclusión de tesis.",
    }.get(evidence_status, "Resultado no disponible.")
    if stacking_better is True:
        return f"{prefix} Stacking lidera frente a los cinco modelos base en esta comparación; la inferencia pareada determina si la mejora es concluyente."
    if stacking_better is False:
        return f"{prefix} {winner or 'Un modelo base'} supera a Stacking; el resultado debe conservarse sin reoptimización retrospectiva."
    return prefix
