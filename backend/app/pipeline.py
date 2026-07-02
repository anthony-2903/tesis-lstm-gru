from __future__ import annotations

import argparse

import pandas as pd

from app.artifacts import build_metrics_summary, make_run_id, persist_models, snapshot_experiment
from app.cleaning.tabular_cleaner import clean_tabular
from app.cleaning.text_cleaner import clean_phishtank
from app.cleaning.timeseries_cleaner import clean_opsd
from app.config import GOLD_DIR, RAW_DIR, RESULTS_DIR, SILVER_DIR, ensure_dirs
from app.ingestion.sources import fetch_mef_brechas, fetch_mef_operadores, fetch_opsd, fetch_phishtank
from app.reports import build_ai_analysis, build_analysis, build_comparison, build_dashboard, build_history
from app.training.trainers import train_tabular_models, train_timeseries_models, train_url_models
from app.utils import read_json, write_json
from app.xai.explainer import build_xai_report


def run_pipeline(mode: str = "sample", limit: int = 10000) -> dict[str, object]:
    ensure_dirs()
    run_id = make_run_id(mode, limit)
    phishtank_raw = fetch_phishtank(mode=mode, limit=limit)
    operadores_raw = fetch_mef_operadores(mode=mode, limit=limit)
    brechas_raw = fetch_mef_brechas(mode=mode, limit=limit)
    opsd_raw = fetch_opsd(mode=mode, limit=limit)

    phishtank_raw.to_csv(RAW_DIR / "phishtank.csv", index=False)
    operadores_raw.to_csv(RAW_DIR / "mef_operadores.csv", index=False)
    brechas_raw.to_csv(RAW_DIR / "mef_brechas.csv", index=False)
    opsd_raw.to_csv(RAW_DIR / "opsd.csv", index=False)

    phishtank = clean_phishtank(phishtank_raw)
    operadores = clean_tabular(operadores_raw)
    brechas = clean_tabular(brechas_raw)
    opsd = clean_opsd(opsd_raw)
    finanzas = _load_data_lake_frame("finanzas")
    if finanzas.empty:
        finanzas = pd.concat([operadores, brechas], ignore_index=True, sort=False)

    phishtank.to_csv(SILVER_DIR / "phishtank.csv", index=False)
    operadores.to_csv(SILVER_DIR / "mef_operadores.csv", index=False)
    brechas.to_csv(SILVER_DIR / "mef_brechas.csv", index=False)
    opsd.to_csv(SILVER_DIR / "opsd.csv", index=False)

    url_result = train_url_models(phishtank)
    ts_result = train_timeseries_models(opsd)
    finance_result = train_tabular_models(finanzas, domain="Finanzas")

    filename = f"pipeline_{mode}_phishtank_mef_opsd"
    merged_models = {}
    for key in url_result.models:
        merged = dict(url_result.models[key])
        if key in ts_result.models:
            merged["rmse"] = ts_result.models[key].get("rmse", merged.get("rmse", 0))
            merged["f1"] = max(float(merged.get("f1", 0)), float(ts_result.models[key].get("f1", 0)))
        merged_models[key] = merged

    analysis = build_analysis(
        filename,
        type(
            "MergedResult",
            (),
            {
                "total_rows": url_result.total_rows + ts_result.total_rows,
                "real_anomalies_count": url_result.real_anomalies_count + ts_result.real_anomalies_count,
                "models": merged_models,
                "timeline": ts_result.timeline or url_result.timeline,
                "samples": url_result.samples + ts_result.samples,
                "processed_records": url_result.processed_records + ts_result.processed_records,
            },
        )(),
    )
    dashboard = build_dashboard(filename, [phishtank, operadores, brechas, opsd])
    comparison = build_comparison(filename, merged_models)
    history = build_history(filename, analysis["processedRecords"])
    xai = build_xai_report(filename, url_result.xai, ts_result.xai)
    ai_analysis = build_ai_analysis(filename, analysis)

    domain_sources = {
        "phishing": {
            "label": "PhishTank - Deteccion de phishing",
            "frames": [phishtank],
            "result": url_result,
            "xai": build_xai_report(f"{filename}_phishing", url_result.xai, {}),
        },
        "energia": {
            "label": "OPSD - Energia y series temporales",
            "frames": [opsd],
            "result": ts_result,
            "xai": build_xai_report(f"{filename}_energia", {}, ts_result.xai),
        },
        "finanzas": {
            "label": "Finanzas - Fraude y registros atipicos",
            "frames": [finanzas],
            "result": finance_result,
            "xai": build_xai_report(f"{filename}_finanzas", finance_result.xai, {}),
        },
    }

    artifacts = {
        "dashboard": dashboard,
        "analysis": analysis,
        "comparison": comparison,
        "history": history,
        "xai": xai,
        "ai_analysis": ai_analysis,
        "domains": {
            "items": [
                {
                    "id": "phishing",
                    "title": "Phishing",
                    "source": "PhishTank",
                    "description": "URLs verificadas, rasgos lexicales y deteccion de phishing.",
                },
                {
                    "id": "energia",
                    "title": "Energia",
                    "source": "Open Power System Data",
                    "description": "Series temporales de consumo/generacion y anomalias por ventana.",
                },
                {
                    "id": "finanzas",
                    "title": "Finanzas publicas",
                    "source": "MEF Datos Abiertos",
                    "description": "Indicadores de brechas y operadores Invierte.pe con limpieza tabular.",
                },
            ]
        },
    }
    for domain, source in domain_sources.items():
        domain_filename = str(source["label"])
        domain_analysis = build_analysis(domain_filename, source["result"])
        artifacts[f"dashboard_{domain}"] = build_dashboard(domain_filename, source["frames"])
        artifacts[f"analysis_{domain}"] = domain_analysis
        artifacts[f"comparison_{domain}"] = build_comparison(domain_filename, source["result"].models)
        artifacts[f"history_{domain}"] = build_history(domain_filename, domain_analysis["processedRecords"])
        artifacts[f"xai_{domain}"] = source["xai"]

    for name, payload in artifacts.items():
        write_json(RESULTS_DIR / f"{name}.json", payload)
    metrics_summary = build_metrics_summary(artifacts)
    model_records = persist_models(
        {
            "phishing": url_result.estimators,
            "energia": ts_result.estimators,
            "finanzas": finance_result.estimators,
        },
        run_id,
    )
    manifest = snapshot_experiment(
        run_id=run_id,
        mode=mode,
        limit=limit,
        model_records=model_records,
        artifact_names=list(artifacts),
        domain_totals={
            "phishing": url_result.total_rows,
            "energia": ts_result.total_rows,
            "finanzas": finance_result.total_rows,
        },
        metrics_summary=metrics_summary,
    )
    artifacts["training_manifest"] = manifest
    return artifacts


def _load_data_lake_frame(domain: str) -> pd.DataFrame:
    path = SILVER_DIR / f"external_{domain}.json"
    if not path.exists():
        return pd.DataFrame()
    payload = read_json(path)
    records = payload.get("records", [])
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "remote"], default="sample")
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    run_pipeline(mode=args.mode, limit=args.limit)
    print(f"Pipeline completado en modo {args.mode} con limite {args.limit}. Resultados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
