from __future__ import annotations

import argparse

from app.cleaning.tabular_cleaner import clean_tabular
from app.cleaning.text_cleaner import clean_phishtank
from app.cleaning.timeseries_cleaner import clean_opsd
from app.config import GOLD_DIR, RAW_DIR, RESULTS_DIR, SILVER_DIR, ensure_dirs
from app.ingestion.sources import fetch_mef_brechas, fetch_mef_operadores, fetch_opsd, fetch_phishtank
from app.reports import build_ai_analysis, build_analysis, build_comparison, build_dashboard, build_history, build_xai
from app.training.trainers import train_timeseries_models, train_url_models
from app.utils import write_json


def run_pipeline(mode: str = "sample") -> dict[str, object]:
    ensure_dirs()
    phishtank_raw = fetch_phishtank(mode=mode)
    operadores_raw = fetch_mef_operadores(mode=mode)
    brechas_raw = fetch_mef_brechas(mode=mode)
    opsd_raw = fetch_opsd(mode=mode)

    phishtank_raw.to_csv(RAW_DIR / "phishtank.csv", index=False)
    operadores_raw.to_csv(RAW_DIR / "mef_operadores.csv", index=False)
    brechas_raw.to_csv(RAW_DIR / "mef_brechas.csv", index=False)
    opsd_raw.to_csv(RAW_DIR / "opsd.csv", index=False)

    phishtank = clean_phishtank(phishtank_raw)
    operadores = clean_tabular(operadores_raw)
    brechas = clean_tabular(brechas_raw)
    opsd = clean_opsd(opsd_raw)

    phishtank.to_csv(SILVER_DIR / "phishtank.csv", index=False)
    operadores.to_csv(SILVER_DIR / "mef_operadores.csv", index=False)
    brechas.to_csv(SILVER_DIR / "mef_brechas.csv", index=False)
    opsd.to_csv(SILVER_DIR / "opsd.csv", index=False)

    url_result = train_url_models(phishtank)
    ts_result = train_timeseries_models(opsd)

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
    xai = build_xai(filename, dashboard["dataset"]["columns"])
    ai_analysis = build_ai_analysis(filename, analysis)

    artifacts = {
        "dashboard": dashboard,
        "analysis": analysis,
        "comparison": comparison,
        "history": history,
        "xai": xai,
        "ai_analysis": ai_analysis,
    }
    for name, payload in artifacts.items():
        write_json(RESULTS_DIR / f"{name}.json", payload)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "remote"], default="sample")
    args = parser.parse_args()
    run_pipeline(mode=args.mode)
    print(f"Pipeline completado en modo {args.mode}. Resultados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
