from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from app.phishing.models import PHISHING_MODEL_IDS
from app.phishing.pipeline import run_phishing_oof_experiment
from app.phishing.runtime import inspect_phishing_training_runtime


THESIS_SEEDS = (42, 52, 62, 72, 82)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta o reanuda el entrenamiento OOF de phishing sin utilizar el conjunto de prueba."
    )
    parser.add_argument("--protocol", choices=("demo", "thesis"), default="thesis")
    parser.add_argument("--execution-run-id", help="Identificador estable; repita el mismo valor para reanudar.")
    parser.add_argument("--output-dir", type=Path, help="Carpeta persistente del experimento, por ejemplo Google Drive.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--demo-max-rows", type=int, default=2_000)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    run_id = args.execution_run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_phishing_thesis")
    preflight = inspect_phishing_training_runtime(protocol=args.protocol, output_dir=args.output_dir)
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return
    if not preflight["ready"]:
        failed = ", ".join(item["id"] for item in preflight["checks"] if not item["passed"])
        raise SystemExit(f"Preflight bloqueado: {failed}")

    seeds = THESIS_SEEDS if args.protocol == "thesis" else (42,)
    demo_max_rows = None if args.protocol == "thesis" else args.demo_max_rows

    def progress(event: dict[str, Any]) -> None:
        summary = {
            key: event.get(key)
            for key in ("event", "completedUnits", "resumedUnits", "totalUnits", "seed", "fold", "modelId", "message")
            if event.get(key) is not None
        }
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    manifest = run_phishing_oof_experiment(
        output_dir=args.output_dir,
        execution_run_id=run_id,
        protocol=args.protocol,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seeds=seeds,
        model_ids=PHISHING_MODEL_IDS,
        demo_max_rows=demo_max_rows,
        progress_callback=progress,
    )
    print(json.dumps({
        "runId": manifest["runId"],
        "status": manifest["status"],
        "manifest": manifest["execution"]["checkpointPath"],
        "resumedUnits": manifest["execution"]["resumedUnits"],
        "testSetUsed": manifest["validation"]["testSetUsed"],
        "stackingReady": manifest["stacking"]["ready"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
