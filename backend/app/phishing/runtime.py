from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

import tensorflow as tf

from app.config import EXPERIMENTS_DIR, SILVER_DIR


GIB = 1024 ** 3


def inspect_phishing_training_runtime(
    *,
    protocol: str = "thesis",
    output_dir: Path | None = None,
    silver_path: Path | None = None,
    assignments_path: Path | None = None,
) -> dict[str, Any]:
    """Return a reproducible, read-only preflight for a long OOF execution."""
    destination = output_dir or EXPERIMENTS_DIR
    source = silver_path or (SILVER_DIR / "phishing.csv")
    assignments = assignments_path or (SILVER_DIR / "phishing_oof_assignments.csv")
    destination.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(destination)
    gpus = tf.config.list_physical_devices("GPU")
    minimum_free_bytes = 2 * GIB if protocol == "thesis" else 512 * 1024 ** 2

    checks = [
        {
            "id": "silver_available",
            "passed": source.is_file(),
            "message": "Dataset silver disponible." if source.is_file() else "Falta el dataset silver de phishing.",
        },
        {
            "id": "oof_assignments_available",
            "passed": assignments.is_file(),
            "message": "Asignaciones OOF disponibles." if assignments.is_file() else "Faltan las asignaciones OOF.",
        },
        {
            "id": "output_writable",
            "passed": os.access(destination, os.W_OK),
            "message": "Directorio de experimentos escribible." if os.access(destination, os.W_OK) else "El directorio de experimentos no permite escritura.",
        },
        {
            "id": "free_disk",
            "passed": disk.free >= minimum_free_bytes,
            "message": f"Espacio libre: {disk.free / GIB:.2f} GiB; mínimo operativo: {minimum_free_bytes / GIB:.2f} GiB.",
        },
    ]
    warnings: list[str] = []
    if protocol == "thesis" and not gpus:
        warnings.append(
            "No se detectó GPU. La corrida de tesis puede ejecutarse en CPU, pero se recomienda Google Colab o WSL2 con GPU."
        )
    if platform.system() == "Windows" and not gpus:
        warnings.append(
            "TensorFlow se está ejecutando en Windows sin acelerador visible; use este entorno para pilotos y validaciones técnicas."
        )

    return {
        "schemaVersion": "1.0.0",
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "ready": all(item["passed"] for item in checks),
        "checks": checks,
        "warnings": warnings,
        "runtime": {
            "platform": platform.platform(),
            "pythonVersion": platform.python_version(),
            "tensorflowVersion": tf.__version__,
            "cpuCount": os.cpu_count(),
            "gpuCount": len(gpus),
            "gpus": [device.name for device in gpus],
            "executable": sys.executable,
        },
        "storage": {
            "outputPath": str(destination.resolve()),
            "freeBytes": int(disk.free),
            "minimumFreeBytes": int(minimum_free_bytes),
        },
        "inputs": {
            "silverPath": str(source.resolve()),
            "assignmentsPath": str(assignments.resolve()),
        },
        "testPolicy": {
            "locked": True,
            "usedByPreflight": False,
            "message": "El preflight no lee ni evalúa el conjunto de prueba.",
        },
    }
