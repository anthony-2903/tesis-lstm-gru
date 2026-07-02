from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = ROOT_DIR / "storage"
RAW_DIR = STORAGE_DIR / "raw"
SILVER_DIR = STORAGE_DIR / "silver"
GOLD_DIR = STORAGE_DIR / "gold"
RESULTS_DIR = STORAGE_DIR / "results"
MODELS_DIR = STORAGE_DIR / "models"
EXPERIMENTS_DIR = STORAGE_DIR / "experiments"


@dataclass(frozen=True)
class SourceConfig:
    phishtank_csv_url: str = "http://data.phishtank.com/data/online-valid.csv"
    urlhaus_text_recent_url: str = "https://urlhaus.abuse.ch/downloads/text_recent/"
    urlhaus_csv_recent_url: str = "https://urlhaus.abuse.ch/downloads/csv_recent/"
    cisa_kev_json_url: str = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
    mef_operadores_resource_id: str = "08daf6b7-0f81-421b-ad8d-f3ae2d777c5e"
    mef_brechas_resource_id: str = "4d32fbdb-b1a3-461a-8276-2386c09d8179"
    mef_datastore_url: str = "https://api.datosabiertos.mef.gob.pe/DatosAbiertos/v1/datastore_search"
    opsd_time_series_url: str = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
    user_agent: str = "tesis-lstm-gru/0.1 local-research"
    google_safe_browsing_api_key: str = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY", "")
    eia_api_key: str = os.getenv("EIA_API_KEY", "")
    entsoe_api_key: str = os.getenv("ENTSOE_API_KEY", "")
    sec_user_agent: str = os.getenv("SEC_USER_AGENT", "tesis-lstm-gru academic-research contact@example.com")


SOURCE_CONFIG = SourceConfig()


def ensure_dirs() -> None:
    for directory in [RAW_DIR, SILVER_DIR, GOLD_DIR, RESULTS_DIR, MODELS_DIR, EXPERIMENTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
