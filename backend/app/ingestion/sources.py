from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from app.config import SOURCE_CONFIG
from app.ingestion.samples import sample_mef_brechas, sample_mef_operadores, sample_opsd, sample_phishtank


def fetch_phishtank(mode: str = "sample", limit: int = 5000) -> pd.DataFrame:
    if mode == "sample":
        return sample_phishtank()
    frames = []
    try:
        response = requests.get(SOURCE_CONFIG.phishtank_csv_url, headers={"User-Agent": SOURCE_CONFIG.user_agent}, timeout=60)
        response.raise_for_status()
        frames.append(pd.read_csv(StringIO(response.text)))
    except Exception:
        frames.append(sample_phishtank())

    try:
        response = requests.get(SOURCE_CONFIG.urlhaus_text_recent_url, headers={"User-Agent": SOURCE_CONFIG.user_agent}, timeout=60)
        response.raise_for_status()
        urls = [line.strip() for line in response.text.splitlines() if line.strip() and not line.startswith("#")]
        frames.append(pd.DataFrame({"url": urls, "source": "urlhaus"}))
    except Exception:
        pass

    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame = frame[frame["url"].notna()].drop_duplicates(subset=["url"]).head(limit)
    frame["label"] = 1
    return frame


def fetch_mef_resource(resource_id: str, fallback: pd.DataFrame, mode: str = "sample", limit: int = 5000) -> pd.DataFrame:
    if mode == "sample":
        return fallback.copy()
    params = {"resource_id": resource_id, "limit": limit}
    response = requests.get(SOURCE_CONFIG.mef_datastore_url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    records = payload.get("result", {}).get("records", [])
    if not records:
        return fallback.copy()
    return pd.DataFrame(records)


def fetch_mef_operadores(mode: str = "sample", limit: int = 5000) -> pd.DataFrame:
    return fetch_mef_resource(SOURCE_CONFIG.mef_operadores_resource_id, sample_mef_operadores(), mode, limit)


def fetch_mef_brechas(mode: str = "sample", limit: int = 5000) -> pd.DataFrame:
    return fetch_mef_resource(SOURCE_CONFIG.mef_brechas_resource_id, sample_mef_brechas(), mode, limit)


def fetch_opsd(mode: str = "sample", limit: int = 5000) -> pd.DataFrame:
    if mode == "sample":
        return sample_opsd()
    return pd.read_csv(SOURCE_CONFIG.opsd_time_series_url, nrows=limit)
