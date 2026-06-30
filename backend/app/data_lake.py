from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from math import ceil
from typing import Any

from app.config import GOLD_DIR, RAW_DIR, SILVER_DIR
from app.external_sources import _fetch_json, _fetch_sec_json, _fetch_text, _normalize_domain
from app.utils import read_json, write_json


DOMAINS = ["phishing", "energia", "finanzas"]
MAX_TARGET = 20000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _domain_path(stage, domain: str):
    return stage / f"external_{domain}.json"


def _record(
    *,
    domain: str,
    source_id: str,
    source_name: str,
    category: str,
    record_id: str | int | None,
    label: str | None,
    value: Any = None,
    date: str | None = None,
    raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": f"{source_id}:{record_id}" if record_id is not None else f"{source_id}:{label}",
        "domain": domain,
        "sourceId": source_id,
        "sourceName": source_name,
        "category": category,
        "label": label,
        "value": value,
        "date": date,
        "raw": raw or {},
    }


def ingest_data_lake(domain: str = "all", target: int = 5000) -> dict[str, Any]:
    safe_target = max(100, min(int(target), MAX_TARGET))
    domains = DOMAINS if domain.strip().lower() == "all" else [_normalize_domain(domain)]
    results = []

    for current_domain in domains:
        if current_domain == "phishing":
            records = _ingest_phishing(safe_target)
        elif current_domain == "energia":
            records = _ingest_energy(safe_target)
        else:
            records = _ingest_finance(safe_target)

        payload = _build_payload(current_domain, safe_target, records)
        write_json(_domain_path(RAW_DIR, current_domain), payload)
        write_json(_domain_path(SILVER_DIR, current_domain), _to_silver(payload))
        write_json(_domain_path(GOLD_DIR, current_domain), _to_gold(payload))
        results.append(_to_gold(payload))

    return {"updatedAt": _now(), "target": safe_target, "domains": results}


def get_data_lake_summary() -> dict[str, Any]:
    domains = []
    total = 0
    for domain in DOMAINS:
        path = _domain_path(GOLD_DIR, domain)
        if path.exists():
            payload = read_json(path)
        else:
            payload = _empty_gold(domain)
        total += int(payload.get("totalRecords", 0))
        domains.append(payload)
    return {"updatedAt": _now(), "totalRecords": total, "domains": domains}


def get_data_lake_records(domain: str, page: int = 1, page_size: int = 100) -> dict[str, Any]:
    normalized = _normalize_domain(domain)
    safe_page = max(1, int(page))
    safe_page_size = max(10, min(int(page_size), 500))
    path = _domain_path(SILVER_DIR, normalized)
    if not path.exists():
        return {
            "domain": normalized,
            "page": safe_page,
            "pageSize": safe_page_size,
            "totalRecords": 0,
            "totalPages": 0,
            "records": [],
        }
    payload = read_json(path)
    records = payload.get("records", [])
    total = len(records)
    start = (safe_page - 1) * safe_page_size
    end = start + safe_page_size
    return {
        "domain": normalized,
        "page": safe_page,
        "pageSize": safe_page_size,
        "totalRecords": total,
        "totalPages": ceil(total / safe_page_size) if total else 0,
        "records": records[start:end],
    }


def _build_payload(domain: str, target: int, records: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(record["sourceId"] for record in records)
    category_counts = Counter(record["category"] for record in records)
    return {
        "domain": domain,
        "target": target,
        "updatedAt": _now(),
        "totalRecords": len(records),
        "sourceBreakdown": dict(source_counts),
        "categoryBreakdown": dict(category_counts),
        "records": records,
    }


def _to_silver(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: value for key, value in payload.items() if key != "records"},
        "records": [
            {key: value for key, value in record.items() if key != "raw"}
            for record in payload.get("records", [])
        ],
    }


def _to_gold(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "records"}


def _empty_gold(domain: str) -> dict[str, Any]:
    return {
        "domain": domain,
        "target": 0,
        "updatedAt": None,
        "totalRecords": 0,
        "sourceBreakdown": {},
        "categoryBreakdown": {},
    }


def _ingest_phishing(target: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    kev_quota = max(100, min(target // 3, 1200))

    try:
        kev = _fetch_json("https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json")
        for item in kev.get("vulnerabilities", [])[:kev_quota]:
            records.append(
                _record(
                    domain="phishing",
                    source_id="cisa_kev",
                    source_name="CISA KEV Catalog",
                    category="exploited_vulnerability",
                    record_id=item.get("cveID"),
                    label=item.get("vulnerabilityName"),
                    date=item.get("dateAdded"),
                    raw=item,
                )
            )
    except Exception:
        pass

    try:
        text = _fetch_text("https://urlhaus.abuse.ch/downloads/text_recent/")
        urls = [line for line in text.splitlines() if line and not line.startswith("#")]
        remaining = max(0, target - len(records))
        for index, url in enumerate(urls[:remaining], start=1):
            records.append(
                _record(
                    domain="phishing",
                    source_id="urlhaus",
                    source_name="URLhaus Recent URLs",
                    category="malicious_url",
                    record_id=index,
                    label=url,
                    raw={"url": url},
                )
            )
    except Exception:
        pass

    return records[:target]


def _ingest_energy(target: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        data = _fetch_json(
            "https://api.worldbank.org/v2/country/all/indicator/EG.ELC.ACCS.ZS",
            {"format": "json", "per_page": min(target, MAX_TARGET)},
        )
        rows = data[1] if isinstance(data, list) and len(data) > 1 else []
        for row in rows[:target]:
            country = row.get("country", {}).get("value")
            date = row.get("date")
            records.append(
                _record(
                    domain="energia",
                    source_id="world_bank_energy",
                    source_name="World Bank Energy Indicator API",
                    category="energy_access_indicator",
                    record_id=f"{row.get('countryiso3code')}:{date}",
                    label=f"{country} {date}",
                    value=row.get("value"),
                    date=date,
                    raw=row,
                )
            )
    except Exception:
        pass
    return records[:target]


def _ingest_finance(target: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sec_quota = target // 2

    try:
        tickers = _fetch_sec_json("https://www.sec.gov/files/company_tickers.json")
        rows = list(tickers.values()) if isinstance(tickers, dict) else []
        for row in rows[:sec_quota]:
            records.append(
                _record(
                    domain="finanzas",
                    source_id="sec_edgar",
                    source_name="SEC EDGAR APIs",
                    category="sec_registrant",
                    record_id=row.get("cik_str"),
                    label=row.get("title"),
                    raw=row,
                )
            )
    except Exception:
        pass

    try:
        remaining = max(0, target - len(records))
        data = _fetch_json(
            "https://api.fdic.gov/banks/institutions",
            {
                "format": "json",
                "limit": remaining,
                "fields": "CERT,NAME,CITY,STALP,ACTIVE,ASSET",
            },
        )
        for row in data.get("data", [])[:remaining]:
            item = row.get("data", {})
            records.append(
                _record(
                    domain="finanzas",
                    source_id="fdic_bankfind",
                    source_name="FDIC BankFind Suite API",
                    category="financial_institution",
                    record_id=item.get("CERT"),
                    label=item.get("NAME"),
                    value=item.get("ASSET"),
                    raw=item,
                )
            )
    except Exception:
        pass

    return records[:target]
