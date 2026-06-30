from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import requests

from app.config import SOURCE_CONFIG


Domain = Literal["phishing", "energia", "finanzas"]

TIMEOUT_SECONDS = 8

SOURCE_CATALOG: dict[str, list[dict[str, Any]]] = {
    "phishing": [
        {
            "id": "google_safe_browsing",
            "name": "Google Safe Browsing API",
            "provider": "Google",
            "url": "https://developers.google.com/safe-browsing/v4",
            "official": True,
            "requiresKey": True,
            "configured": bool(SOURCE_CONFIG.google_safe_browsing_api_key),
            "useCase": "Validar URLs contra listas de phishing, malware y sitios inseguros.",
        },
        {
            "id": "cisa_kev",
            "name": "CISA KEV Catalog",
            "provider": "Cybersecurity and Infrastructure Security Agency",
            "url": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
            "official": True,
            "requiresKey": False,
            "configured": True,
            "useCase": "Agregar contexto oficial de vulnerabilidades explotadas.",
        },
        {
            "id": "urlhaus",
            "name": "URLhaus Recent URLs",
            "provider": "abuse.ch / Spamhaus",
            "url": "https://urlhaus.abuse.ch/downloads/text_recent/",
            "official": False,
            "requiresKey": False,
            "configured": True,
            "useCase": "Ampliar muestras de URLs maliciosas para entrenamiento experimental.",
        },
    ],
    "energia": [
        {
            "id": "eia",
            "name": "EIA Open Data API",
            "provider": "U.S. Energy Information Administration",
            "url": "https://www.eia.gov/opendata/",
            "official": True,
            "requiresKey": True,
            "configured": bool(SOURCE_CONFIG.eia_api_key),
            "useCase": "Series temporales oficiales de electricidad, demanda, generacion y precios.",
        },
        {
            "id": "entsoe",
            "name": "ENTSO-E Transparency Platform",
            "provider": "European Network of Transmission System Operators for Electricity",
            "url": "https://transparency.entsoe.eu/",
            "official": True,
            "requiresKey": True,
            "configured": bool(SOURCE_CONFIG.entsoe_api_key),
            "useCase": "Series horarias de carga, generacion y mercado electrico europeo.",
        },
        {
            "id": "world_bank_energy",
            "name": "World Bank Energy Indicator API",
            "provider": "World Bank",
            "url": "https://api.worldbank.org/v2/country/all/indicator/EG.ELC.ACCS.ZS?format=json&per_page=80",
            "official": True,
            "requiresKey": False,
            "configured": True,
            "useCase": "Contexto energetico global para enriquecer analisis por pais.",
        },
    ],
    "finanzas": [
        {
            "id": "cfpb_complaints",
            "name": "CFPB Consumer Complaint Database",
            "provider": "Consumer Financial Protection Bureau",
            "url": "https://www.consumerfinance.gov/data-research/consumer-complaints/",
            "official": True,
            "requiresKey": False,
            "configured": True,
            "useCase": "Reclamos financieros reales para clasificacion textual de riesgo/fraude.",
        },
        {
            "id": "sec_edgar",
            "name": "SEC EDGAR APIs",
            "provider": "U.S. Securities and Exchange Commission",
            "url": "https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
            "official": True,
            "requiresKey": False,
            "configured": True,
            "useCase": "Metadatos y filings financieros para senales publicas de riesgo.",
        },
        {
            "id": "fdic_bankfind",
            "name": "FDIC BankFind Suite API",
            "provider": "Federal Deposit Insurance Corporation",
            "url": "https://api.fdic.gov/banks/docs",
            "official": True,
            "requiresKey": False,
            "configured": True,
            "useCase": "Instituciones financieras aseguradas, historico bancario y variables de riesgo institucional.",
        },
        {
            "id": "fincen_sar_stats",
            "name": "FinCEN SAR Stats",
            "provider": "Financial Crimes Enforcement Network",
            "url": "https://www.fincen.gov/reports/sar-stats",
            "official": True,
            "requiresKey": False,
            "configured": True,
            "useCase": "Contexto oficial sobre reportes de actividad sospechosa.",
        },
    ],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _headers() -> dict[str, str]:
    return {
        "User-Agent": SOURCE_CONFIG.user_agent,
        "Accept": "application/json,text/plain,*/*",
    }


def _sec_headers() -> dict[str, str]:
    return {
        "User-Agent": SOURCE_CONFIG.sec_user_agent,
        "Accept": "application/json",
        "Host": "www.sec.gov",
    }


def get_source_catalog(domain: str | None = None) -> dict[str, Any]:
    if domain:
        normalized = _normalize_domain(domain)
        return {"updatedAt": _now(), "domain": normalized, "sources": SOURCE_CATALOG[normalized]}
    return {"updatedAt": _now(), "sources": SOURCE_CATALOG}


def fetch_external_data(domain: str, limit: int = 100) -> dict[str, Any]:
    normalized = _normalize_domain(domain)
    safe_limit = max(1, min(limit, 500))
    if normalized == "phishing":
        return _fetch_phishing(safe_limit)
    if normalized == "energia":
        return _fetch_energy(safe_limit)
    return _fetch_finance(safe_limit)


def _normalize_domain(domain: str) -> Domain:
    normalized = domain.strip().lower()
    if normalized in {"phishtank", "phishing"}:
        return "phishing"
    if normalized in {"energia", "energy"}:
        return "energia"
    if normalized in {"finanzas", "finance", "fraude"}:
        return "finanzas"
    raise ValueError(f"Dominio no soportado: {domain}")


def _fetch_json(url: str, params: dict[str, Any] | None = None) -> Any:
    response = requests.get(url, params=params, headers=_headers(), timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _fetch_sec_json(url: str) -> Any:
    response = requests.get(url, headers=_sec_headers(), timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _fetch_text(url: str) -> str:
    response = requests.get(url, headers=_headers(), timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def _source_result(source_id: str, status: str, records: list[dict[str, Any]], error: str | None = None) -> dict[str, Any]:
    source = next(
        item
        for sources in SOURCE_CATALOG.values()
        for item in sources
        if item["id"] == source_id
    )
    return {
        "source": source,
        "status": status,
        "count": len(records),
        "records": records,
        "error": error,
    }


def _fetch_phishing(limit: int) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    try:
        kev = _fetch_json("https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json")
        vulnerabilities = kev.get("vulnerabilities", [])[:limit]
        results.append(
            _source_result(
                "cisa_kev",
                "ok",
                [
                    {
                        "id": item.get("cveID"),
                        "label": item.get("vulnerabilityName"),
                        "vendor": item.get("vendorProject"),
                        "product": item.get("product"),
                        "date": item.get("dateAdded"),
                        "category": "exploited_vulnerability",
                    }
                    for item in vulnerabilities
                ],
            )
        )
    except Exception as exc:
        results.append(_source_result("cisa_kev", "error", [], str(exc)))

    try:
        text = _fetch_text("https://urlhaus.abuse.ch/downloads/text_recent/")
        urls = [line for line in text.splitlines() if line and not line.startswith("#")][:limit]
        results.append(
            _source_result(
                "urlhaus",
                "ok",
                [{"id": index + 1, "url": url, "category": "malicious_url"} for index, url in enumerate(urls)],
            )
        )
    except Exception as exc:
        results.append(_source_result("urlhaus", "error", [], str(exc)))

    return {"domain": "phishing", "updatedAt": _now(), "results": results}


def _fetch_energy(limit: int) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    try:
        data = _fetch_json(
            "https://api.worldbank.org/v2/country/all/indicator/EG.ELC.ACCS.ZS",
            {"format": "json", "per_page": limit},
        )
        rows = data[1] if isinstance(data, list) and len(data) > 1 else []
        results.append(
            _source_result(
                "world_bank_energy",
                "ok",
                [
                    {
                        "country": row.get("country", {}).get("value"),
                        "countryCode": row.get("countryiso3code"),
                        "date": row.get("date"),
                        "value": row.get("value"),
                        "indicator": row.get("indicator", {}).get("value"),
                    }
                    for row in rows[:limit]
                ],
            )
        )
    except Exception as exc:
        results.append(_source_result("world_bank_energy", "error", [], str(exc)))

    if SOURCE_CONFIG.eia_api_key:
        results.append(_fetch_eia(limit))
    else:
        results.append(_source_result("eia", "needs_key", [], "Configura EIA_API_KEY para consumir esta API."))

    if SOURCE_CONFIG.entsoe_api_key:
        results.append(_source_result("entsoe", "configured", [], "Token disponible; falta seleccionar pais y periodo para consulta."))
    else:
        results.append(_source_result("entsoe", "needs_key", [], "Configura ENTSOE_API_KEY para consumir esta API."))

    return {"domain": "energia", "updatedAt": _now(), "results": results}


def _fetch_eia(limit: int) -> dict[str, Any]:
    try:
        data = _fetch_json(
            "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/",
            {
                "api_key": SOURCE_CONFIG.eia_api_key,
                "frequency": "daily",
                "data[0]": "value",
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
                "length": limit,
            },
        )
        rows = data.get("response", {}).get("data", [])
        return _source_result("eia", "ok", rows[:limit])
    except Exception as exc:
        return _source_result("eia", "error", [], str(exc))


def _fetch_finance(limit: int) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    try:
        data = _fetch_json(
            "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/",
            {"size": limit, "sort": "created_date_desc", "format": "json"},
        )
        hits = data.get("hits", {}).get("hits", [])
        results.append(
            _source_result(
                "cfpb_complaints",
                "ok",
                [
                    {
                        "id": hit.get("_id"),
                        "date": hit.get("_source", {}).get("date_received"),
                        "product": hit.get("_source", {}).get("product"),
                        "issue": hit.get("_source", {}).get("issue"),
                        "company": hit.get("_source", {}).get("company"),
                        "state": hit.get("_source", {}).get("state"),
                    }
                    for hit in hits[:limit]
                ],
            )
        )
    except Exception as exc:
        results.append(_source_result("cfpb_complaints", "error", [], str(exc)))

    try:
        tickers = _fetch_sec_json("https://www.sec.gov/files/company_tickers.json")
        rows = list(tickers.values())[:limit] if isinstance(tickers, dict) else []
        results.append(
            _source_result(
                "sec_edgar",
                "ok",
                [
                    {
                        "cik": row.get("cik_str"),
                        "ticker": row.get("ticker"),
                        "title": row.get("title"),
                        "category": "sec_registrant",
                    }
                    for row in rows
                ],
            )
        )
    except Exception as exc:
        results.append(_source_result("sec_edgar", "error", [], str(exc)))

    try:
        data = _fetch_json(
            "https://api.fdic.gov/banks/institutions",
            {
                "format": "json",
                "limit": limit,
                "fields": "CERT,NAME,CITY,STALP,ACTIVE,ASSET",
            },
        )
        rows = data.get("data", [])
        results.append(
            _source_result(
                "fdic_bankfind",
                "ok",
                [
                    {
                        "cert": row.get("data", {}).get("CERT"),
                        "name": row.get("data", {}).get("NAME"),
                        "city": row.get("data", {}).get("CITY"),
                        "state": row.get("data", {}).get("STALP"),
                        "active": row.get("data", {}).get("ACTIVE"),
                        "asset": row.get("data", {}).get("ASSET"),
                    }
                    for row in rows[:limit]
                ],
            )
        )
    except Exception as exc:
        results.append(_source_result("fdic_bankfind", "error", [], str(exc)))

    results.append(
        _source_result(
            "fincen_sar_stats",
            "reference",
            [
                {
                    "label": "SAR Stats",
                    "url": "https://www.fincen.gov/reports/sar-stats",
                    "category": "official_context",
                }
            ],
        )
    )

    return {"domain": "finanzas", "updatedAt": _now(), "results": results}
