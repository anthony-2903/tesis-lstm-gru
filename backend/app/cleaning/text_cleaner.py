from __future__ import annotations

import re
from urllib.parse import urlparse

import pandas as pd

from app.utils import entropy


def clean_phishtank(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["url"] = data["url"].astype(str).str.strip()
    data = data[data["url"].str.len() > 0].drop_duplicates(subset=["url"])
    if "label" not in data.columns:
        data["label"] = 1
    return data.reset_index(drop=True)


def make_benign_urls(target: int = 500) -> pd.DataFrame:
    domains = [
        "https://www.gob.pe",
        "https://www.mef.gob.pe",
        "https://datosabiertos.mef.gob.pe",
        "https://open-power-system-data.org",
        "https://www.wikipedia.org",
        "https://www.google.com",
        "https://github.com",
        "https://www.python.org",
        "https://pandas.pydata.org",
        "https://scikit-learn.org",
        "https://www.cisa.gov",
        "https://www.eia.gov",
        "https://www.sec.gov",
        "https://www.consumerfinance.gov",
        "https://data.worldbank.org",
        "https://transparency.entsoe.eu",
        "https://www.entsoe.eu",
        "https://openei.org",
        "https://www.fdic.gov",
        "https://www.fincen.gov",
    ]
    paths = [
        "",
        "/",
        "/about",
        "/data",
        "/api",
        "/docs",
        "/research",
        "/downloads",
        "/contact",
        "/reports",
    ]
    urls = []
    for domain in domains:
        for path in paths:
            urls.append(f"{domain.rstrip('/')}{path}")
            if len(urls) >= target:
                return pd.DataFrame({"url": urls, "label": 0, "target": "benign"})
    return pd.DataFrame({"url": urls, "label": 0, "target": "benign"})


def build_url_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    labels = frame["label"].astype(int)
    ip_pattern = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
    for url in frame["url"].astype(str):
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = parsed.netloc.lower()
        path = parsed.path.lower()
        rows.append(
            {
                "url_length": len(url),
                "host_length": len(host),
                "path_length": len(path),
                "dot_count": url.count("."),
                "dash_count": url.count("-"),
                "slash_count": url.count("/"),
                "digit_count": sum(ch.isdigit() for ch in url),
                "special_count": sum(not ch.isalnum() for ch in url),
                "entropy": entropy(url),
                "has_ip_host": int(bool(ip_pattern.match(host))),
                "uses_https": int(parsed.scheme == "https"),
                "query_length": len(parsed.query),
            }
        )
    return pd.DataFrame(rows), labels
