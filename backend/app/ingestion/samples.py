from __future__ import annotations

import pandas as pd


def sample_phishtank() -> pd.DataFrame:
    phishing = [
        "https://secure-login-paypal.verify-user-alert.com/session",
        "http://banco-cliente-pe.actualizar-datos.click/login",
        "https://office365-security-check.example-phish.net/auth",
        "http://facturacion-sunat-validacion.xyz/consulta",
        "https://wallet-verify-airdrop.claim-now.top",
    ]
    benign = [
        "https://www.gob.pe/mef",
        "https://www.google.com/search?q=inversion+publica",
        "https://datosabiertos.mef.gob.pe",
        "https://open-power-system-data.org",
        "https://www.wikipedia.org",
    ]
    rows = []
    for idx, url in enumerate(phishing, start=1):
        rows.append({"phish_id": idx, "url": url, "verified": "yes", "online": "yes", "target": "sample", "label": 1})
    for idx, url in enumerate(benign, start=100):
        rows.append({"phish_id": idx, "url": url, "verified": "no", "online": "yes", "target": "benign", "label": 0})
    return pd.DataFrame(rows)


def sample_mef_operadores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"entidad": "Municipalidad Provincial A", "sector": "Gobierno Local", "rol": "UF", "departamento": "Lima"},
            {"entidad": "Gobierno Regional B", "sector": "Gobierno Regional", "rol": "OPMI", "departamento": "Cusco"},
            {"entidad": "Municipalidad Distrital C", "sector": "Gobierno Local", "rol": "UEI", "departamento": "Piura"},
            {"entidad": "Municipalidad Distrital C", "sector": "Gobierno Local", "rol": "UEI", "departamento": "Piura"},
        ]
    )


def sample_mef_brechas() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sector": "Educacion", "servicio": "Primaria", "indicador": "Capacidad instalada inadecuada", "valor": 42.5, "anio": 2023},
            {"sector": "Salud", "servicio": "Hospitales", "indicador": "Capacidad instalada inadecuada", "valor": 31.2, "anio": 2023},
            {"sector": "Transportes", "servicio": "Red vial", "indicador": "Condicion inadecuada", "valor": 58.9, "anio": 2023},
            {"sector": "Saneamiento", "servicio": "Agua potable", "indicador": "Poblacion sin acceso", "valor": 21.1, "anio": 2023},
        ]
    )


def sample_opsd() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=240, freq="h", tz="UTC")
    base = 55 + 8 * pd.Series(range(len(dates))).map(lambda i: __import__("math").sin(i / 24 * 3.14159)).to_numpy()
    load = base + pd.Series(range(len(dates))).map(lambda i: (i % 7) * 0.7).to_numpy()
    load[60] += 24
    load[151] -= 20
    return pd.DataFrame(
        {
            "utc_timestamp": dates.astype(str),
            "DE_load_actual_entsoe_transparency": load,
            "DE_load_forecast_entsoe_transparency": load * 0.98,
            "DE_solar_generation_actual": abs(load * 0.18),
            "DE_wind_generation_actual": abs(load * 0.23),
        }
    )
