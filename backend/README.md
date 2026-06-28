# Backend local para tesis LSTM/GRU/BRNN/Transformer/TCN

Este backend expone los endpoints que consume el frontend:

- `GET /api/dashboard`
- `GET /api/analysis`
- `GET /api/comparison`
- `GET /api/history`
- `GET /api/xai`
- `GET /api/external-sources`
- `GET /api/external-data?domain=phishing`
- `GET /api/ai-analysis?type=general`

## Instalacion

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecutar pipeline inicial

Modo rapido con datos de muestra:

```bash
python -m app.pipeline --mode sample
```

Modo remoto, descargando fuentes reales:

```bash
python -m app.pipeline --mode remote
```

## Levantar API

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

El frontend espera `http://localhost:8000/api`.

## Fuentes externas opcionales

El backend ahora expone fuentes oficiales/institucionales para ampliar data:

- Phishing: CISA KEV y URLhaus. Google Safe Browsing queda listo si configuras `GOOGLE_SAFE_BROWSING_API_KEY`.
- Energia: World Bank Energy funciona sin token. EIA y ENTSO-E quedan listas con `EIA_API_KEY` y `ENTSOE_API_KEY`.
- Finanzas: SEC EDGAR, FDIC BankFind y FinCEN. CFPB queda registrado como fuente oficial; algunos entornos bloquean su API con 403.

Variables opcionales:

```bash
set GOOGLE_SAFE_BROWSING_API_KEY=tu_api_key
set EIA_API_KEY=tu_api_key
set ENTSOE_API_KEY=tu_token
set SEC_USER_AGENT="tu-proyecto tu-correo@example.com"
```
