# Backend local para tesis LSTM/GRU/BRNN/Transformer/TCN

Este backend expone los endpoints que consume el frontend:

- `GET /api/dashboard`
- `GET /api/analysis`
- `GET /api/comparison`
- `GET /api/history`
- `GET /api/xai`
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
