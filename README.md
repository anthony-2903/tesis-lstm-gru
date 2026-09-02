# Tesis LSTM, GRU y Stacking

Dashboard y backend experimental para comparar LSTM, GRU, BiRNN, TCN,
Transformer y Stacking en energía, phishing y fraude financiero.

## Requisitos

- Node.js 20 o superior.
- Python 3.11 compatible con TensorFlow del proyecto.
- Windows, Linux o WSL2. En Windows nativo TensorFlow actual usa CPU; para GPU se
  recomienda WSL2.

## Instalación

```powershell
npm install
python -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
```

Para desarrollo y auditoría de seguridad instale `backend/requirements-dev.txt`,
que también incluye `pip-audit`.

## Ejecución local

Backend:

```powershell
$env:PYTHONPATH = "backend"
backend/.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
```

Frontend, en otra terminal:

```powershell
npm run dev
```

La URL del backend se configura con `VITE_API_URL`; por defecto el frontend usa
`http://localhost:8000/api`.

## Seguridad de despliegue

Copie `.env.example` a un almacén de secretos del entorno y configure
`API_WRITE_TOKEN` para proteger todas las mutaciones y entrenamientos. No incluya
ese secreto en variables `VITE_*`, porque el frontend las publica en el bundle.
El dashboard público puede mantenerse de solo lectura y las operaciones costosas
ejecutarse desde un cliente administrativo que envíe `X-API-Key`. Restrinja
también `CORS_ALLOWED_ORIGINS` a los dominios reales.

## Verificación

```powershell
$env:PYTHONPATH = "backend"
backend/.venv/Scripts/python.exe -m unittest discover -s backend/tests
npm run build
npm audit
```

Las mismas puertas se ejecutan en `.github/workflows/quality.yml` para cada pull
request y actualización de las ramas principales. Dependabot revisa semanalmente
dependencias npm, Python y acciones; una actualización nunca sustituye las pruebas
ni autoriza una evaluación del test científico.

## Protocolo de tesis

Los tres paneles ofrecen pausa cooperativa. La solicitud se persiste, el proceso
activo la observa aunque se ejecute separado de la API, termina la unidad segura
actual y conserva sus checkpoints. El estado `paused` puede reanudarse sin abrir
el test ni repetir unidades cuyo contrato y hash ya fueron verificados.

Cada dominio expone un panel de protocolo completo, persistente y reanudable. El
test final se mantiene bloqueado durante datos, OOF, Stacking, calibración,
ablación, revalidación y XAI. Ninguna ruta implementada autoriza automáticamente
la evaluación final.

El resumen científico conjunto se consulta en `GET /api/thesis/status`. El
dashboard principal lo actualiza automáticamente mientras existe una corrida
activa. Su porcentaje mide solo la ejecución experimental real y asigna el mismo
peso a phishing, energía y finanzas; no mezcla programación ni redacción.

`GET /api/thesis/results-summary` normaliza por dominio una tabla de seis
candidatos (LSTM, GRU, BiRNN, TCN, Transformer y Stacking). La respuesta distingue
resultados demostrativos, candidatos pendientes y evidencia científica sellada;
no expone rutas locales ni presenta una métrica de validación como conclusión final.

Consulte [estado_implementacion_tesis.md](docs/estado_implementacion_tesis.md) y
[especificacion_experimental_stacking.md](docs/especificacion_experimental_stacking.md).
La separación entre entrenamiento y dashboard público se documenta en
[despliegue_seguro.md](docs/despliegue_seguro.md).
