# Arquitectura de entrenamiento con Google Colab

## Objetivo

Separar el dashboard web del entrenamiento pesado de modelos. Vercel debe servir el
frontend y mostrar resultados. Google Colab debe entrenar los modelos reales y
exportar metricas para los graficos.

## Flujo propuesto

```text
CSV/XLSX
  -> limpieza y normalizacion
  -> preparacion de secuencias temporales
  -> entrenamiento en Google Colab
       - LSTM
       - GRU
       - BRNN
       - Transformer
       - TCN
  -> evaluacion
  -> XAI con aproximacion SHAP temporal
  -> exportacion de metrics.json, xai.json y modelos .keras
  -> Supabase/R2 o carpeta de resultados
  -> frontend en Vercel
```

## Responsabilidades por capa

| Capa | Responsabilidad | Ubicacion |
|---|---|---|
| Frontend | Carga visual, dashboard, graficos y reportes | `src/` |
| Limpieza | Eliminar nulos, ordenar registros, detectar tipos | `backend/b_limpieza/` |
| Transformacion | Normalizar columnas numericas | `backend/c_transformacion/` |
| Preparacion | Convertir datos tabulares a secuencias temporales | `backend/colab_training_pipeline.py` |
| Entrenamiento | Construir y entrenar cada arquitectura | `backend/e_entrenamiento/model_architectures.py` |
| Evaluacion | Precision, recall, F1, AUC y matriz de confusion | `backend/colab_training_pipeline.py` |
| XAI | Importancia de variables y pasos temporales | `backend/x_xai/shap_explainer.py` |
| Resultados | Exportar JSON para el dashboard | `training_outputs/metrics.json` |

## Arquitecturas por modelo

### LSTM

Modelo recurrente orientado a dependencias temporales largas.

```text
Input(secuencia)
  -> Masking
  -> LSTM(128, return_sequences=True)
  -> BatchNormalization
  -> LSTM(64)
  -> Dense(64, relu)
  -> Dropout(0.30)
  -> Dense(1, sigmoid)
```

Uso recomendado: baseline fuerte para series con memoria temporal y patrones
que dependen de varios pasos previos.

### GRU

Modelo recurrente mas liviano que LSTM.

```text
Input(secuencia)
  -> Masking
  -> GRU(96, return_sequences=True)
  -> BatchNormalization
  -> GRU(48)
  -> Dense(48, relu)
  -> Dropout(0.25)
  -> Dense(1, sigmoid)
```

Uso recomendado: comparar rendimiento similar a LSTM con menor costo de
entrenamiento e inferencia.

### BRNN

Modelo recurrente bidireccional que procesa la secuencia en sentido directo e
inverso para capturar dependencias anteriores y posteriores dentro de una
ventana temporal.

```text
Input(secuencia)
  -> Masking
  -> Bidirectional(LSTM(96, return_sequences=True))
  -> BatchNormalization
  -> Bidirectional(GRU(48))
  -> Dense(64, relu)
  -> Dropout(0.30)
  -> Dense(1, sigmoid)
```

Uso recomendado: datos donde el contexto completo de la ventana ayuda a
identificar anomalías, fraudes o patrones sospechosos con dependencias
temporales en ambas direcciones.

### Transformer

Modelo basado en auto-atencion para capturar relaciones globales en la secuencia.

```text
Input(secuencia)
  -> Dense(64)
  -> TransformerEncoder
       -> MultiHeadAttention
       -> Add + LayerNormalization
       -> FeedForward 1D
       -> Add + LayerNormalization
  -> TransformerEncoder
  -> GlobalAveragePooling1D
  -> Dense(64, relu)
  -> Dropout(0.35)
  -> Dense(1, sigmoid)
```

Uso recomendado: datos con dependencias mas largas o relaciones no locales.

### TCN

Modelo convolucional temporal con convoluciones causales dilatadas.

```text
Input(secuencia)
  -> TCNBlock(dilation=1)
  -> TCNBlock(dilation=2)
  -> TCNBlock(dilation=4)
  -> TCNBlock(dilation=8)
  -> GlobalAveragePooling1D
  -> Dense(64, relu)
  -> Dropout(0.25)
  -> Dense(1, sigmoid)
```

Uso recomendado: series temporales donde se busca buen equilibrio entre velocidad
y precision.

## Ejecucion en Colab

Desde la raiz del proyecto dentro de Colab:

```bash
pip install -r backend/requirements-colab.txt
python backend/colab_training_pipeline.py \
  --dataset /content/datos.csv \
  --target anomaly \
  --sequence-length 24 \
  --output-dir /content/training_outputs
```

Si el dataset no tiene columna objetivo, se puede omitir `--target`. En ese caso
el pipeline infiere anomalias usando desviacion estadistica de la primera columna
numerica.

## Salida para conectar con el frontend

El entrenamiento genera:

```text
training_outputs/
  metrics.json
  xai.json
  lstm_model.keras
  lstm_predictions.csv
  gru_model.keras
  gru_predictions.csv
  brnn_model.keras
  brnn_predictions.csv
  transformer_model.keras
  transformer_predictions.csv
  tcn_model.keras
  tcn_predictions.csv
```

`metrics.json` es el archivo que debe alimentar los graficos reales del dashboard.
`xai.json` alimenta la vista de interpretabilidad del frontend en `/xai`.
La aplicacion en Vercel no debe entrenar modelos ni calcular SHAP; debe leer estos resultados.

## Siguiente integracion recomendada

1. Subir `metrics.json` a Supabase Storage o Cloudflare R2.
2. Guardar una referencia del experimento en Supabase.
3. Cambiar el frontend para leer metricas reales desde esa URL.
4. Mantener `src/lib/evaluator.ts` solo como modo demo o fallback.
5. Cargar `xai.json` en la vista `/xai` para visualizar interpretabilidad.
