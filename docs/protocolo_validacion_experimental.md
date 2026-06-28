# Protocolo de validación experimental

## Objetivo

Comparar arquitecturas LSTM, GRU, BRNN, Transformer y TCN para detección de anomalías en tres dominios: phishing, energía y finanzas.

## Flujo metodológico

1. Recolectar o preparar datasets por dominio.
2. Limpiar datos nulos, inconsistentes o no utilizables.
3. Transformar los datos según su naturaleza: texto, serie temporal o tabla.
4. Entrenar y evaluar las arquitecturas bajo métricas comparables.
5. Consolidar resultados en el backend.
6. Visualizar métricas, matrices de confusión, historial de anomalías y explicaciones XAI en el dashboard.

## Métricas usadas

- F1-score: balance entre precisión y recall.
- Precisión: proporción de alertas positivas que fueron correctas.
- Recall: proporción de anomalías reales detectadas.
- RMSE: error de predicción para series temporales.
- Matriz de confusión: VP, FP, FN y VN.
- Tiempo de entrenamiento: costo computacional aproximado.
- XAI: importancia de variables y pasos temporales.

## Criterio de conclusión

No se declara un ganador universal. El modelo recomendado depende del dominio, del costo de falsos positivos, del costo de falsos negativos y de los recursos disponibles para entrenamiento o inferencia.
