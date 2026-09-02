# Especificación experimental para modelos base y stacking

## Propósito

Este documento establece el contrato metodológico que debe cumplir el proyecto antes de afirmar que LSTM, GRU, BRNN, TCN, Transformer o Stacking supera a otro modelo. La configuración ejecutable se encuentra en `backend/config/experiments.json` y el formato mínimo de resultados en `backend/config/result-contract.example.json`.

## Alcance y estado actual

El flujo actual de ingesta, limpieza, artefactos, API y dashboard se conserva como base. Los estimadores de `backend/app/training/trainers.py` son prototipos clásicos y no deben presentarse como las cinco redes neuronales finales. Los resultados producidos por ese flujo se considerarán `demo` hasta que el nuevo protocolo los sustituya.

El orden de implementación será:

1. Energía, porque ya existe una serie temporal coherente.
2. Ciberseguridad, después de preparar URLs como secuencias y reunir positivos y negativos reales.
3. Finanzas, cuando exista un dataset de eventos coherente, fechado y con objetivo verificable.
4. Stacking por dominio, después de validar los cinco modelos base.
5. Comparación global como análisis secundario, no como sustituto de los resultados por dominio.

## Preguntas de investigación

### Ciberseguridad

¿El stacking mejora la detección de URLs maliciosas frente a modelos recurrentes, convolucionales y de atención entrenados sobre la misma representación secuencial?

### Energía

¿El stacking reduce el error de predicción y mejora la detección de anomalías basada en residuos en series eléctricas?

### Finanzas

¿El stacking mejora la detección de eventos financieros fraudulentos o anómalos frente a los cinco modelos base?

La pregunta financiera queda bloqueada hasta definir un evento, una entidad, una fecha y una etiqueta real. Mezclar instituciones SEC, registros FDIC y datos administrativos MEF no constituye un dataset de fraude.

## Protocolo sin fuga de información

Cada ejecución debe seguir este orden:

```text
Dataset versionado
  -> test final bloqueado
  -> train/validation del desarrollo
  -> folds internos
  -> ajuste de preprocesadores con cada train fold
  -> entrenamiento de cada modelo base
  -> predicciones out-of-fold
  -> entrenamiento y selección del meta-learner
  -> reentrenamiento sobre train + validation
  -> evaluación única sobre test
```

Reglas obligatorias:

- El escalado, imputación, vocabulario y selección de variables se ajustan solo con entrenamiento.
- Las ventanas temporales no pueden cruzar límites entre particiones.
- El stacking se entrena con predicciones out-of-fold, nunca con predicciones in-sample.
- Los umbrales se calibran con validación.
- Los cinco modelos y los ensambles usan exactamente las mismas particiones externas.
- El conjunto de prueba no participa en hiperparámetros, meta-learner, umbrales ni selección de variables.
- Todos los experimentos registran versión del dataset, configuración, semilla y artefactos.

## Modelos que competirán

### Modelos base

- LSTM.
- GRU.
- BRNN bidireccional.
- TCN causal y dilatada.
- Transformer Encoder.

### Baselines de ensamble

- Media de predicciones para regresión.
- Media de probabilidades para clasificación.
- Voting.
- Promedio ponderado con pesos aprendidos en validación.

### Stacking

El stacking recibe predicciones o probabilidades de los cinco modelos y medidas de desacuerdo. Los primeros meta-learners serán deliberadamente pequeños e interpretables:

- Regresión lineal o logística.
- Ridge.
- Gradient Boosting.

Una red neuronal como meta-learner solo se aceptará si mejora de forma reproducible sin sobreajuste.

## Métricas

### Clasificación

- Principal: PR-AUC.
- Secundarias: F1, precision, recall, ROC-AUC, MCC y balanced accuracy.
- Operativas: tasa de falsos positivos, falsos negativos y coste esperado.

### Predicción energética

- Principal: RMSE.
- Secundarias: MAE, SMAPE, R² y error por horizonte.
- Las métricas de clasificación de anomalías solo se reportan cuando existen etiquetas independientes.

### Eficiencia

- Tiempo real de entrenamiento.
- Latencia por muestra, medida después de calentamiento.
- Tamaño serializado del modelo.
- Memoria máxima cuando el entorno de ejecución permita medirla de forma reproducible.

No se usarán valores escritos manualmente para velocidad, memoria o confianza.

## Repeticiones, boxplots e incertidumbre

Se usarán al menos cinco semillas y cinco folds internos. Cada valor mostrado en un boxplot debe corresponder a un fold o repetición identificable. La tabla final debe incluir:

- Media.
- Mediana.
- Desviación estándar.
- Intervalo de confianza del 95 %.
- Mejor y peor ejecución.
- Número de observaciones experimentales.

Las pruebas estadísticas se definirán según la estructura final de las repeticiones. Además del valor p, se reportará tamaño del efecto y corrección por comparaciones múltiples cuando corresponda.

## Estudio de ablación del stacking

El stacking final se comparará con:

1. Cada modelo base.
2. Media o voting.
3. Promedio ponderado.
4. Stacking completo.
5. Stacking retirando un modelo base a la vez.

También se reportará la correlación entre errores. El stacking solo queda metodológicamente justificado si existe diversidad útil entre los modelos base.

## Criterio de superioridad

El stacking no se declarará ganador por obtener el mejor valor en una única ejecución. Debe cumplir simultáneamente:

- Mejor métrica primaria media sobre las mismas particiones.
- Intervalos de confianza reportados.
- Mejora estadística y tamaño de efecto relevante.
- Estabilidad comparable o superior.
- Ausencia de degradaciones críticas en falsos positivos o falsos negativos.
- Coste operativo documentado.

Si no gana en un dominio, se informará el resultado y se analizará por qué. No se alterarán datos, umbrales o métricas para forzar el ranking.

## Contrato de resultados

Cada ejecución guardará como mínimo:

```text
run_id
protocol_version
domain
dataset_id y dataset_version
split y rangos temporales
model_id y family
seed
hyperparameters
metrics
operational_metrics
predictions_artifact
model_artifact
```

Para stacking también serán obligatorios:

```text
base_models
meta_learner
oof_verified = true
meta_features
```

El dashboard debe consumir estos resultados. No debe inferir ganadores ni crear valores de rendimiento que no existan en los artefactos.

## Puertas de calidad antes de avanzar

### Puerta 1: datos

- Esquema validado.
- Procedencia y licencia documentadas.
- Objetivo real disponible.
- Duplicados entre particiones controlados.
- Tamaño mínimo cumplido.

### Puerta 2: modelo base

- Arquitectura real implementada.
- Prueba de forma de entrada y salida.
- Entrenamiento reproducible.
- Predicciones y métricas persistidas.
- Sin uso del test durante desarrollo.

### Puerta 3: stacking

- Predicciones out-of-fold verificadas.
- Baselines simples disponibles.
- Meta-learner seleccionado con validación.
- Ablación y diversidad calculadas.

### Puerta 4: presentación

- Tablas y gráficos provienen de artefactos reales.
- Boxplots muestran folds o repeticiones reales.
- Resultados demo están identificados.
- Conclusiones reflejan las métricas, incertidumbre y limitaciones.

## Estado implementado y próximo incremento técnico

Energía ya dispone de partición cronológica, modelos base, predicciones OOF y
Stacking. Phishing ya dispone de datos reales auditados, secuencias de caracteres,
cinco folds agrupados por dominio, los cinco modelos base, diagnóstico OOF de
Stacking y selección sobre validación externa. Los modelos finales se reentrenan
con desarrollo, los metamodelos se ajustan exclusivamente con probabilidades OOF
y todos los candidatos se comparan sobre las mismas URLs de validation. El umbral
se calibra maximizando MCC y el test continúa bloqueado.

El estudio de diversidad y ablación de phishing también está implementado:
correlación de probabilidades y residuos, desacuerdo, fallos compartidos, solape
de falsos positivos y negativos, y Stacking retirando una arquitectura base por
vez. Las diferencias de PR-AUC incluyen intervalos bootstrap agrupados por dominio.

La puerta de congelación experimental ya está implementada. Rechaza corridas demo,
exige cinco semillas, todas las filas de desarrollo, cinco folds, cinco modelos,
validación y ablación candidatas de tesis, estabilidad entre semillas, dataset
científicamente apto y verificación completa de hashes. Cuando todos los controles
pasan, genera configuración, inventario, manifiesto y sello inmutables. El test
sigue requiriendo una autorización separada y admite una única evaluación.

La corrida piloto actual queda bloqueada, entre otros motivos, porque usa una
semilla y 500 de 14 000 filas, y porque el dataset todavía presenta acoplamiento
entre fuente y etiqueta. La puerta de datos ya implementa paquetes curados con
SHA-256, cita, licencia y evidencia por fila; exige dos fuentes independientes por
clase, una fuente con ambas etiquetas, negativos verificados y control de atajos
por forma de URL. PhishTank + Tranco sigue siendo piloto y nunca se promociona
automáticamente a ground truth de tesis.

La puerta ya fue cerrada con PhiUSIIL v2 y LegitPhish v2, ambas fuentes mixtas y
CC BY 4.0. Sus versiones, DOI, tamaños y SHA-256 están fijados antes de descargar;
las etiquetas originales se mapean explícitamente al contrato interno. El paquete
curado contiene 80 000 URLs y el silver científico selecciona 10 000 por clase.
Las cinco particiones OOF fueron regeneradas para el nuevo hash, mientras el test
permanece bloqueado y sin evaluar.

El ejecutor de esa corrida ya es reanudable. Cada una de las 125 combinaciones
modelo-fold-semilla persiste modelo, probabilidades y métricas en una unidad
atómica con SHA-256. Una reanudación solo reutiliza unidades cuyo fingerprint de
dataset/configuración, tokenizador, cobertura, etiquetas y artefactos coincide;
las unidades incompletas o dañadas se entrenan otra vez. El preflight registra
plataforma, versiones, GPU y disco, pero no lee ni desbloquea test.

El siguiente paso operativo es ejecutar los cinco modelos base con las
cinco semillas sobre las 14 000 filas de desarrollo de este nuevo dataset. Los
experimentos demo anteriores quedan conservados como historial, pero su linaje no
coincide con el silver actual y la puerta de congelación los rechaza explícitamente.

Finanzas comenzará después de cerrar esta puerta metodológica de phishing. Primero
se definirán evento, entidad, fecha, etiqueta real y fuente versionable; no se
adaptará el pipeline a tablas financieras heterogéneas sin un objetivo verificable.
## Estado implementado de Finanzas

El dominio financiero ya no interpreta registros de MEF, SEC o FDIC como fraude.
Se fijó la tarea `transaction_level_fraud_classification` con el siguiente
contrato mínimo: identificador de transacción, fecha-hora, cliente, terminal,
monto y etiqueta binaria. El primer benchmark reproducible sigue la especificación
pública del Fraud Detection Handbook de ULB y genera localmente escenarios
temporales sin ejecutar archivos pickle remotos.

El silver actual contiene 87 159 transacciones, 2 612 fraudes y particiones
cronológicas 60/20/20. Train y validation alimentan el protocolo de secuencias y
OOF; test permanece bloqueado. Al ser un benchmark
sintético, está habilitado para validar la arquitectura, pero no para sostener por
sí solo la conclusión final de tesis. La puerta final exigirá validación externa
con transacciones reales etiquetadas, licenciadas y versionadas.

El protocolo financiero secuencial ya está implementado: diez variables causales,
ventanas de diez transacciones por cliente y cinco folds OOF temporales expansivos
con un día de purga. Cada fold ajusta su escalador exclusivamente sobre su prefijo
de train. El artefacto contiene 69 756 filas de desarrollo (52 364 de train y
17 392 de validación externa) y la cobertura OOF es de 43 620 transacciones.
Las 17 403 filas de test no están codificadas, escaladas ni evaluadas.

Las arquitecturas LSTM, GRU, BiRNN, TCN y Transformer financieras ya están
implementadas sobre este tensor. El ejecutor genera probabilidades OOF por
transacción, persiste checkpoints atómicos por modelo-fold-semilla y ordena la
comparación por PR-AUC. El primer piloto técnico usó una semilla, una época y 100
holdouts estratificados por fold (500 predicciones por modelo). GRU encabezó ese
piloto con PR-AUC 0,1583, seguido por TCN 0,0926, BiRNN 0,0922, Transformer 0,0652
y LSTM 0,0603. Estos valores verifican la integración, pero no son resultados
finales: proceden de un subconjunto y de un benchmark sintético.

La reanudación real recuperó las 25 unidades verificadas y no reentrenó ninguna.
Con estas probabilidades base disponibles, el siguiente bloque es el Stacking
financiero con cross-fitting temporal y comparación como sexto competidor.

## Puerta de fuente financiera real implementada

La fuente primaria propuesta es IEEE-CIS Fraud Detection. Su adquisicion es
manual y autenticada porque el proveedor la distribuye bajo reglas de
competicion. El repositorio genera un manifiesto curado, pero no descarga ni
redistribuye los datos. Para habilitar la puerta se exige version, fecha,
investigador responsable, aceptacion explicita, archivo local y SHA-256 exacto.

El adaptador usa `TransactionID` como evento, `TransactionDT` como tiempo
relativo, `TransactionAmt` como monto, `isFraud` como etiqueta y un compuesto
anonimizado de `card1/card2/card3/card5/card6` como proxy estable de entidad. La
etiqueta no participa en ese proxy. El archivo test sin etiquetas del proveedor
se excluye; `train_transaction.csv` se divide por tiempos unicos en 60/20/20 y
el ultimo bloque se bloquea como test interno.

La puerta exige ambas clases en cada particion, orden temporal estricto, ausencia
de IDs duplicados, montos y tiempos validos, al menos 100 000 transacciones, 100
fraudes por split y 1 000 entidades. Silver, manifiesto y fuente quedan enlazados
por SHA-256. Cualquier alteracion posterior cierra `readyForThesisTraining`.

ULB/Worldline se mantiene como fuente real para validacion tabular externa. No
abre la puerta secuencial porque por confidencialidad no publica identificadores
de cliente o tarjeta. Esta limitacion se reporta explicitamente y no se resuelve
inventando entidades.

## Validacion temporal y calibracion financiera implementadas

Cada modelo base se reentrena sin consultar `validation` ni test, usando el
numero de epocas congelado desde OOF. El escalador global se ajusta solo sobre
las filas de `train`. Modelos y probabilidades de `validation` se guardan en
checkpoints verificables para reanudar una ejecucion interrumpida.

El meta-learner seleccionado durante OOF se reajusta sobre probabilidades OOF,
sin etiquetas de `validation`. Luego `validation` se divide cronologicamente en
calibracion y seleccion. Un subcorte temporal `fit/check` del primer bloque elige
por Brier score y log-loss entre identidad, Platt e isotonica; alli mismo se fija
el umbral F2. Las etiquetas del segundo bloque no intervienen en esas decisiones.

LSTM, GRU, BiRNN, TCN, Transformer y Stacking se comparan sobre exactamente las
mismas transacciones del segundo bloque. Se reportan PR-AUC, ROC-AUC, precision,
recall, F1, MCC, balanced accuracy, Brier, log-loss y FPR. La diferencia de
PR-AUC entre Stacking y el mejor modelo base incluye bootstrap pareado y
estratificado con intervalo del 95%.

El ganador, calibrador, umbral, linaje y hashes quedan congelados. Sin embargo,
`readyForFinalTestEvaluation` permanece en `false`: las 17 403 transacciones de
test siguen sin codificarse ni evaluarse y el benchmark sintetico no puede
sostener por si solo una conclusion final de tesis.

El piloto temporal real de esta etapa uso una semilla, una epoca, 8 000 filas de
train para cada refit y las 17 392 filas completas de `validation`: 8 696 para
calibracion y 8 696 para seleccion. TCN quedo primero con PR-AUC 0,1980, seguido
por GRU 0,1799, LSTM 0,1442, Transformer 0,0598, BiRNN 0,0580 y Stacking 0,0314.
El ganador congelado fue TCN con calibracion isotonica y umbral F2 0,07143.

Stacking no supero a TCN: delta PR-AUC -0,1666, IC95% bootstrap pareado
[-0,2118; -0,1253] y probabilidad bootstrap de mejora 0%. Este resultado
negativo se conserva sin modificar datos, ranking o umbrales. Los cuatro
artefactos principales superaron la comprobacion SHA-256 y test continuo
bloqueado, no codificado y no evaluado.

## Diversidad y ablacion financiera implementadas

La justificacion empirica del ensamble ya no depende solo de comparar PR-AUC.
Sobre las 8 696 transacciones de seleccion se calculan correlaciones Pearson y
Spearman, correlacion de residuos absolutos, desacuerdo de decisiones, doble
fallo, Jaccard de falsos positivos y negativos, y aciertos exclusivos de cada
modelo. Las probabilidades y umbrales son los calibrados previamente, sin volver
a elegirlos sobre seleccion.

Se evaluan seis configuraciones (completa y cinco leave-one-model-out) para los
dos meta-learners. Cada uno se reajusta solo con probabilidades OOF; despues se
calibra y fija su umbral F2 usando exclusivamente el primer bloque temporal de
`validation`. El segundo bloque se usa para metricas y un bootstrap pareado por
bloques de dia. La regla para retirar una arquitectura exige simultaneamente:
ganancia PR-AUC de al menos 0,001, perdida MCC no mayor que 0,005 e intervalo del
95% de `completo - variante` completamente menor que cero.

El piloto produjo un desacuerdo medio de 0,08740. LSTM y Transformer fueron la
pareja mas complementaria con desacuerdo 0,15536. Para Gradient Boosting, retirar
BiRNN aumento PR-AUC de 0,03138 a 0,03990; el IC95% de la contribucion de BiRNN
fue [-0,01898; -0,00291], por lo que se acepta su retirada en ese meta-learner.
La mejor variante global fue Stacking logistico sin BiRNN (PR-AUC 0,16427), pero
no supero al TCN congelado (0,19802). Este resultado respalda optimizar el
ensamble sin convertirlo indebidamente en ganador.

Los artefactos contienen las 10 parejas, 12 juegos de metricas, probabilidades
crudas y calibradas, 12 modelos, 12 calibradores y hashes SHA-256. La cobertura
es identica entre variantes; test continua bloqueado, no codificado y no usado.

## Puerta de congelamiento financiero implementada

Antes de cualquier evaluacion final se ejecuta una auditoria de 36 controles.
Esta une los manifiestos de secuencias, modelos base, Stacking, validacion y
ablacion; exige fuente real, cinco semillas y folds, cobertura completa,
calibracion cronologica, estabilidad por bloques de dia y los seis candidatos
congelados. Tambien detecta una decision de ablacion que todavia no haya sido
revalidada como sexto competidor, evitando reportar sobre las mismas etiquetas
usadas para escogerla.

Solo una auditoria completa puede crear el paquete inmutable. En ese momento se
verifican hashes y tamanos de todo el inventario, se fija candidato, calibradores,
umbrales, semillas, folds y configuracion del Stacking, y se genera un sello
SHA-256 reutilizable pero no sobrescribible. La autorizacion de test permanece
separada, inicialmente denegada, con maximo de una evaluacion sin reseleccion.

El piloto actual queda bloqueado de forma esperada: supera 23 de 36 controles.
No se interpreta como error del software, sino como evidencia auditable de los
requisitos pendientes: dataset IEEE-CIS, protocolo thesis de cinco semillas,
OOF completo, validacion real y revalidacion de la ablacion.

## Stacking financiero temporal implementado

El metamodelo financiero ya usa las cinco probabilidades OOF y siete variables
de desacuerdo: media, desviación estándar, rango, mínimo, máximo, fracción de
votos y desacuerdo absoluto medio. Se implementaron regresión logística y
Gradient Boosting como meta-learners, con promedio simple, voting y promedio
ponderado como baselines.

El cross-fitting es expansivo: el fold 1 funciona como warm-up y los folds 2–5
solo pueden ajustar el metamodelo con folds anteriores. La cobertura común del
piloto fue de 400 transacciones y 100 quedaron como warm-up. Validation y test no
participaron. Sobre esos folds comunes, GRU obtuvo PR-AUC medio 0,1901 y el mejor
meta-learner, Gradient Boosting, 0,0470. Por tanto, el Stacking aparece como sexto
competidor pero no se declara ganador. Esta diferencia se conservará como
resultado negativo del piloto y se reevaluará con mayor cobertura, cinco semillas
y validación externa.
