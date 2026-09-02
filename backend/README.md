# Backend local para tesis LSTM/GRU/BRNN/Transformer/TCN

Este backend expone los endpoints que consume el frontend:

- `GET /api/dashboard`
- `GET /api/analysis`
- `GET /api/comparison`
- `GET /api/history`
- `GET /api/xai`
- `GET /api/external-sources`
- `GET /api/external-data?domain=phishing&limit=100`
- `POST /api/data-lake/ingest?domain=all&target=5000`
- `GET /api/data-lake/summary`
- `GET /api/data-lake/records?domain=phishing&page=1&pageSize=100`
- `GET /api/ai-analysis?type=general`
- `GET /api/energy/experiments/latest`
- `GET /api/energy/experiments`
- `POST /api/energy/experiments/run`
- `GET /api/phishing/data`
- `POST /api/phishing/data/prepare`
- `GET /api/phishing/sequences`
- `POST /api/phishing/sequences/prepare`
- `GET /api/phishing/experiments/latest`
- `POST /api/phishing/experiments/run`
- `GET /api/phishing/stacking/latest`
- `POST /api/phishing/stacking/run`
- `GET /api/phishing/validation/latest`
- `POST /api/phishing/validation/run`
- `GET /api/phishing/validation/jobs/latest`
- `GET /api/phishing/diversity/latest`
- `POST /api/phishing/diversity/run`
- `GET /api/phishing/freeze/readiness`
- `GET /api/phishing/freeze/latest`
- `POST /api/phishing/freeze/create`

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

## Pipeline energetico experimental v1

### Preparar y auditar OPSD real

La ingesta energética conserva un snapshot versionado por SHA-256, selecciona
el tramo horario continuo más largo con objetivo observado y deja la imputación
de predictores para cada fold de entrenamiento:

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.energy.ingestion
```

Los resultados quedan en:

- `storage/raw/energy/opsd_<version>_<sha>.csv`: fuente original inmutable.
- `storage/raw/energy/latest.json`: procedencia y hash del snapshot.
- `storage/silver/opsd.csv`: serie preparada para entrenamiento.
- `storage/silver/opsd.metadata.json`: versión, hashes y política de selección.
- `storage/results/energy_data_audit.json`: continuidad, faltantes y aptitud.

También puede iniciarse desde la pantalla `/energy` o mediante
`POST /api/energy/data/prepare`; el progreso se consulta en
`GET /api/energy/data/jobs/latest`.

El pipeline nuevo mantiene el conjunto de prueba bloqueado por defecto y compara
un baseline de persistencia con LSTM, GRU, BRNN, TCN y Transformer reales de
TensorFlow/Keras:

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.energy.pipeline `
  --dataset ruta\serie_energia.csv `
  --timestamp timestamp `
  --target DE_load_actual_entsoe_transparency
```

Para comprobar la integracion con los datos pequenos de muestra, sin producir
resultados cientificos:

```powershell
.\.venv\Scripts\python.exe -m app.energy.pipeline --demo --epochs 2
```

Para ejecutar solamente un subconjunto durante desarrollo:

```powershell
.\.venv\Scripts\python.exe -m app.energy.pipeline --demo --epochs 2 --models gru,tcn
```

La evaluacion del test final requiere la opcion explicita `--evaluate-test`. No
debe usarse durante seleccion de hiperparametros. TensorFlow usa CPU en Windows
nativo; para entrenamiento con GPU se recomienda Google Colab o WSL2.

### Walk-forward, predicciones OOF y stacking

El pipeline OOF mantiene el test final fuera del entrenamiento y genera
predicciones por fold para media, promedio ponderado y stacking:

```powershell
.\.venv\Scripts\python.exe -m app.energy.oof_pipeline `
  --dataset ruta\serie_energia.csv `
  --timestamp timestamp `
  --target DE_load_actual_entsoe_transparency `
  --folds 5 `
  --seeds 42,101,202,303,404
```

Smoke test reducido, marcado siempre como demo:

```powershell
.\.venv\Scripts\python.exe -m app.energy.oof_pipeline `
  --demo --window 12 --gap 6 --folds 3 --epochs 1 `
  --seeds 42 --models lstm,gru
```

Los artefactos principales son `oof_predictions.csv`,
`ensemble_oof_predictions.csv`, `anomaly_predictions.csv`,
`anomaly_report.json` y `oof_manifest.json`. Cuando el dataset no contiene
etiquetas independientes de anomalía, el reporte marca las detecciones como
estimadas y omite intencionalmente precision, recall y F1.

### API y dashboard energético

La pantalla `/energy` consulta la corrida OOF más reciente y permite configurar
una nueva ejecución. El entrenamiento se encola en segundo plano para no
bloquear la solicitud HTTP:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/energy/experiments/run `
  -ContentType "application/json" `
  -Body '{"protocol":"demo","source":"sample","folds":3,"epochs":1,"seeds":[42]}'
```

La respuesta contiene un `jobId`. El progreso y la cancelación segura se
gestionan con:

```powershell
Invoke-RestMethod http://localhost:8000/api/energy/jobs/<jobId>
Invoke-RestMethod -Method Post http://localhost:8000/api/energy/jobs/<jobId>/cancel
```

La cancelación de un entrenamiento activo se hace efectiva después de terminar
la combinación modelo-fold-semilla actual, evitando artefactos parcialmente
escritos.

Para consultar resultados sin volver a entrenar:

```powershell
Invoke-RestMethod http://localhost:8000/api/energy/experiments/latest
Invoke-RestMethod http://localhost:8000/api/energy/experiments
```

Una solicitud con `protocol: "thesis"` exige `source: "silver"` y aplica la
validación estricta del dataset. La muestra sintética nunca puede marcarse como
resultado candidato de tesis.

Pruebas del pipeline:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Curación científica del dataset de phishing

PhishTank + Tranco se conserva como dataset de implementación y piloto. Tranco
no se considera ground truth benigno: la ausencia en una blacklist o el ranking
de popularidad no demuestran que una URL sea legítima.

El backend admite un paquete curado local en:

```text
storage/raw/phishing/curated/curation_manifest.json
```

Puede generarse una plantilla sin activar datos ficticios:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/phishing/data/curation/template `
  -ContentType "application/json" `
  -Body '{}'
Invoke-RestMethod http://localhost:8000/api/phishing/data/curation
```

El botón `Construir base académica` —o la siguiente solicitud— descarga y fija
automáticamente dos fuentes mixtas CC BY 4.0:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/phishing/data/prepare `
  -ContentType "application/json" `
  -Body '{"force":false,"perClass":10000,"includeAcademicSources":true}'
```

- PhiUSIIL v2: DOI `10.17632/shwpxscxy2.2`, SHA-256 publicado
  `a236549c...d06d1c6`, etiqueta original `1=legítima, 0=phishing`.
- LegitPhish v2: DOI `10.17632/hx4m73v2sf.2`, SHA-256 publicado
  `8685e790...f09dbcd`, etiqueta original `1=legítima, 0=phishing`.

Los adaptadores invierten explícitamente esas etiquetas al contrato interno
`0=benigna, 1=phishing`, conservan DOI, versión, licencia y snapshot bruto. Una
fila de LegitPhish sin etiqueta se descarta y queda contabilizada; ninguna
etiqueta se imputa. Por fuente y clase se eligen 20 000 URLs mediante orden
SHA-256 determinista, produciendo un paquete curado de 80 000 filas.

Los datasets publicados pueden compartir feeds aguas arriba. Por eso el sistema
no los trata como verdad perpetua: elimina duplicados y conflictos por dominio,
retiene la identidad de fuente y documenta la evidencia como
`published_dataset_ground_truth`. ISCX-URL2016 queda como fuente externa opcional
porque su portal requiere completar un formulario personal.

Cada fuente declara `sourceId`, proveedor, cita, licencia, independencia de
adquisición, etiquetas disponibles, ruta local y SHA-256. Cada CSV usa el
contrato:

```text
url,is_phishing,source_record_id,label_verified,verification_method,verification_reference,verified_at,label_provenance
```

La puerta de tesis exige dos fuentes independientes por clase, al menos una
fuente con ambas clases, evidencia completa para todas las etiquetas seleccionadas
y una diferencia aceptable en la forma de las URLs. Los métodos
`absence_only`, `not_listed`, `assumed_benign`, `tranco_rank` y equivalentes no
se aceptan como verificación negativa. El hash se comprueba antes de leer cada
CSV y las rutas deben permanecer dentro del directorio del paquete.

Cuando existen suficientes negativos verificados, la selección excluye los
negativos presumidos de Tranco y balancea determinísticamente las fuentes. Si no
existe evidencia suficiente, el pipeline puede seguir funcionando como piloto,
pero `readyForThesisTraining` permanece en `false` y la puerta de congelación lo
rechaza.

## Pipeline de phishing y Stacking

La pantalla `/phishing` prepara un conjunto balanceado PhishTank + Tranco,
conserva snapshots con SHA-256 y separa train, validation y test por dominio
registrado. El protocolo de caracteres construye cinco folds OOF sin compartir
dominios y mantiene el test final bloqueado.

El piloto de modelos base se inicia desde el dashboard o mediante
`POST /api/phishing/experiments/run`. Cuando existen probabilidades OOF completas,
el segundo nivel se ejecuta con:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/phishing/stacking/run `
  -ContentType "application/json" `
  -Body '{}'
```

Se comparan promedio simple, Voting, promedio ponderado, Stacking logístico,
Stacking Ridge y Stacking con Gradient Boosting. Los pesos, modelos, métricas y
predicciones se guardan bajo `storage/experiments/<run>/phishing_oof_v1/stacking_v1`
con hashes verificables.

La tabla producida en esta fase es un diagnóstico OOF exploratorio: el ajuste
directo y el holdout del metamodelo no comparten filas, pero todavía no constituye
CV anidado estricto. El ganador y el umbral solo pueden congelarse después de una
comparación común sobre la validación externa. El test no se utiliza en esta fase.

La validación externa se ejecuta en segundo plano:

```powershell
$job = Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/phishing/validation/run `
  -ContentType "application/json" `
  -Body '{}'
Invoke-RestMethod "http://localhost:8000/api/phishing/validation/jobs/$($job.jobId)"
Invoke-RestMethod http://localhost:8000/api/phishing/validation/latest
```

El reentrenamiento utiliza todas las filas de desarrollo de la corrida OOF y el
número mediano de épocas observado en sus folds. Las redes producen probabilidades
sobre validation; los pesos y metamodelos se ajustan solo con OOF. La selección usa
PR-AUC como métrica primaria y calibra el umbral por MCC. Predicciones, matrices de
confusión, modelos y hashes se guardan en
`storage/experiments/<run>/phishing_oof_v1/external_validation_v1`.

Esta etapa selecciona configuración y umbral; no reemplaza la evaluación final.
El conjunto test permanece sin codificar ni evaluar.

### Diversidad y ablación

El estudio posterior reutiliza únicamente OOF y validation; no vuelve a entrenar
las redes ni abre test:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/phishing/diversity/run `
  -ContentType "application/json" `
  -Body '{}'
Invoke-RestMethod http://localhost:8000/api/phishing/diversity/latest
```

Calcula correlaciones de probabilidades y residuos, desacuerdo duro, doble fallo,
solape de falsos positivos y falsos negativos. Para cada meta-learner compara la
configuración completa con cinco variantes leave-one-model-out. La contribución
de cada arquitectura usa diferencias pareadas de PR-AUC con bootstrap agrupado
por dominio registrado. Los 18 metamodelos, 54 000 probabilidades y el informe se
guardan en `storage/experiments/<run>/phishing_oof_v1/diversity_ablation_v1`.

### Puerta de congelación experimental

La puerta se consulta sin modificar artefactos:

```powershell
Invoke-RestMethod http://localhost:8000/api/phishing/freeze/readiness
```

Solo una corrida candidata de tesis puede solicitar el paquete inmutable:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/phishing/freeze/create `
  -ContentType "application/json" `
  -Body '{}'
```

Se exigen cinco semillas, cinco folds, cinco modelos, desarrollo completo, dataset
apto para tesis, validación y ablación completas, estabilidad y hashes válidos.
El paquete contiene `frozen_configuration.json`, `artifact_inventory.json`,
`frozen_pipeline_manifest.json` y `freeze_seal.json`. Si ya existe un paquete con
otro fingerprint, la API rechaza la sobrescritura. Congelar no autoriza ni ejecuta
test; esa acción requiere un paso separado y conserva un máximo de una evaluación.

### Ejecución OOF reanudable de phishing

Antes de una corrida larga, consulte recursos y linaje sin tocar el conjunto de
prueba:

```powershell
Invoke-RestMethod "http://localhost:8000/api/phishing/runtime/preflight?protocol=thesis"
```

Cada combinación semilla-fold-modelo guarda un checkpoint atómico con el modelo,
sus probabilidades OOF, métricas y hashes SHA-256. Si el backend se reinicia o se
cancela una corrida, puede continuarla sin repetir unidades válidas:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/api/phishing/jobs/<jobId>/resume
```

Una unidad solo se reutiliza cuando coinciden el dataset, asignaciones OOF,
configuración, tokenizador, muestras, etiquetas y hashes. Un checkpoint inválido
se vuelve a entrenar. El test permanece bloqueado durante todo este proceso.

Para ejecutar el protocolo completo en Colab o WSL2 con GPU desde `backend/`, use
una carpeta persistente y conserve el mismo identificador al reanudar:

```bash
python -m app.phishing.cli --preflight-only --protocol thesis
python -m app.phishing.cli \
  --protocol thesis \
  --execution-run-id phishing_thesis_v1 \
  --output-dir /content/drive/MyDrive/tesis/phishing_thesis_v1 \
  --epochs 30 --batch-size 128 --patience 5
```

El protocolo de tesis fija cinco modelos, cinco folds y las semillas
`42, 52, 62, 72, 82`: 125 entrenamientos sobre desarrollo. Repetir exactamente
el comando recupera los checkpoints verificados y continúa desde la siguiente
unidad pendiente.

## Finanzas de tesis: Indices Soberanos del MEF

La fuente cientifica principal es el dataset publico **Indices Soberanos** de la
categoria **Endeudamiento y Tesoro Publico** del MEF. El objetivo continuo es el
log-rendimiento porcentual del indice nominal y el horizonte es la siguiente
sesion observada. Las anomalias son estimadas desde residuos OOF; no se interpretan
como fraude confirmado porque la fuente no incluye etiquetas de fraude.

```powershell
$env:PYTHONPATH = "backend"
backend/.venv/Scripts/python.exe -m app.finance.mef_market_data

Invoke-RestMethod http://localhost:8000/api/finance/market-data
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/finance/market-data/prepare `
  -ContentType "application/json" `
  -Body '{"force":false}'

Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/finance/thesis/run `
  -ContentType "application/json" `
  -Body '{"window":20,"horizon":1,"gapSteps":5,"folds":5,"epochs":20,"batchSize":32,"seeds":[42,101,202,303,404],"bootstrapIterations":500}'
```

El pipeline reserva cronologicamente el ultimo 15 % como test bloqueado. Sobre el
85 % de desarrollo ejecuta cinco folds expansivos, cinco semillas y LSTM, GRU,
BiRNN, TCN y Transformer. Stacking compite como sexto candidato por RMSE e incluye
ablacion, bootstrap pareado, XAI, inventario de artefactos y sello pre-test.

La corrida registrada `finance_mef_thesis_20260902_132403_56cfc33b` completo las
125 unidades. En el fold independiente, GRU obtuvo RMSE 0,57109 y Stacking 0,81246;
el IC 95 % del delta `Stacking - GRU` fue [+0,16739; +0,35190]. El paquete
`finance_mef_freeze_20260902_133933` conserva este resultado sin usar test.

## Benchmark financiero transaccional historico

Este benchmark sintetico de clasificación de fraude se conserva para regresion
de software y demostraciones; ya no representa el contrato financiero principal
de la tesis.

```powershell
Invoke-RestMethod http://localhost:8000/api/finance/data
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/finance/data/prepare `
  -ContentType "application/json" `
  -Body '{"days":100,"customers":500,"terminals":200,"seed":42}'

Invoke-RestMethod http://localhost:8000/api/finance/sequences
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/finance/sequences/prepare `
  -ContentType "application/json" `
  -Body '{"window":10,"folds":5,"purgeDays":1}'
```

El benchmark implementa el contrato `transaction_id`, `transaction_time`,
`customer_id`, `terminal_id`, `amount`, `is_fraud`, `fraud_scenario` y `split`.
La división es cronológica 60/20/20 y test permanece bloqueado. El dataset
generado sirve para desarrollar secuencias, OOF y Stacking, pero no se promueve a
resultado final de tesis por ser sintético. La conclusión final requerirá una
validación externa con transacciones reales etiquetadas y licencia verificable.

La metodología se atribuye al Fraud Detection Handbook de ULB. El generador es
local y determinista; no descarga ni ejecuta los archivos pickle publicados por
terceros.

El protocolo secuencial construye diez variables causales y ventanas de diez
transacciones por cliente, con relleno izquierdo y una máscara explícita. Los
cinco folds OOF son temporales y expansivos; cada escalador se ajusta solo con el
prefijo de entrenamiento de su fold y se aplica después al holdout. Las etiquetas
no forman parte de las variables. Validation se conserva como validación externa
del desarrollo y las 17 403 filas de test no se codifican, escalan ni evalúan.

Los cinco modelos base financieros se ejecutan como trabajo reanudable:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/finance/experiments/run `
  -ContentType "application/json" `
  -Body '{"protocol":"demo","epochs":1,"batchSize":128,"patience":1,"seeds":[42],"modelIds":["lstm","gru","brnn","tcn","transformer"],"demoMaxRowsPerFold":1000}'

Invoke-RestMethod http://localhost:8000/api/finance/jobs/latest
Invoke-RestMethod http://localhost:8000/api/finance/experiments/latest
```

Cada unidad modelo-fold-semilla guarda el modelo, sus probabilidades y métricas
con SHA-256. Una reanudación reutiliza únicamente checkpoints cuyo dataset,
configuración, escalador, transacciones, etiquetas y artefactos coinciden. PR-AUC
es la métrica primaria; también se reportan ROC-AUC, precision, recall, F1, MCC,
balanced accuracy, Brier score, log-loss, FPR, tiempo y tamaño del modelo.

El Stacking financiero usa cross-fitting temporal expansivo. El primer fold OOF
es warm-up; para cada fold posterior, los meta-modelos solo reciben predicciones
de folds anteriores. Se comparan promedio simple, voting, promedio ponderado,
regresión logística y Gradient Boosting:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/api/finance/stacking/run
Invoke-RestMethod http://localhost:8000/api/finance/stacking/latest
```

Las meta-features contienen las cinco probabilidades base y siete medidas de
desacuerdo. La etiqueta, validation y test quedan fuera de las variables. La
tabla oficial agrega el mejor candidato provisional de Stacking como sexto
competidor sobre los mismos folds comunes, sin imponer que ocupe el primer lugar.

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

## Fuente financiera real curada (adaptador historico opcional)

El backend incluye una puerta separada para IEEE-CIS y ULB/Worldline. IEEE-CIS
es el candidato principal porque conserva tiempo relativo, etiqueta binaria y
un proxy anonimizado de tarjeta compatible con secuencias. La descarga es
manual: Kaggle exige aceptar sus reglas y el proyecto no redistribuye el archivo.

```powershell
Invoke-RestMethod http://localhost:8000/api/finance/real-data/catalog
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/finance/real-data/template `
  -ContentType "application/json" `
  -Body '{"adapter":"ieee_cis","force":false}'
Invoke-RestMethod http://localhost:8000/api/finance/real-data
```

La plantilla se crea en
`storage/raw/finance/curated/real_finance_manifest.json`. Despues de aceptar las
reglas del proveedor, coloque `train_transaction.csv` en esa carpeta y complete
version, fecha de recuperacion, investigador, aceptacion y SHA-256. El archivo
test de Kaggle no se usa.

El hash exacto puede calcularse localmente, sin cargar el dataset a ningun
servicio externo:

```powershell
Get-FileHash `
  -LiteralPath backend/storage/raw/finance/curated/train_transaction.csv `
  -Algorithm SHA256
```

Copie el valor de `Hash` en `files[0].sha256`, registre el tamano en bytes y no
agregue el CSV al repositorio. `backend/storage/` esta excluido de Git.

La activacion aplica controles minimos de 100 000 filas, 100 fraudes por split y
1 000 entidades anonimizadas:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/finance/real-data/prepare `
  -ContentType "application/json" `
  -Body '{"minimumRows":100000,"minimumFraudPerSplit":100,"minimumEntities":1000}'
```

El adaptador lee solo las columnas necesarias, crea proxies estables de entidad
y contraparte sin usar la etiqueta, conserva el tiempo relativo y genera splits
60/20/20 por valores temporales unicos. Si cambia el manifiesto o el CSV despues
de activarlo, el linaje SHA-256 deja de ser vigente y se bloquean secuencias y
entrenamientos de tesis. ULB/Worldline se cataloga como validacion tabular porque
no publica identificadores compatibles con secuencias por cliente.

## Validacion temporal y calibracion financiera

La etapa posterior al Stacking reentrena las cinco arquitecturas usando solo
`train`, ajusta el meta-learner exclusivamente con probabilidades OOF y divide
`validation` en dos bloques cronologicos. El primer bloque elige entre identidad,
Platt e isotonica y fija el umbral F2; el segundo compara los seis candidatos
sobre exactamente las mismas transacciones.

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/finance/validation/run `
  -ContentType "application/json" `
  -Body '{"demoMaxTrainRows":8000,"demoMaxValidationRows":null,"bootstrapIterations":500}'

Invoke-RestMethod http://localhost:8000/api/finance/validation/jobs/latest
Invoke-RestMethod http://localhost:8000/api/finance/validation/latest
```

El resultado incluye calibracion, umbrales, matrices de confusion y bootstrap
pareado de PR-AUC entre Stacking y el mejor modelo base. Los refits son
reanudables mediante checkpoints SHA-256. La configuracion ganadora queda
congelada, pero test continua sin codificarse ni evaluarse. El benchmark
sintetico no habilita una afirmacion final de tesis: aun se requiere una fuente
financiera real, etiquetada y versionada.

## Diversidad y ablacion financiera

La etapa siguiente cuantifica la complementariedad de las cinco redes sobre la
mitad de seleccion de `validation` y ejecuta una ablacion leave-one-model-out.
Para cada arquitectura retirada se vuelven a ajustar regresion logistica y
Gradient Boosting exclusivamente sobre OOF; cada variante aprende calibrador y
umbral F2 en la primera mitad cronologica y se evalua en la segunda. La
contribucion usa bootstrap pareado por bloques de dia y test permanece bloqueado.

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/finance/diversity/run `
  -ContentType "application/json" `
  -Body '{"bootstrapIterations":300}'

Invoke-RestMethod http://localhost:8000/api/finance/diversity/latest
```

En el piloto actual, el desacuerdo medio por pareja fue 8,74% y la pareja mas
complementaria fue LSTM--Transformer (15,54%). En el meta-learner de referencia
Gradient Boosting, retirar BiRNN elevo PR-AUC de 0,03138 a 0,03990; el intervalo
del 95% para la caida `completo - sin BiRNN` fue [-0,01898; -0,00291], por lo que
la regla conservadora acepta esa ablacion solo como decision de `validation`.

La mejor configuracion explorada fue Stacking logistico sin BiRNN con PR-AUC
0,16427, aun inferior al TCN previamente congelado (0,19802). Por tanto, la
ablacion mejora el Stacking pero no altera artificialmente el ranking general ni
permite declararlo ganador. La pagina Finanzas muestra las diez parejas, las
doce variantes, sus umbrales, contribuciones e intervalos.

## Puerta de congelamiento financiero

El pipeline financiero incluye una puerta final previa al uso de test. La
auditoria exige datos reales curados, secuencias aptas para tesis, cinco modelos,
cinco semillas, cinco folds, cobertura OOF completa, cross-fitting temporal,
refits completos, seis candidatos calibrados, ablacion estable, hashes validos y
test no codificado ni usado. Si una ablacion fue aceptada, esa configuracion debe
volver a competir como sexto candidato antes de poder sellarse.

```powershell
Invoke-RestMethod http://localhost:8000/api/finance/freeze/readiness
Invoke-RestMethod http://localhost:8000/api/finance/freeze/latest
Invoke-RestMethod -Method Post http://localhost:8000/api/finance/freeze/create
```

La creacion vuelve a verificar todos los SHA-256, persiste configuracion e
inventario, genera un sello inmutable e impide sobrescribir un paquete distinto.
El sello no ejecuta el test: deja una autorizacion separada en `false` y limita
la futura evaluacion final a una unica ejecucion sin reseleccion.

Sobre el piloto sintetico actual la puerta aprueba 23 de 36 controles y bloquea
13 correctamente. Los bloqueos incluyen IEEE-CIS pendiente, una sola semilla,
OOF reducido, protocolo demo y la necesidad de revalidar la ablacion aceptada.

## Data lake academico

Para trabajar con miles de registros sin depender de consultas en vivo:

```bash
curl -X POST "http://localhost:8000/api/data-lake/ingest?domain=all&target=5000"
curl "http://localhost:8000/api/data-lake/summary"
curl "http://localhost:8000/api/data-lake/records?domain=phishing&page=1&pageSize=100"
```

Los lotes se guardan en:

- `backend/storage/raw/external_<dominio>.json`
- `backend/storage/silver/external_<dominio>.json`
- `backend/storage/gold/external_<dominio>.json`
