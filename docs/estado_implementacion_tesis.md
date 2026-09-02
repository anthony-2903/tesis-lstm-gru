# Estado de implementación y ruta de cierre de la tesis

Fecha de corte: 2 de septiembre de 2026.

## Estado operativo actual

Phishing y energia completaron sus protocolos cientificos pre-test, incluidas
las 125 unidades (5 modelos x 5 folds x 5 semillas), Stacking, revalidacion,
diversidad/ablacion, XAI, auditoria de linaje y congelacion. Para Finanzas se
selecciono la categoria publica **Endeudamiento y Tesoro Publico** y el dataset
**Indices Soberanos** del MEF. La ingesta, matriz 5x5, Stacking, revalidacion,
XAI, auditoria y congelacion financiera ya estan completas. El test final sigue
bloqueado, no usado y no autorizado.

El avance experimental agregado pre-test es 100 %: los tres dominios estan
completos y sellados. Este porcentaje no incluye la evaluacion final del test,
que requiere una autorizacion explicita y separada, ni la redaccion de tesis.

Los tres dominios disponen de pausa cooperativa separada de la cancelacion. La
orden queda persistida para que un proceso de entrenamiento distinto de la API
la observe, finalice la unidad atomica en curso y cambie a `paused`. Reanudar no
abre el test ni reutiliza checkpoints que fallen su contrato o hash.

## Maquina de estados operativa

| Estado | Significado | Transicion permitida |
|---|---|---|
| `queued` | Trabajo persistido, aun sin ejecutar | `running` |
| `running` | Protocolo activo con test bloqueado | `paused`, `cancelled`, `failed`, espera de datos o `completed` |
| `paused` | Unidad atomica cerrada y checkpoints conservados | nuevo trabajo `queued` con el mismo `executionRunId` |
| `cancelled` / `failed` / `interrupted` | Ejecucion detenida recuperable | nuevo trabajo `queued` desde checkpoints validos |
| espera de datos | No se entrena hasta superar la puerta cientifica o de licencia | nuevo trabajo `queued` al resolver el requisito |
| `completed` | Artefactos pre-test sellados | evaluacion final separada y solo con autorizacion explicita |

Pausa y cancelacion son mutuamente excluyentes: prevalece la primera orden
persistida. La reanudacion crea un nuevo identificador de trabajo, conserva el
`executionRunId` cientifico y vuelve a verificar contrato y hash antes de omitir
una unidad ya calculada.

El supervisor es idempotente y puede reiniciarse sin indicar un trabajo:

```powershell
$env:PYTHONPATH = "backend"
backend/.venv/Scripts/python.exe backend/scripts/run_thesis_chain.py
```

Opcionalmente, `--resume-phishing <jobId>` fija un trabajo preferido. Si existe
uno posterior que sigue activo o ya termino, el supervisor usa el posterior para
no duplicar computo. Un fallo del supervisor queda persistido con PID nulo y no
se presenta como una ejecucion activa.

## Objetivo experimental

El proyecto compara cinco arquitecturas profundas —LSTM, GRU, BiRNN, TCN y
Transformer— y un Stacking como sexto candidato en tres dominios. La hipótesis
no se implementa como una obligación de que Stacking gane: se contrasta sobre
datos no utilizados para seleccionar el metamodelo. Si no supera al mejor modelo
base, ese resultado se conserva y se explica mediante diversidad, ablación y XAI.

## Arquitectura común

```text
fuente versionada y auditada
  -> silver por dominio
  -> test final bloqueado
  -> folds sin fuga
  -> preprocesamiento ajustado solo con train
  -> 5 modelos x 5 folds x 5 semillas
  -> predicciones out-of-fold
  -> Stacking y baselines simples
  -> selección/calibración en desarrollo
  -> ablación y diversidad
  -> revalidación independiente
  -> incertidumbre estadística y XAI
  -> inventario, hashes y sello pre-test
  -> evaluación única de test solo con autorización explícita
```

Cada protocolo completo se ejecuta en segundo plano, persiste su estado y puede
reanudar unidades verificadas por hash. Los paneles del dashboard no generan
métricas: consumen exclusivamente artefactos producidos por el backend.

## Contratos por dominio

| Dominio | Tarea | Fuente/estado de datos | Métrica principal | Unidad de remuestreo | Estado |
|---|---|---|---|---|---|
| Energía | Regresión horaria y anomalías por residuo | OPSD real, 50 400 horas continuas | RMSE | Día calendario | Protocolo 5x5 completo y sellado pre-test |
| Phishing | Clasificación binaria de URL | PhishTank, Tranco, PhiUSIIL y LegitPhish; 20 000 URLs | PR-AUC | Dominio registrable | Protocolo 5x5 completo y sellado pre-test |
| Finanzas | Pronostico del retorno soberano y anomalias por residuo | MEF Indices Soberanos, 3 459 sesiones utiles | RMSE | Sesion financiera observada | Protocolo 5x5 completo y sellado pre-test |

## Energía

- Dataset real OPSD con hash SHA-256, versión de fuente, auditoría de continuidad
  horaria y objetivo no imputado.
- Desarrollo walk-forward expansivo con gap temporal; el 15 % final permanece
  fuera del desarrollo.
- Checkpoint atómico por modelo, fold y semilla. Una reanudación valida contrato,
  timestamps, objetivos, predicciones y hash antes de reutilizar una unidad.
- Stacking Gradient Boosting, promedio y promedio ponderado se comparan sobre los
  mismos pares semilla-fold.
- La configuración completa y cinco variantes leave-one-model-out se seleccionan
  sin utilizar el último fold. Ese último fold compara de forma independiente los
  cinco modelos base y el Stacking optimizado.
- Bootstrap pareado por día reporta delta RMSE, IC 95 % y probabilidad de mejora.
- XAI agrega importancias del metamodelo entre semillas.
- Las anomalías actuales son estimadas a partir de residuos. No se reportan F1,
  precision o recall de anomalías porque OPSD no aporta ground truth independiente.
- La puerta de congelación exige la matriz 5x5 completa, linaje OPSD vigente,
  revalidación independiente, XAI, hashes y test intacto.

La corrida cientifica completa usa cinco folds y cinco semillas. En el bloque
independiente, GRU obtuvo el menor RMSE (827,50) y Stacking quedo segundo
(837,44). El resultado esta sellado como
`energy_freeze_20260824_230905`; no se reoptimiza retrospectivamente y aun no
incluye la evaluacion del test final.

## Phishing

- El silver científico contiene 14 000 URLs de train, 2 996 de validation y
  3 004 de test; no existen dominios compartidos entre particiones.
- Los cinco folds OOF son `StratifiedGroupKFold` por dominio registrable.
- Cada tokenizador se ajusta solo sobre el entrenamiento interno.
- Las 125 unidades guardan modelo Keras, probabilidades, métricas y hash. Se
  unificó el contrato de hash para archivos `.keras` y directorios SavedModel.
- El Stacking usa probabilidades OOF, validación externa común, umbrales calibrados,
  diversidad por pares y ablación leave-one-model-out.
- El bootstrap se agrupa por dominio registrable.
- La puerta final verifica fuente científica, cobertura, modelos, tokenizadores,
  calibradores, metamodelos e inventario completo antes de sellar.

La corrida completa finalizo y quedo sellada como
`phishing-freeze-04c26a73c77c517a`. En la validacion externa por dominio, BiRNN
obtuvo el mayor PR-AUC (0,99656) y Stacking quedo tercero (0,99487). Un primer
intento detecto una diferencia entre el hash del contenedor Keras y un hash de
bytes plano; el contrato fue corregido, probado y la misma ejecucion se reanudo
desde checkpoints. El test final no fue consultado.

## Finanzas

- Fuente oficial: MEF, categoria Endeudamiento y Tesoro Publico, dataset Indices
  Soberanos, recurso `c32056c1-7f91-4d4e-bc76-0c4844ac25fe`.
- La serie auditada contiene 3 459 sesiones entre 29-08-2012 y 02-07-2026. El
  snapshot, silver y metadatos se verifican mediante SHA-256.
- El objetivo es `100 * ln(indice_nominal_t / indice_nominal_t-1)` y el modelo
  pronostica la siguiente sesion observada con una ventana causal de 20 sesiones.
- Desarrollo usa el primer 85 % (2 940 filas); las ultimas 519 forman el test
  bloqueado. Cinco folds expansivos usan una purga de cinco sesiones.
- Tres fechas duplicadas fueron documentadas y consolidadas por media; ninguna
  presento conflicto en el indice nominal objetivo. Siete ceros del indice real
  se tratan como faltantes e imputan solo dentro de cada fold de entrenamiento.
- LSTM, GRU, BiRNN, TCN, Transformer y Stacking compiten por RMSE; tambien se
  reportan MAE, sMAPE, R2, estabilidad por semilla y bootstrap pareado.
- Las anomalias se estiman a partir de residuos OOF. No existen etiquetas de
  fraude, por lo que estan prohibidas las afirmaciones de deteccion de fraude y
  las metricas de clasificacion de anomalias.
- IEEE-CIS y el benchmark sintetico se conservan unicamente como implementacion
  historica; no forman parte del contrato cientifico financiero seleccionado.

La corrida completa quedo sellada como
`finance_mef_freeze_20260902_133933`. En el fold financiero independiente, GRU
obtuvo el menor RMSE (0,57109), Transformer quedo segundo (0,57244) y Stacking
sexto (0,81246). El delta pareado `Stacking - GRU` fue +0,24137, con IC 95 %
[+0,16739; +0,35190] y probabilidad bootstrap de mejora de 0 %. Este resultado
se conserva sin reoptimizacion retrospectiva: la tesis evalua la hipotesis de
superioridad del Stacking, no la presupone.

## Criterio para afirmar que Stacking mejora

Stacking se declarará superior en un dominio solo si:

1. gana la métrica primaria en el bloque independiente;
2. el intervalo pareado del 95 % respalda una mejora, no solo una diferencia puntual;
3. mantiene estabilidad entre cinco semillas;
4. no degrada de forma crítica FPR/recall en clasificación;
5. su coste y latencia están documentados;
6. la configuración fue elegida sin consultar ese bloque ni test.

No se cambiarán datos, umbrales, semillas o métricas después de observar el bloque
independiente para forzar el ranking.

## Trabajo restante antes de la entrega

1. Regenerar las tablas y figuras finales de los tres dominios con los manifiestos
   sellados. Los paneles, contratos XAI y generadores de resultados ya estan
   implementados.
2. Solicitar autorizacion explicita y separada antes de la unica evaluacion del
   test de cada dominio congelado.

La auditoria tecnica fue revalidada el 2 de septiembre de 2026 con 108/108 pruebas
backend aprobadas. La compilacion cliente, SSR, lint, auditoria de dependencias y
configuracion de despliegue seguro se verifican en el cierre tecnico de esta misma
revision. `GET /api/thesis/status` ofrece un resumen de disco de la cadena,
sin depender de que el entrenamiento y la API vivan en el mismo proceso. El avance
global que muestra es exclusivamente experimental y pondera por igual los tres
dominios; no mezcla implementación de software ni redacción de tesis.

La vista de experimentos consume `GET /api/thesis/results-summary` y presenta una
tabla homogénea de seis candidatos junto con boxplots de la métrica primaria por
semilla. Finanzas ahora conserva también las 30 observaciones independientes
(cinco semillas por seis candidatos); su puerta de congelación las exige para que
un promedio no oculte inestabilidad. Una caja con menos de cinco semillas se marca
explícitamente como ilustrativa.
