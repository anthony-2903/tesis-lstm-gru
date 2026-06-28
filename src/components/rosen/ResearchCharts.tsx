import { motion } from "framer-motion";
import type { ComparisonData, DashboardData, EvaluatedData, HistoryItem } from "@/lib/api";

type Dataset = DashboardData["dataset"];
type ModelKey = "lstm" | "gru" | "brnn" | "transformer" | "tcn";

const MODELS: { key: ModelKey; label: string; color: string }[] = [
  { key: "lstm", label: "LSTM", color: "var(--chart-1)" },
  { key: "gru", label: "GRU", color: "var(--chart-2)" },
  { key: "brnn", label: "BRNN", color: "var(--chart-3)" },
  { key: "transformer", label: "Transformer", color: "var(--chart-4)" },
  { key: "tcn", label: "TCN", color: "var(--chart-5)" },
];

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function clamp(value: number, min = 0, max = 100) {
  return Math.max(min, Math.min(max, value));
}

export function MethodologyFunnel({ dataset, quality }: { dataset: Dataset; quality: number }) {
  const originalRows = Math.max(dataset.originalRows, 1);
  const retention = (dataset.cleanedRows / originalRows) * 100;
  const schemaCoverage = clamp((dataset.columns.length / Math.max(dataset.columns.length, 12)) * 100, 66, 100);
  const stages = [
    {
      label: "Datos crudos",
      value: dataset.originalRows.toLocaleString("es-ES"),
      detail: "Entrada inicial recibida por el backend local.",
      width: 100,
      color: "var(--chart-1)",
    },
    {
      label: "Limpieza validada",
      value: dataset.cleanedRows.toLocaleString("es-ES"),
      detail: `${dataset.rowsRemoved.toLocaleString("es-ES")} registros fueron descartados o normalizados.`,
      width: retention,
      color: "var(--chart-2)",
    },
    {
      label: "Esquema utilizable",
      value: `${dataset.columns.length} columnas`,
      detail: "Variables disponibles para entrenamiento, analisis y explicabilidad.",
      width: schemaCoverage,
      color: "var(--chart-3)",
    },
    {
      label: "Evaluacion multimodelo",
      value: "5 modelos",
      detail: "LSTM, GRU, BRNN, Transformer y TCN bajo el mismo protocolo.",
      width: clamp(quality, 58, 100),
      color: "var(--chart-4)",
    },
    {
      label: "Evidencia XAI",
      value: formatPercent(quality / 100),
      detail: "Resultados interpretables listos para justificar la decision final.",
      width: clamp(quality - 4, 54, 96),
      color: "var(--chart-5)",
    },
  ];

  return (
    <section className="card-formal overflow-hidden p-4 sm:p-5">
      <div className="mb-5 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-sm font-bold text-foreground">Funnel metodologico de datos</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Visualizacion tipo RosenCharts del avance desde la fuente hasta la evidencia experimental.
          </p>
        </div>
        <span className="font-data text-[10px] font-bold uppercase tracking-wider text-primary">
          Calidad {quality.toFixed(1)}%
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_280px]">
        <div className="space-y-3">
          {stages.map((stage, index) => (
            <motion.div
              key={stage.label}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.07 }}
              className="grid grid-cols-1 gap-2 sm:grid-cols-[150px_minmax(0,1fr)_90px] sm:items-center"
            >
              <div className="min-w-0">
                <p className="text-xs font-bold text-foreground">{stage.label}</p>
                <p className="mt-0.5 truncate text-[10px] text-muted-foreground" title={stage.detail}>
                  {stage.detail}
                </p>
              </div>
              <div className="h-10 rounded-md border border-border bg-muted/20 p-1">
                <motion.div
                  className="relative h-full rounded-[4px]"
                  style={{ background: stage.color }}
                  initial={{ width: 0 }}
                  animate={{ width: `${stage.width}%` }}
                  transition={{ duration: 0.8, delay: index * 0.08 }}
                >
                  <motion.span
                    className="absolute inset-y-1 right-2 w-8 rounded-full bg-white/25"
                    animate={{ opacity: [0.25, 0.7, 0.25], x: [-6, 0, -6] }}
                    transition={{ duration: 1.6, repeat: Infinity, delay: index * 0.1 }}
                  />
                </motion.div>
              </div>
              <p className="font-data text-xs font-bold text-primary sm:text-right">{stage.value}</p>
            </motion.div>
          ))}
        </div>

        <div className="rounded-md border border-border bg-muted/20 p-4">
          <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Lectura de tesis</p>
          <p className="mt-2 text-sm font-semibold leading-5 text-foreground">
            El frontend no solo muestra datos: documenta el recorrido experimental que conecta preparacion,
            comparacion y explicabilidad.
          </p>
          <div className="mt-4 grid grid-cols-2 gap-2">
            {[
              ["Retencion", `${retention.toFixed(1)}%`],
              ["Variables", dataset.columns.length.toString()],
              ["Modelos", "5"],
              ["XAI", "Activo"],
            ].map(([label, value]) => (
              <div key={label} className="rounded-md border border-border bg-card p-3">
                <p className="text-[10px] font-bold uppercase text-muted-foreground">{label}</p>
                <p className="mt-1 font-data text-sm font-bold text-foreground">{value}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

export function ModelBenchmarkChart({ data }: { data: ComparisonData }) {
  const metricRows = data.comparisonBarData;
  const f1Row = metricRows.find((row) => String(row.metric).toLowerCase().includes("f1")) ?? metricRows[0];
  const precisionRow = metricRows.find((row) => String(row.metric).toLowerCase().includes("precision"));
  const recallRow = metricRows.find((row) => String(row.metric).toLowerCase().includes("recall"));
  const maxTime = Math.max(...data.scatterData.map((item) => item.time), 0.001);
  const rows = MODELS.map((model) => {
    const scatter = data.scatterData.find((item) => item.model === model.label);
    const f1 = Number(f1Row?.[model.label] ?? f1Row?.[model.key] ?? scatter?.accuracy ?? 0);
    const precision = Number(precisionRow?.[model.label] ?? precisionRow?.[model.key] ?? f1);
    const recall = Number(recallRow?.[model.label] ?? recallRow?.[model.key] ?? f1);
    const speed = scatter ? 1 - scatter.time / maxTime : 0.5;
    const score = f1 * 0.45 + precision * 0.25 + recall * 0.2 + speed * 0.1;
    return { ...model, f1, precision, recall, speed, score };
  }).sort((a, b) => b.score - a.score);

  const leader = rows[0];

  return (
    <section className="card-formal p-4 sm:p-6">
      <div className="mb-5 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h3 className="text-sm font-semibold text-foreground">Benchmark tipo RosenCharts</h3>
          <p className="mt-1 text-xs text-muted-foreground">
            Ranking sintetico que combina F1, precision, recall y eficiencia temporal.
          </p>
        </div>
        <span className="w-fit rounded-md border border-primary/20 bg-primary/10 px-3 py-1 text-[10px] font-bold uppercase text-primary">
          Lider: {leader.label}
        </span>
      </div>

      <div className="space-y-4">
        {rows.map((row, index) => (
          <motion.div
            key={row.label}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.06 }}
            className="grid grid-cols-1 gap-2 md:grid-cols-[120px_minmax(0,1fr)_74px]"
          >
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-sm" style={{ background: row.color }} />
              <span className="text-xs font-bold text-foreground">{row.label}</span>
            </div>
            <div className="relative h-9 overflow-hidden rounded-md border border-border bg-muted/20">
              <motion.div
                className="absolute inset-y-0 left-0 rounded-r-md"
                style={{ background: row.color }}
                initial={{ width: 0 }}
                animate={{ width: `${clamp(row.score * 100)}%` }}
                transition={{ duration: 0.8, delay: index * 0.07 }}
              />
              <div className="absolute inset-0 grid grid-cols-4 text-[9px] font-bold uppercase text-foreground/70">
                <span className="flex items-center justify-center border-r border-card/50">F1 {formatPercent(row.f1)}</span>
                <span className="flex items-center justify-center border-r border-card/50">P {formatPercent(row.precision)}</span>
                <span className="flex items-center justify-center border-r border-card/50">R {formatPercent(row.recall)}</span>
                <span className="flex items-center justify-center">T {formatPercent(row.speed)}</span>
              </div>
            </div>
            <p className="font-data text-xs font-bold text-primary md:text-right">{formatPercent(row.score)}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}

export function AnalysisPerformanceHeatmap({ data }: { data: EvaluatedData }) {
  const metrics = [
    { key: "f1", label: "F1" },
    { key: "precision", label: "Precision" },
    { key: "recall", label: "Recall" },
  ] as const;
  const cells = MODELS.flatMap((model) =>
    metrics.map((metric) => ({
      model: model.label,
      metric: metric.label,
      color: model.color,
      value: data.models[model.key][metric.key],
    })),
  );
  const leader = MODELS.map((model) => ({
    label: model.label,
    score:
      data.models[model.key].f1 * 0.5 +
      data.models[model.key].precision * 0.25 +
      data.models[model.key].recall * 0.25,
  })).sort((a, b) => b.score - a.score)[0];

  return (
    <section className="card-formal p-4 sm:p-5">
      <div className="mb-5 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-sm font-bold text-foreground">Heatmap de rendimiento</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Lectura compacta de F1, precision y recall por arquitectura evaluada.
          </p>
        </div>
        <span className="w-fit rounded-md border border-primary/20 bg-primary/10 px-3 py-1 text-[10px] font-bold uppercase text-primary">
          Mejor balance: {leader.label}
        </span>
      </div>

      <div className="responsive-table">
        <div className="min-w-[560px]">
          <div className="grid grid-cols-[120px_repeat(3,minmax(100px,1fr))] gap-2">
            <div />
            {metrics.map((metric) => (
              <div key={metric.key} className="text-center text-[10px] font-bold uppercase text-muted-foreground">
                {metric.label}
              </div>
            ))}
            {MODELS.map((model, rowIndex) => (
              <div key={model.key} className="contents">
                <div className="flex items-center gap-2 rounded-md border border-border bg-muted/20 px-3 text-xs font-bold text-foreground">
                  <span className="h-3 w-3 rounded-sm" style={{ background: model.color }} />
                  {model.label}
                </div>
                {metrics.map((metric, colIndex) => {
                  const cell = cells.find((item) => item.model === model.label && item.metric === metric.label);
                  const value = cell?.value ?? 0;
                  return (
                    <motion.div
                      key={`${model.key}-${metric.key}`}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: rowIndex * 0.05 + colIndex * 0.03 }}
                      className="flex h-16 flex-col items-center justify-center rounded-md border border-border text-center"
                      style={{ background: `color-mix(in oklch, ${model.color} ${clamp(value * 72, 8, 72)}%, transparent)` }}
                      title={`${model.label} ${metric.label}: ${value.toFixed(3)}`}
                    >
                      <span className="font-data text-sm font-bold text-foreground">{value.toFixed(3)}</span>
                      <span className="mt-1 text-[9px] font-bold uppercase text-muted-foreground">{formatPercent(value)}</span>
                    </motion.div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

export function DetectionBalanceChart({ data }: { data: EvaluatedData }) {
  const maxCount = Math.max(data.realAnomaliesCount, ...MODELS.map((model) => data.models[model.key].detectedCount), 1);
  const rows = [
    {
      label: "Anomalias reales",
      value: data.realAnomaliesCount,
      color: "var(--color-anomaly)",
      detail: "Referencia esperada",
    },
    ...MODELS.map((model) => ({
      label: model.label,
      value: data.models[model.key].detectedCount,
      color: model.color,
      detail: `TP ${data.models[model.key].confusionMatrix.tp} / FP ${data.models[model.key].confusionMatrix.fp}`,
    })),
  ];

  return (
    <section className="card-formal p-4 sm:p-5">
      <div className="mb-5">
        <h2 className="text-sm font-bold text-foreground">Balance de deteccion</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Comparacion directa entre la cantidad real de anomalias y lo detectado por cada modelo.
        </p>
      </div>

      <div className="space-y-3">
        {rows.map((row, index) => (
          <motion.div
            key={row.label}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className="grid grid-cols-1 gap-2 sm:grid-cols-[140px_minmax(0,1fr)_82px] sm:items-center"
          >
            <div>
              <p className="text-xs font-bold text-foreground">{row.label}</p>
              <p className="mt-0.5 text-[10px] text-muted-foreground">{row.detail}</p>
            </div>
            <div className="relative h-9 overflow-hidden rounded-md border border-border bg-muted/20">
              <motion.div
                className="absolute inset-y-0 left-0 rounded-r-md"
                style={{ background: row.color }}
                initial={{ width: 0 }}
                animate={{ width: `${clamp((row.value / maxCount) * 100)}%` }}
                transition={{ duration: 0.75, delay: index * 0.06 }}
              />
              <motion.span
                className="absolute inset-y-1 right-2 w-10 rounded-full bg-white/20"
                animate={{ opacity: [0.2, 0.65, 0.2], x: [-5, 0, -5] }}
                transition={{ duration: 1.5, repeat: Infinity, delay: index * 0.08 }}
              />
            </div>
            <p className="font-data text-xs font-bold text-primary sm:text-right">{row.value.toLocaleString("es-ES")}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}

export function AuditHeatmap({ items }: { items: HistoryItem[] }) {
  const domains = Array.from(new Set(items.map((item) => item.domain))).slice(0, 6);
  const models = MODELS.map((model) => model.label).filter((model) => items.some((item) => item.model === model));
  const cells = models.flatMap((model) =>
    domains.map((domain) => {
      const matching = items.filter((item) => item.model === model && item.domain === domain);
      const anomalies = matching.filter((item) => item.predicted !== item.realLabel || item.predicted.toLowerCase().includes("anom")).length;
      const confidence = matching.length
        ? matching.reduce((acc, item) => acc + item.confidence, 0) / matching.length
        : 0;
      return { model, domain, anomalies, total: matching.length, confidence };
    }),
  );
  const maxAnomalies = Math.max(...cells.map((cell) => cell.anomalies), 1);

  if (!items.length || !domains.length || !models.length) {
    return (
      <section className="card-formal p-4 sm:p-5">
        <h2 className="text-sm font-bold text-foreground">Heatmap de auditoria</h2>
        <p className="mt-2 text-xs text-muted-foreground">No hay registros suficientes para construir la matriz visual.</p>
      </section>
    );
  }

  return (
    <section className="card-formal p-4 sm:p-5">
      <div className="mb-5 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-sm font-bold text-foreground">Heatmap de auditoria</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Intensidad por modelo y dominio segun anomalias o discrepancias detectadas.
          </p>
        </div>
        <span className="font-data text-[10px] font-bold uppercase text-muted-foreground">{items.length} registros</span>
      </div>

      <div className="responsive-table">
        <div className="min-w-[620px]">
          <div
            className="grid gap-2"
            style={{ gridTemplateColumns: `120px repeat(${domains.length}, minmax(72px, 1fr))` }}
          >
            <div />
            {domains.map((domain) => (
              <div key={domain} className="truncate text-center text-[10px] font-bold uppercase text-muted-foreground" title={domain}>
                {domain}
              </div>
            ))}
            {models.map((model) => (
              <div key={model} className="contents">
                <div className="flex items-center rounded-md border border-border bg-muted/20 px-3 text-xs font-bold text-foreground">
                  {model}
                </div>
                {domains.map((domain) => {
                  const cell = cells.find((candidate) => candidate.model === model && candidate.domain === domain);
                  const intensity = cell ? cell.anomalies / maxAnomalies : 0;
                  const opacity = cell?.total ? 0.16 + intensity * 0.72 : 0.05;
                  return (
                    <motion.div
                      key={`${model}-${domain}`}
                      initial={{ opacity: 0, scale: 0.96 }}
                      animate={{ opacity: 1, scale: 1 }}
                      title={`${model} / ${domain}: ${cell?.anomalies ?? 0} alertas, confianza ${cell?.confidence.toFixed(1) ?? "0.0"}%`}
                      className="flex h-16 flex-col items-center justify-center rounded-md border border-border text-center"
                      style={{ background: `color-mix(in oklch, var(--color-anomaly) ${opacity * 100}%, transparent)` }}
                    >
                      <span className="font-data text-sm font-bold text-foreground">{cell?.anomalies ?? 0}</span>
                      <span className="mt-1 text-[9px] font-bold uppercase text-muted-foreground">
                        {cell?.total ?? 0} casos
                      </span>
                    </motion.div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
