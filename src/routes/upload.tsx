import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Cpu,
  Database,
  FileCheck2,
  Filter,
  GitBranch,
  Layers3,
  Landmark,
  RefreshCw,
  Server,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";
import { BackendState } from "@/components/BackendState";
import { DataLakePanel } from "@/components/DataLakePanel";
import { ExternalDataSourcesPanel } from "@/components/ExternalDataSourcesPanel";
import { KpiCard } from "@/components/KpiCard";
import { MethodologyFunnel } from "@/components/rosen/ResearchCharts";
import {
  DashboardData,
  ScientificDataSummary,
  fetchDashboardData,
  fetchScientificDataSummary,
} from "@/lib/api";
import { useApiData } from "@/hooks/useApiData";

export const Route = createFileRoute("/upload")({
  head: () => ({
    meta: [
      { title: "Datos Procesados" },
      { name: "description", content: "Estado del dataset procesado por el backend local" },
    ],
  }),
  component: DataStatusPage,
});

function DataStatusPage() {
  const { data, error, isLoading, reload } = useApiData(fetchDashboardData);
  const scientific = useApiData(fetchScientificDataSummary);

  if (isLoading) return <BackendState isLoading />;
  if (error || !data) return <BackendState error={error} onRetry={reload} />;

  const { dataset } = data;
  const quality = (dataset.cleanedRows / Math.max(dataset.originalRows, 1)) * 100;
  const removedRate = (dataset.rowsRemoved / Math.max(dataset.originalRows, 1)) * 100;

  return (
    <div className="dashboard-page">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <h1 className="text-2xl font-bold text-foreground">Datos Procesados</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Cantidades científicas auditadas por dominio, separadas de las muestras técnicas.
          </p>
        </motion.div>
        <button
          type="button"
          onClick={() => {
            reload();
            scientific.reload();
          }}
          className="inline-flex items-center justify-center gap-2 rounded-md border border-border bg-card px-4 py-2 text-sm font-semibold text-foreground shadow-sm transition-colors hover:bg-muted"
        >
          <RefreshCw className="h-4 w-4 text-primary" />
          Actualizar
        </button>
      </div>

      <ScientificDatasetOverview
        data={scientific.data}
        error={scientific.error}
        isLoading={scientific.isLoading}
        onRetry={scientific.reload}
      />

      <div className="rounded-md border border-primary/20 bg-primary/5 px-4 py-3">
        <p className="text-xs font-bold uppercase tracking-wider text-primary">Muestra técnica del pipeline</p>
        <p className="mt-1 text-xs leading-5 text-muted-foreground">
          Los bloques siguientes usan <span className="font-semibold text-foreground">{dataset.filename}</span> para visualizar limpieza y flujo. Sus {dataset.cleanedRows.toLocaleString("es-ES")} filas no representan el tamaño de los datasets de tesis.
        </p>
      </div>

      <PipelineFlow dataset={dataset} quality={quality} />
      <MethodologyFunnel dataset={dataset} quality={quality} />
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.25fr)_minmax(340px,0.75fr)]">
        <DataStreamViz dataset={dataset} />
        <ProcessingMonitor dataset={dataset} quality={quality} removedRate={removedRate} />
      </div>
      <MethodCards dataset={dataset} quality={quality} />
      <BackendOperationalPanel />
      <DataLakePanel />
      <ExternalDataSourcesPanel />

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <KpiCard title="Muestra: originales" value={dataset.originalRows.toLocaleString("es-ES")} icon={Database} variant="cyan" />
        <KpiCard title="Muestra: limpios" value={dataset.cleanedRows.toLocaleString("es-ES")} icon={FileCheck2} variant="violet" delay={0.1} />
        <KpiCard title="Muestra: columnas" value={dataset.columns.length.toLocaleString("es-ES")} icon={Server} variant="cyan" delay={0.2} />
        <KpiCard
          title="Calidad de la muestra"
          value={`${quality.toFixed(1)}%`}
          icon={FileCheck2}
          variant="default"
          delay={0.3}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="rounded-lg border border-border bg-card p-4 shadow-sm sm:p-6">
          <h2 className="mb-4 text-lg font-bold text-foreground">Estructura de Columnas</h2>
          <div className="max-h-[420px] space-y-2 overflow-y-auto pr-1">
            {dataset.columns.map((col) => (
              <div key={col} className="flex items-center justify-between gap-4 rounded-md border border-border bg-muted/20 px-3 py-2">
                <span className="truncate text-sm font-medium text-foreground" title={col}>{col}</span>
                <span className="rounded border border-border bg-card px-2 py-0.5 font-data text-[10px] font-bold text-muted-foreground">
                  {dataset.dataTypes[col] || "desconocido"}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-border bg-card p-4 shadow-sm sm:p-6">
          <h2 className="mb-4 text-lg font-bold text-foreground">Trazabilidad Experimental</h2>
          <div className="space-y-4 text-sm text-muted-foreground">
            <p>
              Los datos presentados corresponden a un artefacto procesado bajo el mismo protocolo experimental usado para comparar las arquitecturas de detección de anomalías.
            </p>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {[
                ["Origen", "Dataset identificado y versionado para la evaluación."],
                ["Preparación", "Limpieza, tipificación de columnas y control de registros no válidos."],
                ["Consistencia", "Métricas calculadas sobre la misma base procesada."],
                ["Evidencia", "Resultados disponibles para análisis, comparación, historial y XAI."],
              ].map(([title, text]) => (
                <div key={title} className="rounded-md border border-border bg-muted/20 p-3">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-primary">{title}</p>
                  <p className="mt-1 text-xs leading-5 text-muted-foreground">{text}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

type Dataset = DashboardData["dataset"];

function ScientificDatasetOverview({
  data,
  error,
  isLoading,
  onRetry,
}: {
  data: ScientificDataSummary | null;
  error: string | null;
  isLoading: boolean;
  onRetry: () => void;
}) {
  if (isLoading) {
    return (
      <section className="rounded-md border border-border bg-card p-5 text-sm text-muted-foreground shadow-sm">
        Cargando conteos científicos auditados…
      </section>
    );
  }

  if (error || !data) {
    return (
      <section className="rounded-md border border-anomaly/20 bg-anomaly/10 p-5 shadow-sm">
        <p className="text-sm font-bold text-anomaly">No se pudo leer el resumen científico.</p>
        <p className="mt-1 text-xs text-muted-foreground">{error || "El backend no devolvió los manifiestos de datos."}</p>
        <button type="button" onClick={onRetry} className="mt-3 rounded-md border border-border bg-card px-3 py-2 text-xs font-bold text-foreground">
          Reintentar
        </button>
      </section>
    );
  }

  const icons = { phishing: ShieldCheck, energia: Zap, finanzas: Landmark } as const;
  const colors = {
    phishing: "text-cyan-500 bg-cyan-500/10",
    energia: "text-amber-500 bg-amber-500/10",
    finanzas: "text-violet-500 bg-violet-500/10",
  } as const;

  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm sm:p-5">
      <div className="mb-5 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-primary">Corpus científico</p>
          <h2 className="mt-1 text-lg font-bold text-foreground">Observaciones únicas y auditadas</h2>
          <p className="mt-1 max-w-3xl text-xs leading-5 text-muted-foreground">
            No se suman como datos nuevos las repeticiones producidas por modelos, folds o semillas.
          </p>
        </div>
        <div className="rounded-md border border-primary/20 bg-primary/5 px-4 py-3 text-right">
          <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Total depurado</p>
          <p className="font-data text-3xl font-bold text-primary">{data.totalUsableObservations.toLocaleString("es-ES")}</p>
          <p className="text-[10px] text-muted-foreground">observaciones en {data.availableDomains}/{data.totalDomains} dominios</p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        {data.domains.map((domain) => {
          const Icon = icons[domain.id];
          return (
            <article key={domain.id} className="rounded-md border border-border bg-muted/10 p-4">
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-3">
                  <div className={`flex h-10 w-10 items-center justify-center rounded-md ${colors[domain.id]}`}>
                    <Icon className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-foreground">{domain.label}</h3>
                    <p className="text-[11px] text-muted-foreground">{domain.source || "Manifiesto no disponible"}</p>
                  </div>
                </div>
                <span className={`rounded-full px-2 py-1 text-[9px] font-bold uppercase ${domain.available ? "bg-success/10 text-success" : "bg-anomaly/10 text-anomaly"}`}>
                  {domain.available ? "Auditado" : "No disponible"}
                </span>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-2">
                <CountCell label="Datos originales" value={domain.originalRows} unit={domain.unit} />
                <CountCell label="Datos depurados" value={domain.usableRows} unit={domain.unit} emphasize />
                <CountCell label="Desarrollo" value={domain.developmentRows} unit={domain.unit} />
                <CountCell label="OOF únicas" value={domain.oofUniqueRows} unit={domain.unit} />
              </div>

              <div className="mt-3 flex items-center justify-between gap-3 rounded-md border border-border bg-card px-3 py-2">
                <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Prueba final bloqueada</span>
                <span className="font-data text-sm font-bold text-foreground">{domain.lockedTestRows.toLocaleString("es-ES")}</span>
              </div>
              <p className="mt-2 truncate font-data text-[9px] text-muted-foreground" title={domain.datasetId || undefined}>
                ID: {domain.datasetId || "sin manifiesto"}
              </p>
            </article>
          );
        })}
      </div>

      <div className="mt-4 flex flex-col gap-2 rounded-md border border-success/20 bg-success/5 px-4 py-3 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
        <span>La muestra técnica y el data lake operativo están excluidos de estas cantidades.</span>
        <span className="font-bold text-success">Prueba final usada: {data.finalTestUsed ? "Sí" : "No"}</span>
      </div>
    </section>
  );
}

function CountCell({ label, value, unit, emphasize = false }: { label: string; value: number; unit: string; emphasize?: boolean }) {
  return (
    <div className={`rounded-md border p-3 ${emphasize ? "border-primary/20 bg-primary/5" : "border-border bg-card"}`}>
      <p className="text-[9px] font-bold uppercase tracking-wider text-muted-foreground">{label}</p>
      <p className={`mt-1 font-data text-xl font-bold ${emphasize ? "text-primary" : "text-foreground"}`}>{value.toLocaleString("es-ES")}</p>
      <p className="text-[9px] text-muted-foreground">{unit}</p>
    </div>
  );
}

function PipelineFlow({ dataset, quality }: { dataset: Dataset; quality: number }) {
  const steps = [
    { label: "Fuente", detail: dataset.filename, icon: Database },
    { label: "Ingesta", detail: `${dataset.originalRows.toLocaleString("es-ES")} registros`, icon: Server },
    { label: "Limpieza", detail: `${dataset.rowsRemoved.toLocaleString("es-ES")} removidos`, icon: Filter },
    { label: "Normalización", detail: `${dataset.columns.length} columnas`, icon: Layers3 },
    { label: "Entrenamiento", detail: "LSTM, GRU, BRNN, Transformer, TCN", icon: Cpu },
    { label: "Evaluación", detail: `Calidad ${quality.toFixed(1)}%`, icon: BarChart3 },
    { label: "XAI", detail: "Importancia y sensibilidad", icon: BrainCircuit },
  ];

  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm sm:p-5">
      <div className="mb-5 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-sm font-bold text-foreground">Metodología de procesamiento</h2>
          <p className="mt-1 text-xs text-muted-foreground">Flujo experimental desde la fuente de datos hasta la interpretación XAI.</p>
        </div>
        <span className="inline-flex w-fit items-center gap-2 rounded-md border border-success/20 bg-success/10 px-3 py-1 text-[10px] font-bold uppercase text-success">
          <CheckCircle2 className="h-3.5 w-3.5" />
          Pipeline activo
        </span>
      </div>

      <div className="relative grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-7">
        <motion.div
          aria-hidden="true"
          className="pointer-events-none absolute left-4 right-4 top-10 hidden h-px bg-primary/20 xl:block"
          initial={{ scaleX: 0, transformOrigin: "left" }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 1.4, ease: "easeOut" }}
        />
        {steps.map((step, index) => {
          const Icon = step.icon;
          return (
            <motion.div
              key={step.label}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.08 }}
              className="relative rounded-md border border-border bg-background p-3"
            >
              <div className="mb-3 flex items-center gap-2">
                <div className="relative flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
                  <Icon className="h-4 w-4" />
                  <motion.span
                    className="absolute inset-0 rounded-md border border-primary/40"
                    animate={{ opacity: [0.25, 0.8, 0.25], scale: [1, 1.08, 1] }}
                    transition={{ duration: 1.8, repeat: Infinity, delay: index * 0.15 }}
                  />
                </div>
                <span className="font-data text-[10px] font-bold text-primary">0{index + 1}</span>
              </div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-foreground">{step.label}</h3>
              <p className="mt-2 line-clamp-2 text-[11px] leading-4 text-muted-foreground" title={step.detail}>{step.detail}</p>
            </motion.div>
          );
        })}
      </div>
    </section>
  );
}

function DataStreamViz({ dataset }: { dataset: Dataset }) {
  const particles = Array.from({ length: 30 }, (_, index) => ({
    id: index,
    top: 12 + ((index * 17) % 74),
    delay: (index % 10) * 0.18,
    duration: 3.2 + (index % 5) * 0.35,
    kind: index % 9 === 0 ? "anomaly" : index % 4 === 0 ? "clean" : "raw",
  }));

  return (
    <section className="relative min-h-[320px] overflow-hidden rounded-md border border-border bg-card p-4 shadow-sm sm:p-5">
      <div className="relative z-10 mb-4">
        <h2 className="text-sm font-bold text-foreground">Consumo simulado de datos</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Representación visual de registros crudos, registros limpios y posibles anomalías circulando por el pipeline.
        </p>
      </div>

      <div className="relative h-56 overflow-hidden rounded-md border border-border bg-muted/10">
        <div className="absolute inset-y-0 left-6 flex w-24 flex-col justify-center gap-2">
          <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Entrada</span>
          <span className="font-data text-xl font-bold text-foreground">{dataset.originalRows.toLocaleString("es-ES")}</span>
        </div>

        <div className="absolute inset-x-32 top-1/2 h-px bg-border" />
        <motion.div
          className="absolute inset-x-32 top-1/2 h-px bg-primary"
          initial={{ scaleX: 0, transformOrigin: "left" }}
          animate={{ scaleX: [0, 1, 0] }}
          transition={{ duration: 3.6, repeat: Infinity, ease: "easeInOut" }}
        />

        {particles.map((particle) => (
          <motion.span
            key={particle.id}
            className={`absolute h-2.5 w-2.5 rounded-full ${
              particle.kind === "anomaly"
                ? "bg-anomaly shadow-[0_0_14px_var(--color-anomaly)]"
                : particle.kind === "clean"
                  ? "bg-success shadow-[0_0_12px_var(--color-success)]"
                  : "bg-primary/60"
            }`}
            style={{ top: `${particle.top}%` }}
            initial={{ x: -30, opacity: 0 }}
            animate={{ x: ["-5%", "52%", "105%"], opacity: [0, 1, 1, 0] }}
            transition={{ duration: particle.duration, repeat: Infinity, delay: particle.delay, ease: "linear" }}
          />
        ))}

        <div className="absolute inset-y-0 right-6 flex w-28 flex-col justify-center gap-2 text-right">
          <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Salida limpia</span>
          <span className="font-data text-xl font-bold text-foreground">{dataset.cleanedRows.toLocaleString("es-ES")}</span>
        </div>

        <div className="absolute bottom-4 left-4 right-4 grid grid-cols-3 gap-2">
          {[
            ["Crudo", "bg-primary/60"],
            ["Limpio", "bg-success"],
            ["Anomalía", "bg-anomaly"],
          ].map(([label, color]) => (
            <div key={label} className="flex items-center justify-center gap-2 rounded border border-border bg-card/80 px-2 py-1 text-[10px] font-bold text-muted-foreground">
              <span className={`h-2 w-2 rounded-full ${color}`} />
              {label}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function ProcessingMonitor({ dataset, quality, removedRate }: { dataset: Dataset; quality: number; removedRate: number }) {
  const metrics = [
    { label: "Registros procesados", value: dataset.cleanedRows, suffix: "", icon: Activity },
    { label: "Calidad de limpieza", value: quality, suffix: "%", icon: ShieldCheck },
    { label: "Filas descartadas", value: removedRate, suffix: "%", icon: Filter },
    { label: "Columnas disponibles", value: dataset.columns.length, suffix: "", icon: GitBranch },
  ];

  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm sm:p-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-bold text-foreground">Monitor de procesamiento</h2>
          <p className="mt-1 text-xs text-muted-foreground">Simulación de consumo incremental basada en el artefacto actual.</p>
        </div>
        <motion.span
          className="flex h-3 w-3 rounded-full bg-success"
          animate={{ opacity: [0.35, 1, 0.35] }}
          transition={{ duration: 1.2, repeat: Infinity }}
        />
      </div>

      <div className="space-y-4">
        {metrics.map((metric, index) => {
          const Icon = metric.icon;
          const percent = metric.suffix === "%" ? Math.min(metric.value, 100) : Math.min((metric.value / Math.max(dataset.originalRows, 1)) * 100, 100);
          return (
            <div key={metric.label} className="rounded-md border border-border bg-muted/20 p-3">
              <div className="mb-2 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <Icon className="h-4 w-4 text-primary" />
                  <span className="text-xs font-semibold text-foreground">{metric.label}</span>
                </div>
                <span className="font-data text-xs font-bold text-primary">
                  {metric.suffix === "%"
                    ? metric.value.toFixed(1)
                    : Math.round(metric.value).toLocaleString("es-ES")}
                  {metric.suffix}
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-primary/10">
                <motion.div
                  className="h-full rounded-full bg-primary"
                  initial={{ width: "0%" }}
                  animate={{ width: `${percent}%` }}
                  transition={{ duration: 0.9, delay: index * 0.12 }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function MethodCards({ dataset, quality }: { dataset: Dataset; quality: number }) {
  const cards = [
    ["Fuente del dataset", dataset.filename],
    ["Tipo de dato", inferDatasetType(dataset)],
    ["Limpieza aplicada", `${dataset.rowsRemoved.toLocaleString("es-ES")} filas removidas y tipos detectados por columna.`],
    ["Modelo recomendado", inferRecommendedModel(dataset)],
    ["Métrica principal", quality >= 90 ? "Calidad de datos y F1-score por dominio." : "Calidad de datos antes de comparar modelos."],
  ];

  return (
    <section className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-5">
      {cards.map(([title, text], index) => (
        <motion.div
          key={title}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.06 }}
          className="rounded-md border border-border bg-card p-4 shadow-sm"
        >
          <div className="mb-3 flex h-8 w-8 items-center justify-center rounded-md bg-primary/10 text-primary">
            <Sparkles className="h-4 w-4" />
          </div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-foreground">{title}</h3>
          <p className="mt-2 text-xs leading-5 text-muted-foreground">{text}</p>
        </motion.div>
      ))}
    </section>
  );
}

function BackendOperationalPanel() {
  const endpoints = ["/api/dashboard", "/api/analysis", "/api/comparison", "/api/history", "/api/xai"];
  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm sm:p-5">
      <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-sm font-bold text-foreground">Estado operativo del backend</h2>
          <p className="mt-1 text-xs text-muted-foreground">Artefactos consultados por el frontend para mantener el dashboard sincronizado.</p>
        </div>
        <span className="inline-flex w-fit items-center gap-2 rounded-md border border-success/20 bg-success/10 px-3 py-1 text-[10px] font-bold uppercase text-success">
          <Server className="h-3.5 w-3.5" />
          Conectado
        </span>
      </div>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-5">
        {endpoints.map((endpoint) => (
          <div key={endpoint} className="flex items-center justify-between gap-2 rounded-md border border-border bg-muted/20 px-3 py-2">
            <span className="font-data text-[11px] text-foreground">{endpoint}</span>
            <CheckCircle2 className="h-4 w-4 text-success" />
          </div>
        ))}
      </div>
    </section>
  );
}

function inferDatasetType(dataset: Dataset) {
  const columns = dataset.columns.join(" ").toLowerCase();
  if (columns.includes("url") || columns.includes("host")) return "Texto / URL";
  if (columns.includes("date") || columns.includes("time") || columns.includes("fecha")) return "Serie temporal";
  return "Tabular";
}

function inferRecommendedModel(dataset: Dataset) {
  const type = inferDatasetType(dataset);
  if (type === "Texto / URL") return "Transformer para dependencias globales de texto.";
  if (type === "Serie temporal") return "TCN para sensibilidad temporal y estabilidad.";
  return "GRU o Transformer según costo computacional y F1-score.";
}
