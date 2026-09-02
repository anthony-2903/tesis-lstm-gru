import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState, type Dispatch, type ReactNode, type SetStateAction } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Database, DownloadCloud, FlaskConical, Play, Settings2, StopCircle, Trophy, Zap } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { BackendState } from "@/components/BackendState";
import { ChartCard } from "@/components/ChartCard";
import { ThesisProtocolPanel } from "@/components/ThesisProtocolPanel";
import { useApiData } from "@/hooks/useApiData";
import {
  cancelEnergyJob,
  cancelEnergyThesisProtocol,
  pauseEnergyThesisProtocol,
  fetchEnergyDatasetStatus,
  fetchEnergyJob,
  fetchLatestEnergyExperiment,
  fetchLatestEnergyOptimizedRevalidation,
  fetchLatestEnergyDataJob,
  fetchLatestEnergyJob,
  fetchLatestEnergyThesisProtocol,
  prepareEnergyDataset,
  runEnergyExperiment,
  resumeEnergyThesisProtocol,
  runEnergyThesisProtocol,
  type EnergyDataPreparationJob,
  type EnergyExperimentJob,
  type EnergyExperimentRequest,
  type EnergyExperimentView,
} from "@/lib/api";

export const Route = createFileRoute("/energy")({
  head: () => ({
    meta: [
      { title: "Experimento energético OOF" },
      { name: "description", content: "Validación walk-forward, Stacking y anomalías del dominio energético." },
    ],
  }),
  component: EnergyExperimentPage,
});

const MODEL_COLORS: Record<string, string> = {
  actual: "var(--foreground)",
  ensemble: "var(--primary)",
  lstm: "var(--chart-1)",
  gru: "var(--chart-2)",
  brnn: "var(--chart-3)",
  transformer: "var(--chart-4)",
  tcn: "var(--chart-5)",
};

const formatNumber = (value: number, digits = 3) =>
  Number.isFinite(value) ? value.toLocaleString("es-ES", { maximumFractionDigits: digits }) : "N/D";

type ModelId = NonNullable<EnergyExperimentRequest["modelIds"]>[number];

interface EnergyExperimentForm {
  protocol: "demo" | "thesis";
  source: "sample" | "silver";
  window: number;
  horizon: number;
  gapSteps: number;
  folds: number;
  epochs: number;
  batchSize: number;
  seeds: string;
  modelIds: ModelId[];
}

const ALL_MODELS: ModelId[] = ["lstm", "gru", "brnn", "tcn", "transformer"];

const INITIAL_FORM: EnergyExperimentForm = {
  protocol: "demo",
  source: "sample",
  window: 12,
  horizon: 1,
  gapSteps: 6,
  folds: 3,
  epochs: 1,
  batchSize: 16,
  seeds: "42",
  modelIds: ALL_MODELS,
};

function EnergyExperimentPage() {
  const { data, error, isLoading, reload } = useApiData(fetchLatestEnergyExperiment);
  const [form, setForm] = useState<EnergyExperimentForm>(INITIAL_FORM);
  const [job, setJob] = useState<EnergyExperimentJob | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const activeJobId = job?.jobId;
  const activeJobStatus = job?.status;

  useEffect(() => {
    fetchLatestEnergyJob()
      .then((latest) => {
        if ("jobId" in latest) setJob(latest);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!activeJobId || !activeJobStatus || !["queued", "running"].includes(activeJobStatus)) return;
    const timer = window.setInterval(() => {
      fetchEnergyJob(activeJobId)
        .then((updated) => {
          setJob(updated);
          if (updated.status === "completed") reload();
        })
        .catch((caught) => setRunError(caught instanceof Error ? caught.message : "No se pudo consultar el progreso."));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [activeJobId, activeJobStatus, reload]);

  const startExperiment = async () => {
    setRunError(null);
    const seeds = [...new Set(form.seeds.split(",").map((value) => Number(value.trim())).filter(Number.isInteger))];
    if (!seeds.length) {
      setRunError("Ingrese al menos una semilla entera.");
      return;
    }
    if (form.modelIds.length < 2) {
      setRunError("Seleccione al menos dos modelos base para habilitar Stacking.");
      return;
    }
    try {
      const created = await runEnergyExperiment({ ...form, seeds });
      setJob(created);
    } catch (caught) {
      setRunError(caught instanceof Error ? caught.message : "No se pudo ejecutar el experimento.");
    }
  };

  const cancelJob = async () => {
    if (!job) return;
    try {
      setJob(await cancelEnergyJob(job.jobId));
    } catch (caught) {
      setRunError(caught instanceof Error ? caught.message : "No se pudo cancelar el experimento.");
    }
  };

  const isRunning = job?.status === "queued" || job?.status === "running";

  if (isLoading) return <BackendState isLoading />;
  if (error || !data) return <BackendState error={error} onRetry={reload} />;

  if (!data.available) {
    return (
      <div className="dashboard-page">
        <PageHeader />
        <EnergyDatasetPanel />
        <EnergyThesisPanel />
        <ExperimentControlPanel form={form} setForm={setForm} isRunning={isRunning} onStart={startExperiment} />
        {job && <JobProgress job={job} onCancel={cancelJob} />}
        <section className="rounded-md border border-dashed border-border bg-card p-8 text-center">
          <FlaskConical className="mx-auto h-10 w-10 text-primary" />
          <h2 className="mt-4 text-lg font-bold text-foreground">Sin resultados OOF todavía</h2>
          <p className="mx-auto mt-2 max-w-xl text-sm text-muted-foreground">{data.message}</p>
        </section>
        {runError && <ErrorNotice message={runError} />}
      </div>
    );
  }

  return (
    <div className="dashboard-page">
      <PageHeader />
      <EnergyDatasetPanel />
      <EnergyThesisPanel />
      <ExperimentControlPanel form={form} setForm={setForm} isRunning={isRunning} onStart={startExperiment} />
      {job && <JobProgress job={job} onCancel={cancelJob} />}
      {runError && <ErrorNotice message={runError} />}
      <MethodologyNotice experiment={data} />
      <EnergyRevalidationPanel />
      <SummaryCards experiment={data} />
      <PredictionChart experiment={data} />
      <ComparisonTable experiment={data} />
      <AnomalyChart experiment={data} />
    </div>
  );
}

function EnergyRevalidationPanel() {
  const { data } = useApiData(fetchLatestEnergyOptimizedRevalidation);
  if (!data?.available) return null;
  const inference = data.independentComparison.pairedInference;
  return (
    <section className={`rounded-md border p-4 ${data.independentComparison.optimizedStackingIsWinner ? "border-success/30 bg-success/10" : "border-warning/30 bg-warning/10"}`}>
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="text-xs font-bold uppercase text-primary">Revalidación independiente del Stacking</p>
          <h2 className="mt-1 text-sm font-bold text-foreground">Configuración: {data.selection.selectedAblationId.replaceAll("_", " ")}</h2>
          <p className="mt-1 text-xs text-muted-foreground">Seleccionada en folds {data.protocol.selectionFolds.join(", ")} y evaluada una sola vez en fold {data.protocol.independentComparisonFold}. Test usado: no.</p>
        </div>
        <div className="text-left lg:text-right">
          <p className="text-sm font-bold text-foreground">{data.independentComparison.optimizedStackingIsWinner ? "Stacking ganó el bloque independiente" : `Ganador independiente: ${data.independentComparison.ranking[0]}`}</p>
          <p className="mt-1 text-xs text-muted-foreground">ΔRMSE vs {inference.referenceId}: {formatNumber(inference.observedDeltaRmse)} · IC95% [{formatNumber(inference.confidenceInterval95[0])}, {formatNumber(inference.confidenceInterval95[1])}]</p>
        </div>
      </div>
      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {data.independentComparison.metrics.map((row, index) => <div key={row.predictorId} className="rounded-md border border-border bg-card p-3"><p className="text-[10px] font-bold uppercase text-muted-foreground">#{index + 1} {row.predictorId.replaceAll("_", " ")}</p><p className="mt-1 font-data text-lg font-bold text-foreground">RMSE {formatNumber(row.rmse)}</p><p className="text-[10px] text-muted-foreground">MAE {formatNumber(row.mae)} · R² {formatNumber(row.r2)}</p></div>)}
      </div>
      <div className="mt-4">
        <p className="text-[10px] font-bold uppercase text-muted-foreground">XAI del metamodelo</p>
        <p className="mt-1 text-xs text-foreground">{data.xai.items.slice(0, 5).map((row) => `${row.feature}: ${(row.importanceMean * 100).toFixed(1)}%`).join(" · ")}</p>
      </div>
    </section>
  );
}

function EnergyThesisPanel() {
  return <ThesisProtocolPanel
    title="Protocolo final de tesis · Energía"
    description="Ejecuta OPSD real → 5 folds walk-forward × 5 semillas × 5 arquitecturas → Stacking → ablación → revalidación independiente → bootstrap diario → XAI → sellado previo al test."
    fetchLatest={fetchLatestEnergyThesisProtocol}
    start={runEnergyThesisProtocol}
    resume={resumeEnergyThesisProtocol}
    pause={pauseEnergyThesisProtocol}
    cancel={cancelEnergyThesisProtocol}
  />;
}

function PageHeader() {
  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Energía: validación OOF y Stacking</h1>
        <p className="mt-1 text-sm text-muted-foreground">Comparación temporal reproducible, sin usar el conjunto de prueba.</p>
      </div>
    </div>
  );
}

function EnergyDatasetPanel() {
  const { data, error, isLoading, reload } = useApiData(fetchEnergyDatasetStatus);
  const [job, setJob] = useState<EnergyDataPreparationJob | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const active = job?.status === "queued" || job?.status === "running";

  useEffect(() => {
    fetchLatestEnergyDataJob().then((latest) => {
      if ("jobId" in latest) setJob(latest);
    }).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => {
      fetchLatestEnergyDataJob().then((latest) => {
        if (!("jobId" in latest)) return;
        setJob(latest);
        if (latest.status === "completed") reload();
      }).catch((caught) => setActionError(caught instanceof Error ? caught.message : "No se pudo consultar la preparación."));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [active, reload]);

  const prepare = async (force: boolean) => {
    setActionError(null);
    try {
      setJob(await prepareEnergyDataset(force));
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "No se pudo preparar OPSD.");
    }
  };

  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-start gap-3">
          <Database className="mt-0.5 h-5 w-5 text-primary" />
          <div>
            <h2 className="text-sm font-bold text-foreground">Dataset energético real</h2>
            {isLoading && <p className="mt-1 text-xs text-muted-foreground">Consultando auditoría…</p>}
            {error && <p className="mt-1 text-xs text-destructive">{error}</p>}
            {data && !data.available && <p className="mt-1 text-xs text-muted-foreground">{data.message}</p>}
            {data?.available && <>
              <p className="mt-1 text-xs text-muted-foreground">{data.datasetId} · {data.silver?.rows.toLocaleString("es-ES")} filas · versión {data.sourceVersion}</p>
              <p className={`mt-1 text-xs font-bold ${data.readyForThesisPilot ? "text-success" : "text-warning"}`}>{data.readyForThesisPilot ? "Apto para piloto científico" : `No apto: ${data.readiness?.reasons.join(" ") || "revise la auditoría"}`}</p>
            </>}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <button type="button" onClick={() => prepare(false)} disabled={active} className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-xs font-bold text-primary-foreground disabled:opacity-60"><DownloadCloud className="h-4 w-4" />{active ? "Preparando…" : data?.available ? "Verificar snapshot" : "Preparar OPSD real"}</button>
          {data?.available && <button type="button" onClick={() => prepare(true)} disabled={active} className="rounded-md border border-border px-3 py-2 text-xs font-bold text-foreground disabled:opacity-60">Forzar descarga</button>}
        </div>
      </div>
      {job && <div className="mt-4">
        <div className="h-2 overflow-hidden rounded-full bg-primary/10"><div className="h-full rounded-full bg-primary transition-all" style={{ width: `${job.progress.percent}%` }} /></div>
        <div className="mt-2 flex justify-between gap-3 text-[10px] font-semibold text-muted-foreground"><span>{job.progress.message}</span><span>{job.progress.percent.toFixed(0)}%</span></div>
        {job.error && <p className="mt-2 text-xs font-semibold text-destructive">{job.error}</p>}
      </div>}
      {actionError && <p className="mt-3 text-xs font-semibold text-destructive">{actionError}</p>}
    </section>
  );
}

function ExperimentControlPanel({
  form,
  setForm,
  isRunning,
  onStart,
}: {
  form: EnergyExperimentForm;
  setForm: Dispatch<SetStateAction<EnergyExperimentForm>>;
  isRunning: boolean;
  onStart: () => void;
}) {
  const updateNumber = (key: keyof EnergyExperimentForm, value: string) =>
    setForm((current) => ({ ...current, [key]: Number(value) }));
  const selectProtocol = (protocol: "demo" | "thesis") => {
    setForm((current) => protocol === "thesis"
      ? { ...current, protocol, source: "silver", folds: 5, epochs: Math.max(current.epochs, 20), seeds: "42,101,202,303,404", modelIds: ALL_MODELS }
      : { ...current, protocol, source: "sample", folds: 3, epochs: 1, seeds: "42" });
  };
  const toggleModel = (model: ModelId) => setForm((current) => ({
    ...current,
    modelIds: current.modelIds.includes(model)
      ? current.modelIds.filter((value) => value !== model)
      : [...current.modelIds, model],
  }));

  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm">
      <div className="mb-4 flex items-center gap-2">
        <Settings2 className="h-4 w-4 text-primary" />
        <div>
          <h2 className="text-sm font-bold text-foreground">Configuración experimental</h2>
          <p className="text-xs text-muted-foreground">Las corridas se procesan en segundo plano y sobreviven a la navegación del dashboard.</p>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-8">
        <Field label="Protocolo">
          <select value={form.protocol} disabled={isRunning} onChange={(event) => selectProtocol(event.target.value as "demo" | "thesis")} className="form-control">
            <option value="demo">Demostración</option><option value="thesis">Tesis</option>
          </select>
        </Field>
        <Field label="Fuente">
          <select value={form.source} disabled={isRunning || form.protocol === "thesis"} onChange={(event) => setForm((current) => ({ ...current, source: event.target.value as "sample" | "silver" }))} className="form-control">
            <option value="sample">Muestra</option><option value="silver">OPSD silver</option>
          </select>
        </Field>
        <NumberField label="Ventana" value={form.window} min={2} disabled={isRunning} onChange={(value) => updateNumber("window", value)} />
        <NumberField label="Horizonte" value={form.horizon} min={1} disabled={isRunning} onChange={(value) => updateNumber("horizon", value)} />
        <NumberField label="Gap" value={form.gapSteps} min={0} disabled={isRunning} onChange={(value) => updateNumber("gapSteps", value)} />
        <NumberField label="Folds" value={form.folds} min={3} max={10} disabled={isRunning || form.protocol === "thesis"} onChange={(value) => updateNumber("folds", value)} />
        <NumberField label="Épocas" value={form.epochs} min={1} max={200} disabled={isRunning} onChange={(value) => updateNumber("epochs", value)} />
        <NumberField label="Batch" value={form.batchSize} min={1} max={512} disabled={isRunning} onChange={(value) => updateNumber("batchSize", value)} />
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_2fr_auto] lg:items-end">
        <Field label="Semillas separadas por coma">
          <input value={form.seeds} disabled={isRunning || form.protocol === "thesis"} onChange={(event) => setForm((current) => ({ ...current, seeds: event.target.value }))} className="form-control" />
        </Field>
        <div>
          <p className="mb-1 text-[10px] font-bold uppercase text-muted-foreground">Modelos base</p>
          <div className="flex flex-wrap gap-2">
            {ALL_MODELS.map((model) => (
              <button key={model} type="button" disabled={isRunning || form.protocol === "thesis"} onClick={() => toggleModel(model)} className={`rounded-md border px-2.5 py-2 text-[10px] font-bold uppercase ${form.modelIds.includes(model) ? "border-primary bg-primary/10 text-primary" : "border-border text-muted-foreground"}`}>{model}</button>
            ))}
          </div>
        </div>
        <button type="button" onClick={onStart} disabled={isRunning} className="inline-flex items-center justify-center gap-2 rounded-md bg-primary px-4 py-2.5 text-sm font-bold text-primary-foreground disabled:cursor-wait disabled:opacity-60">
          <Play className="h-4 w-4" />{isRunning ? "En ejecución" : "Iniciar corrida"}
        </button>
      </div>
    </section>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return <label className="block"><span className="mb-1 block text-[10px] font-bold uppercase text-muted-foreground">{label}</span>{children}</label>;
}

function NumberField({ label, value, min, max, disabled, onChange }: { label: string; value: number; min: number; max?: number; disabled: boolean; onChange: (value: string) => void }) {
  return <Field label={label}><input type="number" value={value} min={min} max={max} disabled={disabled} onChange={(event) => onChange(event.target.value)} className="form-control" /></Field>;
}

function JobProgress({ job, onCancel }: { job: EnergyExperimentJob; onCancel: () => void }) {
  const active = job.status === "queued" || job.status === "running";
  return (
    <section className="rounded-md border border-primary/20 bg-primary/5 p-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs font-bold uppercase text-primary">Trabajo {job.status}</p>
          <p className="mt-1 font-data text-xs text-foreground">{job.jobId}</p>
          <p className="mt-1 text-xs text-muted-foreground">{job.progress.message}</p>
        </div>
        {active && <button type="button" onClick={onCancel} disabled={job.cancelRequested} className="inline-flex items-center justify-center gap-2 rounded-md border border-destructive/30 px-3 py-2 text-xs font-bold text-destructive disabled:opacity-50"><StopCircle className="h-4 w-4" />{job.cancelRequested ? "Cancelación solicitada" : "Cancelar"}</button>}
      </div>
      <div className="mt-3 h-2 overflow-hidden rounded-full bg-primary/10"><div className="h-full rounded-full bg-primary transition-all" style={{ width: `${Math.max(0, Math.min(100, job.progress.percent))}%` }} /></div>
      <div className="mt-2 flex flex-wrap justify-between gap-2 text-[10px] font-semibold text-muted-foreground">
        <span>{job.progress.completedUnits}/{job.progress.totalUnits} entrenamientos</span>
        <span>{job.progress.modelId ? `${job.progress.modelId.toUpperCase()} · fold ${(job.progress.fold ?? 0) + 1} · semilla ${job.progress.seed}` : job.progress.stage}</span>
        <span>{job.progress.percent.toFixed(1)}%</span>
      </div>
      {job.error && <p className="mt-3 text-xs font-semibold text-destructive">{job.error}</p>}
    </section>
  );
}

function MethodologyNotice({ experiment }: { experiment: EnergyExperimentView }) {
  return (
    <section className="rounded-md border border-warning/30 bg-warning/10 p-4">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-warning" />
        <div>
          <p className="text-sm font-bold text-foreground">Estado metodológico: {experiment.run.status}</p>
          <p className="mt-1 text-xs text-muted-foreground">{experiment.methodology.warning}</p>
          <p className="mt-2 text-xs font-semibold text-foreground">
            Test utilizado: {experiment.methodology.testSetUsed ? "sí" : "no"} · Etiquetas de anomalía: {experiment.anomalies.labelType} · F1 de anomalías disponible: {experiment.anomalies.classificationMetricsAvailable ? "sí" : "no"}
          </p>
        </div>
      </div>
    </section>
  );
}

function SummaryCards({ experiment }: { experiment: EnergyExperimentView }) {
  const cards = [
    { label: "Ganador por RMSE", value: experiment.winner.overallDisplayName || "N/D", icon: Trophy },
    { label: "Mejor ensamble", value: experiment.winner.bestEnsembleDisplayName || "N/D", icon: Zap },
    { label: "Folds / semillas", value: `${experiment.validation.folds.length} / ${experiment.validation.seeds.length}`, icon: FlaskConical },
    { label: "Filas validadas", value: experiment.dataset.rows.toLocaleString("es-ES"), icon: CheckCircle2 },
  ];
  return (
    <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {cards.map(({ label, value, icon: Icon }, index) => (
        <motion.article key={label} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.05 }} className="rounded-md border border-border bg-card p-4 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">{label}</p>
            <Icon className="h-4 w-4 text-primary" />
          </div>
          <p className="mt-3 font-data text-xl font-bold text-foreground">{value}</p>
        </motion.article>
      ))}
    </section>
  );
}

function PredictionChart({ experiment }: { experiment: EnergyExperimentView }) {
  const baseModels = experiment.comparison.filter((row) => row.family === "base").map((row) => row.predictorId);
  return (
    <ChartCard title="Valor real frente a predicciones OOF" subtitle={`Último fold comparable · Ensamble: ${experiment.winner.bestEnsembleDisplayName || "N/D"}`}>
      <div className="chart-shell"><div className="chart-min">
        <ResponsiveContainer width="100%" height={390}>
          <LineChart data={experiment.timeline} margin={{ top: 10, right: 20, left: 10, bottom: 35 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
            <XAxis dataKey="timestamp" tick={{ fontSize: 9 }} angle={-20} textAnchor="end" height={65} minTickGap={45} />
            <YAxis tick={{ fontSize: 10 }} width={60} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="actual" name="Valor real" stroke={MODEL_COLORS.actual} strokeWidth={3} dot={false} />
            {baseModels.map((model) => <Line key={model} type="monotone" dataKey={model} name={model.toUpperCase()} stroke={MODEL_COLORS[model]} strokeWidth={1} strokeOpacity={0.55} dot={false} />)}
            <Line type="monotone" dataKey="ensemble" name={experiment.winner.bestEnsembleDisplayName || "Ensamble"} stroke={MODEL_COLORS.ensemble} strokeWidth={3} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div></div>
    </ChartCard>
  );
}

function ComparisonTable({ experiment }: { experiment: EnergyExperimentView }) {
  return (
    <ChartCard title="Competencia entre modelos base y metamodelos" subtitle="Ordenada por RMSE medio; menor es mejor">
      <div className="responsive-table">
        <table className="w-full min-w-[900px] text-xs">
          <thead><tr className="border-b border-border bg-muted/20 text-muted-foreground">
            {["Posición", "Modelo", "Tipo", "RMSE ± DE", "MAE", "sMAPE", "R²", "Evaluaciones", "Entrenamiento"].map((header) => <th key={header} className="px-3 py-3 text-left font-bold uppercase">{header}</th>)}
          </tr></thead>
          <tbody>
            {experiment.comparison.map((row, index) => (
              <tr key={row.predictorId} className={`border-b border-border ${index === 0 ? "bg-primary/5" : "hover:bg-muted/10"}`}>
                <td className="px-3 py-3 font-data font-bold">#{index + 1}</td>
                <td className="px-3 py-3 font-bold text-foreground">{row.displayName}{row.predictorId === "stacking_gradient_boosting" && <span className="ml-2 rounded bg-primary/10 px-1.5 py-0.5 text-[9px] uppercase text-primary">Metamodelo</span>}</td>
                <td className="px-3 py-3 uppercase text-muted-foreground">{row.family === "base" ? "Base" : "Ensamble"}</td>
                <td className="px-3 py-3 font-data font-bold">{formatNumber(row.rmseMean)} ± {formatNumber(row.rmseStd)}</td>
                <td className="px-3 py-3 font-data">{formatNumber(row.maeMean)}</td>
                <td className="px-3 py-3 font-data">{formatNumber(row.smapeMean * 100, 2)}%</td>
                <td className="px-3 py-3 font-data">{formatNumber(row.r2Mean)}</td>
                <td className="px-3 py-3 font-data">{row.foldEvaluations}</td>
                <td className="px-3 py-3 font-data">{row.trainTimeMean == null ? "N/D" : `${formatNumber(row.trainTimeMean, 2)} s`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </ChartCard>
  );
}

function AnomalyChart({ experiment }: { experiment: EnergyExperimentView }) {
  return (
    <ChartCard title="Anomalías estimadas con calibración MAD" subtitle="Umbrales aprendidos únicamente con folds anteriores; no equivalen a etiquetas reales">
      <div className="chart-shell"><div className="chart-min">
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={experiment.anomalies.summaries} margin={{ top: 15, right: 20, left: 10, bottom: 45 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
            <XAxis dataKey="displayName" tick={{ fontSize: 9 }} angle={-20} textAnchor="end" height={60} />
            <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="estimatedAnomalies" name="Anomalías estimadas" fill="var(--primary)" radius={[5, 5, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div></div>
    </ChartCard>
  );
}

function ErrorNotice({ message }: { message: string }) {
  return <div className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">{message}</div>;
}
