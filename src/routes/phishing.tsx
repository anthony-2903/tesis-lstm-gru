import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { AlertTriangle, BrainCircuit, CheckCircle2, Database, DownloadCloud, GitBranch, GitMerge, LockKeyhole, Play, RotateCcw, Share2, ShieldCheck, Snowflake, StopCircle, Trophy } from "lucide-react";
import { BackendState } from "@/components/BackendState";
import { ThesisProtocolPanel } from "@/components/ThesisProtocolPanel";
import { useApiData } from "@/hooks/useApiData";
import {
  fetchLatestPhishingDataJob,
  fetchLatestPhishingExperiment,
  fetchLatestPhishingJob,
  fetchLatestPhishingStacking,
  fetchLatestPhishingExternalValidation,
  fetchLatestPhishingValidationJob,
  fetchLatestPhishingDiversityAblation,
  fetchPhishingFreezeReadiness,
  fetchLatestPhishingFreeze,
  fetchPhishingDatasetStatus,
  fetchPhishingCurationStatus,
  fetchPhishingJob,
  fetchPhishingRuntimePreflight,
  fetchPhishingSequenceStatus,
  fetchPhishingValidationJob,
  preparePhishingDataset,
  preparePhishingSequences,
  runPhishingExperiment,
  runPhishingStacking,
  runPhishingExternalValidation,
  runPhishingDiversityAblation,
  createPhishingFreeze,
  initializePhishingCurationTemplate,
  cancelPhishingJob,
  resumePhishingJob,
  cancelPhishingValidationJob,
  type PhishingDataPreparationJob,
  type PhishingExperimentJob,
  type PhishingExperimentRequest,
  type PhishingExternalValidationJob,
  cancelPhishingThesisProtocol,
  pausePhishingThesisProtocol,
  fetchLatestPhishingThesisProtocol,
  resumePhishingThesisProtocol,
  runPhishingThesisProtocol,
} from "@/lib/api";

export const Route = createFileRoute("/phishing")({
  head: () => ({
    meta: [
      { title: "Datos científicos de phishing" },
      { name: "description", content: "Ingesta reproducible, auditoría y particiones por dominio registrado." },
    ],
  }),
  component: PhishingDataPage,
});

function PhishingDataPage() {
  const { data, error, isLoading, reload } = useApiData(fetchPhishingDatasetStatus);
  const [job, setJob] = useState<PhishingDataPreparationJob | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const active = job?.status === "queued" || job?.status === "running";

  useEffect(() => {
    fetchLatestPhishingDataJob().then((latest) => {
      if ("jobId" in latest) setJob(latest);
    }).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => {
      fetchLatestPhishingDataJob().then((latest) => {
        if (!("jobId" in latest)) return;
        setJob(latest);
        if (latest.status === "completed") reload();
      }).catch((caught) => setActionError(caught instanceof Error ? caught.message : "No se pudo consultar la preparación."));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [active, reload]);

  const prepare = async (force: boolean, includeAcademicSources = false) => {
    setActionError(null);
    try {
      setJob(await preparePhishingDataset(force, 10_000, includeAcademicSources));
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "No se pudo preparar el dataset.");
    }
  };

  if (isLoading) return <BackendState isLoading />;
  if (error || !data) return <BackendState error={error} onRetry={reload} />;

  return (
    <div className="dashboard-page">
      <header>
        <h1 className="text-2xl font-bold text-foreground">Phishing: base de datos científica</h1>
        <p className="mt-1 text-sm text-muted-foreground">Procedencia, etiquetas, deduplicación y particiones sin fuga por dominio registrado.</p>
      </header>

      <ThesisProtocolPanel
        title="Protocolo final de tesis · Phishing"
        description="Ejecuta dataset multifuente sin fuga por dominio → OOF 5×5 → Stacking cross-fit → validación externa → umbrales → ablación y diversidad → sellado previo al test."
        fetchLatest={fetchLatestPhishingThesisProtocol}
        start={runPhishingThesisProtocol}
        resume={resumePhishingThesisProtocol}
        pause={pausePhishingThesisProtocol}
        cancel={cancelPhishingThesisProtocol}
      />

      <section className="rounded-md border border-border bg-card p-4 shadow-sm">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-start gap-3">
            <Database className="mt-0.5 h-5 w-5 text-primary" />
            <div>
              <h2 className="text-sm font-bold text-foreground">PhishTank + Tranco + fuentes académicas</h2>
              {!data.available && <p className="mt-1 text-xs text-muted-foreground">{data.message}</p>}
              {data.available && <>
                <p className="mt-1 text-xs text-muted-foreground">{data.datasetId} · {data.silver?.rows.toLocaleString("es-ES")} URLs · integridad {data.integrityVerified ? "verificada" : "no verificada"}</p>
                <p className={`mt-1 text-xs font-bold ${data.readyForPipelinePilot ? "text-success" : "text-warning"}`}>{data.readyForPipelinePilot ? "Apto para construir y probar el pipeline" : "Aún no cumple el mínimo del piloto"}</p>
              </>}
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={() => prepare(false, true)} disabled={active} className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-xs font-bold text-primary-foreground disabled:opacity-60">
              <ShieldCheck className="h-4 w-4" />{active && job?.includeAcademicSources ? "Curando fuentes…" : "Construir base académica"}
            </button>
            <button type="button" onClick={() => prepare(false)} disabled={active} className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-xs font-bold text-primary-foreground disabled:opacity-60">
              <DownloadCloud className="h-4 w-4" />{active ? "Preparando…" : data.available ? "Verificar piloto" : "Preparar piloto"}
            </button>
            {data.available && <button type="button" onClick={() => prepare(true)} disabled={active} className="rounded-md border border-border px-3 py-2 text-xs font-bold text-foreground disabled:opacity-60">Forzar descarga piloto</button>}
          </div>
        </div>
        {job && <div className="mt-4">
          <div className="h-2 overflow-hidden rounded-full bg-primary/10"><div className="h-full rounded-full bg-primary transition-all" style={{ width: `${job.progress.percent}%` }} /></div>
          <div className="mt-2 flex justify-between gap-3 text-[10px] font-semibold text-muted-foreground"><span>{job.progress.message}</span><span>{job.progress.percent.toFixed(0)}%</span></div>
          {job.error && <p className="mt-2 text-xs font-semibold text-destructive">{job.error}</p>}
        </div>}
        {actionError && <p className="mt-3 text-xs font-semibold text-destructive">{actionError}</p>}
      </section>

      <ScientificCurationPanel />

      {data.available && <SequenceProtocolPanel />}
      {data.available && <BaseModelTrainingPanel />}
      {data.available && <StackingPanel />}
      {data.available && <ExternalValidationPanel />}
      {data.available && <DiversityAblationPanel />}
      {data.available && <FreezeGatePanel />}

      {data.available && <>
        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <StatCard title="URLs phishing" value={data.classDistribution?.positive ?? 0} />
          <StatCard title="Referencias benignas" value={data.classDistribution?.negative ?? 0} />
          <StatCard title="Fuga entre particiones" value={data.leakageAudit?.passed ? "0 dominios" : "Revisar"} success={data.leakageAudit?.passed} />
          <StatCard title="Prueba final utilizada" value={data.readiness?.testSetUsed ? "Sí" : "No"} success={!data.readiness?.testSetUsed} />
        </section>

        <section className="rounded-md border border-border bg-card p-4 shadow-sm">
          <h2 className="flex items-center gap-2 text-sm font-bold text-foreground"><ShieldCheck className="h-4 w-4 text-primary" />Particiones por dominio registrado</h2>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-border text-muted-foreground"><tr><th className="px-3 py-2">Partición</th><th className="px-3 py-2">Filas</th><th className="px-3 py-2">Phishing</th><th className="px-3 py-2">Benignas</th><th className="px-3 py-2">Dominios</th></tr></thead>
              <tbody>{["train", "validation", "test"].map((split) => {
                const item = data.splitDistribution?.[split];
                return <tr key={split} className="border-b border-border/50"><td className="px-3 py-2 font-bold capitalize">{split}</td><td className="px-3 py-2">{item?.rows.toLocaleString("es-ES") ?? "N/D"}</td><td className="px-3 py-2">{item?.positive.toLocaleString("es-ES") ?? "N/D"}</td><td className="px-3 py-2">{item?.negative.toLocaleString("es-ES") ?? "N/D"}</td><td className="px-3 py-2">{item?.groups.toLocaleString("es-ES") ?? "N/D"}</td></tr>;
              })}</tbody>
            </table>
          </div>
        </section>

        <section className={`rounded-md border p-4 ${data.readyForThesisTraining ? "border-success/40 bg-success/5" : "border-warning/40 bg-warning/5"}`}>
          <div className="flex items-start gap-3">
            {data.readyForThesisTraining ? <CheckCircle2 className="mt-0.5 h-5 w-5 text-success" /> : <AlertTriangle className="mt-0.5 h-5 w-5 text-warning" />}
            <div>
              <h2 className="text-sm font-bold text-foreground">Estado metodológico de tesis</h2>
              <p className="mt-1 text-xs text-muted-foreground">La aprobación exige diversidad de procedencia, evidencia verificable por fila y ausencia de atajos evidentes entre fuente, forma de URL y etiqueta.</p>
              {data.biasAudit && <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                <MethodAuditCard label="Fuentes negativas" value={`${data.biasAudit.sourcesPerLabel?.["0"]?.length ?? 0} / 2`} passed={(data.biasAudit.sourcesPerLabel?.["0"]?.length ?? 0) >= 2} />
                <MethodAuditCard label="Fuentes positivas" value={`${data.biasAudit.sourcesPerLabel?.["1"]?.length ?? 0} / 2`} passed={(data.biasAudit.sourcesPerLabel?.["1"]?.length ?? 0) >= 2} />
                <MethodAuditCard label="Negativos con evidencia" value={`${data.biasAudit.verifiedNegativeRows ?? 0} / ${data.biasAudit.negativeRows ?? 0}`} passed={data.biasAudit.negativeLabelsIndependentlyVerified} />
                <MethodAuditCard label="Fuente con ambas clases" value={data.biasAudit.mixedLabelSources?.length ? data.biasAudit.mixedLabelSources.join(", ") : "Ninguna"} passed={!!data.biasAudit.mixedLabelSources?.length} />
              </div>}
              {!!data.readiness?.thesisReasons.length && <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-muted-foreground">{data.readiness.thesisReasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>}
              {data.trancoPermanentListUrl && <a className="mt-3 inline-block text-xs font-bold text-primary hover:underline" href={data.trancoPermanentListUrl} target="_blank" rel="noreferrer">Ver versión permanente de Tranco</a>}
            </div>
          </div>
        </section>
      </>}
    </div>
  );
}

function ScientificCurationPanel() {
  const { data, error, isLoading, reload } = useApiData(fetchPhishingCurationStatus);
  const [isCreating, setIsCreating] = useState(false);
  const [templateResult, setTemplateResult] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const createTemplate = async () => {
    setIsCreating(true);
    setActionError(null);
    try {
      const result = await initializePhishingCurationTemplate();
      setTemplateResult(`${result.exampleManifestPath} · ${result.activation}`);
      reload();
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "No se pudo crear la plantilla de curación.");
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <section className={`rounded-md border bg-card p-4 shadow-sm ${data?.readyForScientificMerge ? "border-success/40" : "border-warning/40"}`}>
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex items-start gap-3">
          {data?.readyForScientificMerge ? <ShieldCheck className="mt-0.5 h-5 w-5 text-success" /> : <AlertTriangle className="mt-0.5 h-5 w-5 text-warning" />}
          <div>
            <h2 className="text-sm font-bold text-foreground">Curación científica y evidencia de etiquetas</h2>
            <p className="mt-1 text-xs text-muted-foreground">
              {isLoading ? "Consultando el paquete curado…" : error ? error : data?.valid
                ? `${data.sources?.length ?? 0} fuentes y ${(data.rows ?? 0).toLocaleString("es-ES")} URLs pasaron la verificación estructural.`
                : data?.message ?? "No hay información de curación."}
            </p>
            {data?.manifestPath && <p className="mt-2 break-all font-mono text-[10px] text-muted-foreground">{data.manifestPath}</p>}
          </div>
        </div>
        <button type="button" onClick={createTemplate} disabled={isCreating} className="rounded-md border border-border px-3 py-2 text-xs font-bold text-foreground disabled:opacity-60">
          {isCreating ? "Creando…" : "Crear plantilla auditable"}
        </button>
      </div>

      {!!data?.sources?.length && <div className="mt-4 overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="border-b border-border text-muted-foreground"><tr><th className="px-3 py-2">Fuente</th><th className="px-3 py-2">Etiquetas</th><th className="px-3 py-2">Filas</th><th className="px-3 py-2">Evidencia + / −</th><th className="px-3 py-2">Independiente</th></tr></thead>
          <tbody>{data.sources.map((source) => <tr key={source.sourceId} className="border-b border-border/50">
            <td className="px-3 py-2"><p className="font-bold text-foreground">{source.provider}</p><p className="text-[10px] text-muted-foreground">{source.sourceId} · {source.license}</p></td>
            <td className="px-3 py-2">{source.labels.join(", ")}</td>
            <td className="px-3 py-2">{source.rows.toLocaleString("es-ES")}</td>
            <td className="px-3 py-2">{source.verifiedPositiveRows.toLocaleString("es-ES")} / {source.positiveRows.toLocaleString("es-ES")} · {source.verifiedNegativeRows.toLocaleString("es-ES")} / {source.negativeRows.toLocaleString("es-ES")}</td>
            <td className="px-3 py-2">{source.independentAcquisition ? "Sí" : "No"}</td>
          </tr>)}</tbody>
        </table>
      </div>}

      {!!data?.requirements?.length && <ul className="mt-4 grid gap-2 text-xs text-muted-foreground md:grid-cols-2">
        {data.requirements.map((requirement) => <li key={requirement} className="rounded-md border border-border/60 p-3">{requirement}</li>)}
      </ul>}
      {!!data?.scientificReasons?.length && <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-warning">
        {data.scientificReasons.map((reason) => <li key={reason}>{reason}</li>)}
      </ul>}
      {templateResult && <p className="mt-3 break-all rounded-md border border-primary/30 bg-primary/5 p-3 text-xs text-muted-foreground">{templateResult}</p>}
      {actionError && <p className="mt-3 text-xs font-semibold text-destructive">{actionError}</p>}
    </section>
  );
}

const PHISHING_MODELS: PhishingExperimentRequest["modelIds"] = ["lstm", "gru", "brnn", "tcn", "transformer"];
const PHISHING_CANDIDATE_NAMES: Record<string, string> = {
  lstm: "LSTM",
  gru: "GRU",
  brnn: "BiRNN",
  tcn: "TCN",
  transformer: "Transformer",
  mean: "Promedio simple",
  voting: "Voting",
  weighted_mean: "Promedio ponderado",
  stacking_logistic: "Stacking logístico",
  stacking_ridge: "Stacking Ridge",
  stacking_gradient_boosting: "Stacking Gradient Boosting",
};
const FREEZE_CHECK_NAMES: Record<string, string> = {
  lineage_available: "Linaje experimental disponible",
  base_status: "Estado candidato de tesis",
  base_protocol: "Protocolo de tesis",
  complete_development: "Desarrollo completo",
  five_seeds: "Cinco semillas",
  five_models: "Cinco arquitecturas",
  five_oof_folds: "Cinco folds OOF",
  dataset_scientific_readiness: "Dataset apto científicamente",
  dataset_lineage_current: "Dataset actual coincide con OOF",
  complete_oof_training_grid: "Matriz OOF completa",
  complete_oof_tokenizers: "Tokenizadores OOF completos",
  complete_oof_coverage: "Cobertura OOF completa",
  base_test_lock: "Test bloqueado en OOF",
  validation_lineage: "Linaje de validación",
  validation_status: "Validación candidata de tesis",
  validation_protocol: "Validación externa común",
  validation_meta_fit: "Meta-ajuste sin etiquetas de validation",
  validation_test_lock: "Test bloqueado en validation",
  complete_final_refit_grid: "Reentrenamientos finales completos",
  all_candidates_frozen: "Todos los candidatos comparados",
  all_thresholds_calibrated: "Umbrales calibrados",
  validation_fitted_objects: "Objetos finales completos",
  diversity_lineage: "Linaje de ablación",
  diversity_status: "Ablación candidata de tesis",
  diversity_test_lock: "Test bloqueado en ablación",
  complete_ablation_grid: "Configuraciones de ablación completas",
  complete_ablation_metrics: "Métricas de ablación completas",
  ablation_fitted_objects: "Metamodelos de ablación completos",
  seed_stability: "Estabilidad entre semillas",
  domain_bootstrap: "Bootstrap por dominio",
  artifact_inventory_complete: "Inventario de artefactos",
  artifact_integrity: "Integridad criptográfica",
};

function BaseModelTrainingPanel() {
  const { data, error, isLoading, reload } = useApiData(fetchLatestPhishingExperiment);
  const { data: runtime, error: runtimeError } = useApiData(fetchPhishingRuntimePreflight);
  const [job, setJob] = useState<PhishingExperimentJob | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [epochs, setEpochs] = useState(1);
  const [demoRows, setDemoRows] = useState(2_000);
  const active = job?.status === "queued" || job?.status === "running";
  const resumable = job?.status === "failed" || job?.status === "cancelled" || job?.status === "interrupted";

  useEffect(() => {
    fetchLatestPhishingJob().then((latest) => {
      if ("jobId" in latest) setJob(latest);
    }).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!job?.jobId || !active) return;
    const timer = window.setInterval(() => {
      fetchPhishingJob(job.jobId).then((updated) => {
        setJob(updated);
        if (updated.status === "completed") reload();
      }).catch((caught) => setActionError(caught instanceof Error ? caught.message : "No se pudo consultar el entrenamiento."));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job?.jobId, active, reload]);

  const run = async () => {
    setActionError(null);
    try {
      setJob(await runPhishingExperiment({
        protocol: "demo",
        epochs,
        batchSize: 64,
        patience: 2,
        seeds: [42],
        modelIds: PHISHING_MODELS,
        demoMaxRows: demoRows,
      }));
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "No se pudo iniciar el entrenamiento.");
    }
  };

  const cancel = async () => {
    if (!job) return;
    try {
      setJob(await cancelPhishingJob(job.jobId));
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "No se pudo cancelar el entrenamiento.");
    }
  };

  const resume = async () => {
    if (!job) return;
    setActionError(null);
    try {
      setJob(await resumePhishingJob(job.jobId));
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "No se pudo reanudar la corrida.");
    }
  };

  const experiment = data && "run" in data ? data : null;
  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div className="flex items-start gap-3">
          <BrainCircuit className="mt-0.5 h-5 w-5 text-primary" />
          <div>
            <h2 className="text-sm font-bold text-foreground">Modelos base OOF</h2>
            <p className="mt-1 text-xs text-muted-foreground">LSTM, GRU, BiRNN, TCN y Transformer con los mismos folds y una probabilidad por URL.</p>
            {isLoading && <p className="mt-1 text-xs text-muted-foreground">Consultando resultados.</p>}
            {error && <p className="mt-1 text-xs text-destructive">{error}</p>}
          </div>
        </div>
        <div className="flex flex-wrap items-end gap-2">
          <label className="text-[10px] font-bold text-muted-foreground">FILAS DEMO<input className="mt-1 block w-24 rounded-md border border-border bg-background px-2 py-2 text-xs text-foreground" type="number" min={500} max={14000} step={500} value={demoRows} disabled={active} onChange={(event) => setDemoRows(Number(event.target.value))} /></label>
          <label className="text-[10px] font-bold text-muted-foreground">ÉPOCAS<input className="mt-1 block w-20 rounded-md border border-border bg-background px-2 py-2 text-xs text-foreground" type="number" min={1} max={20} value={epochs} disabled={active} onChange={(event) => setEpochs(Number(event.target.value))} /></label>
          <button type="button" onClick={run} disabled={active} className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-xs font-bold text-primary-foreground disabled:opacity-60"><Play className="h-4 w-4" />{active ? "Entrenando…" : "Ejecutar piloto OOF"}</button>
          {active && <button type="button" onClick={cancel} className="inline-flex items-center gap-2 rounded-md border border-destructive/40 px-3 py-2 text-xs font-bold text-destructive"><StopCircle className="h-4 w-4" />Cancelar</button>}
          {resumable && <button type="button" onClick={resume} className="inline-flex items-center gap-2 rounded-md border border-primary/40 px-3 py-2 text-xs font-bold text-primary"><RotateCcw className="h-4 w-4" />Reanudar corrida</button>}
        </div>
      </div>
      {runtime && <div className="mt-4 grid gap-2 rounded-md border border-border bg-muted/20 p-3 text-xs md:grid-cols-3">
        <div><span className="font-bold">Preflight tesis:</span> {runtime.ready ? "aprobado" : "bloqueado"}</div>
        <div><span className="font-bold">GPU visible:</span> {runtime.runtime.gpuCount > 0 ? runtime.runtime.gpuCount : "no"}</div>
        <div><span className="font-bold">Disco libre:</span> {(runtime.storage.freeBytes / 1024 ** 3).toFixed(1)} GiB</div>
        {runtime.warnings.length > 0 && <p className="text-muted-foreground md:col-span-3">{runtime.warnings.join(" ")}</p>}
      </div>}
      {runtimeError && <p className="mt-3 text-xs font-semibold text-destructive">No se pudo ejecutar el preflight: {runtimeError}</p>}
      {job && <div className="mt-4">
        <div className="h-2 overflow-hidden rounded-full bg-primary/10"><div className="h-full rounded-full bg-primary transition-all" style={{ width: `${job.progress.percent}%` }} /></div>
        <div className="mt-2 flex justify-between gap-3 text-[10px] font-semibold text-muted-foreground"><span>{job.progress.message}</span><span>{job.progress.completedUnits}/{job.progress.totalUnits} · {job.progress.percent.toFixed(0)}%</span></div>
        {job.progress.resumedUnits > 0 && <p className="mt-1 text-[10px] font-semibold text-primary">{job.progress.resumedUnits} unidades recuperadas desde checkpoints verificados.</p>}
        {job.error && <p className="mt-2 text-xs font-semibold text-destructive">{job.error}</p>}
      </div>}
      {actionError && <p className="mt-3 text-xs font-semibold text-destructive">{actionError}</p>}
      {experiment && <>
        <div className="mt-4 rounded-md border border-warning/30 bg-warning/5 p-3 text-xs text-muted-foreground">{experiment.methodology.warning} Test utilizado: <strong>{experiment.methodology.testSetUsed ? "sí" : "no"}</strong>. Stacking preparado: <strong>{experiment.stacking.ready ? "sí" : "no"}</strong>.</div>
        <div className="mt-4 overflow-x-auto rounded-md border border-border">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-border text-muted-foreground"><tr><th className="px-3 py-2">Modelo</th><th className="px-3 py-2">PR-AUC</th><th className="px-3 py-2">ROC-AUC</th><th className="px-3 py-2">F1*</th><th className="px-3 py-2">MCC*</th><th className="px-3 py-2">Recall*</th><th className="px-3 py-2">FPR*</th><th className="px-3 py-2">Tiempo</th></tr></thead>
            <tbody>{experiment.comparison.map((row, index) => <tr key={row.modelId} className="border-b border-border/50"><td className="px-3 py-2 font-bold">{index === 0 && <Trophy className="mr-1 inline h-3.5 w-3.5 text-warning" />}{row.displayName}</td><td className="px-3 py-2">{row.prAucMean.toFixed(4)} ± {row.prAucStd.toFixed(4)}</td><td className="px-3 py-2">{row.rocAucMean.toFixed(4)}</td><td className="px-3 py-2">{row.f1Mean.toFixed(4)}</td><td className="px-3 py-2">{row.mccMean.toFixed(4)}</td><td className="px-3 py-2">{row.recallMean.toFixed(4)}</td><td className="px-3 py-2">{row.falsePositiveRateMean.toFixed(4)}</td><td className="px-3 py-2">{row.trainTimeSecondsMean.toFixed(1)} s</td></tr>)}</tbody>
          </table>
          <p className="p-3 text-[10px] text-muted-foreground">* Métricas exploratorias con umbral fijo 0.5. El umbral definitivo se elegirá posteriormente usando validación externa.</p>
        </div>
      </>}
    </section>
  );
}

function StackingPanel() {
  const { data, error, isLoading, reload } = useApiData(fetchLatestPhishingStacking);
  const [running, setRunning] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const result = data?.available ? data : null;

  const run = async () => {
    setRunning(true);
    setActionError(null);
    try {
      await runPhishingStacking();
      reload();
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "No se pudo ejecutar Stacking.");
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div className="flex items-start gap-3">
          <GitMerge className="mt-0.5 h-5 w-5 text-primary" />
          <div>
            <h2 className="text-sm font-bold text-foreground">Ensambles y metamodelo de Stacking</h2>
            <p className="mt-1 text-xs text-muted-foreground">Compara los cinco modelos base contra promedio, Voting, promedio ponderado y tres metamodelos usando sus probabilidades OOF.</p>
            {isLoading && <p className="mt-1 text-xs text-muted-foreground">Consultando el diagnóstico OOF.</p>}
            {error && <p className="mt-1 text-xs text-destructive">{error}</p>}
            {data && !data.available && <p className="mt-1 text-xs text-muted-foreground">{data.message}</p>}
          </div>
        </div>
        <button type="button" onClick={run} disabled={running} className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-xs font-bold text-primary-foreground disabled:opacity-60">
          <GitMerge className="h-4 w-4" />{running ? "Calculando…" : result ? "Recalcular Stacking" : "Entrenar Stacking"}
        </button>
      </div>
      {actionError && <p className="mt-3 text-xs font-semibold text-destructive">{actionError}</p>}
      {result && <>
        <div className="mt-4 grid gap-3 lg:grid-cols-3">
          <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">COBERTURA META-OOF</p><p className="mt-1 text-sm font-bold text-foreground">{result.validation.coverageRows.toLocaleString("es-ES")} predicciones</p></div>
          <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">MEJOR STACKING EXPLORATORIO</p><p className="mt-1 text-sm font-bold text-foreground">{PHISHING_CANDIDATE_NAMES[result.recommendation.leadingStackingCandidateId ?? ""] ?? "N/D"}</p></div>
          <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">LÍDER OOF GENERAL</p><p className="mt-1 text-sm font-bold text-foreground">{PHISHING_CANDIDATE_NAMES[result.recommendation.overallOofLeaderId ?? ""] ?? "N/D"}</p></div>
        </div>
        <div className="mt-4 rounded-md border border-warning/30 bg-warning/5 p-3 text-xs text-muted-foreground">
          Diagnóstico exploratorio: no es CV anidado estricto y aún no usa validación externa. Test final bloqueado: <strong>{result.validation.testSetLocked && !result.validation.testSetUsed ? "sí" : "no"}</strong>. La selección definitiva no se hará con esta tabla.
        </div>
        <div className="mt-4 overflow-x-auto rounded-md border border-border">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-border text-muted-foreground"><tr><th className="px-3 py-2">Candidato</th><th className="px-3 py-2">Familia</th><th className="px-3 py-2">PR-AUC</th><th className="px-3 py-2">ROC-AUC</th><th className="px-3 py-2">F1*</th><th className="px-3 py-2">MCC*</th><th className="px-3 py-2">Recall*</th><th className="px-3 py-2">FPR*</th></tr></thead>
            <tbody>{result.comparison.map((row, index) => <tr key={row.candidateId} className="border-b border-border/50"><td className="px-3 py-2 font-bold">{index === 0 && <Trophy className="mr-1 inline h-3.5 w-3.5 text-warning" />}{PHISHING_CANDIDATE_NAMES[row.candidateId] ?? row.candidateId}</td><td className="px-3 py-2 capitalize">{row.family}</td><td className="px-3 py-2">{row.prAucMean.toFixed(4)} ± {row.prAucStd.toFixed(4)}</td><td className="px-3 py-2">{row.rocAucMean.toFixed(4)}</td><td className="px-3 py-2">{row.f1Mean.toFixed(4)}</td><td className="px-3 py-2">{row.mccMean.toFixed(4)}</td><td className="px-3 py-2">{row.recallMean.toFixed(4)}</td><td className="px-3 py-2">{row.falsePositiveRateMean.toFixed(4)}</td></tr>)}</tbody>
          </table>
          <p className="p-3 text-[10px] text-muted-foreground">* Umbral fijo 0.5, únicamente exploratorio. El umbral y el candidato final se seleccionarán con validación externa.</p>
        </div>
      </>}
    </section>
  );
}

function ExternalValidationPanel() {
  const { data, error, isLoading, reload } = useApiData(fetchLatestPhishingExternalValidation);
  const [job, setJob] = useState<PhishingExternalValidationJob | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const active = job?.status === "queued" || job?.status === "running";
  const result = data?.available ? data : null;

  useEffect(() => {
    fetchLatestPhishingValidationJob().then((latest) => {
      if ("jobId" in latest) setJob(latest);
    }).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!job?.jobId || !active) return;
    const timer = window.setInterval(() => {
      fetchPhishingValidationJob(job.jobId).then((updated) => {
        setJob(updated);
        if (updated.status === "completed") reload();
      }).catch((caught) => setActionError(caught instanceof Error ? caught.message : "No se pudo consultar la validación."));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job?.jobId, active, reload]);

  const run = async () => {
    setActionError(null);
    try {
      setJob(await runPhishingExternalValidation());
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "No se pudo iniciar la validación externa.");
    }
  };

  const cancel = async () => {
    if (!job) return;
    try {
      setJob(await cancelPhishingValidationJob(job.jobId));
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "No se pudo cancelar la validación.");
    }
  };

  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div className="flex items-start gap-3">
          <ShieldCheck className="mt-0.5 h-5 w-5 text-primary" />
          <div>
            <h2 className="text-sm font-bold text-foreground">Selección sobre validación externa</h2>
            <p className="mt-1 text-xs text-muted-foreground">Reentrena las redes con desarrollo, ajusta Stacking con OOF y calibra el umbral mediante MCC sobre validation. No evalúa test.</p>
            {isLoading && <p className="mt-1 text-xs text-muted-foreground">Consultando la selección externa.</p>}
            {error && <p className="mt-1 text-xs text-destructive">{error}</p>}
            {data && !data.available && <p className="mt-1 text-xs text-muted-foreground">{data.message}</p>}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <button type="button" onClick={run} disabled={active} className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-xs font-bold text-primary-foreground disabled:opacity-60"><Play className="h-4 w-4" />{active ? "Validando…" : result ? "Repetir validación" : "Ejecutar validación"}</button>
          {active && <button type="button" onClick={cancel} className="inline-flex items-center gap-2 rounded-md border border-destructive/40 px-3 py-2 text-xs font-bold text-destructive"><StopCircle className="h-4 w-4" />Cancelar</button>}
        </div>
      </div>
      {job && <div className="mt-4">
        <div className="h-2 overflow-hidden rounded-full bg-primary/10"><div className="h-full rounded-full bg-primary transition-all" style={{ width: `${job.progress.percent}%` }} /></div>
        <div className="mt-2 flex justify-between gap-3 text-[10px] font-semibold text-muted-foreground"><span>{job.progress.message}</span><span>{job.progress.completedUnits}/{job.progress.totalUnits} · {job.progress.percent.toFixed(0)}%</span></div>
        {job.error && <p className="mt-2 text-xs font-semibold text-destructive">{job.error}</p>}
      </div>}
      {actionError && <p className="mt-3 text-xs font-semibold text-destructive">{actionError}</p>}
      {result && <>
        <div className="mt-4 grid gap-3 lg:grid-cols-5">
          <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">DESARROLLO</p><p className="mt-1 text-sm font-bold text-foreground">{result.dataset.developmentRows.toLocaleString("es-ES")} URLs</p></div>
          <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">VALIDATION</p><p className="mt-1 text-sm font-bold text-foreground">{result.dataset.validationRows.toLocaleString("es-ES")} URLs</p></div>
          <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">CANDIDATO SELECCIONADO</p><p className="mt-1 text-sm font-bold text-foreground">{PHISHING_CANDIDATE_NAMES[result.selection.winnerCandidateId] ?? result.selection.winnerCandidateId}</p></div>
          <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">MEJOR STACKING</p><p className="mt-1 text-sm font-bold text-foreground">{PHISHING_CANDIDATE_NAMES[result.selection.leadingStackingCandidateId ?? ""] ?? "N/D"}</p></div>
          <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">UMBRAL SELECCIONADO</p><p className="mt-1 text-sm font-bold text-foreground">{result.selection.winnerThreshold.toFixed(4)}</p></div>
        </div>
        {result.dataset.demoSubset && <div className="mt-4 rounded-md border border-warning/30 bg-warning/5 p-3 text-xs text-muted-foreground">Corrida demostrativa: el desarrollo usa un subconjunto y persiste el riesgo de acoplamiento entre fuente y etiqueta. No debe citarse como resultado final de tesis.</div>}
        <div className="mt-4 rounded-md border border-success/30 bg-success/5 p-3 text-xs text-muted-foreground">
          Todos los candidatos usaron las mismas URLs de validation. Test bloqueado y sin codificar: <strong>{result.validation.testSetLocked && !result.validation.testFeaturesEncoded && !result.validation.testSetUsed ? "sí" : "no"}</strong>. Estos valores sirven para selección; no son el rendimiento final de tesis.
        </div>
        <div className="mt-4 overflow-x-auto rounded-md border border-border">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-border text-muted-foreground"><tr><th className="px-3 py-2">Candidato</th><th className="px-3 py-2">Familia</th><th className="px-3 py-2">Umbral</th><th className="px-3 py-2">PR-AUC</th><th className="px-3 py-2">F1</th><th className="px-3 py-2">MCC</th><th className="px-3 py-2">Recall</th><th className="px-3 py-2">FPR</th><th className="px-3 py-2">TN/FP/FN/TP</th></tr></thead>
            <tbody>{result.comparison.map((row, index) => <tr key={row.candidateId} className="border-b border-border/50"><td className="px-3 py-2 font-bold">{index === 0 && <Trophy className="mr-1 inline h-3.5 w-3.5 text-warning" />}{PHISHING_CANDIDATE_NAMES[row.candidateId] ?? row.candidateId}</td><td className="px-3 py-2 capitalize">{row.family}</td><td className="px-3 py-2">{row.calibratedThreshold.toFixed(4)}</td><td className="px-3 py-2">{row.prAuc.toFixed(4)}</td><td className="px-3 py-2">{row.f1.toFixed(4)}</td><td className="px-3 py-2">{row.mcc.toFixed(4)}</td><td className="px-3 py-2">{row.recall.toFixed(4)}</td><td className="px-3 py-2">{row.falsePositiveRate.toFixed(4)}</td><td className="px-3 py-2">{row.confusionMatrix.trueNegative}/{row.confusionMatrix.falsePositive}/{row.confusionMatrix.falseNegative}/{row.confusionMatrix.truePositive}</td></tr>)}</tbody>
          </table>
        </div>
      </>}
    </section>
  );
}

function DiversityAblationPanel() {
  const { data, error, isLoading, reload } = useApiData(fetchLatestPhishingDiversityAblation);
  const [running, setRunning] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const result = data?.available ? data : null;
  const referenceRows = result?.ablation.metrics.filter((row) => row.metaModelId === result.ablation.referenceMetaModelId) ?? [];

  const run = async () => {
    setRunning(true);
    setActionError(null);
    try {
      await runPhishingDiversityAblation();
      reload();
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "No se pudo ejecutar diversidad y ablación.");
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div className="flex items-start gap-3">
          <Share2 className="mt-0.5 h-5 w-5 text-primary" />
          <div>
            <h2 className="text-sm font-bold text-foreground">Diversidad y ablación del Stacking</h2>
            <p className="mt-1 text-xs text-muted-foreground">Mide complementariedad y repite el metamodelo retirando una red por vez sobre las mismas URLs de validation.</p>
            {isLoading && <p className="mt-1 text-xs text-muted-foreground">Consultando el estudio.</p>}
            {error && <p className="mt-1 text-xs text-destructive">{error}</p>}
            {data && !data.available && <p className="mt-1 text-xs text-muted-foreground">{data.message}</p>}
          </div>
        </div>
        <button type="button" onClick={run} disabled={running} className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-xs font-bold text-primary-foreground disabled:opacity-60"><Share2 className="h-4 w-4" />{running ? "Analizando…" : result ? "Recalcular estudio" : "Ejecutar estudio"}</button>
      </div>
      {actionError && <p className="mt-3 text-xs font-semibold text-destructive">{actionError}</p>}
      {result && <>
        <div className="mt-4 grid gap-3 lg:grid-cols-4">
          <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">DESACUERDO PROMEDIO</p><p className="mt-1 text-sm font-bold text-foreground">{(result.diversity.meanPairwiseDisagreementRate * 100).toFixed(2)}%</p></div>
          <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">PAR MÁS COMPLEMENTARIO</p><p className="mt-1 text-sm font-bold text-foreground">{result.diversity.mostComplementaryPair ? `${PHISHING_CANDIDATE_NAMES[result.diversity.mostComplementaryPair.modelA]} + ${PHISHING_CANDIDATE_NAMES[result.diversity.mostComplementaryPair.modelB]}` : "N/D"}</p></div>
          <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">CONFIGURACIÓN RECOMENDADA</p><p className="mt-1 text-sm font-bold text-foreground">{result.recommendation.recommendedAblationId === "full" ? "Cinco modelos" : `Sin ${PHISHING_CANDIDATE_NAMES[result.recommendation.removedModelId ?? ""]}`}</p></div>
          <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">SEMILLAS</p><p className="mt-1 text-sm font-bold text-foreground">{result.stability.seedCount} {result.stability.seedLevelInferenceAvailable ? "(estabilidad disponible)" : "(piloto)"}</p></div>
        </div>
        <div className="mt-4 rounded-md border border-warning/30 bg-warning/5 p-3 text-xs text-muted-foreground">Intervalos bootstrap agrupados por dominio sobre validation. Test bloqueado y sin codificar: <strong>{result.validation.testSetLocked && !result.validation.testFeaturesEncoded && !result.validation.testSetUsed ? "sí" : "no"}</strong>. Con una semilla todavía no existe inferencia de estabilidad entre entrenamientos.</div>

        <h3 className="mt-5 text-xs font-bold text-foreground">Diversidad por pareja</h3>
        <div className="mt-2 overflow-x-auto rounded-md border border-border">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-border text-muted-foreground"><tr><th className="px-3 py-2">Pareja</th><th className="px-3 py-2">Corr. prob.</th><th className="px-3 py-2">Corr. residuo</th><th className="px-3 py-2">Desacuerdo</th><th className="px-3 py-2">Doble fallo</th><th className="px-3 py-2">Jaccard FP</th><th className="px-3 py-2">Jaccard FN</th></tr></thead>
            <tbody>{result.diversity.pairs.map((row) => <tr key={`${row.modelA}-${row.modelB}`} className="border-b border-border/50"><td className="px-3 py-2 font-bold">{PHISHING_CANDIDATE_NAMES[row.modelA]} + {PHISHING_CANDIDATE_NAMES[row.modelB]}</td><td className="px-3 py-2">{row.probabilityPearson?.toFixed(4) ?? "N/D"}</td><td className="px-3 py-2">{row.residualPearson?.toFixed(4) ?? "N/D"}</td><td className="px-3 py-2">{(row.hardPredictionDisagreementRate * 100).toFixed(2)}%</td><td className="px-3 py-2">{(row.doubleFaultRate * 100).toFixed(2)}%</td><td className="px-3 py-2">{row.falsePositiveJaccard?.toFixed(4) ?? "N/D"}</td><td className="px-3 py-2">{row.falseNegativeJaccard?.toFixed(4) ?? "N/D"}</td></tr>)}</tbody>
          </table>
        </div>

        <h3 className="mt-5 text-xs font-bold text-foreground">Ablación de {PHISHING_CANDIDATE_NAMES[result.ablation.referenceMetaModelId] ?? result.ablation.referenceMetaModelId}</h3>
        <div className="mt-2 overflow-x-auto rounded-md border border-border">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-border text-muted-foreground"><tr><th className="px-3 py-2">Configuración</th><th className="px-3 py-2">PR-AUC</th><th className="px-3 py-2">Δ PR-AUC</th><th className="px-3 py-2">F1</th><th className="px-3 py-2">MCC</th><th className="px-3 py-2">FPR</th><th className="px-3 py-2">Umbral</th></tr></thead>
            <tbody>{referenceRows.map((row) => <tr key={row.ablationId} className="border-b border-border/50"><td className="px-3 py-2 font-bold">{row.removedModelId ? `Sin ${PHISHING_CANDIDATE_NAMES[row.removedModelId]}` : "Completo"}</td><td className="px-3 py-2">{row.prAuc.toFixed(4)}</td><td className={`px-3 py-2 ${row.prAucDropVsFull > 0 ? "text-success" : row.prAucDropVsFull < 0 ? "text-warning" : ""}`}>{row.prAucDropVsFull.toFixed(5)}</td><td className="px-3 py-2">{row.f1.toFixed(4)}</td><td className="px-3 py-2">{row.mcc.toFixed(4)}</td><td className="px-3 py-2">{row.falsePositiveRate.toFixed(4)}</td><td className="px-3 py-2">{row.calibratedThreshold.toFixed(4)}</td></tr>)}</tbody>
          </table>
        </div>

        <h3 className="mt-5 text-xs font-bold text-foreground">Contribución al metamodelo</h3>
        <div className="mt-2 overflow-x-auto rounded-md border border-border">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-border text-muted-foreground"><tr><th className="px-3 py-2">Modelo retirado</th><th className="px-3 py-2">Caída PR-AUC</th><th className="px-3 py-2">IC 95%</th><th className="px-3 py-2">Caída MCC</th><th className="px-3 py-2">Interpretación</th></tr></thead>
            <tbody>{result.ablation.contribution.map((row) => <tr key={row.modelId} className="border-b border-border/50"><td className="px-3 py-2 font-bold">{PHISHING_CANDIDATE_NAMES[row.modelId]}</td><td className="px-3 py-2">{row.prAucDropWhenRemoved.toFixed(5)}</td><td className="px-3 py-2">[{row.prAucDropCi95.lower.toFixed(5)}, {row.prAucDropCi95.upper.toFixed(5)}]</td><td className="px-3 py-2">{row.mccDropWhenRemoved.toFixed(5)}</td><td className="px-3 py-2">{row.interpretation === "positive_contribution" ? "Contribución positiva" : "Posible redundancia"}</td></tr>)}</tbody>
          </table>
        </div>
      </>}
    </section>
  );
}

function FreezeGatePanel() {
  const { data: readiness, error, isLoading, reload: reloadReadiness } = useApiData(fetchPhishingFreezeReadiness);
  const { data: latestFreeze, reload: reloadFreeze } = useApiData(fetchLatestPhishingFreeze);
  const [creating, setCreating] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const frozen = latestFreeze?.available ? latestFreeze : null;
  const visibleChecks = readiness?.ready ? readiness.checks : readiness?.failedChecks ?? [];

  const create = async () => {
    setCreating(true);
    setActionError(null);
    try {
      await createPhishingFreeze();
      reloadReadiness();
      reloadFreeze();
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "No se pudo congelar el pipeline.");
    } finally {
      setCreating(false);
    }
  };

  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div className="flex items-start gap-3">
          <Snowflake className="mt-0.5 h-5 w-5 text-primary" />
          <div>
            <h2 className="text-sm font-bold text-foreground">Puerta de congelación experimental</h2>
            <p className="mt-1 text-xs text-muted-foreground">Exige protocolo completo, dataset científicamente apto, cinco semillas, integridad total y test sin utilizar.</p>
            {isLoading && <p className="mt-1 text-xs text-muted-foreground">Auditando el linaje experimental.</p>}
            {error && <p className="mt-1 text-xs text-destructive">{error}</p>}
          </div>
        </div>
        <button type="button" onClick={create} disabled={creating || !readiness?.ready || Boolean(frozen)} className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-xs font-bold text-primary-foreground disabled:opacity-60"><Snowflake className="h-4 w-4" />{creating ? "Verificando y sellando…" : frozen ? "Pipeline congelado" : "Crear paquete congelado"}</button>
      </div>

      {readiness && <div className={`mt-4 rounded-md border p-3 text-xs ${readiness.ready ? "border-success/30 bg-success/5" : "border-warning/30 bg-warning/5"}`}>
        <p className="font-bold text-foreground">{readiness.ready ? "Puerta estructural superada" : "Puerta bloqueada correctamente"}</p>
        <p className="mt-1 text-muted-foreground">{readiness.ready ? "Al crear el paquete se verificarán nuevamente todos los hashes antes del sello." : `${readiness.failedChecks.length} controles pendientes. La corrida actual no puede autorizar el test.`}</p>
      </div>}
      {actionError && <p className="mt-3 text-xs font-semibold text-destructive">{actionError}</p>}

      {readiness && <div className="mt-4 grid gap-3 lg:grid-cols-4">
        <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">SEMILLAS EXIGIDAS</p><p className="mt-1 text-sm font-bold text-foreground">{readiness.requirements.minimumSeeds}</p></div>
        <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">FOLDS EXIGIDOS</p><p className="mt-1 text-sm font-bold text-foreground">{readiness.requirements.minimumFolds}</p></div>
        <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">MODELOS EXIGIDOS</p><p className="mt-1 text-sm font-bold text-foreground">{readiness.requirements.requiredModels.length}</p></div>
        <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">TEST</p><p className="mt-1 text-sm font-bold text-success">Bloqueado</p></div>
      </div>}

      {!!visibleChecks.length && <div className="mt-4 overflow-x-auto rounded-md border border-border">
        <table className="w-full text-left text-xs">
          <thead className="border-b border-border text-muted-foreground"><tr><th className="px-3 py-2">Control</th><th className="px-3 py-2">Estado</th><th className="px-3 py-2">Esperado</th><th className="px-3 py-2">Observado</th></tr></thead>
          <tbody>{visibleChecks.map((check) => <tr key={check.checkId} className="border-b border-border/50"><td className="px-3 py-2 font-bold">{FREEZE_CHECK_NAMES[check.checkId] ?? check.checkId}</td><td className={`px-3 py-2 font-bold ${check.passed ? "text-success" : "text-warning"}`}>{check.passed ? "Cumple" : "Pendiente"}</td><td className="max-w-xs px-3 py-2">{formatAuditValue(check.expected)}</td><td className="max-w-xs px-3 py-2">{formatAuditValue(check.observed)}</td></tr>)}</tbody>
        </table>
      </div>}

      {frozen && <div className="mt-4 rounded-md border border-success/30 bg-success/5 p-3 text-xs text-muted-foreground">
        <p className="font-bold text-foreground">{frozen.freezeId}</p>
        <p className="mt-1">Sello verificado: <strong>{frozen.sealVerified ? "sí" : "no"}</strong> · artefactos: <strong>{frozen.gate.artifactCount}</strong> · autorización de test: <strong>{frozen.testAuthorization.granted ? "concedida" : "pendiente"}</strong>.</p>
      </div>}
    </section>
  );
}

function formatAuditValue(value: unknown) {
  if (value === null || value === undefined) return "N/D";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  const serialized = JSON.stringify(value);
  return serialized.length > 180 ? `${serialized.slice(0, 177)}…` : serialized;
}

function SequenceProtocolPanel() {
  const { data, error, isLoading, reload } = useApiData(fetchPhishingSequenceStatus);
  const [preparing, setPreparing] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const prepare = async () => {
    setPreparing(true);
    setActionError(null);
    try {
      await preparePhishingSequences();
      reload();
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "No se pudo preparar el protocolo OOF.");
    } finally {
      setPreparing(false);
    }
  };

  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex items-start gap-3">
          <GitBranch className="mt-0.5 h-5 w-5 text-primary" />
          <div>
            <h2 className="text-sm font-bold text-foreground">Secuencias y validación OOF</h2>
            {isLoading && <p className="mt-1 text-xs text-muted-foreground">Verificando artefactos del protocolo.</p>}
            {error && <p className="mt-1 text-xs text-destructive">{error}</p>}
            {data && !data.available && <p className="mt-1 text-xs text-muted-foreground">{data.message}</p>}
            {data?.available && <>
              <p className="mt-1 text-xs text-muted-foreground">{data.oof?.folds} folds · {data.oof?.coverageRows.toLocaleString("es-ES")} filas OOF · vocabulario {data.tokenization?.outerTokenizer.vocabularySize} · longitud {data.tokenization?.outerTokenizer.maxLength}</p>
              <p className={`mt-1 text-xs font-bold ${data.readyForBaseModelTraining ? "text-success" : "text-warning"}`}>{data.readyForBaseModelTraining ? "Listo para entrenar los modelos base" : data.message || data.readiness?.reasons.join(" ")}</p>
            </>}
          </div>
        </div>
        <button type="button" onClick={prepare} disabled={preparing} className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-xs font-bold text-primary-foreground disabled:opacity-60">
          <GitBranch className="h-4 w-4" />{preparing ? "Preparando…" : data?.available ? "Regenerar protocolo" : "Preparar 5 folds OOF"}
        </button>
      </div>
      {actionError && <p className="mt-3 text-xs font-semibold text-destructive">{actionError}</p>}
      {data?.available && <div className="mt-4 grid gap-4 xl:grid-cols-[1fr_2fr]">
        <div className="rounded-md border border-success/30 bg-success/5 p-3">
          <p className="flex items-center gap-2 text-xs font-bold text-foreground"><LockKeyhole className="h-4 w-4 text-success" />Test bloqueado</p>
          <p className="mt-2 text-xs text-muted-foreground">{data.testLock?.rows.toLocaleString("es-ES")} filas y {data.testLock?.groups.toLocaleString("es-ES")} dominios. No se usó para vocabulario, longitud, OOF ni umbral.</p>
        </div>
        <div className="overflow-x-auto rounded-md border border-border">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-border text-muted-foreground"><tr><th className="px-3 py-2">Fold</th><th className="px-3 py-2">Ajuste</th><th className="px-3 py-2">Holdout</th><th className="px-3 py-2">Negativas</th><th className="px-3 py-2">Positivas</th><th className="px-3 py-2">Solapamiento</th></tr></thead>
            <tbody>{data.oof?.foldsAudit.map((fold) => <tr key={fold.fold} className="border-b border-border/50"><td className="px-3 py-2 font-bold">{fold.fold + 1}</td><td className="px-3 py-2">{fold.fitRows.toLocaleString("es-ES")}</td><td className="px-3 py-2">{fold.holdoutRows.toLocaleString("es-ES")}</td><td className="px-3 py-2">{fold.holdoutNegative.toLocaleString("es-ES")}</td><td className="px-3 py-2">{fold.holdoutPositive.toLocaleString("es-ES")}</td><td className={`px-3 py-2 font-bold ${fold.groupOverlap === 0 ? "text-success" : "text-destructive"}`}>{fold.groupOverlap}</td></tr>)}</tbody>
          </table>
        </div>
      </div>}
    </section>
  );
}

function MethodAuditCard({ label, value, passed }: { label: string; value: string; passed: boolean }) {
  return <div className={`rounded-md border p-3 ${passed ? "border-success/30 bg-success/5" : "border-warning/30 bg-warning/5"}`}>
    <p className="text-[10px] font-bold uppercase tracking-wide text-muted-foreground">{label}</p>
    <p className={`mt-1 text-xs font-bold ${passed ? "text-success" : "text-warning"}`}>{value}</p>
  </div>;
}

function StatCard({ title, value, success }: { title: string; value: number | string; success?: boolean }) {
  return <article className="rounded-md border border-border bg-card p-4 shadow-sm"><p className="text-xs font-semibold text-muted-foreground">{title}</p><p className={`mt-2 text-xl font-bold ${success === undefined ? "text-foreground" : success ? "text-success" : "text-warning"}`}>{typeof value === "number" ? value.toLocaleString("es-ES") : value}</p></article>;
}
