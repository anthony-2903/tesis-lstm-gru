import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { AlertTriangle, CheckCircle2, Database, Landmark, LockKeyhole, Play, Snowflake, TrendingUp } from "lucide-react";
import { BackendState } from "@/components/BackendState";
import { ThesisProtocolPanel } from "@/components/ThesisProtocolPanel";
import { useApiData } from "@/hooks/useApiData";
import {
  cancelFinanceThesisProtocol,
  createFinanceMarketFreeze,
  fetchFinanceMarketDataStatus,
  fetchFinanceMarketFreezeReadiness,
  fetchLatestFinanceMarketExperiment,
  fetchLatestFinanceMarketFreeze,
  fetchLatestFinanceMarketRevalidation,
  fetchLatestFinanceThesisProtocol,
  pauseFinanceThesisProtocol,
  prepareFinanceMarketData,
  resumeFinanceThesisProtocol,
  runFinanceThesisProtocol,
} from "@/lib/api";

export const Route = createFileRoute("/finance")({
  head: () => ({
    meta: [
      { title: "Finanzas - Índices Soberanos MEF" },
      { name: "description", content: "Pronóstico financiero y anomalías residuales con datos públicos del MEF." },
    ],
  }),
  component: FinancePage,
});

const MODEL_NAMES: Record<string, string> = {
  lstm: "LSTM",
  gru: "GRU",
  brnn: "BiRNN",
  tcn: "TCN",
  transformer: "Transformer",
  mean: "Promedio",
  weighted_mean: "Promedio ponderado",
  stacking_gradient_boosting: "Stacking",
  optimized_stacking: "Stacking optimizado",
};

function FinancePage() {
  const { data, error, isLoading, reload } = useApiData(fetchFinanceMarketDataStatus);
  const { data: experiment, reload: reloadExperiment } = useApiData(fetchLatestFinanceMarketExperiment);
  const { data: revalidation, reload: reloadRevalidation } = useApiData(fetchLatestFinanceMarketRevalidation);
  const { data: readiness, reload: reloadReadiness } = useApiData(fetchFinanceMarketFreezeReadiness);
  const { data: frozen, reload: reloadFrozen } = useApiData(fetchLatestFinanceMarketFreeze);
  const [preparing, setPreparing] = useState(false);
  const [freezing, setFreezing] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const refreshEvidence = () => {
    reload();
    reloadExperiment();
    reloadRevalidation();
    reloadReadiness();
    reloadFrozen();
  };

  const prepare = async () => {
    setPreparing(true);
    setActionError(null);
    try {
      await prepareFinanceMarketData(false);
      refreshEvidence();
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "No se pudo preparar la fuente MEF.");
    } finally {
      setPreparing(false);
    }
  };

  const freeze = async () => {
    setFreezing(true);
    setActionError(null);
    try {
      await createFinanceMarketFreeze();
      refreshEvidence();
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "La evidencia aún no supera la puerta de sellado.");
    } finally {
      setFreezing(false);
    }
  };

  if (isLoading) return <BackendState isLoading />;
  if (error || !data) return <BackendState error={error} onRetry={reload} />;

  const completedExperiment = experiment?.available ? experiment : null;
  const completedRevalidation = revalidation?.available ? revalidation : null;
  const split = data.audit?.chronologicalSplit;

  return (
    <div className="dashboard-page space-y-6">
      <header className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex items-start gap-3">
          <Landmark className="mt-1 h-6 w-6 text-primary" />
          <div>
            <h1 className="text-2xl font-bold text-foreground">Finanzas: Índices Soberanos del MEF</h1>
            <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
              Pronóstico del log-rendimiento nominal de la siguiente sesión y detección de anomalías por residuo. Los datos no contienen etiquetas de fraude; por rigor científico, la aplicación no presenta estos eventos como fraude confirmado.
            </p>
          </div>
        </div>
        <button type="button" onClick={prepare} disabled={preparing} className="inline-flex items-center justify-center gap-2 rounded-md bg-primary px-4 py-2 text-xs font-bold text-primary-foreground disabled:opacity-60">
          <Play className="h-4 w-4" />{preparing ? "Verificando MEF..." : data.available ? "Verificar snapshot" : "Preparar datos MEF"}
        </button>
      </header>

      <ThesisProtocolPanel
        title="Protocolo final de tesis — Finanzas"
        description="Encadena snapshot público MEF → auditoría → OOF 5×5 → Stacking → ablación → comparación independiente → bootstrap/XAI → sellado. El 15% final permanece bloqueado."
        fetchLatest={fetchLatestFinanceThesisProtocol}
        start={runFinanceThesisProtocol}
        resume={resumeFinanceThesisProtocol}
        pause={pauseFinanceThesisProtocol}
        cancel={cancelFinanceThesisProtocol}
      />

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
        <StatusCard icon={data.readyForThesisTraining ? CheckCircle2 : AlertTriangle} title="Fuente MEF" value={data.readyForThesisTraining ? "Auditada" : "Pendiente"} />
        <StatusCard icon={Database} title="Observaciones" value={(data.silver?.rows ?? 0).toLocaleString("es-PE")} />
        <StatusCard icon={TrendingUp} title="Objetivo" value="Retorno t+1" />
        <StatusCard icon={LockKeyhole} title="Test final" value={!data.testPolicy.testSetUsed ? "Bloqueado" : "Revisar"} />
        <StatusCard icon={Snowflake} title="Sellado" value={frozen?.freezeId ? "Completo" : "Pendiente"} />
      </div>

      {actionError && <p className="rounded-md border border-destructive/30 bg-destructive/5 p-3 text-xs font-semibold text-destructive">{actionError}</p>}

      <section className="rounded-md border border-border bg-card p-4 shadow-sm">
        <div className="flex items-start gap-3">
          <Database className="mt-0.5 h-5 w-5 text-primary" />
          <div>
            <h2 className="text-sm font-bold text-foreground">Fuente y contrato científico</h2>
            <p className="mt-1 text-xs text-muted-foreground">
              {data.provider ?? "Ministerio de Economía y Finanzas del Perú"} · {data.category ?? "Endeudamiento y Tesoro Público"} · dataset “Índices Soberanos”.
            </p>
          </div>
        </div>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Metric label="Cobertura" value={`${shortDate(data.audit?.source.startAt)} – ${shortDate(data.audit?.source.endAt)}`} />
          <Metric label="SHA-256" value={data.integrityVerified ? "Verificado" : "Pendiente"} />
          <Metric label="Duplicados fuente" value={(data.audit?.source.duplicateTimestamps ?? 0).toLocaleString("es-PE")} />
          <Metric label="Conflictos del objetivo" value={(data.audit?.source.targetDuplicateConflicts ?? 0).toLocaleString("es-PE")} />
        </div>
        {data.sourceUrl && <a href={data.sourceUrl} target="_blank" rel="noreferrer" className="mt-4 inline-block text-xs font-bold text-primary underline">Abrir recurso público utilizado</a>}
      </section>

      <section className="rounded-md border border-border bg-card p-4 shadow-sm">
        <h2 className="text-sm font-bold text-foreground">Partición temporal sin fuga</h2>
        <p className="mt-1 text-xs text-muted-foreground">El desarrollo usa el primer 85%; el último 15% queda reservado para una única evaluación posterior y no participa en selección, Stacking, umbrales ni XAI.</p>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Metric label="Desarrollo" value={(split?.developmentRows ?? 0).toLocaleString("es-PE")} />
          <Metric label="Test bloqueado" value={(split?.lockedTestRows ?? 0).toLocaleString("es-PE")} />
          <Metric label="Test utilizado" value={split?.testSetUsed ? "Sí" : "No"} />
          <Metric label="Autorización" value={split?.testEvaluationAuthorized ? "Concedida" : "No concedida"} />
        </div>
      </section>

      <section className="rounded-md border border-border bg-card p-4 shadow-sm">
        <h2 className="text-sm font-bold text-foreground">Comparación OOF: modelos base y ensambles</h2>
        <p className="mt-1 text-xs text-muted-foreground">RMSE es la métrica primaria y menor es mejor. La tabla se construye sobre los mismos pares semilla/fold.</p>
        {!completedExperiment ? (
          <EmptyEvidence message="La corrida científica 5×5 aún no termina. El piloto técnico no se presenta como resultado de tesis." />
        ) : (
          <div className="mt-4 overflow-x-auto rounded-md border border-border">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-border text-muted-foreground"><tr><th className="px-3 py-2">Posición</th><th className="px-3 py-2">Candidato</th><th className="px-3 py-2">Familia</th><th className="px-3 py-2">RMSE ↓</th><th className="px-3 py-2">MAE ↓</th><th className="px-3 py-2">R² ↑</th><th className="px-3 py-2">Evaluaciones</th></tr></thead>
              <tbody>{completedExperiment.comparison.map((row, index) => <tr key={row.predictorId} className={`border-b border-border/50 ${row.predictorId.includes("stacking") ? "bg-primary/5" : ""}`}><td className="px-3 py-2 font-bold">{index + 1}</td><td className="px-3 py-2 font-bold">{MODEL_NAMES[row.predictorId] ?? row.displayName}</td><td className="px-3 py-2">{row.family === "base" ? "Base" : "Ensamble"}</td><td className="px-3 py-2">{row.rmseMean.toFixed(6)}</td><td className="px-3 py-2">{row.maeMean.toFixed(6)}</td><td className="px-3 py-2">{row.r2Mean.toFixed(4)}</td><td className="px-3 py-2">{row.foldEvaluations}</td></tr>)}</tbody>
            </table>
          </div>
        )}
      </section>

      <section className="rounded-md border border-border bg-card p-4 shadow-sm">
        <h2 className="text-sm font-bold text-foreground">Revalidación independiente, ablación y XAI</h2>
        {!completedRevalidation ? (
          <EmptyEvidence message="Se habilitará al completar las 125 unidades de entrenamiento verificables." />
        ) : (
          <>
            <div className="mt-4 overflow-x-auto rounded-md border border-border">
              <table className="w-full text-left text-xs">
                <thead className="border-b border-border text-muted-foreground"><tr><th className="px-3 py-2">Posición</th><th className="px-3 py-2">Candidato</th><th className="px-3 py-2">RMSE ↓</th><th className="px-3 py-2">MAE ↓</th><th className="px-3 py-2">sMAPE ↓</th><th className="px-3 py-2">R² ↑</th></tr></thead>
                <tbody>{completedRevalidation.independentComparison.metrics.map((row, index) => <tr key={row.predictorId} className={`border-b border-border/50 ${row.predictorId === "optimized_stacking" ? "bg-primary/5" : ""}`}><td className="px-3 py-2 font-bold">{index + 1}</td><td className="px-3 py-2 font-bold">{MODEL_NAMES[row.predictorId] ?? row.predictorId}</td><td className="px-3 py-2">{row.rmse.toFixed(6)}</td><td className="px-3 py-2">{row.mae.toFixed(6)}</td><td className="px-3 py-2">{row.smape.toFixed(4)}</td><td className="px-3 py-2">{row.r2.toFixed(4)}</td></tr>)}</tbody>
              </table>
            </div>
            <p className="mt-3 rounded-md border border-border bg-muted/30 p-3 text-xs text-muted-foreground">
              Stacking ganador: {completedRevalidation.independentComparison.optimizedStackingIsWinner ? "sí" : "no"}. ΔRMSE frente al mejor modelo base: {completedRevalidation.independentComparison.pairedInference.observedDeltaRmse.toFixed(6)}; probabilidad bootstrap de mejora {(completedRevalidation.independentComparison.pairedInference.probabilityCandidateBetter * 100).toFixed(1)}%. El resultado se conserva aunque Stacking no gane.
            </p>
          </>
        )}
      </section>

      <section className="rounded-md border border-border bg-card p-4 shadow-sm">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-sm font-bold text-foreground">Puerta de congelación pre-test</h2>
            <p className="mt-1 text-xs text-muted-foreground">{readiness?.ready ? `${readiness.passedChecks}/${readiness.totalChecks} controles cumplidos.` : `Pendientes: ${readiness?.blockingCheckIds?.join(", ") || "corrida científica"}.`}</p>
          </div>
          <button type="button" onClick={freeze} disabled={!readiness?.ready || freezing} className="inline-flex items-center justify-center gap-2 rounded-md border border-border bg-card px-4 py-2 text-xs font-bold text-foreground disabled:opacity-50"><Snowflake className="h-4 w-4" />{freezing ? "Sellando..." : "Crear sello"}</button>
        </div>
        {frozen?.freezeId && <p className="mt-3 text-xs text-success">Sello vigente: {frozen.freezeId}. Test utilizado: {frozen.testPolicy?.testSetUsed ? "sí" : "no"}.</p>}
      </section>
    </div>
  );
}

function StatusCard({ icon: Icon, title, value }: { icon: typeof Landmark; title: string; value: string }) {
  return <div className="rounded-md border border-border bg-card p-4 shadow-sm"><Icon className="h-5 w-5 text-primary" /><p className="mt-3 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">{title}</p><p className="mt-1 text-sm font-bold text-foreground">{value}</p></div>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="rounded-md border border-border bg-muted/20 p-3"><p className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">{label}</p><p className="mt-1 break-words text-sm font-bold text-foreground">{value}</p></div>;
}

function EmptyEvidence({ message }: { message: string }) {
  return <p className="mt-4 rounded-md border border-warning/30 bg-warning/5 p-3 text-xs text-muted-foreground">{message}</p>;
}

function shortDate(value?: string) {
  if (!value) return "N/D";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString("es-PE");
}
