import { useCallback, useEffect, useState } from "react";
import { CircleCheckBig, Clock3, LockKeyhole, PauseCircle, RefreshCw, TriangleAlert, Workflow } from "lucide-react";
import { fetchThesisExecutionStatus, type ThesisExecutionDomainStatus, type ThesisExecutionStatus } from "@/lib/api";

const ACTIVE = new Set(["queued", "running"]);
const COMPLETE = new Set(["completed"]);
const BLOCKED = new Set(["waiting_for_external_data", "waiting_for_scientific_data", "failed", "cancelled", "interrupted"]);
const PAUSED = new Set(["paused"]);

export function ThesisExecutionRoadmap() {
  const [status, setStatus] = useState<ThesisExecutionStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const reload = useCallback(async () => {
    try {
      setStatus(await fetchThesisExecutionStatus());
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "No se pudo consultar la ejecución científica.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  useEffect(() => {
    if (!status?.domains.some((domain) => ACTIVE.has(domain.status))) return;
    const timer = window.setInterval(() => void reload(), 5_000);
    return () => window.clearInterval(timer);
  }, [reload, status]);

  if (loading && !status) {
    return <section className="rounded-md border border-border bg-card p-4 text-xs text-muted-foreground">Consultando la cadena experimental real…</section>;
  }

  if (!status) {
    return (
      <section className="rounded-md border border-border bg-card p-4">
        <p className="text-xs font-semibold text-muted-foreground">El backend desplegado todavía no expone el tablero científico agregado.</p>
        {error && <p className="mt-2 text-xs text-destructive">{error}</p>}
        <button type="button" onClick={() => void reload()} className="mt-3 inline-flex items-center gap-2 rounded-md border border-border px-3 py-2 text-xs font-bold text-foreground hover:bg-muted">
          <RefreshCw className="h-3.5 w-3.5" /> Reintentar
        </button>
      </section>
    );
  }

  const overallRemaining = status.overall.remainingPercent ?? Math.max(0, 100 - status.overall.percent);

  return (
    <section className="rounded-md border border-primary/20 bg-card p-4 shadow-sm" aria-labelledby="thesis-execution-title">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex items-start gap-3">
          <Workflow className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
          <div>
            <h2 id="thesis-execution-title" className="text-sm font-bold text-foreground">Ejecución científica de los tres dominios</h2>
            <p className="mt-1 max-w-3xl text-xs text-muted-foreground">{status.progressMethod.note} Cada dominio aporta un tercio al porcentaje global.</p>
          </div>
        </div>
        <div className="min-w-48 rounded-md bg-primary/5 px-4 py-3 text-right">
          <p className="font-data text-2xl font-bold text-primary">{status.overall.percent.toFixed(1)}%</p>
          <p className="mt-0.5 text-[10px] font-bold uppercase text-warning">Falta {overallRemaining.toFixed(1)}%</p>
          <p className="text-[10px] font-bold uppercase text-muted-foreground">{status.overall.completedDomains}/{status.overall.totalDomains} dominios sellados</p>
        </div>
      </div>

      <div className="mt-4 h-2 overflow-hidden rounded-full bg-primary/10" aria-label={`Avance científico global: ${status.overall.percent.toFixed(1)} por ciento`}>
        <div className="h-full rounded-full bg-primary transition-all" style={{ width: `${status.overall.percent}%` }} />
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-3">
        {status.domains.map((domain, index) => <DomainProgressCard key={domain.id} domain={domain} order={index + 1} />)}
      </div>

      <div className="mt-4 flex flex-col gap-2 border-t border-border pt-3 text-xs sm:flex-row sm:items-center sm:justify-between">
        <p className="inline-flex items-center gap-2 font-semibold text-success">
          <LockKeyhole className="h-4 w-4" />
          {status.testPolicy.message}
        </p>
        <button type="button" onClick={() => void reload()} className="inline-flex items-center gap-2 self-start text-[10px] font-bold uppercase text-muted-foreground hover:text-foreground">
          <RefreshCw className="h-3.5 w-3.5" /> Actualizar estado
        </button>
      </div>
      {error && <p className="mt-2 text-xs text-warning">La última actualización falló; se conserva el estado anterior. {error}</p>}
    </section>
  );
}

function DomainProgressCard({ domain, order }: { domain: ThesisExecutionDomainStatus; order: number }) {
  const remaining = domain.progress.remainingPercent ?? Math.max(0, 100 - domain.progress.percent);
  const Icon = COMPLETE.has(domain.status) ? CircleCheckBig : PAUSED.has(domain.status) ? PauseCircle : BLOCKED.has(domain.status) ? TriangleAlert : Clock3;
  const tone = COMPLETE.has(domain.status)
    ? "border-success/30 bg-success/5 text-success"
    : PAUSED.has(domain.status)
      ? "border-warning/30 bg-warning/5 text-warning"
      : BLOCKED.has(domain.status)
        ? "border-warning/30 bg-warning/5 text-warning"
        : ACTIVE.has(domain.status)
          ? "border-primary/30 bg-primary/5 text-primary"
          : "border-border bg-muted/20 text-muted-foreground";

  return (
    <article className={`rounded-md border p-3 ${tone}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-2">
          <Icon className="mt-0.5 h-4 w-4 shrink-0" />
          <div>
            <p className="text-xs font-bold text-foreground">{order}. {domain.label}</p>
            <p className="mt-0.5 text-[10px] font-bold uppercase">{statusLabel(domain.status)} · {domain.progress.stage}</p>
          </div>
        </div>
        <div className="text-right">
          <span className="block font-data text-sm font-bold">{domain.progress.percent.toFixed(1)}%</span>
          <span className="block text-[9px] font-bold uppercase opacity-80">Falta {remaining.toFixed(1)}%</span>
        </div>
      </div>
      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-background/70">
        <div className="h-full rounded-full bg-current transition-all" style={{ width: `${domain.progress.percent}%` }} />
      </div>
      <p className="mt-2 text-xs leading-5 text-foreground">{domain.progress.message}</p>
      {domain.blocker && <p className="mt-2 text-xs font-semibold">Bloqueo: {domain.blocker.message}</p>}
      <p className="mt-2 text-[10px] leading-4 text-muted-foreground">Siguiente: {domain.nextAction}</p>
      <a href={domain.route} className="mt-3 inline-flex text-[10px] font-bold uppercase text-primary hover:underline">Ver protocolo del dominio</a>
    </article>
  );
}

function statusLabel(status: ThesisExecutionDomainStatus["status"]) {
  return {
    pending: "Pendiente",
    queued: "En cola",
    running: "Ejecutando",
    completed: "Sellado",
    waiting_for_external_data: "Espera datos externos",
    waiting_for_scientific_data: "Espera datos científicos",
    failed: "Falló",
    cancelled: "Cancelado",
    interrupted: "Interrumpido",
    paused: "Pausado de forma segura",
    unknown: "Estado desconocido",
  }[status];
}
