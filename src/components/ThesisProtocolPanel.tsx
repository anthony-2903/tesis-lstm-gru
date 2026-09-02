import { useEffect, useState } from "react";
import { LockKeyhole, PauseCircle, Play, RotateCcw, ShieldCheck, StopCircle } from "lucide-react";
import type { ThesisProtocolJob, ThesisProtocolJobResponse } from "@/lib/api";

interface ThesisProtocolPanelProps {
  title: string;
  description: string;
  fetchLatest: () => Promise<ThesisProtocolJobResponse>;
  start: () => Promise<ThesisProtocolJob>;
  resume: (jobId: string) => Promise<ThesisProtocolJob>;
  pause?: (jobId: string) => Promise<ThesisProtocolJob>;
  cancel?: (jobId: string) => Promise<ThesisProtocolJob>;
}

const ACTIVE = new Set(["queued", "running"]);
const RESUMABLE = new Set(["failed", "cancelled", "interrupted", "paused", "waiting_for_external_data", "waiting_for_scientific_data"]);

export function ThesisProtocolPanel({ title, description, fetchLatest, start, resume, pause, cancel }: ThesisProtocolPanelProps) {
  const [job, setJob] = useState<ThesisProtocolJob | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionError, setActionError] = useState<string | null>(null);
  const active = job ? ACTIVE.has(job.status) : false;
  const remaining = job ? Math.max(0, 100 - job.progress.percent) : 100;

  useEffect(() => {
    fetchLatest().then((latest) => {
      if ("jobId" in latest) setJob(latest);
    }).catch((caught) => setActionError(caught instanceof Error ? caught.message : "No se pudo consultar el protocolo."))
      .finally(() => setLoading(false));
  }, [fetchLatest]);

  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => {
      fetchLatest().then((latest) => {
        if ("jobId" in latest) setJob(latest);
      }).catch((caught) => setActionError(caught instanceof Error ? caught.message : "No se pudo actualizar el protocolo."));
    }, 2000);
    return () => window.clearInterval(timer);
  }, [active, fetchLatest]);

  const execute = async (operation: () => Promise<ThesisProtocolJob>) => {
    setActionError(null);
    try {
      setJob(await operation());
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "No se pudo ejecutar la accion.");
    }
  };

  return (
    <section className="rounded-md border border-primary/20 bg-primary/5 p-4 shadow-sm">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex items-start gap-3">
          <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
          <div>
            <h2 className="text-sm font-bold text-foreground">{title}</h2>
            <p className="mt-1 max-w-3xl text-xs text-muted-foreground">{description}</p>
            <p className="mt-2 inline-flex items-center gap-1 text-[10px] font-bold uppercase text-success"><LockKeyhole className="h-3.5 w-3.5" />El test final permanece bloqueado</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {!job && <button type="button" disabled={loading} onClick={() => execute(start)} className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-xs font-bold text-primary-foreground disabled:opacity-60"><Play className="h-4 w-4" />Iniciar protocolo completo</button>}
          {job && RESUMABLE.has(job.status) && <button type="button" onClick={() => execute(() => resume(job.jobId))} className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-xs font-bold text-primary-foreground"><RotateCcw className="h-4 w-4" />Reanudar desde checkpoints</button>}
          {job && active && pause && <button type="button" disabled={job.pauseRequested || job.cancelRequested} onClick={() => execute(() => pause(job.jobId))} className="inline-flex items-center gap-2 rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-xs font-bold text-warning disabled:opacity-50"><PauseCircle className="h-4 w-4" />{job.pauseRequested ? "Pausa solicitada" : "Pausar al cerrar unidad"}</button>}
          {job && active && cancel && <button type="button" disabled={job.cancelRequested || job.pauseRequested} onClick={() => execute(() => cancel(job.jobId))} className="inline-flex items-center gap-2 rounded-md border border-destructive/30 px-3 py-2 text-xs font-bold text-destructive disabled:opacity-50"><StopCircle className="h-4 w-4" />Cancelar</button>}
        </div>
      </div>
      {job && <div className="mt-4" aria-live="polite">
        <div className="flex flex-wrap justify-between gap-2 text-[10px] font-bold uppercase text-muted-foreground"><span>{job.status} · {job.progress.stage}</span><span>{job.progress.percent.toFixed(1)}%</span></div>
        <div className="mt-2 h-2 overflow-hidden rounded-full bg-primary/10"><div className="h-full rounded-full bg-primary transition-all" style={{ width: `${Math.max(0, Math.min(100, job.progress.percent))}%` }} /></div>
        <p className="mt-1 text-right text-[10px] font-bold uppercase text-warning">Falta {remaining.toFixed(1)}%</p>
        <p className="mt-2 text-xs text-foreground">{job.progress.message}</p>
        <p className="mt-1 font-data text-[10px] text-muted-foreground">{job.executionRunId} · {job.config.seeds.length} semillas</p>
        {job.blocker && <p className="mt-2 text-xs font-semibold text-warning">Bloqueo: {job.blocker.message}</p>}
        {job.error && <p className="mt-2 text-xs font-semibold text-destructive">{job.error}</p>}
      </div>}
      {actionError && <p className="mt-3 text-xs font-semibold text-destructive">{actionError}</p>}
    </section>
  );
}
