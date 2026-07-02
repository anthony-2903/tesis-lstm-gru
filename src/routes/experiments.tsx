import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { Activity, Archive, BrainCircuit, Clock3, Database, Trophy } from "lucide-react";
import { BackendState } from "@/components/BackendState";
import { fetchExperiments, fetchTrainingManifest, type DomainId, type DomainMetricsSummary } from "@/lib/api";
import { useApiData } from "@/hooks/useApiData";
import type { LucideIcon } from "lucide-react";

export const Route = createFileRoute("/experiments")({
  head: () => ({
    meta: [
      { title: "Experimentos" },
      { name: "description", content: "Historial reproducible de corridas, modelos y metricas." },
    ],
  }),
  component: ExperimentsPage,
});

const DOMAIN_LABELS: Record<DomainId, string> = {
  phishing: "Phishing",
  energia: "Energia",
  finanzas: "Finanzas",
};

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function ExperimentsPage() {
  const { data, error, isLoading, reload } = useApiData(async () => {
    const [manifest, experiments] = await Promise.all([fetchTrainingManifest(), fetchExperiments()]);
    return { manifest, experiments };
  });

  if (isLoading) return <BackendState isLoading />;
  if (error || !data) return <BackendState error={error} onRetry={reload} />;

  const latest = data.manifest;
  const summary = latest.metricsSummary;
  const experimentItems = data.experiments.items;

  return (
    <div className="dashboard-page">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Experimentos reproducibles</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Ultima corrida: <span className="font-data font-semibold text-foreground">{latest.runId}</span>
          </p>
        </div>
        <button
          type="button"
          onClick={reload}
          className="inline-flex items-center justify-center gap-2 rounded-md border border-border bg-card px-4 py-2 text-sm font-semibold text-foreground shadow-sm transition-colors hover:bg-muted"
        >
          <Activity className="h-4 w-4 text-primary" />
          Actualizar
        </button>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Modo" value={latest.mode} icon={Archive} />
        <MetricCard label="Limite de corrida" value={latest.limit.toLocaleString("es-ES")} icon={Database} />
        <MetricCard label="Modelos persistidos" value={latest.models.length.toLocaleString("es-ES")} icon={BrainCircuit} />
        <MetricCard label="Mejor global" value={summary.overallBest ? `${DOMAIN_LABELS[summary.overallBest.domain]} / ${summary.overallBest.model.toUpperCase()}` : "N/D"} icon={Trophy} />
      </div>

      <section className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        {summary.domains.map((domain, index) => (
          <DomainExperimentCard key={domain.domain} domain={domain} index={index} />
        ))}
      </section>

      <section className="rounded-md border border-border bg-card p-4 shadow-sm sm:p-5">
        <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h2 className="text-sm font-bold text-foreground">Historial de corridas</h2>
            <p className="mt-1 text-xs text-muted-foreground">Cada corrida conserva modelos, datos procesados, resultados y manifest.</p>
          </div>
          <span className="font-data text-[10px] font-bold uppercase text-primary">{experimentItems.length} experimento(s)</span>
        </div>

        <div className="responsive-table">
          <table className="w-full min-w-[760px] text-xs">
            <thead>
              <tr className="border-b border-border bg-muted/20 text-muted-foreground">
                <th className="px-3 py-3 text-left font-bold uppercase">Run ID</th>
                <th className="px-3 py-3 text-left font-bold uppercase">Modo</th>
                <th className="px-3 py-3 text-right font-bold uppercase">Limite</th>
                <th className="px-3 py-3 text-right font-bold uppercase">Phishing</th>
                <th className="px-3 py-3 text-right font-bold uppercase">Energia</th>
                <th className="px-3 py-3 text-right font-bold uppercase">Finanzas</th>
                <th className="px-3 py-3 text-left font-bold uppercase">Creado</th>
              </tr>
            </thead>
            <tbody>
              {experimentItems.map((item) => (
                <tr key={item.runId} className="border-b border-border hover:bg-muted/10">
                  <td className="px-3 py-3 font-data text-foreground">{item.runId}</td>
                  <td className="px-3 py-3 font-semibold uppercase text-primary">{item.mode}</td>
                  <td className="px-3 py-3 text-right font-data">{item.limit.toLocaleString("es-ES")}</td>
                  <td className="px-3 py-3 text-right font-data">{(item.domainTotals?.phishing ?? 0).toLocaleString("es-ES")}</td>
                  <td className="px-3 py-3 text-right font-data">{(item.domainTotals?.energia ?? 0).toLocaleString("es-ES")}</td>
                  <td className="px-3 py-3 text-right font-data">{(item.domainTotals?.finanzas ?? 0).toLocaleString("es-ES")}</td>
                  <td className="px-3 py-3 text-muted-foreground">{new Date(item.createdAt).toLocaleString("es-ES")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function MetricCard({ label, value, icon: Icon }: { label: string; value: string; icon: LucideIcon }) {
  return (
    <div className="rounded-md border border-border bg-card p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between gap-3">
        <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">{label}</p>
        <Icon className="h-4 w-4 text-primary" />
      </div>
      <p className="font-data text-xl font-bold text-foreground">{value}</p>
    </div>
  );
}

function DomainExperimentCard({ domain, index }: { domain: DomainMetricsSummary; index: number }) {
  const best = domain.bestModel;
  const rows = Object.entries(domain.models)
    .map(([model, metrics]) => ({ model, ...metrics }))
    .sort((a, b) => b.f1 - a.f1);

  return (
    <motion.article
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
      className="rounded-md border border-border bg-card p-4 shadow-sm"
    >
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-bold text-foreground">{DOMAIN_LABELS[domain.domain]}</h2>
          <p className="mt-1 text-xs text-muted-foreground">{domain.totalRows.toLocaleString("es-ES")} registros evaluados</p>
        </div>
        <span className="inline-flex items-center gap-1 rounded-md border border-primary/20 bg-primary/10 px-2 py-1 text-[10px] font-bold uppercase text-primary">
          <Trophy className="h-3 w-3" />
          {best.model.toUpperCase()}
        </span>
      </div>

      <div className="mb-4 grid grid-cols-3 gap-2">
        <SmallMetric label="F1" value={formatPercent(best.f1)} />
        <SmallMetric label="Precision" value={formatPercent(best.precision)} />
        <SmallMetric label="Recall" value={formatPercent(best.recall)} />
      </div>

      <div className="space-y-2">
        {rows.map((row) => (
          <div key={row.model} className="rounded-md border border-border bg-muted/20 p-2">
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="text-[10px] font-bold uppercase text-foreground">{row.model}</span>
              <span className="font-data text-[10px] text-muted-foreground">
                <Clock3 className="mr-1 inline h-3 w-3" />
                {row.trainTime.toFixed(2)}s
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-primary/10">
              <motion.div
                className="h-full rounded-full bg-primary"
                initial={{ width: 0 }}
                animate={{ width: `${Math.max(2, row.f1 * 100)}%` }}
                transition={{ duration: 0.65 }}
              />
            </div>
          </div>
        ))}
      </div>
    </motion.article>
  );
}

function SmallMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-muted/20 p-2">
      <p className="text-[9px] font-bold uppercase text-muted-foreground">{label}</p>
      <p className="mt-1 font-data text-sm font-bold text-foreground">{value}</p>
    </div>
  );
}
