import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { BackendState } from "@/components/BackendState";
import { HistoryAuditSummary } from "@/components/AcademicPanels";
import { AuditHeatmap } from "@/components/rosen/ResearchCharts";
import { DomainId, fetchHistoryData } from "@/lib/api";
import { DOMAIN_OPTIONS, getInitialDomain } from "@/lib/domains";
import { useApiData } from "@/hooks/useApiData";
import { Download, Filter } from "lucide-react";

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "Historial de Anomalías - LSTM vs GRU vs BRNN" },
      { name: "description", content: "Timeline de anomalías detectadas con filtros" },
    ],
  }),
  component: HistoryPage,
});

function HistoryPage() {
  const [domain, setDomain] = useState<DomainId>(getInitialDomain);
  const { data, error, isLoading, reload } = useApiData(() => fetchHistoryData(domain), [domain]);
  const [domainFilter, setDomainFilter] = useState<string>("all");
  const [modelFilter, setModelFilter] = useState<string>("all");

  const domains = useMemo(() => ["all", ...Array.from(new Set((data?.items || []).map((item) => item.domain)))], [data]);
  const models = useMemo(() => ["all", ...Array.from(new Set((data?.items || []).map((item) => item.model)))], [data]);

  const filtered = useMemo(() => {
    return (data?.items || []).filter((item) => {
      if (domainFilter !== "all" && item.domain !== domainFilter) return false;
      if (modelFilter !== "all" && item.model !== modelFilter) return false;
      return true;
    });
  }, [data, domainFilter, modelFilter]);

  const exportCSV = () => {
    if (!filtered.length) return;
    const headers = "Fecha,Dominio,Dato,Modelo,Confianza,Real,Predicción\n";
    const rows = filtered
      .map((r) => `${r.date},${r.domain},"${r.data.replace(/"/g, '""')}",${r.model},${r.confidence},${r.realLabel},${r.predicted}`)
      .join("\n");
    const blob = new Blob([headers + rows], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `historial_anomalias_${data?.filename || "dataset"}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (isLoading) return <BackendState isLoading />;
  if (error || !data) return <BackendState error={error} onRetry={reload} />;

  return (
    <div className="dashboard-page">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Historial de Anomalías</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Timeline de anomalías identificadas en <span className="font-semibold text-foreground">{data.filename}</span>
          </p>
        </div>
        <button onClick={exportCSV} className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary/90 sm:w-auto">
          <Download className="h-3.5 w-3.5" />
          Exportar CSV
        </button>
      </motion.div>

      <HistoryAuditSummary items={filtered} />
      <AuditHeatmap items={filtered} />

      <div className="flex flex-col gap-4 border-b border-border pb-6 lg:flex-row lg:flex-wrap lg:gap-8">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
          <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Dataset:</span>
          <div className="flex flex-wrap gap-1">
            {DOMAIN_OPTIONS.map((option) => (
              <button key={option.id} onClick={() => setDomain(option.id)} className={`rounded-md px-3 py-1 text-[10px] font-bold transition-all ${domain === option.id ? "bg-primary text-primary-foreground shadow-sm" : "bg-muted/50 text-muted-foreground hover:bg-muted"}`}>
                {option.shortTitle}
              </button>
            ))}
          </div>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
          <Filter className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Dominio:</span>
          <div className="flex flex-wrap gap-1">
            {domains.map((domain) => (
              <button key={domain} onClick={() => setDomainFilter(domain)} className={`rounded-md px-3 py-1 text-[10px] font-bold transition-all ${domainFilter === domain ? "bg-primary text-primary-foreground shadow-sm" : "bg-muted/50 text-muted-foreground hover:bg-muted"}`}>
                {domain === "all" ? "Todos" : domain}
              </button>
            ))}
          </div>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
          <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Modelo:</span>
          <div className="flex flex-wrap gap-1">
            {models.map((model) => (
              <button key={model} onClick={() => setModelFilter(model)} className={`rounded-md px-3 py-1 text-[10px] font-bold transition-all ${modelFilter === model ? "bg-primary text-primary-foreground shadow-sm" : "bg-muted/50 text-muted-foreground hover:bg-muted"}`}>
                {model === "all" ? "Todos" : model}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="space-y-4">
        {filtered.length === 0 ? (
          <div className="rounded-lg border-2 border-dashed bg-muted/5 p-12 text-center text-muted-foreground">
            Ninguna anomalia coincide con los filtros aplicados.
          </div>
        ) : (
          filtered.map((item, i) => (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.02, 0.4) }}
              className="card-formal group flex gap-4 p-4 sm:gap-6 sm:p-5"
            >
              <div className="flex flex-col items-center pt-1.5 font-sans">
                <div className="h-2.5 w-2.5 rotate-45 rounded-sm bg-primary" />
                {i < filtered.length - 1 && <div className="mt-3 w-px flex-1 bg-border transition-colors group-hover:bg-primary/20" />}
              </div>
              <div className="grid min-w-0 flex-1 grid-cols-1 items-center gap-4 md:grid-cols-6 md:gap-6">
                <div>
                  <p className="mb-1 text-[9px] font-bold uppercase tracking-widest text-muted-foreground">Registro / Fecha</p>
                  <p className="font-data text-xs font-medium text-foreground">{item.date}</p>
                </div>
                <div>
                  <span className="rounded-sm border border-primary/20 bg-primary/5 px-2 py-0.5 text-[9px] font-bold uppercase text-primary">{item.domain}</span>
                </div>
                <div className="md:col-span-2">
                  <p className="mb-1 text-[9px] font-bold uppercase tracking-widest text-muted-foreground">Dato de Entrada</p>
                  <p className="truncate font-data text-xs font-medium text-foreground/80" title={item.data}>{item.data}</p>
                </div>
                <div>
                  <p className="mb-1 text-[9px] font-bold uppercase tracking-widest text-muted-foreground">Modelo / Conf.</p>
                  <p className="font-data text-[11px] text-foreground">{item.model} - <span className="font-bold text-primary">{item.confidence}%</span></p>
                </div>
                <div>
                  <p className="mb-1 text-[9px] font-bold uppercase tracking-widest text-muted-foreground">Evaluación</p>
                  <p className="font-data text-[11px] font-bold">
                    <span className="text-foreground/70">{item.realLabel}</span>
                    <span className="font-normal text-muted-foreground"> {"->"} </span>
                    <span className={item.realLabel === item.predicted ? "text-success" : "text-anomaly"}>{item.predicted}</span>
                  </p>
                </div>
              </div>
            </motion.div>
          ))
        )}
      </div>
    </div>
  );
}
