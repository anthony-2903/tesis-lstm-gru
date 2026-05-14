import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { motion } from "framer-motion";
import { ChartCard } from "@/components/ChartCard";
import { anomalyHistory } from "@/lib/mock-data";
import { Download, Filter } from "lucide-react";

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "Historial de Anomalías — LSTM vs GRU" },
      { name: "description", content: "Timeline de anomalías detectadas con filtros" },
    ],
  }),
  component: HistoryPage,
});

function HistoryPage() {
  const [domainFilter, setDomainFilter] = useState<"all" | "PhishTank" | "Energía" | "Finanzas">("all");
  const [modelFilter, setModelFilter] = useState<"all" | "LSTM" | "GRU" | "CNN">("all");

  const filtered = anomalyHistory.filter((item) => {
    if (domainFilter !== "all" && item.domain !== domainFilter) return false;
    if (modelFilter !== "all" && item.model !== modelFilter) return false;
    return true;
  });

  const exportCSV = () => {
    const headers = "Fecha,Dominio,Dato,Modelo,Confianza,Real,Predicción\n";
    const rows = filtered.map((r) => `${r.date},${r.domain},"${r.data}",${r.model},${r.confidence},${r.realLabel},${r.predicted}`).join("\n");
    const blob = new Blob([headers + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "historial_anomalias.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Historial de Anomalías</h1>
          <p className="text-sm text-muted-foreground mt-1">Timeline de todas las anomalías detectadas</p>
        </div>
        <button onClick={exportCSV} className="w-full sm:w-auto flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 transition-colors">
          <Download className="h-3.5 w-3.5" />
          Exportar CSV
        </button>
      </motion.div>

      {/* Filters */}
      <div className="flex flex-col lg:flex-row gap-4 lg:gap-8 border-b border-border pb-6">
        <div className="flex items-center gap-3">
          <Filter className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">Dominio:</span>
          <div className="flex flex-wrap gap-1">
            {(["all", "PhishTank", "Energía", "Finanzas"] as const).map((d) => (
              <button key={d} onClick={() => setDomainFilter(d)} className={`px-3 py-1 rounded-md text-[10px] font-bold transition-all ${domainFilter === d ? "bg-primary text-primary-foreground shadow-sm" : "bg-muted/50 text-muted-foreground hover:bg-muted"}`}>
                {d === "all" ? "Todos" : d}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">Modelo:</span>
          <div className="flex flex-wrap gap-1">
            {(["all", "LSTM", "GRU", "CNN"] as const).map((m) => (
              <button key={m} onClick={() => setModelFilter(m)} className={`px-3 py-1 rounded-md text-[10px] font-bold transition-all ${modelFilter === m ? "bg-primary text-primary-foreground shadow-sm" : "bg-muted/50 text-muted-foreground hover:bg-muted"}`}>
                {m === "all" ? "Todos" : m}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div className="space-y-4">
        {filtered.map((item, i) => (
          <motion.div
            key={item.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.03 }}
            className="flex gap-6 card-formal p-5 group"
          >
            <div className="flex flex-col items-center pt-1.5">
              <div className={`w-2.5 h-2.5 rounded-sm rotate-45 ${item.domain === "PhishTank" ? "bg-primary" : item.domain === "Energía" ? "bg-secondary" : "bg-accent"}`} />
              {i < filtered.length - 1 && <div className="w-px flex-1 bg-border mt-3 group-hover:bg-primary/20 transition-colors" />}
            </div>
            <div className="flex-1 grid grid-cols-1 md:grid-cols-6 gap-6 items-center">
              <div>
                <p className="text-[9px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Fecha de Registro</p>
                <p className="text-xs font-data text-foreground font-medium">{item.date}</p>
              </div>
              <div>
                <span className={`text-[9px] px-2 py-0.5 rounded-sm font-bold uppercase border ${
                  item.domain === "PhishTank" ? "bg-primary/5 text-primary border-primary/20" : 
                  item.domain === "Energía" ? "bg-secondary/5 text-secondary border-secondary/20" : 
                  "bg-accent/5 text-accent border-accent/20"
                }`}>
                  {item.domain}
                </span>
              </div>
              <div className="md:col-span-2">
                <p className="text-[9px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Dato de Entrada</p>
                <p className="text-xs font-data text-foreground/80 truncate font-medium">{item.data}</p>
              </div>
              <div>
                <p className="text-[9px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Inferencia / Conf.</p>
                <p className="text-[11px] font-data text-foreground">{item.model} — <span className="text-primary font-bold">{item.confidence}%</span></p>
              </div>
              <div>
                <p className="text-[9px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Evaluación</p>
                <p className="text-[11px] font-data font-bold">
                  <span className="text-foreground/70">{item.realLabel}</span>
                  <span className="text-muted-foreground font-normal"> → </span>
                  <span className={item.realLabel === item.predicted ? "text-success" : "text-anomaly"}>{item.predicted}</span>
                </p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
