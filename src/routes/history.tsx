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
  const [domainFilter, setDomainFilter] = useState<"all" | "PhishTank" | "OPSD">("all");
  const [modelFilter, setModelFilter] = useState<"all" | "LSTM" | "GRU">("all");

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
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Historial de Anomalías</h1>
          <p className="text-sm text-muted-foreground mt-1">Timeline de todas las anomalías detectadas</p>
        </div>
        <button onClick={exportCSV} className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 transition-colors">
          <Download className="h-3.5 w-3.5" />
          Exportar CSV
        </button>
      </motion.div>

      {/* Filters */}
      <div className="flex flex-wrap gap-4">
        <div className="flex items-center gap-2">
          <Filter className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">Dominio:</span>
          {(["all", "PhishTank", "OPSD"] as const).map((d) => (
            <button key={d} onClick={() => setDomainFilter(d)} className={`px-3 py-1 rounded-md text-[11px] font-medium transition-all ${domainFilter === d ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"}`}>
              {d === "all" ? "Todos" : d}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Modelo:</span>
          {(["all", "LSTM", "GRU"] as const).map((m) => (
            <button key={m} onClick={() => setModelFilter(m)} className={`px-3 py-1 rounded-md text-[11px] font-medium transition-all ${modelFilter === m ? "bg-secondary text-secondary-foreground" : "bg-muted text-muted-foreground"}`}>
              {m === "all" ? "Todos" : m}
            </button>
          ))}
        </div>
      </div>

      {/* Timeline */}
      <div className="space-y-3">
        {filtered.map((item, i) => (
          <motion.div
            key={item.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.05 }}
            className="flex gap-4 rounded-xl border border-border bg-card p-4 hover:border-primary/30 transition-colors"
          >
            <div className="flex flex-col items-center">
              <div className={`w-3 h-3 rounded-full ${item.domain === "PhishTank" ? "bg-primary" : "bg-secondary"}`} />
              {i < filtered.length - 1 && <div className="w-px flex-1 bg-border mt-1" />}
            </div>
            <div className="flex-1 grid grid-cols-1 md:grid-cols-6 gap-2 items-center">
              <div>
                <p className="text-[10px] text-muted-foreground">Fecha</p>
                <p className="text-xs font-data text-foreground">{item.date}</p>
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground">Dominio</p>
                <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${item.domain === "PhishTank" ? "bg-primary/15 text-primary" : "bg-secondary/15 text-secondary"}`}>
                  {item.domain}
                </span>
              </div>
              <div className="md:col-span-2">
                <p className="text-[10px] text-muted-foreground">Dato</p>
                <p className="text-xs font-data text-foreground truncate">{item.data}</p>
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground">Modelo / Confianza</p>
                <p className="text-xs font-data text-foreground">{item.model} — <span className="text-primary">{item.confidence}%</span></p>
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground">Real → Pred</p>
                <p className="text-xs font-data">
                  <span className="text-foreground">{item.realLabel}</span>
                  <span className="text-muted-foreground"> → </span>
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
