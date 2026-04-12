import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { motion } from "framer-motion";
import { KpiCard } from "@/components/KpiCard";
import { ChartCard } from "@/components/ChartCard";
import { kpiData } from "@/lib/mock-data";
import { Database, AlertTriangle, Trophy, Zap } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Dashboard — LSTM vs GRU Detección de Anomalías" },
      { name: "description", content: "Dashboard interactivo para comparar modelos LSTM y GRU en detección de anomalías" },
    ],
  }),
  component: HomePage,
});

const domainDistribution = [
  { name: "PhishTank", value: 12722, color: "oklch(0.85 0.18 195)" },
  { name: "OPSD", value: 12131, color: "oklch(0.55 0.2 290)" },
];

const recentActivity = [
  { domain: "PhishTank", count: 45 },
  { domain: "OPSD", count: 32 },
];

function HomePage() {
  const [selectedDomain, setSelectedDomain] = useState<"all" | "phishtank" | "opsd">("all");

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
        <h1 className="text-2xl font-bold text-foreground">Resumen General</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Comparación de modelos LSTM y GRU para detección de anomalías en secuencias textuales y series temporales
        </p>
      </motion.div>

      {/* Domain selector */}
      <div className="flex gap-2">
        {(["all", "phishtank", "opsd"] as const).map((d) => (
          <button
            key={d}
            onClick={() => setSelectedDomain(d)}
            className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all ${
              selectedDomain === d
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-muted-foreground hover:bg-muted/80"
            }`}
          >
            {d === "all" ? "Todos" : d === "phishtank" ? "PhishTank" : "OPSD"}
          </button>
        ))}
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <KpiCard
          title="Registros Analizados"
          value={kpiData.totalRecords.toLocaleString()}
          subtitle="En ambos dominios"
          icon={Database}
          variant="cyan"
          delay={0}
        />
        <KpiCard
          title="Anomalías Detectadas"
          value={(kpiData.totalAnomalies.phishtank + kpiData.totalAnomalies.opsd).toLocaleString()}
          subtitle={`PhishTank: ${kpiData.totalAnomalies.phishtank} | OPSD: ${kpiData.totalAnomalies.opsd}`}
          icon={AlertTriangle}
          variant="violet"
          delay={0.1}
        />
        <KpiCard
          title="Mejor F1-Score"
          value={kpiData.bestF1.phishtank.score.toFixed(3)}
          subtitle={`${kpiData.bestF1.phishtank.model} — PhishTank`}
          icon={Trophy}
          variant="cyan"
          delay={0.2}
        />
        <KpiCard
          title="Inferencia Promedio"
          value={`${kpiData.avgInferenceTime} ms`}
          subtitle="Tiempo promedio por predicción"
          icon={Zap}
          variant="default"
          delay={0.3}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartCard title="Distribución por Dominio" delay={0.4}>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={domainDistribution}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={90}
                dataKey="value"
                stroke="none"
              >
                {domainDistribution.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ backgroundColor: "oklch(0.18 0.02 260)", border: "1px solid oklch(0.28 0.02 260)", borderRadius: "8px", fontSize: "12px", color: "oklch(0.95 0.01 250)" }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex justify-center gap-6 mt-2">
            {domainDistribution.map((d) => (
              <div key={d.name} className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: d.color }} />
                <span className="text-xs text-muted-foreground">{d.name}: {d.value.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </ChartCard>

        <ChartCard title="Modelo Ganador por Métrica" delay={0.5}>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={[
              { metric: "F1 (PT)", LSTM: 0.964, GRU: 0.934 },
              { metric: "AUC (PT)", LSTM: 0.983, GRU: 0.969 },
              { metric: "Velocidad", LSTM: 0.72, GRU: 0.89 },
              { metric: "Memoria", LSTM: 0.65, GRU: 0.82 },
            ]}>
              <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.28 0.02 260)" />
              <XAxis dataKey="metric" tick={{ fontSize: 11, fill: "oklch(0.65 0.02 250)" }} />
              <YAxis tick={{ fontSize: 11, fill: "oklch(0.65 0.02 250)" }} />
              <Tooltip contentStyle={{ backgroundColor: "oklch(0.18 0.02 260)", border: "1px solid oklch(0.28 0.02 260)", borderRadius: "8px", fontSize: "12px", color: "oklch(0.95 0.01 250)" }} />
              <Bar dataKey="LSTM" fill="oklch(0.85 0.18 195)" radius={[4, 4, 0, 0]} />
              <Bar dataKey="GRU" fill="oklch(0.55 0.2 290)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
}
