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
      { title: "Dashboard — LSTM vs GRU vs CNN Detección de Anomalías" },
      { name: "description", content: "Dashboard interactivo para comparar modelos LSTM, GRU y CNN en detección de anomalías" },
    ],
  }),
  component: HomePage,
});

const domainDistribution = [
  { name: "PhishTank", value: 12722, color: "var(--chart-1)" },
  { name: "Energía", value: 12131, color: "var(--chart-2)" },
  { name: "Finanzas", value: 13441, color: "var(--chart-3)" },
];

function HomePage() {
  const [selectedDomain, setSelectedDomain] = useState<"all" | "phishtank" | "energia" | "finanzas">("all");

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
        <h1 className="text-2xl font-bold text-foreground">Resumen General</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Comparación de modelos LSTM, GRU y CNN para detección de anomalías en secuencias textuales, series temporales y transacciones
        </p>
      </motion.div>

      {/* Domain selector */}
      <div className="flex flex-wrap gap-2">
        {(["all", "phishtank", "energia", "finanzas"] as const).map((d) => (
          <button
            key={d}
            onClick={() => setSelectedDomain(d)}
            className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all ${
              selectedDomain === d
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-muted-foreground hover:bg-muted/80"
            }`}
          >
            {d === "all" ? "Todos" : d === "phishtank" ? "PhishTank" : d === "energia" ? "Energía" : "Finanzas"}
          </button>
        ))}
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <KpiCard
          title="Registros Analizados"
          value={kpiData.totalRecords.toLocaleString("es-ES")}
          subtitle="En los 3 dominios"
          icon={Database}
          variant="cyan"
          delay={0}
        />
        <KpiCard
          title="Anomalías Detectadas"
          value={(kpiData.totalAnomalies.phishtank + kpiData.totalAnomalies.energia + kpiData.totalAnomalies.finanzas).toLocaleString("es-ES")}
          subtitle={`PT: ${kpiData.totalAnomalies.phishtank} | EN: ${kpiData.totalAnomalies.energia} | FN: ${kpiData.totalAnomalies.finanzas}`}
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
          subtitle="CNN optimizado — Global"
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
                contentStyle={{ backgroundColor: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: "4px", fontSize: "12px" }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex justify-center gap-6 mt-2">
            {domainDistribution.map((d) => (
              <div key={d.name} className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: d.color }} />
                <span className="text-xs text-muted-foreground">{d.name}: {d.value.toLocaleString("es-ES")}</span>
              </div>
            ))}
          </div>
        </ChartCard>

        <ChartCard title="Modelo Ganador por Métrica" delay={0.5}>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={[
              { metric: "F1 (PT)", LSTM: 0.964, GRU: 0.934, CNN: 0.951 },
              { metric: "F1 (FN)", LSTM: 0.903, GRU: 0.917, CNN: 0.952 },
              { metric: "Velocidad", LSTM: 0.72, GRU: 0.89, CNN: 0.98 },
              { metric: "Memoria", LSTM: 0.65, GRU: 0.82, CNN: 0.94 },
            ]}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
              <XAxis dataKey="metric" tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ backgroundColor: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: "4px", fontSize: "12px" }}
              />
              <Bar dataKey="LSTM" fill="var(--chart-1)" radius={[2, 2, 0, 0]} />
              <Bar dataKey="GRU" fill="var(--chart-2)" radius={[2, 2, 0, 0]} />
              <Bar dataKey="CNN" fill="var(--chart-3)" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
}
