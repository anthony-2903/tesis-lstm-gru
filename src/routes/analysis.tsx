import { createFileRoute } from "@tanstack/react-router";
import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { ChartCard } from "@/components/ChartCard";
import { KpiCard } from "@/components/KpiCard";
import {
  phishtankUrls, phishtankMetrics, phishtankBarData, confusionMatrix,
  generateTimeSeriesData, opsdMetrics
} from "@/lib/mock-data";
import { Shield, Zap, TrendingUp, AlertCircle } from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, ScatterChart, Scatter, Cell
} from "recharts";

export const Route = createFileRoute("/analysis")({
  head: () => ({
    meta: [
      { title: "Análisis por Dominio — LSTM vs GRU" },
      { name: "description", content: "Análisis detallado de PhishTank y Open Power System Data" },
    ],
  }),
  component: AnalysisPage,
});

function ConfusionMatrixViz({ title, matrix }: { title: string; matrix: { tp: number; fp: number; fn: number; tn: number } }) {
  const cells = [
    { label: "VP", value: matrix.tp, row: 0, col: 0, color: "bg-success/20 text-success" },
    { label: "FP", value: matrix.fp, row: 0, col: 1, color: "bg-anomaly/20 text-anomaly" },
    { label: "FN", value: matrix.fn, row: 1, col: 0, color: "bg-warning/20 text-warning" },
    { label: "VN", value: matrix.tn, row: 1, col: 1, color: "bg-primary/20 text-primary" },
  ];

  return (
    <div>
      <p className="text-xs font-semibold text-foreground mb-2">{title}</p>
      <div className="grid grid-cols-2 gap-1">
        {cells.map((c) => (
          <div key={c.label} className={`rounded-lg p-3 text-center ${c.color}`}>
            <p className="text-[10px] uppercase tracking-wider opacity-70">{c.label}</p>
            <p className="text-lg font-bold font-data">{c.value.toLocaleString()}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function AnalysisPage() {
  const [tab, setTab] = useState<"phishtank" | "opsd">("phishtank");
  const timeSeriesData = useMemo(() => generateTimeSeriesData(), []);
  const filteredTS = timeSeriesData.slice(0, 90); // show 3 months

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <h1 className="text-2xl font-bold text-foreground">Análisis por Dominio</h1>
        <p className="text-sm text-muted-foreground mt-1">Resultados detallados por modelo y dominio</p>
      </motion.div>

      <div className="flex gap-2">
        <button onClick={() => setTab("phishtank")} className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all ${tab === "phishtank" ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"}`}>
          PhishTank (URLs)
        </button>
        <button onClick={() => setTab("opsd")} className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all ${tab === "opsd" ? "bg-secondary text-secondary-foreground" : "bg-muted text-muted-foreground"}`}>
          Open Power System
        </button>
      </div>

      {tab === "phishtank" ? (
        <motion.div key="phishtank" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="space-y-4">
          {/* Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <KpiCard title="Precision LSTM" value={phishtankMetrics.lstm.precision.toFixed(3)} icon={Shield} variant="cyan" />
            <KpiCard title="Recall LSTM" value={phishtankMetrics.lstm.recall.toFixed(3)} icon={TrendingUp} variant="cyan" delay={0.05} />
            <KpiCard title="Precision GRU" value={phishtankMetrics.gru.precision.toFixed(3)} icon={Shield} variant="violet" delay={0.1} />
            <KpiCard title="Recall GRU" value={phishtankMetrics.gru.recall.toFixed(3)} icon={TrendingUp} variant="violet" delay={0.15} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Bar chart */}
            <ChartCard title="Detecciones por Modelo" delay={0.2}>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={phishtankBarData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.28 0.02 260)" />
                  <XAxis dataKey="name" tick={{ fontSize: 10, fill: "oklch(0.65 0.02 250)" }} />
                  <YAxis tick={{ fontSize: 10, fill: "oklch(0.65 0.02 250)" }} />
                  <Tooltip contentStyle={{ backgroundColor: "oklch(0.18 0.02 260)", border: "1px solid oklch(0.28 0.02 260)", borderRadius: "8px", fontSize: "12px", color: "oklch(0.95 0.01 250)" }} />
                  <Bar dataKey="LSTM" fill="oklch(0.85 0.18 195)" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="GRU" fill="oklch(0.55 0.2 290)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* Confusion matrices */}
            <ChartCard title="Matrices de Confusión" delay={0.3}>
              <div className="grid grid-cols-2 gap-4">
                <ConfusionMatrixViz title="LSTM" matrix={confusionMatrix.lstm} />
                <ConfusionMatrixViz title="GRU" matrix={confusionMatrix.gru} />
              </div>
            </ChartCard>
          </div>

          {/* URL table */}
          <ChartCard title="URLs Analizadas (Muestra)" delay={0.4}>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 px-3 text-muted-foreground font-medium">URL</th>
                    <th className="text-center py-2 px-3 text-muted-foreground font-medium">Real</th>
                    <th className="text-center py-2 px-3 text-muted-foreground font-medium">LSTM</th>
                    <th className="text-center py-2 px-3 text-muted-foreground font-medium">GRU</th>
                    <th className="text-center py-2 px-3 text-muted-foreground font-medium">Anomalía</th>
                  </tr>
                </thead>
                <tbody>
                  {phishtankUrls.map((row) => (
                    <tr key={row.id} className="border-b border-border/50 hover:bg-muted/30 transition-colors">
                      <td className="py-2 px-3 font-data text-foreground truncate max-w-[300px]">{row.url}</td>
                      <td className="py-2 px-3 text-center">
                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${row.real === "phishing" ? "bg-anomaly/20 text-anomaly" : "bg-success/20 text-success"}`}>
                          {row.real}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-center">
                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${row.lstm === "phishing" ? "bg-anomaly/20 text-anomaly" : "bg-success/20 text-success"}`}>
                          {row.lstm}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-center">
                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${row.gru === "phishing" ? "bg-anomaly/20 text-anomaly" : "bg-success/20 text-success"}`}>
                          {row.gru}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-center">
                        {row.anomaly ? <AlertCircle className="inline h-4 w-4 text-anomaly" /> : <span className="text-muted-foreground">—</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </ChartCard>
        </motion.div>
      ) : (
        <motion.div key="opsd" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <KpiCard title="RMSE LSTM" value={opsdMetrics.lstm.rmse.toFixed(2)} icon={Zap} variant="cyan" />
            <KpiCard title="MAE LSTM" value={opsdMetrics.lstm.mae.toFixed(2)} icon={TrendingUp} variant="cyan" delay={0.05} />
            <KpiCard title="RMSE GRU" value={opsdMetrics.gru.rmse.toFixed(2)} icon={Zap} variant="violet" delay={0.1} />
            <KpiCard title="MAE GRU" value={opsdMetrics.gru.mae.toFixed(2)} icon={TrendingUp} variant="violet" delay={0.15} />
          </div>

          <ChartCard title="Consumo Eléctrico — Real vs Predicción" subtitle="Primeros 90 días del período de prueba" delay={0.2}>
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={filteredTS}>
                <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.28 0.02 260)" />
                <XAxis dataKey="date" tick={{ fontSize: 9, fill: "oklch(0.65 0.02 250)" }} interval={14} />
                <YAxis tick={{ fontSize: 10, fill: "oklch(0.65 0.02 250)" }} />
                <Tooltip contentStyle={{ backgroundColor: "oklch(0.18 0.02 260)", border: "1px solid oklch(0.28 0.02 260)", borderRadius: "8px", fontSize: "11px", color: "oklch(0.95 0.01 250)" }} />
                <Line type="monotone" dataKey="actual" stroke="oklch(0.65 0.02 250)" strokeWidth={1.5} dot={false} name="Real" />
                <Line type="monotone" dataKey="lstm" stroke="oklch(0.85 0.18 195)" strokeWidth={2} dot={false} name="LSTM" />
                <Line type="monotone" dataKey="gru" stroke="oklch(0.55 0.2 290)" strokeWidth={2} dot={false} name="GRU" />
                <Scatter
                  dataKey="actual"
                  data={filteredTS.filter(d => d.anomaly)}
                  fill="oklch(0.65 0.25 25)"
                  name="Anomalía"
                />
              </LineChart>
            </ResponsiveContainer>
            <div className="flex gap-4 justify-center mt-2">
              <div className="flex items-center gap-1.5"><div className="w-3 h-0.5 bg-muted-foreground" /><span className="text-[10px] text-muted-foreground">Real</span></div>
              <div className="flex items-center gap-1.5"><div className="w-3 h-0.5 bg-primary" /><span className="text-[10px] text-muted-foreground">LSTM</span></div>
              <div className="flex items-center gap-1.5"><div className="w-3 h-0.5 bg-secondary" /><span className="text-[10px] text-muted-foreground">GRU</span></div>
              <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-anomaly" /><span className="text-[10px] text-muted-foreground">Anomalía</span></div>
            </div>
          </ChartCard>
        </motion.div>
      )}
    </div>
  );
}
