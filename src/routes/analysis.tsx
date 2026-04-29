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
import { AiAnalysis } from "@/components/AiAnalysis";

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
            <p className="text-lg font-bold font-data">{c.value.toLocaleString("es-ES")}</p>
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
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                  <XAxis dataKey="name" tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: "4px", fontSize: "12px" }}
                    itemStyle={{ fontSize: "12px", fontWeight: "bold" }}
                  />
                  <Bar dataKey="LSTM" fill="var(--chart-1)" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="GRU" fill="var(--chart-2)" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* Confusion matrices */}
            <ChartCard title="Matrices de Confusión" delay={0.3}>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <ConfusionMatrixViz title="LSTM" matrix={confusionMatrix.lstm} />
                <ConfusionMatrixViz title="GRU" matrix={confusionMatrix.gru} />
              </div>
            </ChartCard>
          </div>

          {/* URL table */}
          <ChartCard title="URLs Analizadas (Muestra)" delay={0.4}>
            <div className="overflow-x-auto -mx-6 px-6">
              <table className="w-full text-xs min-w-[600px]">
                <thead>
                  <tr className="border-b border-border bg-muted/20">
                    <th className="text-left py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">URL</th>
                    <th className="text-center py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">Real</th>
                    <th className="text-center py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">LSTM</th>
                    <th className="text-center py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">GRU</th>
                    <th className="text-center py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">Anomalía</th>
                  </tr>
                </thead>
                <tbody>
                  {phishtankUrls.map((row) => (
                    <tr key={row.id} className="border-b border-border hover:bg-muted/10 transition-colors">
                      <td className="py-2.5 px-4 font-data text-foreground/80 truncate max-w-[200px] sm:max-w-[300px]">{row.url}</td>
                      <td className="py-2.5 px-4 text-center">
                        <span className={`px-2 py-0.5 rounded-sm text-[10px] font-bold uppercase ${row.real === "phishing" ? "bg-anomaly/10 text-anomaly border border-anomaly/20" : "bg-success/10 text-success border border-success/20"}`}>
                          {row.real}
                        </span>
                      </td>
                      <td className="py-2.5 px-4 text-center">
                        <span className={`px-2 py-0.5 rounded-sm text-[10px] font-bold uppercase ${row.lstm === "phishing" ? "bg-anomaly/10 text-anomaly border border-anomaly/20" : "bg-success/10 text-success border border-success/20"}`}>
                          {row.lstm}
                        </span>
                      </td>
                      <td className="py-2.5 px-4 text-center">
                        <span className={`px-2 py-0.5 rounded-sm text-[10px] font-bold uppercase ${row.gru === "phishing" ? "bg-anomaly/10 text-anomaly border border-anomaly/20" : "bg-success/10 text-success border border-success/20"}`}>
                          {row.gru}
                        </span>
                      </td>
                      <td className="py-2.5 px-4 text-center">
                        {row.anomaly ? <AlertCircle className="inline h-3.5 w-3.5 text-anomaly" /> : <span className="text-muted-foreground">—</span>}
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
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                <XAxis dataKey="date" tick={{ fontSize: 9, fill: "var(--color-muted-foreground)" }} interval={14} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ backgroundColor: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: "4px", fontSize: "11px" }} />
                <Line type="monotone" dataKey="actual" stroke="var(--color-muted-foreground)" strokeWidth={1.5} dot={false} name="Real" />
                <Line type="monotone" dataKey="lstm" stroke="var(--chart-1)" strokeWidth={2} dot={false} name="LSTM" />
                <Line type="monotone" dataKey="gru" stroke="var(--chart-2)" strokeWidth={2} dot={false} name="GRU" />
                <Scatter
                  dataKey="actual"
                  data={filteredTS.filter(d => d.anomaly)}
                  fill="var(--color-anomaly)"
                  name="Anomalía"
                />
              </LineChart>
            </ResponsiveContainer>
            <div className="flex gap-4 justify-center mt-4">
              <div className="flex items-center gap-1.5"><div className="w-3 h-0.5 bg-muted-foreground" /><span className="text-[10px] text-muted-foreground font-bold uppercase tracking-tighter">Real</span></div>
              <div className="flex items-center gap-1.5"><div className="w-3 h-0.5 bg-primary" /><span className="text-[10px] text-muted-foreground font-bold uppercase tracking-tighter">LSTM</span></div>
              <div className="flex items-center gap-1.5"><div className="w-3 h-0.5 bg-secondary" /><span className="text-[10px] text-muted-foreground font-bold uppercase tracking-tighter">GRU</span></div>
              <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-anomaly" /><span className="text-[10px] text-muted-foreground font-bold uppercase tracking-tighter">Anomalía</span></div>
            </div>
          </ChartCard>

          <ChartCard title="Registro de Puntos Anómalos" subtitle="Identificación exacta de las fechas con comportamiento atípico en la serie temporal" delay={0.3}>
            <div className="overflow-x-auto -mx-6 px-6">
              <table className="w-full text-xs min-w-[600px]">
                <thead>
                  <tr className="border-b border-border bg-muted/20">
                    <th className="text-left py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">Fecha del Evento</th>
                    <th className="text-center py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">Valor Real</th>
                    <th className="text-center py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">Esperado (LSTM)</th>
                    <th className="text-center py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">Est. Desviación</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTS.filter(d => d.anomaly).map((row, idx) => (
                    <tr key={idx} className="border-b border-border hover:bg-muted/10 transition-colors">
                      <td className="py-2.5 px-4 font-data text-foreground/80 flex items-center gap-2">
                        <AlertCircle className="h-3.5 w-3.5 text-anomaly" />
                        {row.date}
                      </td>
                      <td className="py-2.5 px-4 text-center font-bold text-anomaly">
                        {row.actual.toLocaleString("es-ES")} MWh
                      </td>
                      <td className="py-2.5 px-4 text-center text-muted-foreground">
                        {row.lstm.toLocaleString("es-ES")} MWh
                      </td>
                      <td className="py-2.5 px-4 text-center">
                        <span className="px-2 py-0.5 rounded-sm lg bg-anomaly/10 text-anomaly border border-anomaly/20 font-bold">
                          + {Math.abs(row.actual - row.lstm).toFixed(1)}
                        </span>
                      </td>
                    </tr>
                  ))}
                  {filteredTS.filter(d => d.anomaly).length === 0 && (
                    <tr>
                      <td colSpan={4} className="py-6 text-center text-muted-foreground">Ninguna anomalía detectada en el periodo visible.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </ChartCard>

          <AiAnalysis type="opsd" />
        </motion.div>
      )}
      {tab === "phishtank" && <AiAnalysis type="phishtank" />}
    </div>
  );
}
