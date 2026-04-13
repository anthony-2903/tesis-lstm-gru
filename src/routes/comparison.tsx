import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { ChartCard } from "@/components/ChartCard";
import { radarData, comparisonBarData, scatterData, comparisonTable } from "@/lib/mock-data";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Cell, ZAxis
} from "recharts";
import { AiAnalysis } from "@/components/AiAnalysis";

export const Route = createFileRoute("/comparison")({
  head: () => ({
    meta: [
      { title: "Comparativa LSTM vs GRU" },
      { name: "description", content: "Comparación multidimensional de modelos LSTM y GRU" },
    ],
  }),
  component: ComparisonPage,
});

const tooltipStyle = {
  backgroundColor: "oklch(0.18 0.02 260)",
  border: "1px solid oklch(0.28 0.02 260)",
  borderRadius: "8px",
  fontSize: "11px",
  color: "oklch(0.95 0.01 250)",
};

function ComparisonPage() {
  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <h1 className="text-2xl font-bold text-foreground">Comparativa LSTM vs GRU</h1>
        <p className="text-sm text-muted-foreground mt-1">Vista central de la tesis — análisis multidimensional</p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Radar chart */}
        <ChartCard title="Comparación Multidimensional" subtitle="Normalizado 0-1, mayor es mejor" delay={0.1}>
          <ResponsiveContainer width="100%" height={320}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="oklch(0.28 0.02 260)" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: "oklch(0.65 0.02 250)" }} />
              <PolarRadiusAxis tick={{ fontSize: 9, fill: "oklch(0.5 0.02 250)" }} domain={[0, 1]} />
              <Radar name="LSTM" dataKey="LSTM" stroke="oklch(0.85 0.18 195)" fill="oklch(0.85 0.18 195)" fillOpacity={0.2} strokeWidth={2} />
              <Radar name="GRU" dataKey="GRU" stroke="oklch(0.55 0.2 290)" fill="oklch(0.55 0.2 290)" fillOpacity={0.2} strokeWidth={2} />
              <Tooltip contentStyle={tooltipStyle} />
            </RadarChart>
          </ResponsiveContainer>
          <div className="flex gap-6 justify-center mt-2">
            <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-primary" /><span className="text-xs text-muted-foreground">LSTM</span></div>
            <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-secondary" /><span className="text-xs text-muted-foreground">GRU</span></div>
          </div>
        </ChartCard>

        {/* Scatter: time vs accuracy */}
        <ChartCard title="Tiempo de Entrenamiento vs Precisión" subtitle="Cada punto = modelo + dominio" delay={0.2}>
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.28 0.02 260)" />
              <XAxis type="number" dataKey="time" name="Tiempo (s)" tick={{ fontSize: 10, fill: "oklch(0.65 0.02 250)" }} label={{ value: "Tiempo (s)", position: "insideBottom", offset: -5, fontSize: 10, fill: "oklch(0.5 0.02 250)" }} />
              <YAxis type="number" dataKey="accuracy" name="F1-Score" tick={{ fontSize: 10, fill: "oklch(0.65 0.02 250)" }} domain={[0.85, 1]} label={{ value: "F1-Score", angle: -90, position: "insideLeft", fontSize: 10, fill: "oklch(0.5 0.02 250)" }} />
              <ZAxis range={[200, 200]} />
              <Tooltip contentStyle={tooltipStyle} formatter={(value: unknown) => String(typeof value === 'number' ? value.toFixed(3) : value)} />
              <Scatter data={scatterData} name="Modelos">
                {scatterData.map((entry, i) => (
                  <Cell key={i} fill={entry.model.startsWith("LSTM") ? "oklch(0.85 0.18 195)" : "oklch(0.55 0.2 290)"} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-3 justify-center mt-2">
            {scatterData.map((d) => (
              <span key={d.model} className="text-[10px] text-muted-foreground font-data">
                {d.model}: {d.accuracy.toFixed(3)} / {d.time}s
              </span>
            ))}
          </div>
        </ChartCard>
      </div>

      {/* Grouped bar chart */}
      <ChartCard title="Métricas de Desempeño Lado a Lado" delay={0.3}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={comparisonBarData}>
            <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.28 0.02 260)" />
            <XAxis dataKey="metric" tick={{ fontSize: 9, fill: "oklch(0.65 0.02 250)" }} />
            <YAxis tick={{ fontSize: 10, fill: "oklch(0.65 0.02 250)" }} />
            <Tooltip contentStyle={tooltipStyle} />
            <Bar dataKey="LSTM" fill="oklch(0.85 0.18 195)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="GRU" fill="oklch(0.55 0.2 290)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Comparison table */}
      <ChartCard title="Tabla Comparativa Final" subtitle="El color indica el modelo ganador por criterio" delay={0.4}>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-2.5 px-3 text-muted-foreground font-medium">Métrica</th>
                <th className="text-center py-2.5 px-3 text-muted-foreground font-medium">LSTM</th>
                <th className="text-center py-2.5 px-3 text-muted-foreground font-medium">GRU</th>
                <th className="text-center py-2.5 px-3 text-muted-foreground font-medium">Ganador</th>
              </tr>
            </thead>
            <tbody>
              {comparisonTable.map((row) => (
                <tr key={row.metric} className="border-b border-border/50 hover:bg-muted/30 transition-colors">
                  <td className="py-2.5 px-3 font-medium text-foreground">{row.metric}</td>
                  <td className={`py-2.5 px-3 text-center font-data ${row.winner === "LSTM" ? "text-primary font-bold" : "text-muted-foreground"}`}>
                    {typeof row.lstm === "number" && row.lstm < 1 ? row.lstm.toFixed(3) : row.lstm}
                  </td>
                  <td className={`py-2.5 px-3 text-center font-data ${row.winner === "GRU" ? "text-secondary font-bold" : "text-muted-foreground"}`}>
                    {typeof row.gru === "number" && row.gru < 1 ? row.gru.toFixed(3) : row.gru}
                  </td>
                  <td className="py-2.5 px-3 text-center">
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${row.winner === "LSTM" ? "bg-primary/20 text-primary" : "bg-secondary/20 text-secondary"}`}>
                      {row.winner}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </ChartCard>

      <AiAnalysis type="general" />
    </div>
  );
}
