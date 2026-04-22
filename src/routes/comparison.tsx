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
  backgroundColor: "var(--color-card)",
  border: "1px solid var(--color-border)",
  borderRadius: "4px",
  fontSize: "11px",
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
              <PolarGrid stroke="var(--color-border)" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} />
              <PolarRadiusAxis tick={{ fontSize: 9, fill: "var(--color-muted-foreground)" }} domain={[0, 1]} axisLine={false} />
              <Radar name="LSTM" dataKey="LSTM" stroke="var(--chart-1)" fill="var(--chart-1)" fillOpacity={0.15} strokeWidth={2} />
              <Radar name="GRU" dataKey="GRU" stroke="var(--chart-2)" fill="var(--chart-2)" fillOpacity={0.15} strokeWidth={2} />
              <Tooltip contentStyle={tooltipStyle} />
            </RadarChart>
          </ResponsiveContainer>
          <div className="flex gap-6 justify-center mt-4">
            <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm bg-primary" /><span className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">LSTM</span></div>
            <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm bg-secondary" /><span className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">GRU</span></div>
          </div>
        </ChartCard>

        {/* Scatter: time vs accuracy */}
        <ChartCard title="Tiempo de Entrenamiento vs Precisión" subtitle="Cada punto = modelo + dominio" delay={0.2}>
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
              <XAxis type="number" dataKey="time" name="Tiempo (s)" tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} label={{ value: "Tiempo (s)", position: "insideBottom", offset: -5, fontSize: 10, fill: "var(--color-muted-foreground)" }} />
              <YAxis type="number" dataKey="accuracy" name="F1-Score" tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} domain={[0.85, 1]} axisLine={false} tickLine={false} label={{ value: "F1-Score", angle: -90, position: "insideLeft", fontSize: 10, fill: "var(--color-muted-foreground)" }} />
              <ZAxis range={[100, 100]} />
              <Tooltip contentStyle={tooltipStyle} formatter={(value: unknown) => String(typeof value === 'number' ? value.toFixed(3) : value)} />
              <Scatter data={scatterData} name="Modelos">
                {scatterData.map((entry, i) => (
                  <Cell key={i} fill={entry.model.startsWith("LSTM") ? "var(--chart-1)" : "var(--chart-2)"} />
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
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
            <XAxis dataKey="metric" tick={{ fontSize: 9, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={tooltipStyle} />
            <Bar dataKey="LSTM" fill="var(--chart-1)" radius={[2, 2, 0, 0]} />
            <Bar dataKey="GRU" fill="var(--chart-2)" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Comparison table */}
      <ChartCard title="Tabla Comparativa Final" subtitle="El color indica el modelo ganador por criterio" delay={0.4}>
        <div className="overflow-x-auto -mx-6 px-6">
          <table className="w-full text-xs min-w-[500px]">
            <thead>
              <tr className="border-b border-border bg-muted/20">
                <th className="text-left py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">Métrica</th>
                <th className="text-center py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">LSTM</th>
                <th className="text-center py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">GRU</th>
                <th className="text-center py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">Ganador</th>
              </tr>
            </thead>
            <tbody>
              {comparisonTable.map((row) => (
                <tr key={row.metric} className="border-b border-border hover:bg-muted/10 transition-colors">
                  <td className="py-2.5 px-4 font-bold text-foreground/80">{row.metric}</td>
                  <td className={`py-2.5 px-4 text-center font-data ${row.winner === "LSTM" ? "text-primary font-bold" : "text-muted-foreground"}`}>
                    {typeof row.lstm === "number" && row.lstm < 1 ? row.lstm.toFixed(3) : row.lstm}
                  </td>
                  <td className={`py-2.5 px-4 text-center font-data ${row.winner === "GRU" ? "text-primary font-bold" : "text-muted-foreground"}`}>
                    {typeof row.gru === "number" && row.gru < 1 ? row.gru.toFixed(3) : row.gru}
                  </td>
                  <td className="py-2.5 px-4 text-center">
                    <span className={`px-2 py-0.5 rounded-sm text-[10px] font-bold uppercase ${row.winner === "LSTM" ? "bg-primary/10 text-primary border border-primary/20" : "bg-primary/10 text-primary border border-primary/20"}`}>
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
