import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { ChartCard } from "@/components/ChartCard";
import { radarData, comparisonBarData, scatterData, comparisonTable } from "@/lib/mock-data";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Cell, ZAxis, Legend
} from "recharts";
import { AiAnalysis } from "@/components/AiAnalysis";

export const Route = createFileRoute("/comparison")({
  head: () => ({
    meta: [
      { title: "Comparativa LSTM vs GRU vs Transformer vs TCN" },
      { name: "description", content: "Análisis multidimensional de modelos avanzados" },
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
        <h1 className="text-2xl font-bold text-foreground">Comparativa Multimodelo</h1>
        <p className="text-sm text-muted-foreground mt-1">Análisis de LSTM, GRU, Transformer y TCN</p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Radar chart */}
        <ChartCard title="Perfil Multidimensional" subtitle="Métricas normalizadas (0-1)" delay={0.1}>
          <ResponsiveContainer width="100%" height={320}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="var(--color-border)" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} />
              <PolarRadiusAxis tick={{ fontSize: 9, fill: "var(--color-muted-foreground)" }} domain={[0, 1]} axisLine={false} />
              <Radar name="LSTM" dataKey="LSTM" stroke="var(--chart-1)" fill="var(--chart-1)" fillOpacity={0.1} strokeWidth={2} />
              <Radar name="GRU" dataKey="GRU" stroke="var(--chart-2)" fill="var(--chart-2)" fillOpacity={0.1} strokeWidth={2} />
              <Radar name="Transformer" dataKey="Transformer" stroke="var(--chart-4)" fill="var(--chart-4)" fillOpacity={0.1} strokeWidth={2} />
              <Radar name="TCN" dataKey="TCN" stroke="var(--chart-5)" fill="var(--chart-5)" fillOpacity={0.1} strokeWidth={2} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend verticalAlign="bottom" wrapperStyle={{ fontSize: '10px', paddingTop: '20px' }} />
            </RadarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Scatter: time vs accuracy */}
        <ChartCard title="Eficiencia: Tiempo vs F1-Score" subtitle="Distribución por modelo y dataset" delay={0.2}>
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
              <XAxis type="number" dataKey="time" name="Tiempo (s)" tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} label={{ value: "Tiempo (s)", position: "insideBottom", offset: -5, fontSize: 10, fill: "var(--color-muted-foreground)" }} />
              <YAxis type="number" dataKey="accuracy" name="F1-Score" tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} domain={[0.8, 1]} axisLine={false} tickLine={false} label={{ value: "F1-Score", angle: -90, position: "insideLeft", fontSize: 10, fill: "var(--color-muted-foreground)" }} />
              <ZAxis range={[100, 100]} />
              <Tooltip contentStyle={tooltipStyle} formatter={(value: unknown) => String(typeof value === 'number' ? value.toFixed(3) : value)} />
              <Scatter data={scatterData} name="Modelos">
                {scatterData.map((entry, i) => {
                  let color = "var(--chart-1)";
                  if (entry.model.includes("GRU")) color = "var(--chart-2)";
                  if (entry.model.includes("Transf")) color = "var(--chart-4)";
                  if (entry.model.includes("TCN")) color = "var(--chart-5)";
                  return <Cell key={i} fill={color} />;
                })}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-x-4 gap-y-2 justify-center mt-2 px-4">
            {["LSTM", "GRU", "Transformer", "TCN"].map((m, idx) => (
              <div key={m} className="flex items-center gap-1.5">
                <div className={`w-2 h-2 rounded-full bg-[var(--chart-${idx === 2 ? 4 : idx === 3 ? 5 : idx + 1})]`} />
                <span className="text-[10px] text-muted-foreground font-bold">{m}</span>
              </div>
            ))}
          </div>
        </ChartCard>
      </div>

      {/* Grouped bar chart */}
      <ChartCard title="Métricas Comparativas por Dominio" delay={0.3}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={comparisonBarData}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
            <XAxis dataKey="metric" tick={{ fontSize: 9, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={tooltipStyle} />
            <Bar dataKey="LSTM" fill="var(--chart-1)" radius={[2, 2, 0, 0]} />
            <Bar dataKey="GRU" fill="var(--chart-2)" radius={[2, 2, 0, 0]} />
            <Bar dataKey="Transformer" fill="var(--chart-4)" radius={[2, 2, 0, 0]} />
            <Bar dataKey="TCN" fill="var(--chart-5)" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Comparison table */}
      <ChartCard title="Benchmarks de Arquitectura" subtitle="Desempeño final consolidado" delay={0.4}>
        <div className="overflow-x-auto -mx-6 px-6">
          <table className="w-full text-xs min-w-[600px]">
            <thead>
              <tr className="border-b border-border bg-muted/20">
                <th className="text-left py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">Métrica</th>
                <th className="text-center py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">LSTM</th>
                <th className="text-center py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">GRU</th>
                <th className="text-center py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">Transformer</th>
                <th className="text-center py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">TCN</th>
                <th className="text-center py-3 px-4 text-muted-foreground font-bold uppercase tracking-wider">Líder</th>
              </tr>
            </thead>
            <tbody>
              {comparisonTable.map((row) => (
                <tr key={row.metric} className="border-b border-border hover:bg-muted/10 transition-colors">
                  <td className="py-2.5 px-4 font-bold text-foreground/80">{row.metric}</td>
                  <td className={`py-2.5 px-4 text-center font-data ${row.winner === "LSTM" ? "text-primary font-bold underline decoration-primary/30" : "text-muted-foreground"}`}>
                    {typeof row.lstm === "number" && row.lstm < 1 ? row.lstm.toFixed(3) : row.lstm}
                  </td>
                  <td className={`py-2.5 px-4 text-center font-data ${row.winner === "GRU" ? "text-primary font-bold underline decoration-primary/30" : "text-muted-foreground"}`}>
                    {typeof row.gru === "number" && row.gru < 1 ? row.gru.toFixed(3) : row.gru}
                  </td>
                  <td className={`py-2.5 px-4 text-center font-data ${row.winner === "Transformer" ? "text-primary font-bold underline decoration-primary/30" : "text-muted-foreground"}`}>
                    {typeof row.transformer === "number" && row.transformer < 1 ? row.transformer.toFixed(3) : row.transformer}
                  </td>
                  <td className={`py-2.5 px-4 text-center font-data ${row.winner === "TCN" ? "text-primary font-bold underline decoration-primary/30" : "text-muted-foreground"}`}>
                    {typeof row.tcn === "number" && row.tcn < 1 ? row.tcn.toFixed(3) : row.tcn}
                  </td>
                  <td className="py-2.5 px-4 text-center">
                    <span className="px-2 py-0.5 rounded-sm text-[10px] font-bold uppercase bg-primary/10 text-primary border border-primary/20">
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
