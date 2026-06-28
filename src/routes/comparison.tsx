import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { motion } from "framer-motion";
import { ChartCard } from "@/components/ChartCard";
import { AiAnalysis } from "@/components/AiAnalysis";
import { BackendState } from "@/components/BackendState";
import { ConclusionPanel } from "@/components/ConclusionPanel";
import { MetricGuide } from "@/components/MetricGuide";
import { DomainId, fetchComparisonData } from "@/lib/api";
import { DOMAIN_OPTIONS, getDomainOption, getInitialDomain } from "@/lib/domains";
import { useApiData } from "@/hooks/useApiData";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

export const Route = createFileRoute("/comparison")({
  head: () => ({
    meta: [
      { title: "Comparativa LSTM vs GRU vs BRNN vs Transformer vs TCN" },
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

const modelColor = (model: string) => {
  if (model === "GRU") return "var(--chart-2)";
  if (model === "BRNN") return "var(--chart-3)";
  if (model === "Transformer") return "var(--chart-4)";
  if (model === "TCN") return "var(--chart-5)";
  return "var(--chart-1)";
};

function ComparisonPage() {
  const [domain, setDomain] = useState<DomainId>(getInitialDomain);
  const selected = getDomainOption(domain);
  const { data, error, isLoading, reload } = useApiData(() => fetchComparisonData(domain), [domain]);

  if (isLoading) return <BackendState isLoading />;
  if (error || !data) return <BackendState error={error} onRetry={reload} />;

  const { filename, radarData, comparisonBarData, scatterData, comparisonTable } = data;

  return (
    <div className="dashboard-page">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <h1 className="text-2xl font-bold text-foreground">Comparativa Multimodelo</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Dataset: <span className="font-semibold text-foreground">{filename}</span>
        </p>
      </motion.div>

      <div className="flex flex-wrap gap-2 rounded-md border border-border bg-muted/30 p-1">
        {DOMAIN_OPTIONS.map((option) => (
          <button
            key={option.id}
            onClick={() => setDomain(option.id)}
            className={`rounded-md px-3 py-2 text-xs font-bold transition-all ${
              domain === option.id ? "bg-card text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {option.shortTitle}
          </button>
        ))}
      </div>

      <MetricGuide />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard title="Perfil Multidimensional" subtitle="Métricas normalizadas (0-1)" delay={0.1}>
          <div className="chart-shell">
          <div className="chart-min">
          <ResponsiveContainer width="100%" height={320}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="var(--color-border)" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} />
              <PolarRadiusAxis tick={{ fontSize: 9, fill: "var(--color-muted-foreground)" }} domain={[0, 1]} axisLine={false} />
              {["LSTM", "GRU", "BRNN", "Transformer", "TCN"].map((model) => (
                <Radar key={model} name={model} dataKey={model} stroke={modelColor(model)} fill={modelColor(model)} fillOpacity={0.1} strokeWidth={2} />
              ))}
              <Tooltip contentStyle={tooltipStyle} />
              <Legend verticalAlign="bottom" wrapperStyle={{ fontSize: "10px", paddingTop: "20px" }} />
            </RadarChart>
          </ResponsiveContainer>
          </div>
          </div>
        </ChartCard>

        <ChartCard title="Eficiencia: Tiempo vs F1-Score" subtitle="Rendimiento consolidado" delay={0.2}>
          <div className="chart-shell">
          <div className="chart-min">
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
              <XAxis type="number" dataKey="time" name="Tiempo (s)" tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} />
              <YAxis type="number" dataKey="accuracy" name="F1-Score" tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} domain={[0.5, 1]} axisLine={false} tickLine={false} />
              <ZAxis range={[150, 150]} />
              <Tooltip contentStyle={tooltipStyle} formatter={(value: unknown) => String(typeof value === "number" ? value.toFixed(3) : value)} />
              <Scatter data={scatterData} name="Modelos">
                {scatterData.map((entry) => <Cell key={entry.model} fill={modelColor(entry.model)} />)}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          </div>
          </div>
        </ChartCard>
      </div>

      <ChartCard title="Métricas Comparativas de Clasificación" delay={0.3}>
        <div className="chart-shell">
        <div className="chart-min">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={comparisonBarData}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
            <XAxis dataKey="metric" tick={{ fontSize: 9, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={tooltipStyle} />
            {["LSTM", "GRU", "BRNN", "Transformer", "TCN"].map((model) => (
              <Bar key={model} dataKey={model} fill={modelColor(model)} radius={[2, 2, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
        </div>
        </div>
      </ChartCard>

      <ChartCard title="Benchmarks de Arquitectura" subtitle="Desempeño final consolidado" delay={0.4}>
        <div className="responsive-table -mx-4 px-4 sm:-mx-6 sm:px-6">
          <table className="w-full min-w-[720px] text-xs">
            <thead>
              <tr className="border-b border-border bg-muted/20">
                {["Métrica", "LSTM", "GRU", "BRNN", "Transformer", "TCN", "Líder"].map((header) => (
                  <th key={header} className="px-4 py-3 text-center font-bold uppercase tracking-wider text-muted-foreground first:text-left">{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {comparisonTable.map((row) => (
                <tr key={row.metric} className="border-b border-border transition-colors hover:bg-muted/10">
                  <td className="px-4 py-2.5 font-bold text-foreground/80">{row.metric}</td>
                  {(["lstm", "gru", "brnn", "transformer", "tcn"] as const).map((key) => (
                    <td key={key} className={`px-4 py-2.5 text-center font-data ${row.winner.toLowerCase() === key ? "font-bold text-primary underline decoration-primary/30" : "text-muted-foreground"}`}>
                      {row[key] < 10 ? row[key].toFixed(3) : row[key]}
                    </td>
                  ))}
                  <td className="px-4 py-2.5 text-center">
                    <span className="rounded-sm border border-primary/20 bg-primary/10 px-2 py-0.5 text-[10px] font-bold uppercase text-primary">{row.winner}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </ChartCard>

      <ConclusionPanel />

      <AiAnalysis type={selected.aiType} />
    </div>
  );
}
