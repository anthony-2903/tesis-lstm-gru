import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { motion } from "framer-motion";
import { ChartCard } from "@/components/ChartCard";
import { KpiCard } from "@/components/KpiCard";
import { Shield, Zap, TrendingUp, AlertCircle, Landmark, Activity, BarChart3, CheckCircle2, Download, LockKeyhole, Trophy } from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend
} from "recharts";
import { AiAnalysis } from "@/components/AiAnalysis";
import { BackendState } from "@/components/BackendState";
import { MetricGuide } from "@/components/MetricGuide";
import { AnalysisInterpretationPanel, ConfusionMatrixGuide, ModelRankingPanel } from "@/components/AcademicPanels";
import { AnalysisPerformanceHeatmap, DetectionBalanceChart } from "@/components/rosen/ResearchCharts";
import { DomainId, fetchAnalysisData, fetchFinanceDatasetStatus, fetchLatestFinanceExperiment, fetchLatestFinanceStacking, fetchLatestFinanceValidation, type FinanceDatasetStatus } from "@/lib/api";
import { getDomainOption, getInitialDomain } from "@/lib/domains";
import { useApiData } from "@/hooks/useApiData";

function toCsv(rows: Record<string, unknown>[]) {
  if (!rows.length) return "";
  const headers = Array.from(rows.reduce((set, row) => {
    Object.keys(row).forEach((key) => set.add(key));
    return set;
  }, new Set<string>()));
  const escapeCell = (value: unknown) => `"${String(value ?? "").replace(/"/g, '""')}"`;
  return [headers.join(","), ...rows.map((row) => headers.map((header) => escapeCell(row[header])).join(","))].join("\n");
}

export const Route = createFileRoute("/analysis")({
  head: () => ({
    meta: [
      { title: "Análisis Detallado — LSTM vs GRU vs BRNN vs Transformer vs TCN" },
      { name: "description", content: "Evaluación profunda de modelos sobre el dataset cargado" },
    ],
  }),
  component: AnalysisPage,
});

// ─────────────────────────────────────────────────────────
function ConfusionMatrixViz({
  title,
  matrix,
  colorClass,
}: {
  title: string;
  matrix: { tp: number; fp: number; fn: number; tn: number };
  colorClass: string;
}) {
  const cells = [
    { label: "VP", value: matrix.tp, color: "bg-success/20 text-success" },
    { label: "FP", value: matrix.fp, color: "bg-anomaly/20 text-anomaly" },
    { label: "FN", value: matrix.fn, color: "bg-warning/20 text-warning" },
    { label: "VN", value: matrix.tn, color: colorClass },
  ];

  return (
    <div className="flex-1">
      <p className="mb-2 text-center text-[10px] font-bold uppercase tracking-widest text-muted-foreground">{title}</p>
      <div className="grid grid-cols-2 gap-1">
        {cells.map((c) => (
          <div key={c.label} className={`rounded p-2 text-center ${c.color}`}>
            <p className="text-[8px] uppercase tracking-wider opacity-70">{c.label}</p>
            <p className="font-data text-sm font-bold">{c.value.toLocaleString("es-ES")}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function PredictionDeviationChart({
  title,
  subtitle,
  data,
  delay = 0.3,
}: {
  title: string;
  subtitle: string;
  data: {
    date: string;
    actual: number;
    lstm: number;
    gru: number;
    brnn: number;
    transformer: number;
    tcn: number;
  }[];
  delay?: number;
}) {
  return (
    <ChartCard title={title} subtitle={subtitle} delay={delay}>
      <div className="chart-shell">
      <div className="chart-min">
      <ResponsiveContainer width="100%" height={540}>
        <LineChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 40 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
          <XAxis dataKey="date" tick={{ fontSize: 9, fill: "var(--color-muted-foreground)" }} interval={Math.ceil(data.length / 12)} angle={-30} textAnchor="end" height={55} />
          <YAxis tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} width={60} />
          <Tooltip contentStyle={{ backgroundColor: "var(--color-card)", borderRadius: "8px", fontSize: "11px" }} />
          <Legend iconType="circle" verticalAlign="top" wrapperStyle={{ fontSize: "11px", paddingBottom: "12px" }} />
          <Line type="monotone" dataKey="brnn" stroke="var(--chart-3)" strokeWidth={1.5} dot={false} name="BRNN" />
          <Line type="monotone" dataKey="gru" stroke="var(--chart-2)" strokeWidth={1.5} dot={false} name="GRU" />
          <Line type="monotone" dataKey="lstm" stroke="var(--chart-1)" strokeWidth={1.5} dot={false} name="LSTM" />
          <Line type="monotone" dataKey="tcn" stroke="var(--chart-5)" strokeWidth={1.5} dot={false} name="TCN" />
          <Line type="monotone" dataKey="transformer" stroke="var(--chart-4)" strokeWidth={1.5} dot={false} name="Transformer" />
          <Line type="monotone" dataKey="actual" stroke="var(--foreground)" strokeWidth={2} dot={false} name="Valor Real" />
        </LineChart>
      </ResponsiveContainer>
      </div>
      </div>
    </ChartCard>
  );
}

function FinanceAnalysisGate({ data, onTabChange }: { data: FinanceDatasetStatus; onTabChange: (domain: DomainId) => void }) {
  const { data: experiment, error: experimentError, isLoading: experimentLoading } = useApiData(fetchLatestFinanceExperiment);
  const { data: stacking } = useApiData(fetchLatestFinanceStacking);
  const { data: temporalValidation } = useApiData(fetchLatestFinanceValidation);
  const result = experiment?.available ? experiment : null;
  const stackingResult = stacking?.available ? stacking : null;
  const validationResult = temporalValidation?.available ? temporalValidation : null;
  const ranking = validationResult
    ? validationResult.comparison.map((row) => ({ id: row.candidateId, name: analysisFinanceCandidateName(row.candidateId, row.sourceCandidateId ?? undefined), prAucMean: row.prAuc, prAucStd: 0, rocAucMean: row.rocAuc, f1Mean: row.f1, recallMean: row.recall, mccMean: row.mcc, brierScoreMean: row.brierScore, falsePositiveRateMean: row.falsePositiveRate }))
    : stackingResult
      ? stackingResult.sixModelComparison.map((row) => ({ ...row, id: row.candidateId, name: analysisFinanceCandidateName(row.candidateId, row.sourceCandidateId) }))
      : result?.comparison.map((row) => ({ ...row, id: row.modelId, name: row.displayName })) ?? [];
  const coverageRows = validationResult?.selection.rows ?? stackingResult?.validation.coverageRows ?? result?.dataset.oofRowsUsedPerSeed ?? 0;
  const testUsed = validationResult?.validation.testSetUsed ?? stackingResult?.validation.testSetUsed ?? result?.validation.testSetUsed ?? false;
  return (
    <div className="dashboard-page space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Análisis financiero</h1>
        <p className="mt-1 text-sm text-muted-foreground">Detección temporal de fraude por transacción. Las tablas administrativas ya no se presentan como fraude etiquetado.</p>
      </div>
      <div className="flex w-full flex-wrap gap-2 rounded-xl border border-border bg-muted/30 p-1 sm:w-fit">
        <button onClick={() => onTabChange("phishing")} className="flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-bold text-muted-foreground hover:text-foreground"><Shield className="h-3.5 w-3.5" />Phishing</button>
        <button onClick={() => onTabChange("energia")} className="flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-bold text-muted-foreground hover:text-foreground"><Zap className="h-3.5 w-3.5" />Energía</button>
        <button className="flex items-center gap-2 rounded-lg bg-card px-4 py-2 text-xs font-bold text-primary shadow-sm"><Landmark className="h-3.5 w-3.5" />Finanzas</button>
      </div>
      <section className="rounded-md border border-border bg-card p-5 shadow-sm">
        <div className="flex items-start gap-3">
          {data.readyForPipelinePilot ? <CheckCircle2 className="mt-0.5 h-5 w-5 text-success" /> : <AlertCircle className="mt-0.5 h-5 w-5 text-warning" />}
          <div>
            <h2 className="text-sm font-bold text-foreground">{data.readyForPipelinePilot ? "Benchmark listo para secuencias y OOF" : "Preparación financiera pendiente"}</h2>
            <p className="mt-1 text-xs text-muted-foreground">{data.message}</p>
          </div>
        </div>
        {data.available && <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">TRANSACCIONES</p><p className="mt-1 text-lg font-bold">{(data.audit?.rows ?? 0).toLocaleString("es-ES")}</p></div>
          <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">FRAUDES</p><p className="mt-1 text-lg font-bold">{(data.audit?.fraudRows ?? 0).toLocaleString("es-ES")}</p></div>
          <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">TASA DE FRAUDE</p><p className="mt-1 text-lg font-bold">{((data.audit?.fraudRate ?? 0) * 100).toFixed(2)}%</p></div>
          <div className="rounded-md border border-border p-3"><p className="text-[10px] font-bold text-muted-foreground">TEST</p><p className="mt-1 flex items-center gap-1 text-sm font-bold"><LockKeyhole className="h-4 w-4" />Bloqueado</p></div>
        </div>}
      </section>
      {ranking.length ? <section className="rounded-md border border-border bg-card p-5 shadow-sm">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h2 className="text-sm font-bold text-foreground">{stackingResult ? "Ranking financiero: cinco modelos base + Stacking" : "Ranking financiero de modelos base"}</h2>
            <p className="mt-1 text-xs text-muted-foreground">Comparación temporal OOF ordenada por PR-AUC sobre las mismas transacciones y folds evaluables.</p>
          </div>
          <span className="rounded-full border border-warning/30 bg-warning/5 px-3 py-1 text-[10px] font-bold text-warning">PILOTO TÉCNICO</span>
        </div>
        <div className="mt-4 overflow-x-auto rounded-md border border-border">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-border text-muted-foreground"><tr><th className="px-3 py-2">Posición</th><th className="px-3 py-2">Modelo</th><th className="px-3 py-2">PR-AUC</th><th className="px-3 py-2">ROC-AUC</th><th className="px-3 py-2">F1</th><th className="px-3 py-2">Recall</th><th className="px-3 py-2">MCC</th><th className="px-3 py-2">Brier ↓</th><th className="px-3 py-2">FPR</th></tr></thead>
            <tbody>{ranking.map((row, index) => <tr key={row.id} className={`border-b border-border/50 ${row.id === "stacking" ? "bg-primary/5" : ""}`}><td className="px-3 py-2 font-bold">{index + 1}</td><td className="px-3 py-2 font-bold">{index === 0 && <Trophy className="mr-1 inline h-3.5 w-3.5 text-warning" />}{row.name}</td><td className="px-3 py-2">{row.prAucMean.toFixed(4)} ± {row.prAucStd.toFixed(4)}</td><td className="px-3 py-2">{row.rocAucMean.toFixed(4)}</td><td className="px-3 py-2">{row.f1Mean.toFixed(4)}</td><td className="px-3 py-2">{row.recallMean.toFixed(4)}</td><td className="px-3 py-2">{row.mccMean.toFixed(4)}</td><td className="px-3 py-2">{row.brierScoreMean.toFixed(4)}</td><td className="px-3 py-2">{row.falsePositiveRateMean.toFixed(4)}</td></tr>)}</tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-muted-foreground">Cobertura común: {coverageRows.toLocaleString("es-ES")} transacciones · test utilizado: {testUsed ? "sí" : "no"} · cross-fitting temporal: {stackingResult?.validation.strictTemporalCrossFit ? "verificado" : "pendiente"}.</p>
        <p className="mt-3 rounded-md border border-warning/30 bg-warning/5 p-3 text-xs text-muted-foreground">{stackingResult ? (stackingResult.recommendation.stackingBeatsBestBaseOnMeanPrAuc ? "Stacking supera provisionalmente al mejor modelo base; falta validación externa." : "Stacking no supera al mejor modelo base en este piloto. El resultado se conserva sin modificar el ranking.") : result?.methodology.warning}</p>
      </section> : <section className="rounded-md border border-warning/30 bg-warning/5 p-4">
        <h2 className="text-sm font-bold text-foreground">Ranking financiero pendiente</h2>
        <p className="mt-2 text-xs text-muted-foreground">{experimentLoading ? "Consultando experimentos..." : experimentError ?? "Ejecute los cinco modelos base desde el módulo financiero para construir una comparación OOF válida."}</p>
      </section>}
      <a href="/finance" className="inline-flex rounded-md bg-primary px-4 py-2 text-xs font-bold text-primary-foreground">Abrir preparación financiera</a>
    </div>
  );
}

function analysisFinanceCandidateName(candidateId: string, sourceCandidateId?: string) {
  const names: Record<string, string> = {
    lstm: "LSTM",
    gru: "GRU",
    brnn: "BiRNN",
    tcn: "TCN",
    transformer: "Transformer",
    stacking: "Stacking",
    stacking_logistic: "Stacking logístico",
    stacking_gradient_boosting: "Stacking Gradient Boosting",
  };
  if (candidateId === "stacking" && sourceCandidateId) return `Stacking (${names[sourceCandidateId] ?? sourceCandidateId})`;
  return names[candidateId] ?? candidateId;
}

function AnalysisPage() {
  const [tab, setTab] = useState<DomainId>(getInitialDomain);
  const selected = getDomainOption(tab);
  const { data: evaluated, error, isLoading, reload } = useApiData(() => fetchAnalysisData(tab), tab);
  const { data: financeStatus, error: financeError, isLoading: financeLoading, reload: reloadFinance } = useApiData(fetchFinanceDatasetStatus);

  if (tab === "finanzas") {
    if (financeLoading) return <BackendState isLoading />;
    if (financeError || !financeStatus) return <BackendState error={financeError} onRetry={reloadFinance} />;
    return <FinanceAnalysisGate data={financeStatus} onTabChange={setTab} />;
  }

  if (isLoading) return <BackendState isLoading />;
  if (error || !evaluated) return <BackendState error={error} onRetry={reload} />;

  const { models, timeline, samples, realAnomaliesCount } = evaluated;

  // 1. Datos para la barra de PhishTank (NLP)
  const phishtankBarData = [
    { name: "Anomalías (Reales)", count: realAnomaliesCount },
    { name: "Detectado LSTM", count: models.lstm.detectedCount },
    { name: "Detectado GRU", count: models.gru.detectedCount },
    { name: "Detectado BRNN", count: models.brnn.detectedCount },
    { name: "Detectado Transf.", count: models.transformer.detectedCount },
    { name: "Detectado TCN", count: models.tcn.detectedCount },
  ];

  const handleExport = () => {
    if (!evaluated) return;
    const csvContent = toCsv((evaluated.processedRecords || evaluated.samples) as Record<string, unknown>[]);
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `resultados_entrenamiento_${evaluated.filename || "dataset"}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="dashboard-page">
      {/* Encabezado */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <h1 className="text-2xl font-bold text-foreground">Análisis de Anomalías y Resultados</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Dataset Evaluado: <span className="font-semibold text-foreground">{evaluated.filename}</span>
          </p>
        </motion.div>
        
        <motion.button
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          onClick={handleExport}
          className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-success px-4 py-2 font-medium text-success-foreground shadow-sm transition-colors hover:bg-success/90 sm:w-auto"
        >
          <Download className="w-4 h-4" />
          Exportar Resultados (CSV)
        </motion.button>
      </div>

      {/* Selector de Pestañas */}
      <div className="flex w-full flex-wrap gap-2 rounded-xl border border-border bg-muted/30 p-1 sm:w-fit">
        <button
          onClick={() => setTab("phishing")}
          className={`px-4 py-2 rounded-lg text-xs font-bold transition-all flex items-center gap-2 ${
            tab === "phishing" ? "bg-card text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Shield className="h-3.5 w-3.5" />
          PhishTank (NLP / Texto)
        </button>
        <button
          onClick={() => setTab("energia")}
          className={`px-4 py-2 rounded-lg text-xs font-bold transition-all flex items-center gap-2 ${
            tab === "energia" ? "bg-card text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Zap className="h-3.5 w-3.5" />
          Energía (Series Temporales)
        </button>
        <button
          onClick={() => setTab("finanzas")}
          className={`px-4 py-2 rounded-lg text-xs font-bold transition-all flex items-center gap-2 ${
            tab === "finanzas" ? "bg-card text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Landmark className="h-3.5 w-3.5" />
          Finanzas (Fraude / Transacciones)
        </button>
      </div>

      <section className="rounded-md border border-warning/40 bg-warning/10 p-4">
        <div className="flex items-start gap-3">
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-warning" />
          <div>
            <h2 className="text-sm font-bold text-foreground">Vista histórica de demostración</h2>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">
              Estas métricas verifican la visualización del pipeline legado y no son resultados finales de tesis: no representan la matriz 5×5, no incluyen Stacking como sexto candidato ni una revalidación independiente. Las conclusiones defendibles se generan en el protocolo sellado de cada dominio.
            </p>
            <a href={tab === "phishing" ? "/phishing" : "/energy"} className="mt-3 inline-flex rounded-md bg-primary px-3 py-2 text-xs font-bold text-primary-foreground">
              Abrir protocolo científico de {tab === "phishing" ? "phishing" : "energía"}
            </a>
          </div>
        </div>
      </section>

      <MetricGuide />
      <AnalysisInterpretationPanel domain={tab} data={evaluated} />
      <ModelRankingPanel domain={tab} data={evaluated} />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(360px,0.8fr)]">
        <AnalysisPerformanceHeatmap data={evaluated} />
        <DetectionBalanceChart data={evaluated} />
      </div>
      <ConfusionMatrixGuide />

      {/* ── SECCIÓN: PHISHTANK (TEXTO) ── */}
      {tab === "phishing" && (
        <motion.div key="phishing" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <KpiCard title="F1-Score (Transformer)" value={models.transformer.f1.toFixed(3)} icon={Activity} variant="cyan" />
            <KpiCard title="Precisión Phishing" value={`${(models.transformer.precision * 100).toFixed(1)}%`} icon={Shield} variant="violet" delay={0.1} />
            <KpiCard title="Recall Phishing" value={`${(models.transformer.recall * 100).toFixed(1)}%`} icon={TrendingUp} variant="default" delay={0.2} />
          </div>

          <PredictionDeviationChart
            title="Prediccion y Desviacion de URLs Phishing"
            subtitle={`Senal lexical completa - ${evaluated.totalRows.toLocaleString("es-ES")} registros agrupados en ${timeline.length} puntos`}
            data={timeline}
          />

          <ChartCard title="Eficacia de Detección (Total)" subtitle="Anomalías reales vs. detectadas por modelo" delay={0.4}>
            <div className="chart-shell">
            <div className="chart-min">
            <ResponsiveContainer width="100%" height={420}>
              <BarChart data={phishtankBarData} margin={{ top: 10, right: 30, left: 10, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }} angle={-20} textAnchor="end" height={55} />
                <YAxis tick={{ fontSize: 11 }} width={55} />
                <Tooltip cursor={{ fill: "rgba(0,0,0,0.05)" }} />
                <Bar dataKey="count" fill="var(--primary)" radius={[6, 6, 0, 0]} label={{ position: "top", fontSize: 11, fill: "var(--color-muted-foreground)" }} />
              </BarChart>
            </ResponsiveContainer>
            </div>
            </div>
          </ChartCard>

          <ChartCard title="Análisis de Matrices de Confusión" subtitle="VP=Verdadero Positivo · FP=Falso Positivo · FN=Falso Negativo · VN=Verdadero Negativo" delay={0.5}>
            <div className="grid grid-cols-1 gap-3 py-4 sm:grid-cols-2 lg:grid-cols-5 lg:gap-6">
              <ConfusionMatrixViz title="LSTM" matrix={models.lstm.confusionMatrix} colorClass="bg-primary/20 text-primary" />
              <ConfusionMatrixViz title="GRU" matrix={models.gru.confusionMatrix} colorClass="bg-secondary/20 text-secondary" />
              <ConfusionMatrixViz title="BRNN" matrix={models.brnn.confusionMatrix} colorClass="bg-chart-3/20 text-chart-3" />
              <ConfusionMatrixViz title="Transformer (Best)" matrix={models.transformer.confusionMatrix} colorClass="bg-chart-4/20 text-chart-4" />
              <ConfusionMatrixViz title="TCN" matrix={models.tcn.confusionMatrix} colorClass="bg-chart-5/20 text-chart-5" />
            </div>
          </ChartCard>

          <ChartCard title="Muestreo de Clasificación de Datos de Texto" subtitle={`${samples.length} registros representativos`} delay={0.6}>
            <div className="responsive-table max-h-[520px] overflow-y-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border bg-muted/20 text-muted-foreground">
                    <th className="text-left py-3 px-4 uppercase font-bold tracking-wider">Muestra de Texto</th>
                    <th className="text-center py-3 px-4 uppercase font-bold tracking-wider">Estado Real</th>
                    <th className="text-center py-3 px-4 uppercase font-bold tracking-wider">LSTM</th>
                    <th className="text-center py-3 px-4 uppercase font-bold tracking-wider">GRU</th>
                    <th className="text-center py-3 px-4 uppercase font-bold tracking-wider">BRNN</th>
                    <th className="text-center py-3 px-4 uppercase font-bold tracking-wider">Transformer</th>
                    <th className="text-center py-3 px-4 uppercase font-bold tracking-wider">TCN</th>
                  </tr>
                </thead>
                <tbody>
                  {samples.map((r) => (
                    <tr key={r.id} className="border-b border-border hover:bg-muted/10 transition-colors">
                      <td className="py-2.5 px-4 font-data text-foreground/70 truncate max-w-[300px]" title={r.label}>{r.label}</td>
                      <td className="py-2.5 px-4 text-center">
                        <span className={`px-2 py-0.5 rounded-sm font-bold uppercase text-[9px] ${
                          r.real === "anomalía" ? "bg-anomaly/10 text-anomaly" : "bg-success/10 text-success"
                        }`}>{r.real}</span>
                      </td>
                      <td className={`py-2.5 px-4 text-center font-bold ${r.lstm === r.real ? "text-success" : "text-anomaly"}`}>{r.lstm}</td>
                      <td className={`py-2.5 px-4 text-center font-bold ${r.gru === r.real ? "text-success" : "text-anomaly"}`}>{r.gru}</td>
                      <td className={`py-2.5 px-4 text-center font-bold ${r.brnn === r.real ? "text-success" : "text-anomaly"}`}>{r.brnn}</td>
                      <td className={`py-2.5 px-4 text-center font-bold ${r.transformer === r.real ? "text-success" : "text-anomaly"}`}>{r.transformer}</td>
                      <td className={`py-2.5 px-4 text-center font-bold ${r.tcn === r.real ? "text-success" : "text-anomaly"}`}>{r.tcn}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </ChartCard>
        </motion.div>
      )}

      {/* ── SECCIÓN: ENERGÍA (SERIES TEMPORALES) ── */}
      {tab === "energia" && (
        <motion.div key="energia" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <KpiCard title="Error RMSE (TCN)" value={models.tcn.rmse?.toFixed(2) || "0.00"} icon={BarChart3} variant="cyan" />
            <KpiCard title="Precisión de Anomalía" value={`${(models.tcn.precision * 100).toFixed(1)}%`} icon={Zap} variant="violet" delay={0.1} />
            <KpiCard title="Detecciones Reales" value={`${models.tcn.confusionMatrix.tp} / ${realAnomaliesCount}`} icon={Activity} variant="default" delay={0.2} />
          </div>

          <PredictionDeviationChart
            title="Prediccion y Desviacion de Consumo"
            subtitle={`Serie temporal completa - ${evaluated.totalRows.toLocaleString("es-ES")} registros agrupados en ${timeline.length} puntos`}
            data={timeline}
          />

          <ChartCard title="Distribución de Anomalías Detectadas" subtitle="Comparativa entre todos los modelos" delay={0.4}>
            <div className="chart-shell">
            <div className="chart-min">
            <ResponsiveContainer width="100%" height={420}>
              <BarChart data={phishtankBarData} margin={{ top: 10, right: 30, left: 10, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} angle={-20} textAnchor="end" height={55} />
                <YAxis tick={{ fontSize: 11 }} width={55} />
                <Tooltip />
                <Bar dataKey="count" fill="var(--secondary)" radius={[6, 6, 0, 0]} label={{ position: "top", fontSize: 11, fill: "var(--color-muted-foreground)" }} />
              </BarChart>
            </ResponsiveContainer>
            </div>
            </div>
          </ChartCard>

          <ChartCard title="Matrices de Validación" subtitle="VP · FP · FN · VN por cada arquitectura" delay={0.5}>
            <div className="grid grid-cols-1 gap-3 py-4 sm:grid-cols-2 lg:grid-cols-5 lg:gap-6">
              <ConfusionMatrixViz title="TCN (Mejor)" matrix={models.tcn.confusionMatrix} colorClass="bg-chart-5/20 text-chart-5" />
              <ConfusionMatrixViz title="Transformer" matrix={models.transformer.confusionMatrix} colorClass="bg-chart-4/20 text-chart-4" />
              <ConfusionMatrixViz title="BRNN" matrix={models.brnn.confusionMatrix} colorClass="bg-chart-3/20 text-chart-3" />
              <ConfusionMatrixViz title="GRU" matrix={models.gru.confusionMatrix} colorClass="bg-secondary/20 text-secondary" />
              <ConfusionMatrixViz title="LSTM" matrix={models.lstm.confusionMatrix} colorClass="bg-primary/20 text-primary" />
            </div>
          </ChartCard>

          <ChartCard title="Registro de Mediciones Anómalas" subtitle={`${samples.length} registros con detalle de clasificación`} delay={0.6}>
            <div className="responsive-table max-h-[540px] overflow-y-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border bg-muted/20 text-muted-foreground">
                    <th className="text-left py-3 px-4 uppercase font-bold">Registro / Fecha</th>
                    <th className="text-center py-3 px-4 uppercase font-bold">Valor</th>
                    <th className="text-center py-3 px-4 uppercase font-bold">Real</th>
                    <th className="text-center py-3 px-4 uppercase font-bold">TCN (Best)</th>
                    <th className="text-center py-3 px-4 uppercase font-bold">Alerta</th>
                  </tr>
                </thead>
                <tbody>
                  {samples.map((s) => (
                    <tr key={s.id} className="border-b border-border hover:bg-muted/10 transition-colors">
                      <td className="py-2.5 px-4 font-data">{s.label}</td>
                      <td className="py-2.5 px-4 text-center font-bold text-foreground">{s.value.toFixed(2)}</td>
                      <td className="py-2.5 px-4 text-center">
                        <span className={`px-2 py-0.5 rounded-sm font-bold uppercase text-[9px] ${
                          s.real === "anomalía" ? "bg-anomaly/10 text-anomaly" : "bg-success/10 text-success"
                        }`}>{s.real}</span>
                      </td>
                      <td className={`py-2.5 px-4 text-center font-bold ${s.tcn === s.real ? "text-success" : "text-anomaly"}`}>{s.tcn}</td>
                      <td className="py-2.5 px-4 text-center">
                        {s.real === "anomalía" && <AlertCircle className="h-4 w-4 text-anomaly inline" />}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </ChartCard>
        </motion.div>
      )}

      {/* ── SECCIÓN: FINANZAS (FRAUDE/TRANSACCIONES) ── */}
      {tab === "finanzas" && (
        <motion.div key="finanzas" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <KpiCard title="F1-Score (Transf.)" value={models.transformer.f1.toFixed(3)} icon={Landmark} variant="cyan" />
            <KpiCard title="Fraude Detectado" value={`${models.transformer.confusionMatrix.tp} / ${realAnomaliesCount}`} icon={TrendingUp} variant="violet" delay={0.1} />
            <KpiCard title="Falsos Positivos" value={`${models.transformer.confusionMatrix.fp} (Mínimo)`} icon={AlertCircle} variant="default" delay={0.2} />
          </div>

          <PredictionDeviationChart
            title="Prediccion y Desviacion de Riesgo Financiero"
            subtitle={`Score de anomalia completo - ${evaluated.totalRows.toLocaleString("es-ES")} registros agrupados en ${timeline.length} puntos`}
            data={timeline}
          />

          <ChartCard title="Volumen de Fraude Clasificado" subtitle="Comparativa de eficiencia arquitectónica" delay={0.4}>
            <div className="chart-shell">
            <div className="chart-min">
            <ResponsiveContainer width="100%" height={420}>
              <BarChart data={phishtankBarData} margin={{ top: 10, right: 30, left: 10, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} angle={-20} textAnchor="end" height={55} />
                <YAxis tick={{ fontSize: 11 }} width={55} />
                <Tooltip />
                <Bar dataKey="count" fill="var(--chart-3)" radius={[6, 6, 0, 0]} label={{ position: "top", fontSize: 11, fill: "var(--color-muted-foreground)" }} />
              </BarChart>
            </ResponsiveContainer>
            </div>
            </div>
          </ChartCard>

          <ChartCard title="Matrices de Confusión (Fraude Financiero)" delay={0.5}>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5 lg:gap-4">
              <ConfusionMatrixViz title="Transformer" matrix={models.transformer.confusionMatrix} colorClass="bg-chart-4/20 text-chart-4" />
              <ConfusionMatrixViz title="BRNN" matrix={models.brnn.confusionMatrix} colorClass="bg-chart-3/20 text-chart-3" />
              <ConfusionMatrixViz title="TCN" matrix={models.tcn.confusionMatrix} colorClass="bg-chart-5/20 text-chart-5" />
              <ConfusionMatrixViz title="GRU" matrix={models.gru.confusionMatrix} colorClass="bg-secondary/20 text-secondary" />
              <ConfusionMatrixViz title="LSTM" matrix={models.lstm.confusionMatrix} colorClass="bg-primary/20 text-primary" />
            </div>
          </ChartCard>

          <ChartCard title="Detección de Transacciones Fraudulentas" subtitle={`${samples.length} registros con clasificación completa`} delay={0.6}>
            <div className="responsive-table max-h-[540px] overflow-y-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border bg-muted/20 text-muted-foreground">
                    <th className="text-left py-3 px-4 uppercase font-bold">Identificador</th>
                    <th className="text-right py-3 px-4 uppercase font-bold">Monto / Valor</th>
                    <th className="text-center py-3 px-4 uppercase font-bold">Real</th>
                    <th className="text-center py-3 px-4 uppercase font-bold">Transf. (Best)</th>
                    <th className="text-center py-3 px-4 uppercase font-bold">Impacto</th>
                  </tr>
                </thead>
                <tbody>
                  {samples.map((t) => (
                    <tr key={t.id} className="border-b border-border hover:bg-muted/10 transition-colors">
                      <td className="py-2.5 px-4 font-data">{t.label}</td>
                      <td className="py-2.5 px-4 text-right font-bold text-foreground">{t.value.toLocaleString("es-ES", { minimumFractionDigits: 2 })}</td>
                      <td className="py-2.5 px-4 text-center">
                        <span className={`px-2 py-0.5 rounded-sm font-bold uppercase text-[9px] ${
                          t.real === "anomalía" ? "bg-anomaly/10 text-anomaly border border-anomaly/20" : "bg-success/10 text-success border border-success/20"
                        }`}>{t.real}</span>
                      </td>
                      <td className={`py-2.5 px-4 text-center font-bold ${t.transformer === t.real ? "text-success" : "text-anomaly"}`}>{t.transformer}</td>
                      <td className="py-2.5 px-4 text-center">
                        {t.real === "anomalía" ? <span className="text-anomaly font-bold">ALTO</span> : <span className="text-muted-foreground opacity-30">BAJO</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </ChartCard>
        </motion.div>
      )}

      <AiAnalysis type={selected.aiType} />
    </div>
  );
}


