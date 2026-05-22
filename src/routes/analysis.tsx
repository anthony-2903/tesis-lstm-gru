import { createFileRoute, Link } from "@tanstack/react-router";
import { useState, useMemo, useEffect } from "react";
import { motion } from "framer-motion";
import { ChartCard } from "@/components/ChartCard";
import { KpiCard } from "@/components/KpiCard";
import { useDataStore } from "@/lib/dataStore";
import { evaluateDataset } from "@/lib/evaluator";
import { Shield, Zap, TrendingUp, AlertCircle, Landmark, Activity, BarChart3, UploadCloud } from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, AreaChart, Area, Legend
} from "recharts";
import { AiAnalysis } from "@/components/AiAnalysis";

export const Route = createFileRoute("/analysis")({
  head: () => ({
    meta: [
      { title: "Análisis Detallado — LSTM vs GRU vs Transformer vs TCN" },
      { name: "description", content: "Evaluación profunda de modelos sobre el dataset cargado" },
    ],
  }),
  component: AnalysisPage,
});

// ─────────────────────────────────────────────────────────
// Componente de Matriz de Confusión
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
      <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2 text-center">
        {title}
      </p>
      <div className="grid grid-cols-2 gap-1">
        {cells.map((c) => (
          <div key={c.label} className={`rounded p-3 text-center ${c.color}`}>
            <p className="text-[8px] uppercase tracking-wider opacity-70">{c.label}</p>
            <p className="text-base font-bold font-data">{c.value.toLocaleString("es-ES")}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────
// Estado vacío (sin dataset cargado)
// ─────────────────────────────────────────────────────────
function EmptyState() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="flex flex-col items-center justify-center min-h-[60vh] text-center gap-6"
    >
      <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center">
        <UploadCloud className="w-10 h-10 text-primary" />
      </div>
      <div>
        <h2 className="text-2xl font-bold text-foreground mb-2">No hay datos cargados</h2>
        <p className="text-muted-foreground max-w-md">
          Sube un archivo CSV o Excel desde la sección de{" "}
          <strong>Gestión de Datos</strong> para ejecutar el análisis comparativo de modelos con tu dataset.
        </p>
      </div>
      <Link
        to="/upload"
        className="inline-flex items-center gap-2 bg-primary text-primary-foreground px-6 py-2.5 rounded-lg font-medium hover:bg-primary/90 transition-colors"
      >
        <UploadCloud className="w-4 h-4" />
        Cargar dataset
      </Link>
    </motion.div>
  );
}

// ─────────────────────────────────────────────────────────
// Página principal de análisis
// ─────────────────────────────────────────────────────────
// Mapa dominio → pestaña activa
const DOMAIN_TO_TAB: Record<string, "phishtank" | "energia" | "finanzas"> = {
  phishing: "phishtank",
  energia: "energia",
  finanzas: "finanzas",
  general: "phishtank",
};

// Pestañas habilitadas por dominio ("general" habilita todas)
const DOMAIN_TABS: Record<string, ("phishtank" | "energia" | "finanzas")[]> = {
  phishing: ["phishtank"],
  energia: ["energia"],
  finanzas: ["finanzas"],
  general: ["phishtank", "energia", "finanzas"],
};

function AnalysisPage() {
  const { dataset } = useDataStore();
  const [tab, setTab] = useState<"phishtank" | "energia" | "finanzas">("phishtank");

  // Auto-seleccionar la pestaña correcta cuando cambia el dataset
  useEffect(() => {
    if (dataset) {
      setTab(DOMAIN_TO_TAB[dataset.domain] ?? "phishtank");
    }
  }, [dataset]);

  const enabledTabs = dataset ? (DOMAIN_TABS[dataset.domain] ?? ["phishtank", "energia", "finanzas"]) : ["phishtank", "energia", "finanzas"];

  // Evaluación puramente matemática del dataset cargado
  const evaluated = useMemo(() => {
    if (!dataset) return null;
    return evaluateDataset(dataset);
  }, [dataset]);

  if (!dataset || !evaluated) return <EmptyState />;

  const { models, timeline, samples, realAnomaliesCount } = evaluated;

  // 1. Datos para la barra de PhishTank (NLP)
  const phishtankBarData = [
    { name: "Anomalías (Reales)", count: realAnomaliesCount },
    { name: "Detectado LSTM", count: models.lstm.detectedCount },
    { name: "Detectado GRU", count: models.gru.detectedCount },
    { name: "Detectado Transf.", count: models.transformer.detectedCount },
    { name: "Detectado TCN", count: models.tcn.detectedCount },
  ];

  // 2. Mapeo del timeline para PhishTank (ocurrencias binarias)
  const phishtankTimeline = timeline.map((t) => ({
    date: t.date,
    anomalies: t.anomalies,
    lstm: t.lstm > t.actual ? 1 : 0,
    gru: t.gru > t.actual ? 1 : 0,
    transformer: t.transformer > t.actual ? 1 : 0,
    tcn: t.tcn > t.actual ? 1 : 0,
  }));

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto px-1">
      {/* Encabezado */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <h1 className="text-2xl font-bold text-foreground">Análisis de Anomalías</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Dataset Evaluado: <span className="font-semibold text-foreground">{dataset.filename}</span>
        </p>
      </motion.div>

      {/* Selector de Pestañas */}
      <div className="flex flex-wrap gap-2 bg-muted/30 p-1.5 rounded-xl w-fit border border-border">
        {([
          { key: "phishtank", icon: Shield, label: "PhishTank (NLP / Texto)" },
          { key: "energia",   icon: Zap,    label: "Energía (Series Temporales)" },
          { key: "finanzas",  icon: Landmark,label: "Finanzas (Fraude / Transacciones)" },
        ] as const).map(({ key, icon: Icon, label }) => {
          const active = tab === key;
          const enabled = enabledTabs.includes(key);
          return (
            <button
              key={key}
              onClick={() => enabled && setTab(key)}
              title={!enabled ? "No aplica para el dominio detectado en este dataset" : undefined}
              className={`px-5 py-2.5 rounded-lg text-xs font-bold transition-all flex items-center gap-2
                ${ active ? "bg-card text-primary shadow-sm"
                  : enabled ? "text-muted-foreground hover:text-foreground"
                  : "text-muted-foreground/30 cursor-not-allowed line-through"}`}
            >
              <Icon className="h-4 w-4" />
              {label}
              {!enabled && <span className="text-[9px] font-normal ml-1 opacity-60">(N/A)</span>}
            </button>
          );
        })}
      </div>

      {/* ── SECCIÓN: PHISHTANK (TEXTO) ── */}
      {tab === "phishtank" && (
        <motion.div key="phishtank" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          {/* KPI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <KpiCard title="F1-Score (Transformer)" value={models.transformer.f1.toFixed(3)} icon={Activity} variant="cyan" />
            <KpiCard title="Precisión Phishing" value={`${(models.transformer.precision * 100).toFixed(1)}%`} icon={Shield} variant="violet" delay={0.1} />
            <KpiCard title="Recall Phishing" value={`${(models.transformer.recall * 100).toFixed(1)}%`} icon={TrendingUp} variant="default" delay={0.2} />
          </div>

          {/* Gráfico de Línea Temporal a Ancho Completo */}
          <ChartCard title="Frecuencia Temporal de Ataques" subtitle="Detecciones de anomalías distribuidas a lo largo de todo el dataset" delay={0.3}>
            <ResponsiveContainer width="100%" height={420}>
              <AreaChart data={phishtankTimeline} margin={{ left: 10, right: 10, top: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                <XAxis dataKey="date" tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} />
                <YAxis tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} />
                <Tooltip contentStyle={{ backgroundColor: "var(--color-card)", borderRadius: "8px" }} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: "11px", paddingTop: "15px" }} />
                <Area type="monotone" dataKey="lstm" stroke="var(--chart-1)" fill="var(--chart-1)" fillOpacity={0.06} strokeWidth={2} name="LSTM" />
                <Area type="monotone" dataKey="gru" stroke="var(--chart-2)" fill="var(--chart-2)" fillOpacity={0.06} strokeWidth={2} name="GRU" />
                <Area type="monotone" dataKey="transformer" stroke="var(--chart-4)" fill="var(--chart-4)" fillOpacity={0.06} strokeWidth={2} name="Transformer" />
                <Area type="monotone" dataKey="tcn" stroke="var(--chart-5)" fill="var(--chart-5)" fillOpacity={0.06} strokeWidth={2} name="TCN" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Gráfico de Eficacia (Barras) a Ancho Completo */}
          <ChartCard title="Eficacia de Detección (Total)" subtitle="Anomalías reales vs. detectadas por cada modelo" delay={0.4}>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={phishtankBarData} margin={{ left: 10, right: 10, top: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip cursor={{ fill: "var(--color-muted/10)" }} />
                <Bar dataKey="count" fill="var(--primary)" radius={[6, 6, 0, 0]} maxBarSize={60} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Matrices de Confusión */}
          <ChartCard title="Análisis de Matrices de Confusión" delay={0.5}>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 py-2">
              <ConfusionMatrixViz title="LSTM" matrix={models.lstm.confusionMatrix} colorClass="bg-primary/20 text-primary" />
              <ConfusionMatrixViz title="GRU" matrix={models.gru.confusionMatrix} colorClass="bg-secondary/20 text-secondary" />
              <ConfusionMatrixViz title="Transformer (Best)" matrix={models.transformer.confusionMatrix} colorClass="bg-chart-4/20 text-chart-4" />
              <ConfusionMatrixViz title="TCN" matrix={models.tcn.confusionMatrix} colorClass="bg-chart-5/20 text-chart-5" />
            </div>
          </ChartCard>

          {/* Tabla de clasificación */}
          <ChartCard title="Muestreo de Clasificación de Datos de Texto" delay={0.6}>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border bg-muted/20 text-muted-foreground">
                    <th className="text-left py-3.5 px-4 uppercase font-bold tracking-wider">Muestra de Texto</th>
                    <th className="text-center py-3.5 px-4 uppercase font-bold tracking-wider">Estado Real</th>
                    <th className="text-center py-3.5 px-4 uppercase font-bold tracking-wider">LSTM</th>
                    <th className="text-center py-3.5 px-4 uppercase font-bold tracking-wider">GRU</th>
                    <th className="text-center py-3.5 px-4 uppercase font-bold tracking-wider">Transformer</th>
                    <th className="text-center py-3.5 px-4 uppercase font-bold tracking-wider">TCN</th>
                  </tr>
                </thead>
                <tbody>
                  {samples.map((r) => (
                    <tr key={r.id} className="border-b border-border hover:bg-muted/10 transition-colors">
                      <td className="py-3 px-4 font-data text-foreground/70 truncate max-w-[300px]" title={r.label}>{r.label}</td>
                      <td className="py-3 px-4 text-center">
                        <span className={`px-2 py-0.5 rounded-sm font-bold uppercase text-[9px] ${
                          r.real === "anomalía" ? "bg-anomaly/10 text-anomaly" : "bg-success/10 text-success"
                        }`}>{r.real}</span>
                      </td>
                      <td className={`py-3 px-4 text-center font-bold ${r.lstm === r.real ? "text-success" : "text-anomaly"}`}>{r.lstm}</td>
                      <td className={`py-3 px-4 text-center font-bold ${r.gru === r.real ? "text-success" : "text-anomaly"}`}>{r.gru}</td>
                      <td className={`py-3 px-4 text-center font-bold ${r.transformer === r.real ? "text-success" : "text-anomaly"}`}>{r.transformer}</td>
                      <td className={`py-3 px-4 text-center font-bold ${r.tcn === r.real ? "text-success" : "text-anomaly"}`}>{r.tcn}</td>
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
          {/* KPI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <KpiCard title="Error RMSE (TCN)" value={models.tcn.rmse?.toFixed(2) || "0.00"} icon={BarChart3} variant="cyan" />
            <KpiCard title="Precisión de Anomalía" value={`${(models.tcn.precision * 100).toFixed(1)}%`} icon={Zap} variant="violet" delay={0.1} />
            <KpiCard title="Detecciones Reales" value={`${models.tcn.confusionMatrix.tp} / ${realAnomaliesCount}`} icon={Activity} variant="default" delay={0.2} />
          </div>

          {/* Gráfico de Consumo Ancho Completo */}
          <ChartCard title="Predicción y Desviación de Consumo" subtitle="Señales del dataset y correspondencia de modelos para picos de anomalías" delay={0.3}>
            <ResponsiveContainer width="100%" height={450}>
              <LineChart data={timeline} margin={{ left: 10, right: 10, top: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                <XAxis dataKey="date" tick={{ fontSize: 9, fill: "var(--color-muted-foreground)" }} />
                <YAxis tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} />
                <Tooltip />
                <Legend iconType="circle" wrapperStyle={{ fontSize: "11px", paddingTop: "15px" }} />
                <Line type="monotone" dataKey="actual" stroke="var(--foreground)" strokeWidth={2} dot={false} name="Valor Real" />
                <Line type="monotone" dataKey="lstm" stroke="var(--chart-1)" strokeWidth={1.5} dot={false} name="LSTM" />
                <Line type="monotone" dataKey="gru" stroke="var(--chart-2)" strokeWidth={1.5} dot={false} name="GRU" />
                <Line type="monotone" dataKey="transformer" stroke="var(--chart-4)" strokeWidth={1.5} dot={false} name="Transformer" />
                <Line type="monotone" dataKey="tcn" stroke="var(--chart-5)" strokeWidth={2} dot={false} name="TCN (Best)" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Distribución y Matrices */}
          <div className="grid grid-cols-1 gap-6">
            <ChartCard title="Distribución de Anomalías detectadas" delay={0.4}>
              <ResponsiveContainer width="100%" height={380}>
                <BarChart data={phishtankBarData} margin={{ left: 10, right: 10, top: 10, bottom: 5 }}>
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="var(--secondary)" radius={[6, 6, 0, 0]} maxBarSize={60} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Matrices de Validación" delay={0.5}>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 py-2">
                <ConfusionMatrixViz title="TCN (Mejor)" matrix={models.tcn.confusionMatrix} colorClass="bg-chart-5/20 text-chart-5" />
                <ConfusionMatrixViz title="Transformer" matrix={models.transformer.confusionMatrix} colorClass="bg-chart-4/20 text-chart-4" />
                <ConfusionMatrixViz title="GRU" matrix={models.gru.confusionMatrix} colorClass="bg-secondary/20 text-secondary" />
                <ConfusionMatrixViz title="LSTM" matrix={models.lstm.confusionMatrix} colorClass="bg-primary/20 text-primary" />
              </div>
            </ChartCard>
          </div>

          {/* Registro de Mediciones */}
          <ChartCard title="Registro de Mediciones Anómalas" delay={0.6}>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border bg-muted/20 text-muted-foreground">
                    <th className="text-left py-3.5 px-4 uppercase font-bold">Registro / Fecha</th>
                    <th className="text-center py-3.5 px-4 uppercase font-bold">Valor</th>
                    <th className="text-center py-3.5 px-4 uppercase font-bold">Real</th>
                    <th className="text-center py-3.5 px-4 uppercase font-bold">TCN (Best)</th>
                    <th className="text-center py-3.5 px-4 uppercase font-bold">Alerta</th>
                  </tr>
                </thead>
                <tbody>
                  {samples.map((s) => (
                    <tr key={s.id} className="border-b border-border hover:bg-muted/10 transition-colors">
                      <td className="py-3 px-4 font-data">{s.label}</td>
                      <td className="py-3 px-4 text-center font-bold text-foreground">{s.value.toFixed(2)}</td>
                      <td className="py-3 px-4 text-center">
                        <span className={`px-2 py-0.5 rounded-sm font-bold uppercase text-[9px] ${
                          s.real === "anomalía" ? "bg-anomaly/10 text-anomaly" : "bg-success/10 text-success"
                        }`}>{s.real}</span>
                      </td>
                      <td className={`py-3 px-4 text-center font-bold ${s.tcn === s.real ? "text-success" : "text-anomaly"}`}>{s.tcn}</td>
                      <td className="py-3 px-4 text-center">
                        {s.real === "anomalía" && <AlertCircle className="h-4 w-4 text-anomaly inline animate-bounce" />}
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
          {/* KPI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <KpiCard title="F1-Score (Transf.)" value={models.transformer.f1.toFixed(3)} icon={Landmark} variant="cyan" />
            <KpiCard title="Fraude Detectado" value={`${models.transformer.confusionMatrix.tp} / ${realAnomaliesCount}`} icon={TrendingUp} variant="violet" delay={0.1} />
            <KpiCard title="Falsos Positivos" value={`${models.transformer.confusionMatrix.fp} (Mínimo)`} icon={AlertCircle} variant="default" delay={0.2} />
          </div>

          {/* Timeline de Fraude Ancho Completo */}
          <ChartCard title="Timeline de Riesgo de Fraude" subtitle="Variación del score de anomalía a lo largo de todo el dataset" delay={0.3}>
            <ResponsiveContainer width="100%" height={450}>
              <LineChart data={timeline} margin={{ left: 10, right: 10, top: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                <XAxis dataKey="date" tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} />
                <YAxis />
                <Tooltip />
                <Legend iconType="circle" wrapperStyle={{ fontSize: "11px", paddingTop: "15px" }} />
                <Line type="stepAfter" dataKey="transformer" stroke="var(--chart-4)" strokeWidth={3} dot={false} name="Score Transf." />
                <Line type="stepAfter" dataKey="tcn" stroke="var(--chart-5)" strokeWidth={2} dot={false} strokeDasharray="5 5" name="Score TCN" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          {/* Gráfico y Matrices */}
          <div className="grid grid-cols-1 gap-6">
            <ChartCard title="Volumen de Fraude Clasificado" subtitle="Comparativa de eficiencia de anomalías detectadas" delay={0.4}>
              <ResponsiveContainer width="100%" height={380}>
                <BarChart data={phishtankBarData} margin={{ left: 10, right: 10, top: 10, bottom: 5 }}>
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="var(--chart-3)" radius={[6, 6, 0, 0]} maxBarSize={60} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Matrices de Confusión (Fraude Financiero)" delay={0.5}>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 py-2">
                <ConfusionMatrixViz title="Transformer" matrix={models.transformer.confusionMatrix} colorClass="bg-chart-4/20 text-chart-4" />
                <ConfusionMatrixViz title="TCN" matrix={models.tcn.confusionMatrix} colorClass="bg-chart-5/20 text-chart-5" />
                <ConfusionMatrixViz title="GRU" matrix={models.gru.confusionMatrix} colorClass="bg-secondary/20 text-secondary" />
                <ConfusionMatrixViz title="LSTM" matrix={models.lstm.confusionMatrix} colorClass="bg-primary/20 text-primary" />
              </div>
            </ChartCard>
          </div>

          {/* Tabla de Transacciones */}
          <ChartCard title="Detección de Transacciones Fraudulentas" delay={0.6}>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border bg-muted/20 text-muted-foreground">
                    <th className="text-left py-3.5 px-4 uppercase font-bold">Identificador</th>
                    <th className="text-right py-3.5 px-4 uppercase font-bold">Monto / Valor</th>
                    <th className="text-center py-3.5 px-4 uppercase font-bold">Real</th>
                    <th className="text-center py-3.5 px-4 uppercase font-bold">Transf. (Best)</th>
                    <th className="text-center py-3.5 px-4 uppercase font-bold">Impacto</th>
                  </tr>
                </thead>
                <tbody>
                  {samples.map((t) => (
                    <tr key={t.id} className="border-b border-border hover:bg-muted/10 transition-colors">
                      <td className="py-3 px-4 font-data">{t.label}</td>
                      <td className="py-3 px-4 text-right font-bold text-foreground">{t.value.toLocaleString("es-ES", { minimumFractionDigits: 2 })}</td>
                      <td className="py-3 px-4 text-center">
                        <span className={`px-2 py-0.5 rounded-sm font-bold uppercase text-[9px] ${
                          t.real === "anomalía" ? "bg-anomaly/10 text-anomaly border border-anomaly/20" : "bg-success/10 text-success border border-success/20"
                        }`}>{t.real === "anomalía" ? "fraude" : "normal"}</span>
                      </td>
                      <td className={`py-3 px-4 text-center font-bold ${t.transformer === t.real ? "text-success" : "text-anomaly"}`}>{t.transformer}</td>
                      <td className="py-3 px-4 text-center">
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

      <AiAnalysis type={tab} />
    </div>
  );
}
