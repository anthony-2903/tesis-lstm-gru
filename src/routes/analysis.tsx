import { createFileRoute } from "@tanstack/react-router";
import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { ChartCard } from "@/components/ChartCard";
import { KpiCard } from "@/components/KpiCard";
import {
  phishtankMetrics, phishtankBarData, phishtankConfusionMatrix, phishtankUrls,
  energyMetrics, energyBarData, energyConfusionMatrix, energySamples,
  financeMetrics, financeBarData, financeConfusionMatrix, financeTransactions,
  generateTimeSeriesData, generatePhishTankTimeline, generateFinanceTimeline
} from "@/lib/mock-data";
import { Shield, Zap, TrendingUp, AlertCircle, Landmark, Clock, Activity, BarChart3 } from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, AreaChart, Area, Legend, Scatter
} from "recharts";
import { AiAnalysis } from "@/components/AiAnalysis";

export const Route = createFileRoute("/analysis")({
  head: () => ({
    meta: [
      { title: "Análisis Detallado — LSTM vs GRU vs Transformer vs TCN" },
      { name: "description", content: "Evaluación profunda de modelos en PhishTank, Energía y Finanzas" },
    ],
  }),
  component: AnalysisPage,
});

function ConfusionMatrixViz({ title, matrix, colorClass }: { title: string; matrix: { tp: number; fp: number; fn: number; tn: number }; colorClass: string }) {
  const cells = [
    { label: "VP", value: matrix.tp, color: "bg-success/20 text-success" },
    { label: "FP", value: matrix.fp, color: "bg-anomaly/20 text-anomaly" },
    { label: "FN", value: matrix.fn, color: "bg-warning/20 text-warning" },
    { label: "VN", value: matrix.tn, color: colorClass },
  ];

  return (
    <div className="flex-1">
      <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2 text-center">{title}</p>
      <div className="grid grid-cols-2 gap-1">
        {cells.map((c) => (
          <div key={c.label} className={`rounded p-2 text-center ${c.color}`}>
            <p className="text-[8px] uppercase tracking-wider opacity-70">{c.label}</p>
            <p className="text-sm font-bold font-data">{c.value.toLocaleString()}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function AnalysisPage() {
  const [tab, setTab] = useState<"phishtank" | "energia" | "finanzas">("phishtank");
  
  const phishtankTimeline = useMemo(() => generatePhishTankTimeline(), []);
  const energyData = useMemo(() => generateTimeSeriesData(), []);
  const financeTimeline = useMemo(() => generateFinanceTimeline(), []);

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <h1 className="text-2xl font-bold text-foreground">Análisis de Anomalías</h1>
        <p className="text-sm text-muted-foreground mt-1">Evaluación técnica y comparativa de arquitecturas neuronales</p>
      </motion.div>

      {/* Domain Selector */}
      <div className="flex flex-wrap gap-2 bg-muted/30 p-1 rounded-xl w-fit border border-border">
        <button onClick={() => setTab("phishtank")} className={`px-4 py-2 rounded-lg text-xs font-bold transition-all flex items-center gap-2 ${tab === "phishtank" ? "bg-card text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"}`}>
          <Shield className="h-3.5 w-3.5" />
          PhishTank (URLs)
        </button>
        <button onClick={() => setTab("energia")} className={`px-4 py-2 rounded-lg text-xs font-bold transition-all flex items-center gap-2 ${tab === "energia" ? "bg-card text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"}`}>
          <Zap className="h-3.5 w-3.5" />
          Energía (Series)
        </button>
        <button onClick={() => setTab("finanzas")} className={`px-4 py-2 rounded-lg text-xs font-bold transition-all flex items-center gap-2 ${tab === "finanzas" ? "bg-card text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"}`}>
          <Landmark className="h-3.5 w-3.5" />
          Finanzas (Fraude)
        </button>
      </div>

      {/* --- PHISHTANK SECTION --- */}
      {tab === "phishtank" && (
        <motion.div key="phishtank" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <KpiCard title="F1-Score (Transformer)" value="0.978" icon={Activity} variant="cyan" />
            <KpiCard title="Precisión Phishing" value="97.5%" icon={Shield} variant="violet" delay={0.1} />
            <KpiCard title="Recall Phishing" value="98.2%" icon={TrendingUp} variant="default" delay={0.2} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ChartCard title="Frecuencia Temporal de Ataques" subtitle="Detecciones diarias identificadas" delay={0.3}>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={phishtankTimeline}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                  <XAxis dataKey="date" hide />
                  <YAxis tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} />
                  <Tooltip contentStyle={{ backgroundColor: "var(--color-card)", borderRadius: "8px" }} />
                  <Area type="monotone" dataKey="lstm" stroke="var(--chart-1)" fill="var(--chart-1)" fillOpacity={0.1} strokeWidth={2} name="LSTM" />
                  <Area type="monotone" dataKey="gru" stroke="var(--chart-2)" fill="var(--chart-2)" fillOpacity={0.1} strokeWidth={2} name="GRU" />
                  <Area type="monotone" dataKey="transformer" stroke="var(--chart-4)" fill="var(--chart-4)" fillOpacity={0.1} strokeWidth={2} name="Transformer" />
                  <Area type="monotone" dataKey="tcn" stroke="var(--chart-5)" fill="var(--chart-5)" fillOpacity={0.1} strokeWidth={2} name="TCN" />
                  <Scatter name="Anomalías Reales" dataKey="anomalies" fill="var(--anomaly)" shape="star" />
                </AreaChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Eficacia de Detección (Total)" subtitle="Anomalías reales vs. detectadas" delay={0.4}>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={phishtankBarData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip cursor={{ fill: 'var(--color-muted/10)' }} />
                  <Bar dataKey="count" fill="var(--primary)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          <ChartCard title="Análisis de Matrices de Confusión" delay={0.5}>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <ConfusionMatrixViz title="LSTM" matrix={phishtankConfusionMatrix.lstm} colorClass="bg-primary/20 text-primary" />
              <ConfusionMatrixViz title="GRU" matrix={phishtankConfusionMatrix.gru} colorClass="bg-secondary/20 text-secondary" />
              <ConfusionMatrixViz title="Transformer (Best)" matrix={phishtankConfusionMatrix.transformer} colorClass="bg-chart-4/20 text-chart-4" />
              <ConfusionMatrixViz title="TCN" matrix={phishtankConfusionMatrix.tcn} colorClass="bg-chart-5/20 text-chart-5" />
            </div>
          </ChartCard>

          <ChartCard title="Muestreo de Clasificación de URLs" delay={0.6}>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border bg-muted/20 text-muted-foreground">
                    <th className="text-left py-3 px-4 uppercase font-bold tracking-wider">Dirección URL</th>
                    <th className="text-center py-3 px-4 uppercase font-bold tracking-wider">Estado Real</th>
                    <th className="text-center py-3 px-4 uppercase font-bold tracking-wider">LSTM</th>
                    <th className="text-center py-3 px-4 uppercase font-bold tracking-wider">GRU</th>
                    <th className="text-center py-3 px-4 uppercase font-bold tracking-wider">Transformer</th>
                    <th className="text-center py-3 px-4 uppercase font-bold tracking-wider">TCN</th>
                  </tr>
                </thead>
                <tbody>
                  {phishtankUrls.map((r) => (
                    <tr key={r.id} className="border-b border-border hover:bg-muted/10 transition-colors">
                      <td className="py-2.5 px-4 font-data text-foreground/70 truncate max-w-[300px]">{r.url}</td>
                      <td className="py-2.5 px-4 text-center">
                        <span className={`px-2 py-0.5 rounded-sm font-bold uppercase text-[9px] ${r.real === 'phishing' ? 'bg-anomaly/10 text-anomaly' : 'bg-success/10 text-success'}`}>{r.real}</span>
                      </td>
                      <td className={`py-2.5 px-4 text-center font-bold ${r.lstm === r.real ? 'text-success' : 'text-anomaly'}`}>{r.lstm}</td>
                      <td className={`py-2.5 px-4 text-center font-bold ${r.gru === r.real ? 'text-success' : 'text-anomaly'}`}>{r.gru}</td>
                      <td className={`py-2.5 px-4 text-center font-bold ${r.transformer === r.real ? 'text-success' : 'text-anomaly'}`}>{r.transformer}</td>
                      <td className={`py-2.5 px-4 text-center font-bold ${r.tcn === r.real ? 'text-success' : 'text-anomaly'}`}>{r.tcn}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </ChartCard>
        </motion.div>
      )}

      {/* --- ENERGY SECTION --- */}
      {tab === "energia" && (
        <motion.div key="energia" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <KpiCard title="Error RMSE (TCN)" value="14.80" icon={BarChart3} variant="cyan" />
            <KpiCard title="Precisión de Anomalía" value="95.2%" icon={Zap} variant="violet" delay={0.1} />
            <KpiCard title="Detecciones Reales" value="365/389" icon={Activity} variant="default" delay={0.2} />
          </div>

          <ChartCard title="Predicción y Desviación de Consumo" subtitle="Detección de picos anómalos" delay={0.3}>
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={energyData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                <XAxis dataKey="date" tick={{ fontSize: 9 }} interval={10} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
                <Line type="monotone" dataKey="actual" stroke="var(--foreground)" strokeWidth={1.5} dot={false} name="Consumo Real" />
                <Line type="monotone" dataKey="lstm" stroke="var(--chart-1)" strokeWidth={2} dot={false} name="LSTM" />
                <Line type="monotone" dataKey="gru" stroke="var(--chart-2)" strokeWidth={2} dot={false} name="GRU" />
                <Line type="monotone" dataKey="transformer" stroke="var(--chart-4)" strokeWidth={2} dot={false} name="Transformer" />
                <Line type="monotone" dataKey="tcn" stroke="var(--chart-5)" strokeWidth={2} dot={false} name="TCN" />
                <Scatter name="Anomalía Detectada" data={energyData.filter(d => d.anomaly)} fill="var(--anomaly)" shape="circle" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ChartCard title="Distribución de Anomalías Energéticas" delay={0.4}>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={energyBarData}>
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="var(--secondary)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Matrices de Validación (Energía)" delay={0.5}>
              <div className="grid grid-cols-2 gap-4">
                <ConfusionMatrixViz title="TCN (Mejor)" matrix={energyConfusionMatrix.tcn} colorClass="bg-chart-5/20 text-chart-5" />
                <ConfusionMatrixViz title="Transformer" matrix={energyConfusionMatrix.transformer} colorClass="bg-chart-4/20 text-chart-4" />
                <ConfusionMatrixViz title="GRU" matrix={energyConfusionMatrix.gru} colorClass="bg-secondary/20 text-secondary" />
                <ConfusionMatrixViz title="LSTM" matrix={energyConfusionMatrix.lstm} colorClass="bg-primary/20 text-primary" />
              </div>
            </ChartCard>
          </div>

          <ChartCard title="Registro de Mediciones Anómalas" delay={0.6}>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border bg-muted/20 text-muted-foreground">
                    <th className="text-left py-3 px-4 uppercase font-bold">Timestamp</th>
                    <th className="text-center py-3 px-4 uppercase font-bold">Valor (kW)</th>
                    <th className="text-center py-3 px-4 uppercase font-bold">Real</th>
                    <th className="text-center py-3 px-4 uppercase font-bold">TCN (Best)</th>
                    <th className="text-center py-3 px-4 uppercase font-bold">Alerta</th>
                  </tr>
                </thead>
                <tbody>
                  {energySamples.map((s) => (
                    <tr key={s.id} className="border-b border-border hover:bg-muted/10 transition-colors">
                      <td className="py-2.5 px-4 font-data">{s.date}</td>
                      <td className="py-2.5 px-4 text-center font-bold text-foreground">{s.value}</td>
                      <td className="py-2.5 px-4 text-center">
                        <span className={`px-2 py-0.5 rounded-sm font-bold uppercase text-[9px] ${s.real === 'anomalía' ? 'bg-anomaly/10 text-anomaly' : 'bg-success/10 text-success'}`}>{s.real}</span>
                      </td>
                      <td className={`py-2.5 px-4 text-center font-bold ${s.tcn === s.real ? 'text-success' : 'text-anomaly'}`}>{s.tcn}</td>
                      <td className="py-2.5 px-4 text-center">
                        {s.anomaly && <AlertCircle className="h-4 w-4 text-anomaly inline" />}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </ChartCard>
        </motion.div>
      )}

      {/* --- FINANCE SECTION --- */}
      {tab === "finanzas" && (
        <motion.div key="finanzas" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <KpiCard title="F1-Score (Transf.)" value="0.965" icon={Landmark} variant="cyan" />
            <KpiCard title="Fraude Detectado" value="810/842" icon={TrendingUp} variant="violet" delay={0.1} />
            <KpiCard title="Falsos Positivos" value="27 (Mínimo)" icon={AlertCircle} variant="default" delay={0.2} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ChartCard title="Timeline de Riesgo de Fraude" subtitle="Variación del score de anomalía" delay={0.3}>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={financeTimeline}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                  <XAxis dataKey="date" hide />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Line type="stepAfter" dataKey="transformer" stroke="var(--chart-4)" strokeWidth={3} dot={false} name="Score Transf." />
                  <Line type="stepAfter" dataKey="tcn" stroke="var(--chart-5)" strokeWidth={2} dot={false} strokeDasharray="5 5" name="Score TCN" />
                  <Scatter name="Fraude Detectado" data={financeTimeline.filter(d => d.anomaly)} fill="var(--anomaly)" shape="diamond" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Volumen de Fraude Clasificado" subtitle="Comparativa de eficiencia arquitectónica" delay={0.4}>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={financeBarData}>
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="var(--chart-3)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          <ChartCard title="Matrices de Confusión (Fraude Financiero)" delay={0.5}>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <ConfusionMatrixViz title="Transformer" matrix={financeConfusionMatrix.transformer} colorClass="bg-chart-4/20 text-chart-4" />
              <ConfusionMatrixViz title="TCN" matrix={financeConfusionMatrix.tcn} colorClass="bg-chart-5/20 text-chart-5" />
              <ConfusionMatrixViz title="GRU" matrix={financeConfusionMatrix.gru} colorClass="bg-secondary/20 text-secondary" />
              <ConfusionMatrixViz title="LSTM" matrix={financeConfusionMatrix.lstm} colorClass="bg-primary/20 text-primary" />
            </div>
          </ChartCard>

          <ChartCard title="Detección de Transacciones Fraudulentas" delay={0.6}>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border bg-muted/20 text-muted-foreground">
                    <th className="text-left py-3 px-4 uppercase font-bold">ID Transacción</th>
                    <th className="text-right py-3 px-4 uppercase font-bold">Monto ($)</th>
                    <th className="text-center py-3 px-4 uppercase font-bold">Real</th>
                    <th className="text-center py-3 px-4 uppercase font-bold">Transf. (Best)</th>
                    <th className="text-center py-3 px-4 uppercase font-bold">Impacto</th>
                  </tr>
                </thead>
                <tbody>
                  {financeTransactions.map((t) => (
                    <tr key={t.id} className="border-b border-border hover:bg-muted/10 transition-colors">
                      <td className="py-2.5 px-4 font-data">{t.txn}</td>
                      <td className="py-2.5 px-4 text-right font-bold text-foreground">{t.amount.toFixed(2)}</td>
                      <td className="py-2.5 px-4 text-center">
                        <span className={`px-2 py-0.5 rounded-sm font-bold uppercase text-[9px] ${t.real === 'fraude' ? 'bg-anomaly/10 text-anomaly border border-anomaly/20' : 'bg-success/10 text-success border border-success/20'}`}>{t.real}</span>
                      </td>
                      <td className={`py-2.5 px-4 text-center font-bold ${t.transformer === t.real ? 'text-success' : 'text-anomaly'}`}>{t.transformer}</td>
                      <td className="py-2.5 px-4 text-center">
                        {t.real === 'fraude' ? <span className="text-anomaly font-bold">ALTO</span> : <span className="text-muted-foreground opacity-30">BAJO</span>}
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
