import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { BrainCircuit, FileJson, Network, Sparkles, UploadCloud } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChartCard } from "@/components/ChartCard";
import { KpiCard } from "@/components/KpiCard";
import { formatImportance, normalizeXaiReport, type XaiReport } from "@/lib/xai";

export const Route = createFileRoute("/xai")({
  head: () => ({
    meta: [
      { title: "XAI SHAP - Interpretabilidad de Modelos" },
      { name: "description", content: "Interpretacion XAI mediante SHAP temporal para modelos de anomalias" },
    ],
  }),
  component: XaiPage,
});

const modelColors: Record<string, string> = {
  lstm: "var(--chart-1)",
  gru: "var(--chart-2)",
  transformer: "var(--chart-4)",
  tcn: "var(--chart-5)",
};

function EmptyXaiState({ onFile }: { onFile: (file: File) => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex min-h-[64vh] flex-col items-center justify-center gap-6 text-center"
    >
      <div className="flex h-20 w-20 items-center justify-center rounded-full border border-primary/20 bg-primary/10 text-primary">
        <BrainCircuit className="h-10 w-10" />
      </div>
      <div>
        <h1 className="text-2xl font-bold text-foreground">Interpretabilidad XAI</h1>
        <p className="mt-2 max-w-xl text-sm text-muted-foreground">
          Carga el archivo `xai.json` generado en Google Colab para visualizar las variables y pasos temporales que mas influyeron en las predicciones.
        </p>
      </div>
      <label className="inline-flex cursor-pointer items-center gap-2 rounded-md bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground shadow-sm transition-colors hover:bg-primary/90">
        <UploadCloud className="h-4 w-4" />
        Cargar xai.json
        <input
          type="file"
          accept=".json,application/json"
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onFile(file);
          }}
        />
      </label>
    </motion.div>
  );
}

function XaiPage() {
  const [report, setReport] = useState<XaiReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    try {
      setError(null);
      const parsed = JSON.parse(await file.text());
      setReport(normalizeXaiReport(parsed));
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo leer el archivo XAI.");
    }
  };

  const topFeature = report?.global_feature_importance[0];
  const topStep = report?.global_temporal_importance[0];

  const featureData = useMemo(
    () =>
      (report?.global_feature_importance || []).slice(0, 12).map((item) => ({
        name: item.feature.length > 26 ? `${item.feature.slice(0, 26)}...` : item.feature,
        fullName: item.feature,
        importance: item.importance,
      })),
    [report],
  );

  const temporalData = useMemo(
    () =>
      (report?.global_temporal_importance || [])
        .slice()
        .sort((a, b) => a.step.localeCompare(b.step, undefined, { numeric: true }))
        .map((item) => ({
          step: item.step,
          importance: item.importance,
        })),
    [report],
  );

  if (!report) {
    return (
      <div className="space-y-4">
        {error && (
          <div className="rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm font-medium text-destructive">
            {error}
          </div>
        )}
        <EmptyXaiState onFile={handleFile} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <h1 className="text-2xl font-bold text-foreground">XAI con SHAP Temporal</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Dataset interpretado: <span className="font-semibold text-foreground">{report.dataset}</span>
          </p>
        </motion.div>

        <label className="inline-flex cursor-pointer items-center gap-2 rounded-md border border-border bg-card px-4 py-2 text-sm font-semibold text-foreground shadow-sm transition-colors hover:bg-muted">
          <FileJson className="h-4 w-4 text-primary" />
          Cambiar JSON
          <input
            type="file"
            accept=".json,application/json"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) handleFile(file);
            }}
          />
        </label>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <KpiCard title="Variable Principal" value={topFeature?.feature || "N/D"} icon={Sparkles} variant="cyan" />
        <KpiCard title="Paso Temporal Clave" value={topStep?.step || "N/D"} icon={Network} variant="violet" delay={0.1} />
        <KpiCard title="Variables Evaluadas" value={report.feature_count.toLocaleString("es-ES")} icon={BrainCircuit} variant="default" delay={0.2} />
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <ChartCard title="Importancia Global de Variables" subtitle="Promedio de impacto XAI entre modelos" delay={0.2}>
          <ResponsiveContainer width="100%" height={470}>
            <BarChart data={featureData} layout="vertical" margin={{ top: 10, right: 28, left: 28, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--color-border)" />
              <XAxis type="number" tick={{ fontSize: 10 }} />
              <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 10 }} />
              <Tooltip
                formatter={(value) => [formatImportance(Number(value)), "Importancia"]}
                labelFormatter={(_, payload) => payload?.[0]?.payload?.fullName || ""}
                contentStyle={{ backgroundColor: "var(--color-card)", borderRadius: 8, fontSize: 11 }}
              />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]} fill="var(--primary)" />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Importancia Temporal" subtitle="Pasos previos que mas alteran la probabilidad de anomalia" delay={0.3}>
          <ResponsiveContainer width="100%" height={470}>
            <BarChart data={temporalData} margin={{ top: 10, right: 28, left: 10, bottom: 35 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
              <XAxis dataKey="step" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={48} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip
                formatter={(value) => [formatImportance(Number(value)), "Importancia"]}
                contentStyle={{ backgroundColor: "var(--color-card)", borderRadius: 8, fontSize: 11 }}
              />
              <Bar dataKey="importance" fill="var(--secondary)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      <ChartCard title="Comparacion XAI por Modelo" subtitle="Variable y paso temporal dominante en cada arquitectura" delay={0.4}>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
          {report.model_comparison.map((item) => (
            <div key={item.model_key} className="rounded-md border border-border bg-muted/20 p-4">
              <div className="flex items-center justify-between gap-3">
                <p className="text-sm font-bold text-foreground">{item.model}</p>
                <span
                  className="h-3 w-3 rounded-full"
                  style={{ backgroundColor: modelColors[item.model_key] || "var(--primary)" }}
                />
              </div>
              <div className="mt-4 space-y-3">
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Variable dominante</p>
                  <p className="mt-1 break-words text-sm font-semibold text-foreground">{item.top_feature || "N/D"}</p>
                </div>
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Paso temporal</p>
                  <p className="mt-1 text-sm font-semibold text-primary">{item.top_step || "N/D"}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </ChartCard>

      <ChartCard title="Detalle por Modelo" subtitle="Top 8 variables explicativas por arquitectura" delay={0.5}>
        <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
          {Object.values(report.models).map((model) => {
            const rows = model.feature_importance.slice(0, 8);
            return (
              <div key={model.model_key} className="rounded-md border border-border bg-card p-4">
                <h3 className="text-sm font-bold text-foreground">{model.model}</h3>
                <p className="mt-1 text-xs text-muted-foreground">{model.description || "Importancia por enmascaramiento temporal."}</p>
                <div className="mt-4 space-y-2">
                  {rows.map((row) => (
                    <div key={`${model.model_key}-${row.feature}`} className="grid grid-cols-[minmax(0,1fr)_72px] items-center gap-3">
                      <span className="truncate text-xs font-medium text-muted-foreground" title={row.feature}>
                        {row.feature}
                      </span>
                      <span className="text-right font-data text-xs font-bold text-foreground">
                        {formatImportance(row.importance)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </ChartCard>
    </div>
  );
}
