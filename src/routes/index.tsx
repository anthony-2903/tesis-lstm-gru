import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { KpiCard } from "@/components/KpiCard";
import { ChartCard } from "@/components/ChartCard";
import { BackendState } from "@/components/BackendState";
import { fetchDashboardData } from "@/lib/api";
import { useApiData } from "@/hooks/useApiData";
import { Columns, Database, FileSearch } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Dashboard - Analisis de Dataset" },
      { name: "description", content: "Visualizacion interactiva de resultados procesados por el backend local" },
    ],
  }),
  component: HomePage,
});

const COLORS = [
  "var(--chart-1)", "var(--chart-2)", "var(--chart-3)",
  "var(--chart-4)", "var(--chart-5)",
];

const TYPE_COLORS: Record<string, string> = {
  numero: "var(--chart-1)",
  "número": "var(--chart-1)",
  texto: "var(--chart-2)",
  fecha: "var(--chart-3)",
  vacio: "var(--chart-4)",
  "vacío": "var(--chart-4)",
};

function HomePage() {
  const { data, error, isLoading, reload } = useApiData(fetchDashboardData);

  if (isLoading) return <BackendState isLoading />;
  if (error || !data) return <BackendState error={error} onRetry={reload} />;

  const { dataset, typeDistribution, columnBarData, numericDistribution, numericColumn } = data;

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
        <h1 className="text-2xl font-bold text-foreground">Resumen del Dataset</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Archivo procesado: <span className="font-medium text-foreground">{dataset.filename}</span>
        </p>
      </motion.div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <KpiCard title="Total de Registros" value={dataset.originalRows.toLocaleString("es-ES")} subtitle="Filas recibidas por backend" icon={Database} variant="cyan" />
        <KpiCard title="Registros Limpios" value={dataset.cleanedRows.toLocaleString("es-ES")} subtitle={`Removidos: ${dataset.rowsRemoved.toLocaleString("es-ES")} nulos`} icon={FileSearch} variant="violet" delay={0.1} />
        <KpiCard title="Columnas" value={dataset.columns.length.toLocaleString("es-ES")} subtitle="Campos detectados" icon={Columns} variant="cyan" delay={0.2} />
        <KpiCard title="Calidad de Datos" value={`${((dataset.cleanedRows / Math.max(dataset.originalRows, 1)) * 100).toFixed(1)}%`} subtitle="Registros sin valores nulos" icon={Database} variant="default" delay={0.3} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard title="Tipos de Columna" delay={0.4}>
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={typeDistribution} cx="50%" cy="50%" innerRadius={55} outerRadius={85} dataKey="value" stroke="none">
                {typeDistribution.map((entry, i) => (
                  <Cell key={entry.name} fill={TYPE_COLORS[entry.name] ?? COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ backgroundColor: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: "6px", fontSize: "12px" }} />
              <Legend iconSize={10} wrapperStyle={{ fontSize: "11px" }} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Registros por Columna" delay={0.5}>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={columnBarData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--color-border)" />
              <XAxis type="number" tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="col" tick={{ fontSize: 9, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} width={95} />
              <Tooltip contentStyle={{ backgroundColor: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: "6px", fontSize: "12px" }} />
              <Bar dataKey="registros" radius={[0, 4, 4, 0]}>
                {columnBarData.map((entry, i) => (
                  <Cell key={`${entry.col}-${i}`} fill={TYPE_COLORS[entry.tipo] ?? COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {numericDistribution.length > 0 && (
          <div className="lg:col-span-2">
            <ChartCard title={`Distribucion de valores${numericColumn ? `: ${numericColumn}` : ""}`} delay={0.6}>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={numericDistribution}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                  <XAxis dataKey="rango" tick={{ fontSize: 9, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ backgroundColor: "var(--color-card)", border: "1px solid var(--color-border)", borderRadius: "6px", fontSize: "12px" }} />
                  <Bar dataKey="cantidad" fill="var(--chart-1)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>
        )}
      </div>
    </div>
  );
}
