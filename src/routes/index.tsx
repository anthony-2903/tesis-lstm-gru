import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { KpiCard } from "@/components/KpiCard";
import { ChartCard } from "@/components/ChartCard";
import { useDataStore } from "@/lib/dataStore";
import { Database, Columns, FileSearch, UploadCloud } from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend,
} from "recharts";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Dashboard — Análisis de Dataset" },
      { name: "description", content: "Visualización interactiva de los datos cargados por el usuario" },
    ],
  }),
  component: HomePage,
});

// ─────────────────────────────────────────────────────────
// Colores para gráficos
// ─────────────────────────────────────────────────────────
const COLORS = [
  "var(--chart-1)", "var(--chart-2)", "var(--chart-3)",
  "var(--chart-4)", "var(--chart-5)",
];

const TYPE_COLORS: Record<string, string> = {
  número: "var(--chart-1)",
  texto: "var(--chart-2)",
  fecha: "var(--chart-3)",
  vacío: "var(--chart-4)",
};

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
          <strong>Gestión de Datos</strong> para visualizar las estadísticas y gráficos de tu dataset.
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
// Página principal
// ─────────────────────────────────────────────────────────
function HomePage() {
  const { dataset } = useDataStore();

  if (!dataset) return <EmptyState />;

  // Distribución de tipos de dato para el PieChart
  const typeCount: Record<string, number> = {};
  for (const t of Object.values(dataset.dataTypes)) {
    typeCount[t] = (typeCount[t] || 0) + 1;
  }
  const typeDistribution = Object.entries(typeCount).map(([name, value]) => ({ name, value }));

  // Columnas del dataset para el BarChart horizontal
  const columnBarData = dataset.columns.slice(0, 12).map((col) => ({
    col: col.length > 14 ? col.slice(0, 14) + "…" : col,
    tipo: dataset.dataTypes[col],
    registros: dataset.cleanedRows,
  }));

  // Distribución numérica de la primera columna numérica
  const numericCol = dataset.columns.find((c) => dataset.dataTypes[c] === "número");
  const numericDist = numericCol
    ? (() => {
        const vals = dataset.allData
          .map((r) => Number(r[numericCol]))
          .filter((n) => !isNaN(n));
        if (vals.length === 0) return [];
        const min = Math.min(...vals);
        const max = Math.max(...vals);
        const buckets = 10;
        const step = (max - min) / buckets || 1;
        const counts = Array.from({ length: buckets }, (_, i) => ({
          rango: `${(min + i * step).toFixed(1)}`,
          cantidad: 0,
        }));
        for (const v of vals) {
          const idx = Math.min(Math.floor((v - min) / step), buckets - 1);
          counts[idx].cantidad++;
        }
        return counts;
      })()
    : [];

  return (
    <div className="space-y-6">
      {/* Encabezado */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
        <h1 className="text-2xl font-bold text-foreground">Resumen del Dataset</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Archivo: <span className="font-medium text-foreground">{dataset.filename}</span>
        </p>
      </motion.div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <KpiCard
          title="Total de Registros"
          value={dataset.originalRows.toLocaleString("es-ES")}
          subtitle="Filas en el archivo original"
          icon={Database}
          variant="cyan"
          delay={0}
        />
        <KpiCard
          title="Registros Limpios"
          value={dataset.cleanedRows.toLocaleString("es-ES")}
          subtitle={`Removidos: ${dataset.rowsRemoved.toLocaleString("es-ES")} nulos`}
          icon={FileSearch}
          variant="violet"
          delay={0.1}
        />
        <KpiCard
          title="Columnas"
          value={dataset.columns.length}
          subtitle="Campos detectados en el archivo"
          icon={Columns}
          variant="cyan"
          delay={0.2}
        />
        <KpiCard
          title="Calidad de Datos"
          value={`${((dataset.cleanedRows / dataset.originalRows) * 100).toFixed(1)}%`}
          subtitle="Registros sin valores nulos"
          icon={Database}
          variant="default"
          delay={0.3}
        />
      </div>

      {/* Gráficos */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        {/* Distribución de tipos de dato */}
        <ChartCard title="Tipos de Columna" delay={0.4}>
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie
                data={typeDistribution}
                cx="50%"
                cy="50%"
                innerRadius={55}
                outerRadius={85}
                dataKey="value"
                stroke="none"
              >
                {typeDistribution.map((entry, i) => (
                  <Cell key={i} fill={TYPE_COLORS[entry.name] ?? COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--color-card)",
                  border: "1px solid var(--color-border)",
                  borderRadius: "6px",
                  fontSize: "12px",
                }}
              />
              <Legend iconSize={10} wrapperStyle={{ fontSize: "11px" }} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Columnas del dataset */}
        <ChartCard title="Registros por Columna (primeras 12)" delay={0.5}>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={columnBarData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--color-border)" />
              <XAxis
                type="number"
                tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="col"
                tick={{ fontSize: 9, fill: "var(--color-muted-foreground)" }}
                axisLine={false}
                tickLine={false}
                width={95}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--color-card)",
                  border: "1px solid var(--color-border)",
                  borderRadius: "6px",
                  fontSize: "12px",
                }}
              />
              <Bar dataKey="registros" radius={[0, 4, 4, 0]}>
                {columnBarData.map((entry, i) => (
                  <Cell key={i} fill={TYPE_COLORS[entry.tipo] ?? COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Distribución numérica — ocupa las 2 columnas */}
        {numericDist.length > 0 && numericCol && (
          <div className="lg:col-span-2">
            <ChartCard title={`Distribución de valores: ${numericCol}`} delay={0.6}>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={numericDist}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--color-border)" />
                  <XAxis
                    dataKey="rango"
                    tick={{ fontSize: 9, fill: "var(--color-muted-foreground)" }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "var(--color-card)",
                      border: "1px solid var(--color-border)",
                      borderRadius: "6px",
                      fontSize: "12px",
                    }}
                  />
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
