import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { Database, FileCheck2, RefreshCw, Server } from "lucide-react";
import { BackendState } from "@/components/BackendState";
import { KpiCard } from "@/components/KpiCard";
import { fetchDashboardData } from "@/lib/api";
import { useApiData } from "@/hooks/useApiData";

export const Route = createFileRoute("/upload")({
  head: () => ({
    meta: [
      { title: "Datos Procesados" },
      { name: "description", content: "Estado del dataset procesado por el backend local" },
    ],
  }),
  component: DataStatusPage,
});

function DataStatusPage() {
  const { data, error, isLoading, reload } = useApiData(fetchDashboardData);

  if (isLoading) return <BackendState isLoading />;
  if (error || !data) return <BackendState error={error} onRetry={reload} />;

  const { dataset } = data;

  return (
    <div className="dashboard-page">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <h1 className="text-2xl font-bold text-foreground">Datos Procesados</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Fuente actual del backend local: <span className="font-semibold text-foreground">{dataset.filename}</span>
          </p>
        </motion.div>
        <button
          type="button"
          onClick={reload}
          className="inline-flex items-center justify-center gap-2 rounded-md border border-border bg-card px-4 py-2 text-sm font-semibold text-foreground shadow-sm transition-colors hover:bg-muted"
        >
          <RefreshCw className="h-4 w-4 text-primary" />
          Actualizar
        </button>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <KpiCard title="Registros Originales" value={dataset.originalRows.toLocaleString("es-ES")} icon={Database} variant="cyan" />
        <KpiCard title="Registros Limpios" value={dataset.cleanedRows.toLocaleString("es-ES")} icon={FileCheck2} variant="violet" delay={0.1} />
        <KpiCard title="Columnas" value={dataset.columns.length.toLocaleString("es-ES")} icon={Server} variant="cyan" delay={0.2} />
        <KpiCard
          title="Calidad de Datos"
          value={`${((dataset.cleanedRows / Math.max(dataset.originalRows, 1)) * 100).toFixed(1)}%`}
          icon={FileCheck2}
          variant="default"
          delay={0.3}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="rounded-lg border border-border bg-card p-4 shadow-sm sm:p-6">
          <h2 className="mb-4 text-lg font-bold text-foreground">Estructura de Columnas</h2>
          <div className="max-h-[420px] space-y-2 overflow-y-auto pr-1">
            {dataset.columns.map((col) => (
              <div key={col} className="flex items-center justify-between gap-4 rounded-md border border-border bg-muted/20 px-3 py-2">
                <span className="truncate text-sm font-medium text-foreground" title={col}>{col}</span>
                <span className="rounded border border-border bg-card px-2 py-0.5 font-data text-[10px] font-bold text-muted-foreground">
                  {dataset.dataTypes[col] || "desconocido"}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-border bg-card p-4 shadow-sm sm:p-6">
          <h2 className="mb-4 text-lg font-bold text-foreground">Flujo Actual</h2>
          <div className="space-y-4 text-sm text-muted-foreground">
            <p>
              El frontend ya no recibe archivos del usuario. Solo consulta los resultados que el backend local preparo previamente.
            </p>
            <p>
              Endpoints esperados: <span className="font-data text-foreground">/api/dashboard</span>,{" "}
              <span className="font-data text-foreground">/api/analysis</span>,{" "}
              <span className="font-data text-foreground">/api/comparison</span>,{" "}
              <span className="font-data text-foreground">/api/history</span> y{" "}
              <span className="font-data text-foreground">/api/xai</span>.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
