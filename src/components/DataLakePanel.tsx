import { useState } from "react";
import { Archive, CheckCircle2, Database, Loader2, RefreshCw } from "lucide-react";
import type { DataLakeSummary } from "@/lib/api";
import { fetchDataLakeSummary, ingestDataLake } from "@/lib/api";
import { useApiData } from "@/hooks/useApiData";

const DOMAIN_LABELS: Record<string, string> = {
  phishing: "Phishing",
  energia: "Energia",
  finanzas: "Finanzas",
};

export function DataLakePanel() {
  const { data, error, isLoading, reload } = useApiData<DataLakeSummary>(fetchDataLakeSummary);
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestError, setIngestError] = useState<string | null>(null);

  const updateLake = async () => {
    setIsIngesting(true);
    setIngestError(null);
    try {
      await ingestDataLake("all", 5000);
      reload();
    } catch (err) {
      setIngestError(err instanceof Error ? err.message : "No se pudo actualizar el data lake.");
    } finally {
      setIsIngesting(false);
    }
  };

  return (
    <section className="card-formal p-4 sm:p-5">
      <div className="mb-5 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-md bg-primary/10 text-primary">
            <Archive className="h-4 w-4" />
          </div>
          <h2 className="text-sm font-bold text-foreground">Data lake academico</h2>
          <p className="mt-1 max-w-3xl text-xs leading-5 text-muted-foreground">
            Almacenamiento local por lotes para trabajar con miles de registros sin cargar todo en vivo.
          </p>
        </div>
        <button
          type="button"
          onClick={updateLake}
          disabled={isIngesting}
          className="inline-flex w-full items-center justify-center gap-2 rounded-md border border-border bg-card px-4 py-2 text-xs font-bold uppercase text-foreground shadow-sm transition-colors hover:bg-muted disabled:opacity-60 sm:w-auto"
        >
          {isIngesting ? <Loader2 className="h-4 w-4 animate-spin text-primary" /> : <RefreshCw className="h-4 w-4 text-primary" />}
          {isIngesting ? "Actualizando..." : "Actualizar lote 5,000"}
        </button>
      </div>

      {isLoading && (
        <div className="flex items-center gap-2 rounded-md border border-border bg-muted/20 p-4 text-xs font-semibold text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin text-primary" />
          Leyendo resumen local...
        </div>
      )}

      {(error || ingestError) && (
        <div className="mb-4 rounded-md border border-anomaly/20 bg-anomaly/10 p-4 text-xs text-anomaly">
          {error || ingestError}
        </div>
      )}

      {data && (
        <>
          <div className="mb-4 grid grid-cols-1 gap-3 md:grid-cols-[1.2fr_repeat(3,1fr)]">
            <div className="rounded-md border border-border bg-muted/20 p-4">
              <div className="mb-2 flex items-center justify-between gap-2">
                <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Total almacenado</p>
                <Database className="h-4 w-4 text-primary" />
              </div>
              <p className="font-data text-3xl font-bold text-foreground">{data.totalRecords.toLocaleString("es-ES")}</p>
              <p className="mt-1 text-[11px] text-muted-foreground">Registros listos para paginacion y procesamiento.</p>
            </div>
            {data.domains.map((domain) => (
              <div key={domain.domain} className="rounded-md border border-border bg-muted/20 p-4">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">{DOMAIN_LABELS[domain.domain]}</p>
                  <CheckCircle2 className={`h-4 w-4 ${domain.totalRecords ? "text-success" : "text-muted-foreground"}`} />
                </div>
                <p className="font-data text-2xl font-bold text-foreground">{domain.totalRecords.toLocaleString("es-ES")}</p>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  {domain.updatedAt ? `Actualizado: ${new Date(domain.updatedAt).toLocaleDateString("es-ES")}` : "Sin lote guardado"}
                </p>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
            {data.domains.map((domain) => (
              <div key={`${domain.domain}-sources`} className="rounded-md border border-border bg-card p-4">
                <h3 className="text-xs font-bold uppercase tracking-wider text-foreground">{DOMAIN_LABELS[domain.domain]}</h3>
                <div className="mt-3 space-y-2">
                  {Object.entries(domain.sourceBreakdown).length === 0 ? (
                    <p className="text-xs text-muted-foreground">Ejecuta la actualizacion para poblar este dominio.</p>
                  ) : (
                    Object.entries(domain.sourceBreakdown).map(([source, count]) => (
                      <div key={source} className="flex items-center justify-between gap-3 rounded-md border border-border bg-muted/20 px-3 py-2">
                        <span className="truncate font-data text-[11px] text-foreground">{source}</span>
                        <span className="font-data text-xs font-bold text-primary">{count.toLocaleString("es-ES")}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
