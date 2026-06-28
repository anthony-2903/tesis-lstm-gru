import { useState } from "react";
import { motion } from "framer-motion";
import { AlertCircle, CheckCircle2, DatabaseZap, ExternalLink, KeyRound, Loader2 } from "lucide-react";
import type { DomainId, ExternalData, ExternalSourceResult } from "@/lib/api";
import { fetchExternalData } from "@/lib/api";
import { DOMAIN_OPTIONS } from "@/lib/domains";
import { useApiData } from "@/hooks/useApiData";

function statusLabel(status: ExternalSourceResult["status"]) {
  if (status === "ok") return "Consumida";
  if (status === "needs_key") return "Requiere token";
  if (status === "configured") return "Configurada";
  if (status === "reference") return "Referencia";
  return "Error";
}

function statusClass(status: ExternalSourceResult["status"]) {
  if (status === "ok") return "border-success/20 bg-success/10 text-success";
  if (status === "needs_key") return "border-warning/30 bg-warning/10 text-warning";
  if (status === "reference" || status === "configured") return "border-primary/20 bg-primary/10 text-primary";
  return "border-anomaly/20 bg-anomaly/10 text-anomaly";
}

function summarizeRecord(record: Record<string, unknown>) {
  const values = Object.values(record)
    .filter((value) => value !== null && value !== undefined && String(value).trim().length > 0)
    .slice(0, 3)
    .map(String);
  return values.join(" / ");
}

function SourceCard({ result, index }: { result: ExternalSourceResult; index: number }) {
  const Icon = result.status === "needs_key" ? KeyRound : result.status === "error" ? AlertCircle : CheckCircle2;
  const preview = result.records.slice(0, 3);

  return (
    <motion.article
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
      className="rounded-md border border-border bg-card p-4 shadow-sm"
    >
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-sm font-bold text-foreground">{result.source.name}</h3>
            {result.source.official && (
              <span className="rounded-sm border border-primary/20 bg-primary/10 px-2 py-0.5 text-[9px] font-bold uppercase text-primary">
                Oficial
              </span>
            )}
          </div>
          <p className="mt-1 text-xs text-muted-foreground">{result.source.provider}</p>
        </div>
        <span className={`inline-flex shrink-0 items-center gap-1 rounded-md border px-2 py-1 text-[10px] font-bold uppercase ${statusClass(result.status)}`}>
          <Icon className="h-3 w-3" />
          {statusLabel(result.status)}
        </span>
      </div>

      <p className="min-h-10 text-xs leading-5 text-muted-foreground">{result.source.useCase}</p>

      <div className="mt-4 grid grid-cols-2 gap-2">
        <div className="rounded-md border border-border bg-muted/20 p-3">
          <p className="text-[10px] font-bold uppercase text-muted-foreground">Registros</p>
          <p className="mt-1 font-data text-sm font-bold text-foreground">{result.count}</p>
        </div>
        <div className="rounded-md border border-border bg-muted/20 p-3">
          <p className="text-[10px] font-bold uppercase text-muted-foreground">Acceso</p>
          <p className="mt-1 text-xs font-bold text-foreground">{result.source.requiresKey ? "Token" : "Publico"}</p>
        </div>
      </div>

      {result.error && <p className="mt-3 rounded-md border border-border bg-muted/20 p-2 text-[11px] text-muted-foreground">{result.error}</p>}

      {preview.length > 0 && (
        <div className="mt-4 space-y-2">
          {preview.map((record, recordIndex) => (
            <p key={recordIndex} className="truncate rounded-md border border-border bg-muted/20 px-3 py-2 font-data text-[10px] text-foreground/80" title={summarizeRecord(record)}>
              {summarizeRecord(record)}
            </p>
          ))}
        </div>
      )}

      <a
        href={result.source.url}
        target="_blank"
        rel="noreferrer"
        className="mt-4 inline-flex items-center gap-2 text-xs font-bold text-primary hover:underline"
      >
        Ver API
        <ExternalLink className="h-3.5 w-3.5" />
      </a>
    </motion.article>
  );
}

export function ExternalDataSourcesPanel() {
  const [domain, setDomain] = useState<DomainId>("phishing");
  const { data, error, isLoading, reload } = useApiData<ExternalData>(() => fetchExternalData(domain), [domain]);

  return (
    <section className="card-formal p-4 sm:p-5">
      <div className="mb-5 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-md bg-primary/10 text-primary">
            <DatabaseZap className="h-4 w-4" />
          </div>
          <h2 className="text-sm font-bold text-foreground">Fuentes externas oficiales</h2>
          <p className="mt-1 max-w-3xl text-xs leading-5 text-muted-foreground">
            APIs conectadas para ampliar la data experimental sin perder trazabilidad metodologica.
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          {DOMAIN_OPTIONS.map((option) => (
            <button
              key={option.id}
              type="button"
              onClick={() => setDomain(option.id)}
              className={`rounded-md px-3 py-2 text-[10px] font-bold uppercase transition-colors ${
                domain === option.id ? "bg-primary text-primary-foreground" : "bg-muted/50 text-muted-foreground hover:text-foreground"
              }`}
            >
              {option.shortTitle}
            </button>
          ))}
          <button
            type="button"
            onClick={reload}
            className="rounded-md border border-border bg-card px-3 py-2 text-[10px] font-bold uppercase text-foreground hover:bg-muted"
          >
            Actualizar
          </button>
        </div>
      </div>

      {isLoading && (
        <div className="flex items-center gap-2 rounded-md border border-border bg-muted/20 p-4 text-xs font-semibold text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin text-primary" />
          Consultando fuentes externas...
        </div>
      )}

      {error && (
        <div className="rounded-md border border-anomaly/20 bg-anomaly/10 p-4 text-xs text-anomaly">
          No se pudieron consultar las fuentes externas: {error}
        </div>
      )}

      {data && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          {data.results.map((result, index) => (
            <SourceCard key={result.source.id} result={result} index={index} />
          ))}
        </div>
      )}
    </section>
  );
}
