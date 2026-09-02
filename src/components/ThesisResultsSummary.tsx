import { useCallback, useEffect, useState } from "react";
import { FlaskConical, LockKeyhole, RefreshCw, TriangleAlert, Trophy } from "lucide-react";
import { fetchThesisResultsSummary, type ThesisResultDomainSummary, type ThesisResultsSummary as Summary } from "@/lib/api";

const BOX_COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)", "var(--color-primary)"];

export function ThesisResultsSummary() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    try {
      setSummary(await fetchThesisResultsSummary());
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "No se pudo consultar la tabla científica.");
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  useEffect(() => {
    if (!summary || summary.readiness.allDomainsFrozen) return;
    const timer = window.setInterval(() => void reload(), 15_000);
    return () => window.clearInterval(timer);
  }, [reload, summary]);

  if (!summary) {
    return (
      <section className="rounded-md border border-border bg-card p-4 shadow-sm">
        <p className="text-xs text-muted-foreground">{error ?? "Construyendo la comparación científica de seis candidatos…"}</p>
        {error && <button type="button" onClick={() => void reload()} className="mt-3 inline-flex items-center gap-2 text-xs font-bold text-primary hover:underline"><RefreshCw className="h-3.5 w-3.5" />Reintentar</button>}
      </section>
    );
  }

  return (
    <section className="rounded-md border border-primary/20 bg-card p-4 shadow-sm sm:p-5" aria-labelledby="thesis-results-title">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-3">
          <FlaskConical className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
          <div>
            <h2 id="thesis-results-title" className="text-sm font-bold text-foreground">{summary.title}</h2>
            <p className="mt-1 max-w-4xl text-xs text-muted-foreground">La tabla separa demostración, candidato científico y evidencia sellada. Un primer lugar pre-test no equivale al resultado final.</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="rounded-full border border-success/30 bg-success/5 px-3 py-1 text-[10px] font-bold uppercase text-success">{summary.readiness.eligibleDomains}/{summary.readiness.totalDomains} sellados</span>
          <button type="button" onClick={() => void reload()} aria-label="Actualizar comparación" className="text-muted-foreground hover:text-foreground"><RefreshCw className="h-4 w-4" /></button>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-3">
        {summary.domains.map((domain) => <DomainTable key={domain.id} domain={domain} />)}
      </div>

      <div className="mt-4 border-t border-border pt-3">
        <p className="inline-flex items-center gap-2 text-xs font-semibold text-success"><LockKeyhole className="h-4 w-4" />Test final sin utilizar; la evaluación única continúa bloqueada.</p>
        <ul className="mt-2 space-y-1 text-[10px] leading-4 text-muted-foreground">
          {summary.interpretationPolicy.map((rule) => <li key={rule}>• {rule}</li>)}
        </ul>
      </div>
      {error && <p className="mt-2 text-xs text-warning">Se conserva la última tabla disponible. {error}</p>}
    </section>
  );
}

function DomainTable({ domain }: { domain: ThesisResultDomainSummary }) {
  if (!domain.available) {
    return (
      <article className="rounded-md border border-dashed border-border bg-muted/10 p-4">
        <h3 className="text-xs font-bold text-foreground">{domain.label}</h3>
        <p className="mt-3 inline-flex items-start gap-2 text-xs text-muted-foreground"><TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />{domain.interpretation}</p>
      </article>
    );
  }

  return (
    <article className="overflow-hidden rounded-md border border-border">
      <div className="border-b border-border bg-muted/20 p-3">
        <div className="flex items-start justify-between gap-2">
          <div>
            <h3 className="text-xs font-bold text-foreground">{domain.label}</h3>
            <p className="mt-0.5 text-[10px] text-muted-foreground">{metricLabel(domain)} · {domain.comparisonSplit}</p>
          </div>
          <EvidenceBadge status={domain.evidenceStatus} />
        </div>
      </div>
      <div className="responsive-table">
        <table className="w-full min-w-[360px] text-[10px]">
          <thead><tr className="border-b border-border text-muted-foreground"><th className="px-2 py-2 text-center">#</th><th className="px-2 py-2 text-left">Candidato</th><th className="px-2 py-2 text-left">Familia</th><th className="px-2 py-2 text-right">{domain.primaryMetric === "rmse" ? "RMSE ↓" : "PR-AUC ↑"}</th></tr></thead>
          <tbody>
            {domain.comparison.map((candidate) => (
              <tr key={candidate.candidateId} className={`border-b border-border/60 last:border-0 ${candidate.candidateId === "stacking" ? "bg-primary/5 font-bold" : ""}`}>
                <td className="px-2 py-2 text-center font-data">{candidate.rank}</td>
                <td className="px-2 py-2 uppercase text-foreground">{candidate.candidateId}{candidate.rank === 1 && <Trophy className="ml-1 inline h-3 w-3 text-warning" />}</td>
                <td className="px-2 py-2 text-muted-foreground">{candidate.family === "stacking" ? "Metamodelo" : "Base"}</td>
                <td className="px-2 py-2 text-right font-data text-foreground">{formatMetric(candidate.primaryValue)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="border-t border-border bg-muted/10 p-3">
        <p className="text-[10px] leading-4 text-muted-foreground">{domain.interpretation}</p>
        {domain.allEvaluatedCandidatesWinner && domain.allEvaluatedCandidatesWinner !== domain.sixCandidateWinner && (
          <p className="mt-1 text-[10px] font-semibold text-warning">Incluyendo baselines auxiliares, el ganador fue {domain.allEvaluatedCandidatesWinner.toUpperCase()}.</p>
        )}
      </div>
      <SeedBoxPlot domain={domain} />
    </article>
  );
}

function SeedBoxPlot({ domain }: { domain: ThesisResultDomainSummary }) {
  const distributions = domain.seedDistributions;
  if (!distributions.length) {
    return <div className="border-t border-border p-3 text-[10px] text-muted-foreground">La corrida disponible todavía no contiene métricas comparables por semilla; el protocolo científico las generará antes del sello.</div>;
  }

  const width = 720;
  const height = 260;
  const margin = { top: 18, right: 18, bottom: 48, left: 58 };
  const values = distributions.flatMap((item) => item.values);
  const observedMin = Math.min(...values);
  const observedMax = Math.max(...values);
  const observedRange = observedMax - observedMin;
  const padding = observedRange > 0 ? observedRange * 0.12 : Math.max(Math.abs(observedMax) * 0.05, 0.01);
  const lower = observedMin - padding;
  const upper = observedMax + padding;
  const chartHeight = height - margin.top - margin.bottom;
  const chartWidth = width - margin.left - margin.right;
  const y = (value: number) => margin.top + (upper - value) / (upper - lower) * chartHeight;
  const slot = chartWidth / distributions.length;
  const ticks = Array.from({ length: 5 }, (_, index) => lower + (upper - lower) * index / 4);
  const minimumSeeds = Math.min(...distributions.map((item) => item.seedCount));

  return (
    <div className="border-t border-border p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <p className="text-[10px] font-bold uppercase text-foreground">Distribución por semilla · {domain.primaryMetric === "rmse" ? "RMSE" : "PR-AUC"}</p>
        <span className={`text-[9px] font-bold uppercase ${minimumSeeds >= 5 ? "text-success" : "text-warning"}`}>{minimumSeeds} semilla(s)</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Boxplots por semilla para ${domain.label}`} className="h-auto w-full min-w-[360px] overflow-visible">
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} stroke="var(--color-border)" strokeDasharray="4 4" />
            <text x={margin.left - 8} y={y(tick) + 4} textAnchor="end" fontSize="10" fill="var(--color-muted-foreground)">{tick.toFixed(domain.primaryMetric === "rmse" ? 3 : 4)}</text>
          </g>
        ))}
        {distributions.map((item, index) => {
          const x = margin.left + slot * (index + 0.5);
          const boxWidth = Math.min(46, slot * 0.52);
          const color = BOX_COLORS[index % BOX_COLORS.length];
          const boxTop = y(item.q3);
          const boxBottom = y(item.q1);
          return (
            <g key={item.candidateId}>
              <title>{`${item.candidateId}: mediana ${item.median.toFixed(5)}, rango ${item.minimum.toFixed(5)}–${item.maximum.toFixed(5)}, ${item.seedCount} semillas`}</title>
              <line x1={x} x2={x} y1={y(item.maximum)} y2={y(item.minimum)} stroke={color} strokeWidth="2" />
              <line x1={x - boxWidth * 0.3} x2={x + boxWidth * 0.3} y1={y(item.maximum)} y2={y(item.maximum)} stroke={color} strokeWidth="2" />
              <line x1={x - boxWidth * 0.3} x2={x + boxWidth * 0.3} y1={y(item.minimum)} y2={y(item.minimum)} stroke={color} strokeWidth="2" />
              <rect x={x - boxWidth / 2} y={boxTop} width={boxWidth} height={Math.max(2, boxBottom - boxTop)} fill={color} fillOpacity="0.28" stroke={color} strokeWidth="2" rx="2" />
              <line x1={x - boxWidth / 2} x2={x + boxWidth / 2} y1={y(item.median)} y2={y(item.median)} stroke={color} strokeWidth="3" />
              {item.values.map((value, valueIndex) => (
                <circle key={`${item.candidateId}-${item.seeds[valueIndex]}`} cx={x + (valueIndex - (item.values.length - 1) / 2) * 4} cy={y(value)} r="2.5" fill={color} />
              ))}
              <text x={x} y={height - 19} textAnchor="middle" fontSize="10" fontWeight="700" fill="var(--color-foreground)">{item.candidateId === "stacking" ? "STACK" : item.candidateId.toUpperCase()}</text>
            </g>
          );
        })}
        <line x1={margin.left} x2={margin.left} y1={margin.top} y2={height - margin.bottom} stroke="var(--color-muted-foreground)" />
        <line x1={margin.left} x2={width - margin.right} y1={height - margin.bottom} y2={height - margin.bottom} stroke="var(--color-muted-foreground)" />
      </svg>
      {minimumSeeds < 5 && <p className="mt-1 text-[10px] font-semibold text-warning">Con menos de cinco semillas, la caja es solo ilustrativa y no demuestra estabilidad.</p>}
    </div>
  );
}

function EvidenceBadge({ status }: { status: ThesisResultDomainSummary["evidenceStatus"] }) {
  const content = {
    unavailable: ["No disponible", "border-border text-muted-foreground"],
    demonstration_only: ["Demo", "border-warning/30 bg-warning/5 text-warning"],
    scientific_candidate_pending_freeze: ["Candidato", "border-primary/30 bg-primary/5 text-primary"],
    scientific_frozen_pre_test: ["Sellado", "border-success/30 bg-success/5 text-success"],
  }[status];
  return <span className={`shrink-0 rounded-full border px-2 py-1 text-[9px] font-bold uppercase ${content[1]}`}>{content[0]}</span>;
}

function metricLabel(domain: ThesisResultDomainSummary) {
  return domain.primaryMetric === "rmse" ? "Error de predicción" : "Detección de clase minoritaria";
}

function formatMetric(value: number) {
  return Number.isFinite(value) ? value.toFixed(4) : "N/D";
}
