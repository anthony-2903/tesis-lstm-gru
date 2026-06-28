import {
  AlertTriangle,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ClipboardList,
  Gauge,
  HelpCircle,
  Layers3,
  Scale,
  ShieldAlert,
  Target,
  Trophy,
} from "lucide-react";
import type { ComparisonData, DomainId, EvaluatedData, HistoryItem, ModelPerformance } from "@/lib/api";
import { DOMAIN_OPTIONS, getDomainOption } from "@/lib/domains";

type ModelKey = "lstm" | "gru" | "brnn" | "transformer" | "tcn";

const MODEL_KEYS: ModelKey[] = ["lstm", "gru", "brnn", "transformer", "tcn"];
const MODEL_LABELS: Record<ModelKey, string> = {
  lstm: "LSTM",
  gru: "GRU",
  brnn: "BRNN",
  transformer: "Transformer",
  tcn: "TCN",
};

export function ResearchFramingPanel() {
  const items = [
    ["Pregunta de investigación", "¿Qué arquitectura detecta anomalías con mejor equilibrio entre exactitud, costo computacional e interpretabilidad según el tipo de dato?"],
    ["Hipótesis", "El desempeño no depende de un ganador universal: cambia según si el dominio es textual, temporal o tabular."],
    ["Alcance experimental", "La evidencia se organiza por dominio, métricas de clasificación, sensibilidad temporal e importancia de variables."],
  ];

  return (
    <section className="grid grid-cols-1 gap-3 lg:grid-cols-3">
      {items.map(([title, text], index) => (
        <div key={title} className="rounded-md border border-border bg-card p-4 shadow-sm">
          <div className="mb-3 flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary/10 text-primary">
              {index === 0 ? <HelpCircle className="h-4 w-4" /> : index === 1 ? <Target className="h-4 w-4" /> : <Layers3 className="h-4 w-4" />}
            </div>
            <h2 className="text-xs font-bold uppercase tracking-wider text-foreground">{title}</h2>
          </div>
          <p className="text-xs leading-5 text-muted-foreground">{text}</p>
        </div>
      ))}
    </section>
  );
}

export function DomainEvidenceTable({ activeDomain }: { activeDomain: DomainId }) {
  const rows = DOMAIN_OPTIONS.map((domain) => ({
    ...domain,
    leader: domain.id === "phishing" ? "Transformer" : domain.id === "energia" ? "TCN" : "Transformer",
    metric: domain.id === "energia" ? "RMSE / Recall" : "F1-score / Precisión",
  }));

  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm sm:p-5">
      <div className="mb-4">
        <h2 className="text-sm font-bold text-foreground">Evidencia por dominio</h2>
        <p className="mt-1 text-xs text-muted-foreground">Resumen ejecutivo que conecta dataset, arquitectura líder y criterio principal de evaluación.</p>
      </div>
      <div className="responsive-table">
        <table className="w-full min-w-[720px] text-xs">
          <thead>
            <tr className="border-b border-border bg-muted/20">
              {["Dominio", "Fuente", "Modelo líder", "Métrica principal", "Lectura"].map((header) => (
                <th key={header} className="px-4 py-3 text-left font-bold uppercase tracking-wider text-muted-foreground">{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id} className={activeDomain === row.id ? "bg-primary/5" : ""}>
                <td className="px-4 py-3 font-bold text-foreground">{row.title}</td>
                <td className="px-4 py-3 text-muted-foreground">{row.source}</td>
                <td className="px-4 py-3 font-data font-bold text-primary">{row.leader}</td>
                <td className="px-4 py-3 text-muted-foreground">{row.metric}</td>
                <td className="px-4 py-3 text-muted-foreground">{row.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function AnalysisInterpretationPanel({ domain, data }: { domain: DomainId; data: EvaluatedData }) {
  const ranking = rankModels(data.models, domain);
  const best = ranking[0];
  const fnTotal = MODEL_KEYS.reduce((sum, key) => sum + data.models[key].confusionMatrix.fn, 0);
  const fpTotal = MODEL_KEYS.reduce((sum, key) => sum + data.models[key].confusionMatrix.fp, 0);
  const risk = fnTotal > fpTotal ? "falsos negativos" : "falsos positivos";

  return (
    <section className="grid grid-cols-1 gap-3 xl:grid-cols-3">
      <InsightCard
        icon={Trophy}
        title="Interpretación del resultado"
        text={`Para ${getDomainOption(domain).shortTitle}, el modelo con mejor lectura global es ${best.label}. La conclusión debe contrastarse con la matriz de confusión y el costo del error.`}
      />
      <InsightCard
        icon={Gauge}
        title="Semáforo de desempeño"
        text={`${performanceLevel(best.score)}. El puntaje resume F1-score y recall; en series temporales también considera RMSE cuando está disponible.`}
      />
      <InsightCard
        icon={ShieldAlert}
        title="Riesgo experimental"
        text={`El principal riesgo observado se concentra en ${risk}. Para defensa, explica si es más costoso alertar de más o dejar pasar una anomalía.`}
      />
    </section>
  );
}

export function ModelRankingPanel({ domain, data }: { domain: DomainId; data: EvaluatedData }) {
  const ranking = rankModels(data.models, domain);
  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm sm:p-5">
      <div className="mb-4 flex items-center gap-2">
        <Trophy className="h-4 w-4 text-primary" />
        <h2 className="text-sm font-bold text-foreground">Ranking de modelos por dominio</h2>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-5">
        {ranking.map((item, index) => (
          <div key={item.key} className="rounded-md border border-border bg-muted/20 p-3">
            <span className="font-data text-[10px] font-bold text-primary">#{index + 1}</span>
            <p className="mt-2 text-sm font-bold text-foreground">{item.label}</p>
            <p className="mt-1 font-data text-xs text-muted-foreground">{item.score.toFixed(3)}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

export function ConfusionMatrixGuide() {
  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm sm:p-5">
      <h2 className="text-sm font-bold text-foreground">Cómo leer la matriz de confusión</h2>
      <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-4">
        {[
          ["VP", "Anomalías correctamente detectadas."],
          ["FP", "Alertas generadas sobre casos normales."],
          ["FN", "Anomalías reales no detectadas."],
          ["VN", "Casos normales correctamente descartados."],
        ].map(([label, text]) => (
          <div key={label} className="rounded-md border border-border bg-muted/20 p-3">
            <p className="font-data text-sm font-bold text-primary">{label}</p>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">{text}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

export function ComparisonDecisionPanel({ data }: { data: ComparisonData }) {
  const rows = data.comparisonTable.map((row) => ({
    metric: row.metric,
    winner: row.winner,
    criterion: metricCriterion(row.metric),
  }));

  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm sm:p-5">
      <div className="mb-4">
        <h2 className="text-sm font-bold text-foreground">Ganador por criterio</h2>
        <p className="mt-1 text-xs text-muted-foreground">La decisión se descompone por métrica para evitar una conclusión única sin contexto.</p>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        {rows.map((row) => (
          <div key={row.metric} className="rounded-md border border-border bg-muted/20 p-3">
            <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">{row.metric}</p>
            <p className="mt-2 font-data text-sm font-bold text-primary">{row.winner}</p>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">{row.criterion}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

export function RadarGuidePanel() {
  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm sm:p-5">
      <div className="flex items-start gap-3">
        <BarChart3 className="mt-0.5 h-4 w-4 text-primary" />
        <div>
          <h2 className="text-sm font-bold text-foreground">Lectura del radar comparativo</h2>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">
            Cada eje representa una métrica normalizada entre 0 y 1. Un área más amplia indica un perfil más equilibrado, pero la recomendación final debe priorizar el costo del error en el dominio evaluado.
          </p>
        </div>
      </div>
    </section>
  );
}

export function ModelStrengthsPanel() {
  const rows = [
    ["LSTM", "Memoria de largo plazo", "Mayor costo de entrenamiento."],
    ["GRU", "Buen balance entre velocidad y precisión", "Puede perder patrones complejos."],
    ["BRNN", "Contexto bidireccional", "Menos útil para inferencia estrictamente causal."],
    ["Transformer", "Dependencias globales y texto", "Mayor demanda de datos y cómputo."],
    ["TCN", "Estabilidad en series temporales", "Menos intuitivo para texto puro."],
  ];

  return (
    <section className="grid grid-cols-1 gap-3 md:grid-cols-5">
      {rows.map(([model, strength, limit]) => (
        <div key={model} className="rounded-md border border-border bg-card p-4 shadow-sm">
          <p className="font-data text-sm font-bold text-primary">{model}</p>
          <p className="mt-2 text-xs font-semibold text-foreground">{strength}</p>
          <p className="mt-2 text-xs leading-5 text-muted-foreground">{limit}</p>
        </div>
      ))}
    </section>
  );
}

export function NoUniversalWinnerPanel() {
  return (
    <section className="rounded-md border border-primary/20 bg-primary/5 p-4 shadow-sm sm:p-5">
      <div className="flex items-start gap-3">
        <Scale className="mt-0.5 h-4 w-4 text-primary" />
        <div>
          <h2 className="text-sm font-bold text-foreground">No existe ganador universal</h2>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">
            El objetivo de la tesis no es coronar un único modelo para todos los casos, sino justificar qué arquitectura conviene bajo condiciones de datos, costo de error y recursos disponibles.
          </p>
        </div>
      </div>
    </section>
  );
}

export function XaiReadingPanel({ topFeature, topStep }: { topFeature?: string; topStep?: string }) {
  return (
    <section className="grid grid-cols-1 gap-3 lg:grid-cols-3">
      <InsightCard
        icon={BrainCircuit}
        title="Qué aporta XAI"
        text="XAI permite explicar qué variables o pasos temporales sostienen la predicción, agregando trazabilidad a la comparación de modelos."
      />
      <InsightCard
        icon={Target}
        title="Interpretación dominante"
        text={topFeature ? `La variable con mayor impacto global es ${topFeature}. Debe revisarse junto con el modelo líder del dominio.` : "Este dominio no presenta variables globales; la lectura debe centrarse en sensibilidad temporal o detalle por modelo."}
      />
      <InsightCard
        icon={ClipboardList}
        title="Cómo leer la gráfica"
        text={topStep ? `Los pasos temporales muestran cuánto cambia la predicción al alterar el historial. El paso clave actual es ${topStep}.` : "Si no hay pasos temporales, el dominio se interpreta por variables tabulares o textuales."}
      />
    </section>
  );
}

export function HistoryAuditSummary({ items }: { items: HistoryItem[] }) {
  const total = items.length;
  const matches = items.filter((item) => item.realLabel === item.predicted).length;
  const avgConfidence = total ? items.reduce((sum, item) => sum + Number(item.confidence || 0), 0) / total : 0;
  const highSeverity = items.filter((item) => item.realLabel !== item.predicted || item.confidence >= 90).length;

  return (
    <section className="grid grid-cols-1 gap-3 md:grid-cols-4">
      {[
        ["Eventos auditados", total.toLocaleString("es-ES"), "Registros visibles tras aplicar filtros."],
        ["Coincidencias", matches.toLocaleString("es-ES"), "Predicción igual a la etiqueta real."],
        ["Confianza promedio", `${avgConfidence.toFixed(1)}%`, "Promedio de confianza del modelo."],
        ["Severidad alta", highSeverity.toLocaleString("es-ES"), "Errores o decisiones con confianza elevada."],
      ].map(([title, value, text]) => (
        <div key={title} className="rounded-md border border-border bg-card p-4 shadow-sm">
          <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">{title}</p>
          <p className="mt-2 font-data text-2xl font-bold text-foreground">{value}</p>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">{text}</p>
        </div>
      ))}
    </section>
  );
}

export function SelectorScenarioPanel() {
  const scenarios = [
    ["Máxima precisión", "Prioriza recall y F1-score cuando omitir anomalías tiene alto costo."],
    ["Bajo recurso", "Prefiere modelos compactos como GRU cuando el costo computacional limita el despliegue."],
    ["Tiempo real", "Favorece inferencia estable y rápida, especialmente para señales temporales."],
  ];
  return (
    <section className="grid grid-cols-1 gap-3 md:grid-cols-3">
      {scenarios.map(([title, text]) => (
        <div key={title} className="rounded-md border border-border bg-card p-4 shadow-sm">
          <p className="text-xs font-bold uppercase tracking-wider text-primary">{title}</p>
          <p className="mt-2 text-xs leading-5 text-muted-foreground">{text}</p>
        </div>
      ))}
    </section>
  );
}

function InsightCard({ icon: Icon, title, text }: { icon: typeof Target; title: string; text: string }) {
  return (
    <div className="rounded-md border border-border bg-card p-4 shadow-sm">
      <div className="mb-3 flex items-center gap-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary/10 text-primary">
          <Icon className="h-4 w-4" />
        </div>
        <h2 className="text-xs font-bold uppercase tracking-wider text-foreground">{title}</h2>
      </div>
      <p className="text-xs leading-5 text-muted-foreground">{text}</p>
    </div>
  );
}

function rankModels(models: EvaluatedData["models"], domain: DomainId) {
  return MODEL_KEYS.map((key) => {
    const model = models[key] as ModelPerformance;
    const rmseScore = model.rmse ? 1 / (1 + model.rmse) : 0;
    const score = domain === "energia"
      ? (model.f1 || 0) * 0.35 + (model.recall || 0) * 0.35 + rmseScore * 0.3
      : (model.f1 || 0) * 0.5 + (model.precision || 0) * 0.25 + (model.recall || 0) * 0.25;
    return { key, label: MODEL_LABELS[key], score };
  }).sort((a, b) => b.score - a.score);
}

function performanceLevel(score: number) {
  if (score >= 0.85) return "Desempeño alto";
  if (score >= 0.7) return "Desempeño medio";
  return "Desempeño bajo";
}

function metricCriterion(metric: string) {
  const normalized = metric.toLowerCase();
  if (normalized.includes("precision")) return "Controla falsos positivos.";
  if (normalized.includes("recall")) return "Reduce anomalías omitidas.";
  if (normalized.includes("tiempo") || normalized.includes("velocidad")) return "Evalúa costo operativo.";
  if (normalized.includes("f1")) return "Balancea precisión y recall.";
  return "Criterio complementario de comparación.";
}
