import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { motion } from "framer-motion";
import { ConclusionPanel } from "@/components/ConclusionPanel";
import { Brain, Cpu, Gauge, Scale } from "lucide-react";

export const Route = createFileRoute("/selector")({
  head: () => ({
    meta: [
      { title: "Selector de Modelo — LSTM vs GRU" },
      { name: "description", content: "Método de selección automática basado en criterios del usuario" },
    ],
  }),
  component: SelectorPage,
});

type SeqType = "textual" | "temporal";
type Resources = "alto" | "medio" | "bajo";
type Priority = "precision" | "velocidad" | "balance";

interface Recommendation {
  model: string;
  reasons: string[];
  score: number;
}

function getRecommendation(seq: SeqType, res: Resources, pri: Priority): Recommendation {
  let lstmScore = 0;
  let gruScore = 0;

  // Sequence type
  if (seq === "textual") { lstmScore += 3; gruScore += 2; }
  else { lstmScore += 2; gruScore += 3; }

  // Resources
  if (res === "alto") { lstmScore += 2; gruScore += 2; }
  else if (res === "medio") { lstmScore += 1; gruScore += 2; }
  else { lstmScore += 0; gruScore += 3; }

  // Priority
  if (pri === "precision") { lstmScore += 3; gruScore += 2; }
  else if (pri === "velocidad") { lstmScore += 1; gruScore += 3; }
  else { lstmScore += 2; gruScore += 2; }

  const isLSTM = lstmScore > gruScore;
  const model = isLSTM ? "LSTM" : "GRU";
  const score = isLSTM ? lstmScore : gruScore;

  const reasons: string[] = [];
  if (isLSTM) {
    reasons.push("Mayor F1-Score en secuencias textuales (0.964 vs 0.934)");
    reasons.push("Mejor AUC-ROC para clasificación binaria de URLs");
    if (pri === "precision") reasons.push("LSTM sobresale cuando la precisión es la prioridad máxima");
    if (res === "alto") reasons.push("Con recursos altos, la mayor complejidad de LSTM se justifica");
  } else {
    reasons.push("Menor RMSE y MAE en series temporales (16.95 vs 18.42)");
    reasons.push("36% más rápido en entrenamiento (265s vs 415s)");
    reasons.push("30% menos uso de memoria GPU (876MB vs 1248MB)");
    if (pri === "velocidad") reasons.push("GRU ofrece la mejor relación velocidad/rendimiento");
    if (res === "bajo") reasons.push("GRU es ideal para entornos con recursos limitados");
  }

  return { model, reasons, score };
}

function SelectorPage() {
  const [seqType, setSeqType] = useState<SeqType | null>(null);
  const [resources, setResources] = useState<Resources | null>(null);
  const [priority, setPriority] = useState<Priority | null>(null);
  const [result, setResult] = useState<Recommendation | null>(null);

  const canSubmit = seqType && resources && priority;

  const handleSubmit = () => {
    if (canSubmit) {
      setResult(getRecommendation(seqType, resources, priority));
    }
  };

  const options = [
    {
      label: "Tipo de Secuencia",
      icon: Brain,
      choices: [
        { value: "textual" as SeqType, label: "Textual", desc: "URLs, texto, cadenas de caracteres" },
        { value: "temporal" as SeqType, label: "Numérica/Temporal", desc: "Series de tiempo, sensores, consumo" },
      ],
      selected: seqType,
      onSelect: (v: SeqType) => { setSeqType(v); setResult(null); },
    },
    {
      label: "Recursos Disponibles",
      icon: Cpu,
      choices: [
        { value: "alto" as Resources, label: "Alto", desc: "GPU dedicada, >16GB VRAM" },
        { value: "medio" as Resources, label: "Medio", desc: "GPU compartida, 8-16GB VRAM" },
        { value: "bajo" as Resources, label: "Bajo", desc: "CPU o GPU limitada, <8GB" },
      ],
      selected: resources,
      onSelect: (v: Resources) => { setResources(v); setResult(null); },
    },
    {
      label: "Prioridad",
      icon: Gauge,
      choices: [
        { value: "precision" as Priority, label: "Precisión", desc: "Maximizar detección correcta" },
        { value: "velocidad" as Priority, label: "Velocidad", desc: "Minimizar tiempo de inferencia" },
        { value: "balance" as Priority, label: "Balance", desc: "Equilibrio entre ambas" },
      ],
      selected: priority,
      onSelect: (v: Priority) => { setPriority(v); setResult(null); },
    },
  ];

  return (
    <div className="dashboard-page">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <h1 className="text-2xl font-bold text-foreground">Método de Selección</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Aporte diferencial de la tesis — recomendación basada en resultados experimentales
        </p>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {options.map((opt, oi) => (
          <motion.div
            key={opt.label}
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: oi * 0.1 }}
            className="card-formal p-4 sm:p-6"
          >
            <div className="flex items-center gap-2 mb-4 border-b border-border pb-3">
              <opt.icon className="h-4 w-4 text-primary" />
              <h3 className="text-xs font-bold text-foreground uppercase tracking-widest">{opt.label}</h3>
            </div>
            <div className="space-y-2">
              {opt.choices.map((choice) => (
                <button
                  key={choice.value}
                  onClick={() => opt.onSelect(choice.value as never)}
                  className={`w-full text-left rounded-md p-3 transition-all border ${
                    opt.selected === choice.value
                      ? "border-primary bg-primary/5 shadow-sm"
                      : "border-border hover:bg-muted/30"
                  }`}
                >
                  <p className={`text-[13px] font-bold ${opt.selected === choice.value ? "text-primary" : "text-foreground"}`}>{choice.label}</p>
                  <p className="text-[10px] text-muted-foreground mt-0.5 font-medium">{choice.desc}</p>
                </button>
              ))}
            </div>
          </motion.div>
        ))}
      </div>

      <div className="flex justify-center py-4">
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          className={`w-full rounded-lg px-6 py-3 text-xs font-bold uppercase tracking-widest transition-all sm:w-auto sm:px-10 ${
            canSubmit
              ? "bg-primary text-primary-foreground shadow-md hover:bg-primary/95 active:scale-95"
              : "bg-muted text-muted-foreground cursor-not-allowed border border-border"
          }`}
        >
          <Scale className="inline h-4 w-4 mr-2" />
          Generar Recomendación Técnica
        </button>
      </div>

      {result && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="card-formal border-l-4 border-l-primary p-4 sm:p-8"
        >
          <div className="mb-6 flex items-start gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 text-primary border border-primary/20">
              <Brain className="h-6 w-6" />
            </div>
            <div>
              <p className="text-[10px] text-muted-foreground uppercase font-bold tracking-widest mb-1">Modelo Recomendado</p>
              <h2 className="text-2xl font-bold text-foreground sm:text-3xl">
                Arquitectura <span className="text-primary">{result.model}</span>
              </h2>
            </div>
          </div>
          <div className="space-y-4">
            <p className="text-[10px] font-bold text-foreground uppercase tracking-wider border-b border-border pb-2">Justificación Técnica Experimental:</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {result.reasons.map((r, i) => (
                <div key={i} className="flex items-start gap-3 p-3 rounded-md bg-muted/30 border border-border/50">
                  <div className="mt-1 w-1.5 h-1.5 rounded-full flex-shrink-0 bg-primary" />
                  <p className="text-[11px] text-muted-foreground leading-relaxed font-medium">{r}</p>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      )}

      <ConclusionPanel />
    </div>
  );
}
