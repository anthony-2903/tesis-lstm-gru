import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { motion } from "framer-motion";
import { ChartCard } from "@/components/ChartCard";
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
    <div className="space-y-6">
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
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: oi * 0.1 }}
            className="rounded-xl border border-border bg-card p-5"
          >
            <div className="flex items-center gap-2 mb-4">
              <opt.icon className="h-4 w-4 text-primary" />
              <h3 className="text-sm font-semibold text-foreground">{opt.label}</h3>
            </div>
            <div className="space-y-2">
              {opt.choices.map((choice) => (
                <button
                  key={choice.value}
                  onClick={() => opt.onSelect(choice.value as never)}
                  className={`w-full text-left rounded-lg p-3 transition-all border ${
                    opt.selected === choice.value
                      ? "border-primary bg-primary/10"
                      : "border-border hover:border-muted-foreground/30"
                  }`}
                >
                  <p className={`text-sm font-medium ${opt.selected === choice.value ? "text-primary" : "text-foreground"}`}>{choice.label}</p>
                  <p className="text-[10px] text-muted-foreground mt-0.5">{choice.desc}</p>
                </button>
              ))}
            </div>
          </motion.div>
        ))}
      </div>

      <div className="flex justify-center">
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          className={`px-8 py-3 rounded-xl text-sm font-semibold transition-all ${
            canSubmit
              ? "gradient-cyan text-cyan-foreground hover:opacity-90 card-glow"
              : "bg-muted text-muted-foreground cursor-not-allowed"
          }`}
        >
          <Scale className="inline h-4 w-4 mr-2" />
          Obtener Recomendación
        </button>
      </div>

      {result && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className={`rounded-xl border p-6 ${result.model === "LSTM" ? "border-primary/40 card-glow bg-primary/5" : "border-secondary/40 card-glow-violet bg-secondary/5"}`}
        >
          <div className="flex items-center gap-3 mb-4">
            <div className={`flex h-12 w-12 items-center justify-center rounded-xl ${result.model === "LSTM" ? "gradient-cyan" : "gradient-violet"}`}>
              <Brain className={`h-6 w-6 ${result.model === "LSTM" ? "text-cyan-foreground" : "text-violet-foreground"}`} />
            </div>
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wider">Modelo Recomendado</p>
              <p className={`text-2xl font-bold ${result.model === "LSTM" ? "text-primary" : "text-secondary"}`}>{result.model}</p>
            </div>
          </div>
          <div className="space-y-2">
            <p className="text-xs font-semibold text-foreground mb-2">Justificación:</p>
            {result.reasons.map((r, i) => (
              <div key={i} className="flex items-start gap-2">
                <div className={`mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0 ${result.model === "LSTM" ? "bg-primary" : "bg-secondary"}`} />
                <p className="text-xs text-muted-foreground">{r}</p>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}
