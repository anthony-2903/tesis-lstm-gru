import { BarChart3, BookOpenCheck, ClipboardCheck, Target } from "lucide-react";

const objectiveItems = [
  {
    title: "Objetivo general",
    text: "Comparar LSTM, GRU, BRNN, Transformer y TCN en detección de anomalías para determinar qué arquitectura conviene según el dominio de datos.",
    icon: Target,
  },
  {
    title: "Dominios evaluados",
    text: "Phishing basado en texto, consumo energético como serie temporal y transacciones financieras con señales tabulares.",
    icon: ClipboardCheck,
  },
  {
    title: "Criterio de comparación",
    text: "Se contrastan métricas de clasificación, error, tiempo de entrenamiento, matrices de confusión e interpretabilidad XAI.",
    icon: BarChart3,
  },
  {
    title: "Aporte de la tesis",
    text: "La plataforma convierte los resultados experimentales en evidencia visual y en una recomendación técnica por escenario.",
    icon: BookOpenCheck,
  },
];

export function ResearchOverview() {
  return (
    <section className="grid grid-cols-1 gap-3 lg:grid-cols-4">
      {objectiveItems.map((item) => {
        const Icon = item.icon;
        return (
          <div key={item.title} className="rounded-md border border-border bg-card p-4 shadow-sm">
            <div className="mb-3 flex items-center gap-2">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
                <Icon className="h-4 w-4" />
              </div>
              <h2 className="text-xs font-bold uppercase tracking-wider text-foreground">{item.title}</h2>
            </div>
            <p className="text-xs leading-5 text-muted-foreground">{item.text}</p>
          </div>
        );
      })}
    </section>
  );
}
