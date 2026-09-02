const metricItems = [
  ["PR-AUC", "Métrica principal en phishing; evalúa el ranking de la clase positiva bajo desbalance sin fijar primero un umbral."],
  ["RMSE", "Métrica principal en energía y finanzas; penaliza con más fuerza los errores grandes de pronóstico y se minimiza."],
  ["F1-score", "Métrica secundaria que equilibra precisión y recall en un umbral calibrado exclusivamente con desarrollo."],
  ["Precisión", "Mide cuántas alertas positivas fueron correctas y ayuda a controlar falsos positivos."],
  ["Recall", "Mide cuántas anomalías reales fueron encontradas; es clave cuando omitir una anomalía tiene alto costo."],
  ["IC 95 %", "Cuantifica incertidumbre mediante remuestreo pareado por dominio o bloque temporal; una diferencia puntual no basta."],
  ["VP/FP/FN/VN", "La matriz de confusión muestra aciertos, falsas alarmas y anomalías omitidas para auditar el umbral."],
  ["XAI", "Explica variables, modelos base o pasos temporales; no sustituye la revalidación independiente."],
];

export function MetricGuide() {
  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm sm:p-5">
      <div className="mb-4">
        <h2 className="text-sm font-bold text-foreground">Guía de métricas</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Interpretación rápida de los indicadores usados para comparar las arquitecturas bajo el protocolo de tesis.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        {metricItems.map(([title, text]) => (
          <div key={title} className="rounded-md border border-border bg-muted/20 p-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-primary">{title}</h3>
            <p className="mt-2 text-xs leading-5 text-muted-foreground">{text}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
