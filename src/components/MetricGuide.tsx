const metricItems = [
  ["F1-score", "Equilibra precisión y recall. Es la métrica principal cuando las anomalías son menos frecuentes que los casos normales."],
  ["Precisión", "Mide cuántas alertas positivas fueron correctas. Ayuda a controlar falsos positivos."],
  ["Recall", "Mide cuántas anomalías reales fueron encontradas. Es clave cuando perder una anomalía tiene alto costo."],
  ["RMSE", "Resume el error de predicción en series temporales. Un valor menor indica mejor ajuste numérico."],
  ["VP/FP/FN/VN", "La matriz de confusión muestra aciertos, falsas alarmas y anomalías omitidas para auditar el resultado."],
  ["XAI", "Explica qué variables o pasos temporales influyen más en la predicción del modelo."],
];

export function MetricGuide() {
  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm sm:p-5">
      <div className="mb-4">
        <h2 className="text-sm font-bold text-foreground">Guía de métricas</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Interpretación rápida de los indicadores usados para comparar las arquitecturas.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
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
