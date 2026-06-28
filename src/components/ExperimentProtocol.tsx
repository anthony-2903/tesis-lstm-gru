const protocolItems = [
  ["Datos", "Se documenta el archivo procesado, cantidad de registros, columnas detectadas y calidad luego de la limpieza."],
  ["Preparación", "El backend centraliza limpieza, normalización y construcción de muestras para que las rutas usen una misma fuente de resultados."],
  ["Entrenamiento", "Cada arquitectura se compara con métricas equivalentes y se conserva el tiempo de ejecución cuando está disponible."],
  ["Validación", "La lectura final combina F1-score, precisión, recall, RMSE, matriz de confusión, detecciones reales y XAI."],
];

export function ExperimentProtocol() {
  return (
    <section className="rounded-md border border-border bg-card p-4 shadow-sm sm:p-5">
      <div className="mb-4">
        <h2 className="text-sm font-bold text-foreground">Protocolo experimental</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Resumen metodológico para sustentar la comparación de modelos durante la defensa.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        {protocolItems.map(([title, text], index) => (
          <div key={title} className="rounded-md border border-border bg-muted/20 p-3">
            <span className="font-data text-[10px] font-bold text-primary">0{index + 1}</span>
            <h3 className="mt-2 text-xs font-bold uppercase tracking-wider text-foreground">{title}</h3>
            <p className="mt-2 text-xs leading-5 text-muted-foreground">{text}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
