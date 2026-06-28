const conclusions = [
  "Transformer suele destacar en dominios textuales por su capacidad de capturar dependencias globales.",
  "TCN es una alternativa fuerte para series temporales cuando se busca estabilidad y menor error de predicción.",
  "GRU mantiene una buena relación entre rendimiento y costo computacional en escenarios con recursos limitados.",
  "La recomendación final debe leerse por dominio, no como un ganador absoluto para todos los datos.",
];

export function ConclusionPanel() {
  return (
    <section className="rounded-md border border-primary/20 bg-primary/5 p-4 shadow-sm sm:p-5">
      <h2 className="text-sm font-bold text-foreground">Conclusiones para la defensa</h2>
      <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
        {conclusions.map((item) => (
          <div key={item} className="rounded-md border border-primary/10 bg-card/70 p-3">
            <p className="text-xs leading-5 text-muted-foreground">{item}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
