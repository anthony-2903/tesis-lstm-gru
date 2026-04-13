import { phishtankMetrics, opsdMetrics, kpiData, confusionMatrix } from "./mock-data";

export const generateAiAnalysis = async (type: "general" | "phishtank" | "opsd") => {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 2000));
  const sections = [];

  if (type === "general" || type === "phishtank") {
    sections.push(`
### 🛡️ Análisis de PhishTank (Detección de URLs)
El modelo **LSTM** ha demostrado un desempeño superior en la clasificación de URLs maliciosas con un **F1-Score de ${phishtankMetrics.lstm.f1.toFixed(3)}**, superando al GRU (${phishtankMetrics.gru.f1.toFixed(3)}). 
- **Fortaleza**: El LSTM captura mejor las dependencias secuenciales largas en las cadenas de URL, lo que reduce los falsos negativos a solo **${confusionMatrix.lstm.fn}**.
- **Observación**: Aunque el GRU es un **${((1 - kpiData.avgInferenceTime / 14.2) * 100).toFixed(1)}%** más rápido en inferencia, para seguridad crítica como el phishing, la precisión del LSTM justifica el costo computacional adicional.`);
  }

  if (type === "general" || type === "opsd") {
    sections.push(`
### ⚡ Análisis de Open Power System (Series Temporales)
En el dominio de consumo eléctrico, los resultados se invierten a favor de la arquitectura **GRU**.
- **Precisión**: El GRU alcanzó un **RMSE de ${opsdMetrics.gru.rmse.toFixed(2)}**, siendo más preciso que el LSTM (${opsdMetrics.lstm.rmse.toFixed(2)}).
- **Eficiencia**: La menor complejidad estructural del GRU le permite converger más rápido en datos numéricos continuos, capturando las estacionalidades diarias con menor error absoluto medio (MAE: ${opsdMetrics.gru.mae.toFixed(2)}).`);
  }

  if (type === "general") {
    sections.push(`
### 📈 Conclusión de la Tesis
Tras procesar **${kpiData.totalRecords.toLocaleString()}** registros, se concluye que no existe un "ganador único". La elección del modelo depende estrictamente de la naturaleza del dato:
1. **Datos Textuales (NLP)**: LSTM es la opción predilecta por su capacidad de memoria selectiva.
2. **Datos Numéricos (Series Temporales)**: GRU ofrece un balance óptimo entre precisión y velocidad de respuesta.
3. **Optimización**: Se recomienda el uso de GRU en entornos con recursos limitados (IoT/Edge) debido a su menor consumo de memoria GPU.`);
  }

  return sections.join("\n\n");
};
