import { phishtankMetrics, energyMetrics, kpiData, phishtankConfusionMatrix, financeMetrics } from "./mock-data";

export const generateAiAnalysis = async (type: "general" | "phishtank" | "energia" | "finanzas") => {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 1500));
  const sections = [];

  if (type === "general" || type === "phishtank") {
    sections.push(`
### 🛡️ Análisis de PhishTank (NLP / URLs)
El modelo **LSTM** mantiene el liderazgo en detección de phishing con un **F1-Score de ${phishtankMetrics.lstm.f1.toFixed(3)}**. 
- **CNN vs RNN**: La **CNN** demostró ser una alternativa competitiva (${phishtankMetrics.cnn.f1.toFixed(3)}) con una velocidad de inferencia sustancialmente mayor.
- **Detección**: Se redujeron los falsos negativos a solo **${phishtankConfusionMatrix.lstm.fn}** mediante el uso de celdas de memoria selectiva.`);
  }

  if (type === "general" || type === "energia") {
    sections.push(`
### ⚡ Análisis de Energía (Series Temporales)
En este dominio numérico, la arquitectura **GRU** supera a sus competidores.
- **Precisión**: Alcanzó un **RMSE de ${energyMetrics.gru.rmse.toFixed(2)}**, demostrando que para series temporales con estacionalidad marcada, las puertas de actualización del GRU son más eficientes que las del LSTM (${energyMetrics.lstm.rmse.toFixed(2)}).`);
  }

  if (type === "general" || type === "finanzas") {
    sections.push(`
### 💰 Análisis de Finanzas (Fraude)
La sorpresa arquitectónica se dio en el dominio financiero, donde la **CNN** dominó ampliamente.
- **Rendimiento**: La **CNN** alcanzó un **F1-Score de ${financeMetrics.cnn.f1.toFixed(3)}**, superando la capacidad secuencial de las RNN.
- **Conclusión**: Esto sugiere que en la detección de fraude, los patrones locales (capturados por filtros convolucionales) son más determinantes que las dependencias a largo plazo.`);
  }

  if (type === "general") {
    sections.push(`
### 📈 Conclusión Global de la Tesis
Tras procesar **${kpiData.totalRecords.toLocaleString()}** registros, se concluye:
1. **NLP**: LSTM es superior en semántica compleja.
2. **Series Temporales**: GRU es el más balanceado.
3. **Optimización**: La **CNN** es la más eficiente en entornos de alta transaccionalidad (Finanzas) y recursos limitados.`);
  }

  return sections.join("\n\n");
};
