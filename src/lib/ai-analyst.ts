import { phishtankMetrics, energyMetrics, kpiData, phishtankConfusionMatrix, financeMetrics } from "./mock-data";

export const generateAiAnalysis = async (type: "general" | "phishtank" | "energia" | "finanzas") => {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 1500));
  const sections = [];

  if (type === "general" || type === "phishtank") {
    sections.push(`
### 🛡️ Análisis de PhishTank (NLP / URLs)
El modelo **Transformer** ha establecido un nuevo estándar en detección de phishing con un **F1-Score de ${phishtankMetrics.transformer.f1.toFixed(3)}**. 
- **Atención Global**: La capacidad del Transformer para analizar la URL completa mediante mecanismos de atención permite detectar variaciones sutiles que las RNN (LSTM/GRU) suelen pasar por alto.
- **Reducción de Errores**: Se redujeron los falsos negativos a solo **${phishtankConfusionMatrix.transformer.fn}**, el nivel más bajo registrado en el proyecto.`);
  }

  if (type === "general" || type === "energia") {
    sections.push(`
### ⚡ Análisis de Energía (Series Temporales)
En el dominio de series temporales puras, la arquitectura **TCN** demuestra una eficiencia superior.
- **Precisión Convolucional**: Alcanzó un **RMSE de ${energyMetrics.tcn.rmse.toFixed(2)}**, superando a la GRU (${energyMetrics.gru.rmse.toFixed(2)}). Esto confirma que las convoluciones causales dilatadas son más efectivas para capturar la estacionalidad energética sin el riesgo de desvanecimiento de gradiente.`);
  }

  if (type === "general" || type === "finanzas") {
    sections.push(`
### 💰 Análisis de Finanzas (Fraude)
El modelo **Transformer** vuelve a destacar en el dominio financiero por su capacidad de identificar dependencias complejas entre transacciones.
- **Rendimiento**: Logró un **F1-Score de ${financeMetrics.transformer.f1.toFixed(3)}**.
- **Contexto TCN**: La **TCN** se posiciona como la mejor alternativa para despliegues en tiempo real, ofreciendo un balance óptimo entre precisión (${financeMetrics.tcn.f1.toFixed(3)}) y velocidad de inferencia.`);
  }

  if (type === "general") {
    sections.push(`
### 📈 Conclusión Global de la Tesis
Tras procesar **${kpiData.totalRecords.toLocaleString()}** registros, los hallazgos principales son:
1. **Liderazgo en Precisión**: El **Transformer** es la arquitectura más potente para clasificación semántica y de fraude.
2. **Eficiencia Temporal**: La **TCN** es la más balanceada para series temporales numéricas, ofreciendo la mayor velocidad de inferencia.
3. **Legado RNN**: LSTM y GRU siguen siendo sólidos, pero presentan dificultades ante dependencias de muy largo plazo en comparación con las nuevas arquitecturas integradas.`);
  }

  return sections.join("\n\n");
};
