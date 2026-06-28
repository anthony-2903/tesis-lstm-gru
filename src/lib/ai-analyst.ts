import { fetchAiAnalysis } from "./api";

export const generateAiAnalysis = async (type: "general" | "phishtank" | "energia" | "finanzas") => {
  try {
    return await fetchAiAnalysis(type);
  } catch {
    return [
      "### Backend local pendiente",
      "El frontend ya no genera análisis desde un dataset cargado en el navegador.",
      "Cuando el backend exponga `/api/ai-analysis?type=...`, este bloque mostrará la síntesis técnica generada con los resultados procesados.",
    ].join("\n");
  }
};
