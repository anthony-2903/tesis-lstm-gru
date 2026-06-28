import { fetchAiAnalysis } from "./api";

export const generateAiAnalysis = async (type: "general" | "phishtank" | "energia" | "finanzas") => {
  try {
    return await fetchAiAnalysis(type);
  } catch {
    return [
      "### Backend local pendiente",
      "El frontend ya no genera an?lisis desde un dataset cargado en el navegador.",
      "Cuando el backend exponga `/api/ai-analysis?type=...`, este bloque mostrar? la s?ntesis t?cnica generada con los resultados procesados.",
    ].join("\n");
  }
};
