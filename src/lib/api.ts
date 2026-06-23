const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

export interface DashboardData {
  dataset: {
    filename: string;
    originalRows: number;
    cleanedRows: number;
    rowsRemoved: number;
    columns: string[];
    dataTypes: Record<string, string>;
  };
  typeDistribution: { name: string; value: number }[];
  columnBarData: { col: string; tipo: string; registros: number }[];
  numericDistribution: { rango: string; cantidad: number }[];
  numericColumn?: string | null;
}

export interface ConfusionMatrix {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

export interface ModelPerformance {
  f1: number;
  precision: number;
  recall: number;
  rmse?: number;
  confusionMatrix: ConfusionMatrix;
  detectedCount: number;
}

export interface EvaluatedData {
  filename: string;
  totalRows: number;
  realAnomaliesCount: number;
  models: {
    lstm: ModelPerformance;
    gru: ModelPerformance;
    brnn: ModelPerformance;
    transformer: ModelPerformance;
    tcn: ModelPerformance;
  };
  timeline: {
    date: string;
    actual: number;
    anomalies: number;
    lstm: number;
    gru: number;
    brnn: number;
    transformer: number;
    tcn: number;
  }[];
  samples: {
    id: number;
    label: string;
    value: number;
    real: string;
    lstm: string;
    gru: string;
    brnn: string;
    transformer: string;
    tcn: string;
  }[];
  processedRecords?: Record<string, unknown>[];
}

export interface ComparisonData {
  filename: string;
  radarData: Record<string, string | number>[];
  comparisonBarData: Record<string, string | number>[];
  scatterData: { model: string; time: number; accuracy: number }[];
  comparisonTable: {
    metric: string;
    lstm: number;
    gru: number;
    brnn: number;
    transformer: number;
    tcn: number;
    winner: string;
  }[];
}

export interface HistoryItem {
  id: number;
  date: string;
  domain: string;
  data: string;
  model: string;
  confidence: number;
  realLabel: string;
  predicted: string;
}

export interface HistoryData {
  filename: string;
  items: HistoryItem[];
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`);
  if (!response.ok) {
    throw new Error(`Backend local respondio ${response.status} en ${path}`);
  }
  return response.json() as Promise<T>;
}

export function fetchDashboardData() {
  return fetchJson<DashboardData>("/dashboard");
}

export function fetchAnalysisData() {
  return fetchJson<EvaluatedData>("/analysis");
}

export function fetchComparisonData() {
  return fetchJson<ComparisonData>("/comparison");
}

export function fetchHistoryData() {
  return fetchJson<HistoryData>("/history");
}

export function fetchXaiData() {
  return fetchJson<unknown>("/xai");
}

export async function fetchAiAnalysis(type: "general" | "phishtank" | "energia" | "finanzas") {
  const query = new URLSearchParams({ type });
  const data = await fetchJson<{ analysis: string }>(`/ai-analysis?${query.toString()}`);
  return data.analysis;
}
