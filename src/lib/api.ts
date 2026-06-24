const LOCAL_API_URL = "http://localhost:8000/api";
const PRODUCTION_API_URL = "https://name-tesis-lstm-gru-backend.onrender.com/api";

function resolveApiUrl() {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }

  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    if (host === "localhost" || host === "127.0.0.1") {
      return LOCAL_API_URL;
    }
  }

  return PRODUCTION_API_URL;
}

const API_URL = resolveApiUrl();
export type DomainId = "phishing" | "energia" | "finanzas";

export interface DomainOption {
  id: DomainId;
  title: string;
  source: string;
  description: string;
}

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

function withDomain(path: string, domain?: DomainId) {
  if (!domain) return path;
  const query = new URLSearchParams({ domain });
  return `${path}?${query.toString()}`;
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`);
  if (!response.ok) {
    throw new Error(`Backend respondio ${response.status} en ${path}`);
  }
  return response.json() as Promise<T>;
}

export function fetchDomains() {
  return fetchJson<{ items: DomainOption[] }>("/domains");
}

export function fetchDashboardData(domain?: DomainId) {
  return fetchJson<DashboardData>(withDomain("/dashboard", domain));
}

export function fetchAnalysisData(domain?: DomainId) {
  return fetchJson<EvaluatedData>(withDomain("/analysis", domain));
}

export function fetchComparisonData(domain?: DomainId) {
  return fetchJson<ComparisonData>(withDomain("/comparison", domain));
}

export function fetchHistoryData(domain?: DomainId) {
  return fetchJson<HistoryData>(withDomain("/history", domain));
}

export function fetchXaiData(domain?: DomainId) {
  return fetchJson<unknown>(withDomain("/xai", domain));
}

export async function fetchAiAnalysis(type: "general" | "phishtank" | "energia" | "finanzas") {
  const query = new URLSearchParams({ type });
  const data = await fetchJson<{ analysis: string }>(`/ai-analysis?${query.toString()}`);
  return data.analysis;
}
