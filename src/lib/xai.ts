export interface XaiImportanceItem {
  feature?: string;
  step?: string;
  importance: number;
}

export interface XaiModelExplanation {
  model_key: string;
  model: string;
  method: string;
  description?: string;
  top_feature?: string | null;
  top_step?: string | null;
  feature_importance: Array<XaiImportanceItem & { feature: string; feature_index?: number }>;
  temporal_importance: Array<XaiImportanceItem & { step: string; step_index?: number }>;
}

export interface XaiReport {
  dataset: string;
  method: string;
  feature_count: number;
  sequence_length: number;
  global_feature_importance: Array<XaiImportanceItem & { feature: string }>;
  global_temporal_importance: Array<XaiImportanceItem & { step: string }>;
  model_comparison: {
    model_key: string;
    model: string;
    top_feature?: string | null;
    top_step?: string | null;
  }[];
  models: Record<string, XaiModelExplanation>;
}

export const formatImportance = (value: number) => {
  if (!Number.isFinite(value)) return "0.0000";
  return value.toFixed(4);
};

export const normalizeXaiReport = (raw: unknown): XaiReport => {
  const report = raw as Partial<XaiReport>;
  if (!report || typeof report !== "object") {
    throw new Error("El archivo XAI no tiene un formato JSON valido.");
  }

  if (!Array.isArray(report.global_feature_importance) || !Array.isArray(report.global_temporal_importance)) {
    throw new Error("El archivo no contiene importancias XAI globales.");
  }

  return {
    dataset: report.dataset || "dataset",
    method: report.method || "temporal_masking_shap_approximation",
    feature_count: Number(report.feature_count || 0),
    sequence_length: Number(report.sequence_length || 0),
    global_feature_importance: report.global_feature_importance,
    global_temporal_importance: report.global_temporal_importance,
    model_comparison: Array.isArray(report.model_comparison) ? report.model_comparison : [],
    models: report.models || {},
  };
};
