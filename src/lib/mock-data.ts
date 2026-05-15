// Mock data for the LSTM vs GRU vs Transformer vs TCN anomaly detection dashboard

export const kpiData = {
  totalRecords: 38294,
  totalAnomalies: { phishtank: 1247, energia: 389, finanzas: 842 },
  bestF1: { 
    phishtank: { model: "Transformer", score: 0.978 }, 
    energia: { model: "TCN", score: 0.945 },
    finanzas: { model: "Transformer", score: 0.965 }
  },
  avgInferenceTime: 12.4, // ms
};

// --- PHISHTANK DATA ---
export const phishtankMetrics = {
  lstm: { precision: 0.957, recall: 0.971, f1: 0.964 },
  gru: { precision: 0.941, recall: 0.928, f1: 0.934 },
  transformer: { precision: 0.975, recall: 0.982, f1: 0.978 },
  tcn: { precision: 0.948, recall: 0.955, f1: 0.951 },
};

export const phishtankBarData = [
  { name: "Anomalía (Real)", count: 1247 },
  { name: "Detectado LSTM", count: 1198 },
  { name: "Detectado GRU", count: 1156 },
  { name: "Detectado Transf.", count: 1224 },
  { name: "Detectado TCN", count: 1182 },
];

export const phishtankConfusionMatrix = {
  lstm: { tp: 1198, fp: 52, fn: 49, tn: 11423 },
  gru: { tp: 1156, fp: 87, fn: 91, tn: 11389 },
  transformer: { tp: 1224, fp: 31, fn: 23, tn: 11444 },
  tcn: { tp: 1182, fp: 70, fn: 65, tn: 11405 },
};

export const confusionMatrix = phishtankConfusionMatrix;

export const phishtankUrls = [
  { id: 1, url: "http://secure-paypal-login.com/verify", real: "phishing", lstm: "phishing", gru: "phishing", transformer: "phishing", tcn: "phishing", anomaly: true },
  { id: 2, url: "https://www.google.com/search?q=test", real: "legit", lstm: "legit", gru: "legit", transformer: "legit", tcn: "legit", anomaly: false },
  { id: 3, url: "http://amaz0n-security.net/update", real: "phishing", lstm: "phishing", gru: "legit", transformer: "phishing", tcn: "phishing", anomaly: true },
  { id: 4, url: "https://github.com/tensorflow/keras", real: "legit", lstm: "legit", gru: "legit", transformer: "legit", tcn: "legit", anomaly: false },
  { id: 5, url: "http://bank0famerica-login.xyz/auth", real: "phishing", lstm: "phishing", gru: "phishing", transformer: "phishing", tcn: "phishing", anomaly: true },
];

// --- ENERGY DATA ---
export const energyMetrics = {
  lstm: { precision: 0.892, recall: 0.875, f1: 0.883, rmse: 18.42 },
  gru: { precision: 0.925, recall: 0.912, f1: 0.918, rmse: 16.95 },
  transformer: { precision: 0.918, recall: 0.905, f1: 0.911, rmse: 17.20 },
  tcn: { precision: 0.952, recall: 0.938, f1: 0.945, rmse: 14.80 },
};

export const energyBarData = [
  { name: "Anomalía (Real)", count: 389 },
  { name: "Detectado LSTM", count: 342 },
  { name: "Detectado GRU", count: 355 },
  { name: "Detectado Transf.", count: 351 },
  { name: "Detectado TCN", count: 365 },
];

export const energyConfusionMatrix = {
  lstm: { tp: 342, fp: 41, fn: 47, tn: 11701 },
  gru: { tp: 355, fp: 28, fn: 34, tn: 11714 },
  transformer: { tp: 351, fp: 31, fn: 38, tn: 11711 },
  tcn: { tp: 365, fp: 18, fn: 24, tn: 11724 },
};

export const energySamples = [
  { id: 1, date: "2024-03-15 14:00", value: 742.5, real: "anomalía", lstm: "anomalía", gru: "anomalía", transformer: "anomalía", tcn: "anomalía", anomaly: true },
  { id: 2, date: "2024-03-15 15:00", value: 450.2, real: "normal", lstm: "normal", gru: "normal", transformer: "normal", tcn: "normal", anomaly: false },
  { id: 3, date: "2024-03-15 16:00", value: 812.9, real: "anomalía", lstm: "anomalía", gru: "anomalía", transformer: "anomalía", tcn: "anomalía", anomaly: true },
  { id: 4, date: "2024-03-15 17:00", value: 462.1, real: "normal", lstm: "normal", gru: "normal", transformer: "normal", tcn: "normal", anomaly: false },
  { id: 5, date: "2024-03-15 18:00", value: 448.7, real: "normal", lstm: "normal", gru: "normal", transformer: "normal", tcn: "normal", anomaly: false },
];

// --- FINANCE DATA ---
export const financeMetrics = {
  lstm: { precision: 0.912, recall: 0.895, f1: 0.903 },
  gru: { precision: 0.925, recall: 0.910, f1: 0.917 },
  transformer: { precision: 0.968, recall: 0.962, f1: 0.965 },
  tcn: { precision: 0.958, recall: 0.946, f1: 0.952 },
};

export const financeBarData = [
  { name: "Fraude (Real)", count: 842 },
  { name: "Detectado LSTM", count: 742 },
  { name: "Detectado GRU", count: 765 },
  { name: "Detectado Transf.", count: 810 },
  { name: "Detectado TCN", count: 802 },
];

export const financeConfusionMatrix = {
  lstm: { tp: 742, fp: 145, fn: 108, tn: 12105 },
  gru: { tp: 765, fp: 110, fn: 85, tn: 12140 },
  transformer: { tp: 810, fp: 27, fn: 32, tn: 12223 },
  tcn: { tp: 802, fp: 70, fn: 48, tn: 12180 },
};

export const financeTransactions = [
  { id: 1, txn: "TXN_9482_AD", amount: 12450.00, real: "fraude", lstm: "fraude", gru: "fraude", transformer: "fraude", tcn: "fraude", anomaly: true },
  { id: 2, txn: "TXN_1029_BS", amount: 45.20, real: "normal", lstm: "normal", gru: "normal", transformer: "normal", tcn: "normal", anomaly: false },
  { id: 3, txn: "TXN_5521_CQ", amount: 8900.50, real: "fraude", lstm: "normal", gru: "normal", transformer: "fraude", tcn: "fraude", anomaly: true },
  { id: 4, txn: "TXN_8820_ZZ", amount: 120.00, real: "normal", lstm: "normal", gru: "normal", transformer: "normal", tcn: "normal", anomaly: false },
  { id: 5, txn: "TXN_3310_PL", amount: 15400.00, real: "fraude", lstm: "fraude", gru: "fraude", transformer: "fraude", tcn: "fraude", anomaly: true },
];

// --- GENERATORS ---
export const generateTimeSeriesData = () => {
  const data = [];
  const startDate = new Date("2024-01-01");
  for (let i = 0; i < 90; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    const base = 450 + Math.sin(i * 0.1) * 50;
    const actual = base + (Math.random() - 0.5) * 40;
    const isAnomaly = Math.random() < 0.05;
    const val = isAnomaly ? actual + 100 : actual;
    data.push({
      date: date.toISOString().split("T")[0],
      actual: Math.round(val * 10) / 10,
      lstm: Math.round((base + (Math.random() - 0.5) * 15) * 10) / 10,
      gru: Math.round((base + (Math.random() - 0.5) * 18) * 10) / 10,
      transformer: Math.round((base + (Math.random() - 0.5) * 12) * 10) / 10,
      tcn: Math.round((base + (Math.random() - 0.5) * 14) * 10) / 10,
      anomaly: isAnomaly,
    });
  }
  return data;
};

export const generatePhishTankTimeline = () => {
  const data = [];
  const startDate = new Date("2024-03-01");
  for (let i = 0; i < 30; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    const total = 50 + Math.floor(Math.random() * 30);
    const anomalies = Math.floor(total * (0.05 + Math.random() * 0.1));
    data.push({
      date: date.toISOString().split("T")[0],
      anomalies,
      lstm: Math.max(0, anomalies + (Math.random() > 0.8 ? 1 : 0)),
      gru: Math.max(0, anomalies - (Math.random() > 0.7 ? 1 : 0)),
      transformer: Math.max(0, anomalies + (Math.random() > 0.9 ? 1 : 0)),
      tcn: Math.max(0, anomalies + (Math.random() > 0.9 ? 1 : -1)),
    });
  }
  return data;
};

export const generateFinanceTimeline = () => {
  const data = [];
  const startDate = new Date("2024-03-01");
  for (let i = 0; i < 30; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    const fraudScore = Math.random() * 100;
    data.push({
      date: date.toISOString().split("T")[0],
      score: Math.round(fraudScore),
      lstm: Math.round(fraudScore + (Math.random() - 0.5) * 10),
      gru: Math.round(fraudScore + (Math.random() - 0.5) * 12),
      transformer: Math.round(fraudScore + (Math.random() - 0.5) * 5),
      tcn: Math.round(fraudScore + (Math.random() - 0.5) * 8),
      anomaly: fraudScore > 85,
    });
  }
  return data;
};

// --- GLOBAL COMPARISON DATA ---
export const radarData = [
  { metric: "F1-Score", LSTM: 0.964, GRU: 0.934, Transformer: 0.978, TCN: 0.951 },
  { metric: "AUC-ROC", LSTM: 0.983, GRU: 0.969, Transformer: 0.991, TCN: 0.978 },
  { metric: "Precision", LSTM: 0.957, GRU: 0.941, Transformer: 0.975, TCN: 0.948 },
  { metric: "Recall", LSTM: 0.971, GRU: 0.928, Transformer: 0.982, TCN: 0.955 },
  { metric: "Velocidad", LSTM: 0.62, GRU: 0.78, Transformer: 0.45, TCN: 0.96 },
  { metric: "Eficiencia Mem.", LSTM: 0.55, GRU: 0.72, Transformer: 0.35, TCN: 0.92 },
];

export const comparisonBarData = [
  { metric: "F1 (PhishTank)", LSTM: 0.964, GRU: 0.934, Transformer: 0.978, TCN: 0.951 },
  { metric: "F1 (Finanzas)", LSTM: 0.903, GRU: 0.917, Transformer: 0.965, TCN: 0.952 },
  { metric: "F1 (Energía)", LSTM: 0.883, GRU: 0.918, Transformer: 0.911, TCN: 0.945 },
];

export const scatterData = [
  { model: "LSTM-PhishTank", time: 342, accuracy: 0.964, domain: "PhishTank" },
  { model: "GRU-PhishTank", time: 218, accuracy: 0.934, domain: "PhishTank" },
  { model: "Transf-PhishTank", time: 512, accuracy: 0.978, domain: "PhishTank" },
  { model: "TCN-PhishTank", time: 145, accuracy: 0.951, domain: "PhishTank" },
  { model: "LSTM-Energía", time: 489, accuracy: 0.883, domain: "Energía" },
  { model: "GRU-Energía", time: 312, accuracy: 0.918, domain: "Energía" },
  { model: "Transf-Energía", time: 642, accuracy: 0.911, domain: "Energía" },
  { model: "TCN-Energía", time: 198, accuracy: 0.945, domain: "Energía" },
  { model: "LSTM-Finanzas", time: 412, accuracy: 0.903, domain: "Finanzas" },
  { model: "GRU-Finanzas", time: 275, accuracy: 0.917, domain: "Finanzas" },
  { model: "Transf-Finanzas", time: 580, accuracy: 0.965, domain: "Finanzas" },
  { model: "TCN-Finanzas", time: 165, accuracy: 0.952, domain: "Finanzas" },
];

export const comparisonTable = [
  { metric: "Precision Global", lstm: 0.942, gru: 0.938, transformer: 0.971, tcn: 0.945, winner: "Transformer" },
  { metric: "F1-Score Promedio", lstm: 0.916, gru: 0.923, transformer: 0.951, tcn: 0.949, winner: "Transformer" },
  { metric: "Tiempo Entrenamiento (s)", lstm: 415.5, gru: 265.0, transformer: 1240.2, tcn: 169.3, winner: "TCN" },
  { metric: "Memoria GPU (MB)", lstm: 1248, gru: 876, transformer: 4250, tcn: 452, winner: "TCN" },
  { metric: "Tiempo Inferencia (ms)", lstm: 14.2, gru: 10.6, transformer: 22.4, tcn: 4.8, winner: "TCN" },
];

export const anomalyHistory = [
  { id: 1, date: "2024-03-15", domain: "PhishTank", data: "http://secure-paypal-login.com/verify", model: "Transformer", confidence: 99.1, realLabel: "phishing", predicted: "phishing" },
  { id: 2, date: "2024-03-15", domain: "Finanzas", data: "TXN_9482_AD", model: "TCN", confidence: 98.5, realLabel: "fraude", predicted: "fraude" },
  { id: 3, date: "2024-03-14", domain: "Energía", data: "2024-03-14 14:30:00", model: "GRU", confidence: 89.3, realLabel: "anomalía", predicted: "anomalía" },
];
