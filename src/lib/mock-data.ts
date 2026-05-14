// Mock data for the LSTM vs GRU vs CNN anomaly detection dashboard

export const kpiData = {
  totalRecords: 38294,
  totalAnomalies: { phishtank: 1247, energia: 389, finanzas: 842 },
  bestF1: { 
    phishtank: { model: "LSTM", score: 0.964 }, 
    energia: { model: "GRU", score: 0.938 },
    finanzas: { model: "CNN", score: 0.952 }
  },
  avgInferenceTime: 10.8, // ms
};

// --- PHISHTANK DATA ---
export const phishtankMetrics = {
  lstm: { precision: 0.957, recall: 0.971, f1: 0.964 },
  gru: { precision: 0.941, recall: 0.928, f1: 0.934 },
  cnn: { precision: 0.948, recall: 0.955, f1: 0.951 },
};

export const phishtankBarData = [
  { name: "Anomalía (Real)", count: 1247 },
  { name: "Detectado LSTM", count: 1198 },
  { name: "Detectado GRU", count: 1156 },
  { name: "Detectado CNN", count: 1182 },
];

export const phishtankConfusionMatrix = {
  lstm: { tp: 1198, fp: 52, fn: 49, tn: 11423 },
  gru: { tp: 1156, fp: 87, fn: 91, tn: 11389 },
  cnn: { tp: 1182, fp: 70, fn: 65, tn: 11405 },
};

export const confusionMatrix = phishtankConfusionMatrix;

export const phishtankUrls = [
  { id: 1, url: "http://secure-paypal-login.com/verify", real: "phishing", lstm: "phishing", gru: "phishing", cnn: "phishing", anomaly: true },
  { id: 2, url: "https://www.google.com/search?q=test", real: "legit", lstm: "legit", gru: "legit", cnn: "legit", anomaly: false },
  { id: 3, url: "http://amaz0n-security.net/update", real: "phishing", lstm: "phishing", gru: "legit", cnn: "phishing", anomaly: true },
  { id: 4, url: "https://github.com/tensorflow/keras", real: "legit", lstm: "legit", gru: "legit", cnn: "legit", anomaly: false },
  { id: 5, url: "http://bank0famerica-login.xyz/auth", real: "phishing", lstm: "phishing", gru: "phishing", cnn: "phishing", anomaly: true },
];

// --- ENERGY DATA ---
export const energyMetrics = {
  lstm: { precision: 0.892, recall: 0.875, f1: 0.883, rmse: 18.42 },
  gru: { precision: 0.925, recall: 0.912, f1: 0.918, rmse: 16.95 },
  cnn: { precision: 0.854, recall: 0.820, f1: 0.836, rmse: 21.34 },
};

export const energyBarData = [
  { name: "Anomalía (Real)", count: 389 },
  { name: "Detectado LSTM", count: 342 },
  { name: "Detectado GRU", count: 355 },
  { name: "Detectado CNN", count: 319 },
];

export const energyConfusionMatrix = {
  lstm: { tp: 342, fp: 41, fn: 47, tn: 11701 },
  gru: { tp: 355, fp: 28, fn: 34, tn: 11714 },
  cnn: { tp: 319, fp: 55, fn: 70, tn: 11687 },
};

export const energySamples = [
  { id: 1, date: "2024-03-15 14:00", value: 742.5, real: "anomalía", lstm: "anomalía", gru: "anomalía", cnn: "anomalía", anomaly: true },
  { id: 2, date: "2024-03-15 15:00", value: 450.2, real: "normal", lstm: "normal", gru: "normal", cnn: "normal", anomaly: false },
  { id: 3, date: "2024-03-15 16:00", value: 812.9, real: "anomalía", lstm: "anomalía", gru: "anomalía", cnn: "normal", anomaly: true },
  { id: 4, date: "2024-03-15 17:00", value: 462.1, real: "normal", lstm: "normal", gru: "normal", cnn: "normal", anomaly: false },
  { id: 5, date: "2024-03-15 18:00", value: 448.7, real: "normal", lstm: "normal", gru: "normal", cnn: "normal", anomaly: false },
];

// --- FINANCE DATA ---
export const financeMetrics = {
  lstm: { precision: 0.912, recall: 0.895, f1: 0.903 },
  gru: { precision: 0.925, recall: 0.910, f1: 0.917 },
  cnn: { precision: 0.958, recall: 0.946, f1: 0.952 },
};

export const financeBarData = [
  { name: "Fraude (Real)", count: 842 },
  { name: "Detectado LSTM", count: 742 },
  { name: "Detectado GRU", count: 765 },
  { name: "Detectado CNN", count: 802 },
];

export const financeConfusionMatrix = {
  lstm: { tp: 742, fp: 145, fn: 108, tn: 12105 },
  gru: { tp: 765, fp: 110, fn: 85, tn: 12140 },
  cnn: { tp: 802, fp: 70, fn: 48, tn: 12180 },
};

export const financeTransactions = [
  { id: 1, txn: "TXN_9482_AD", amount: 12450.00, real: "fraude", lstm: "fraude", gru: "fraude", cnn: "fraude", anomaly: true },
  { id: 2, txn: "TXN_1029_BS", amount: 45.20, real: "normal", lstm: "normal", gru: "normal", cnn: "normal", anomaly: false },
  { id: 3, txn: "TXN_5521_CQ", amount: 8900.50, real: "fraude", lstm: "normal", gru: "normal", cnn: "fraude", anomaly: true },
  { id: 4, txn: "TXN_8820_ZZ", amount: 120.00, real: "normal", lstm: "normal", gru: "normal", cnn: "normal", anomaly: false },
  { id: 5, txn: "TXN_3310_PL", amount: 15400.00, real: "fraude", lstm: "fraude", gru: "fraude", cnn: "fraude", anomaly: true },
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
      cnn: Math.round((base + (Math.random() - 0.5) * 22) * 10) / 10,
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
      cnn: Math.max(0, anomalies + (Math.random() > 0.9 ? 1 : -1)),
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
      cnn: Math.round(fraudScore + (Math.random() - 0.5) * 5),
      anomaly: fraudScore > 85,
    });
  }
  return data;
};

// --- GLOBAL COMPARISON DATA ---
export const radarData = [
  { metric: "F1-Score", LSTM: 0.964, GRU: 0.934, CNN: 0.952 },
  { metric: "AUC-ROC", LSTM: 0.983, GRU: 0.969, CNN: 0.976 },
  { metric: "Precision", LSTM: 0.957, GRU: 0.941, CNN: 0.948 },
  { metric: "Recall", LSTM: 0.971, GRU: 0.928, CNN: 0.955 },
  { metric: "Velocidad", LSTM: 0.72, GRU: 0.89, CNN: 0.98 },
  { metric: "Eficiencia Mem.", LSTM: 0.65, GRU: 0.82, CNN: 0.94 },
];

export const comparisonBarData = [
  { metric: "F1 (PhishTank)", LSTM: 0.964, GRU: 0.934, CNN: 0.951 },
  { metric: "F1 (Finanzas)", LSTM: 0.903, GRU: 0.917, CNN: 0.952 },
  { metric: "F1 (Energía)", LSTM: 0.883, GRU: 0.918, CNN: 0.836 },
];

export const scatterData = [
  { model: "LSTM-PhishTank", time: 342, accuracy: 0.964, domain: "PhishTank" },
  { model: "GRU-PhishTank", time: 218, accuracy: 0.934, domain: "PhishTank" },
  { model: "CNN-PhishTank", time: 145, accuracy: 0.951, domain: "PhishTank" },
  { model: "LSTM-Energía", time: 489, accuracy: 0.883, domain: "Energía" },
  { model: "GRU-Energía", time: 312, accuracy: 0.918, domain: "Energía" },
  { model: "CNN-Energía", time: 198, accuracy: 0.836, domain: "Energía" },
  { model: "LSTM-Finanzas", time: 412, accuracy: 0.903, domain: "Finanzas" },
  { model: "GRU-Finanzas", time: 275, accuracy: 0.917, domain: "Finanzas" },
  { model: "CNN-Finanzas", time: 165, accuracy: 0.952, domain: "Finanzas" },
];

export const comparisonTable = [
  { metric: "Precision Global", lstm: 0.942, gru: 0.938, cnn: 0.945, winner: "CNN" },
  { metric: "F1-Score Promedio", lstm: 0.916, gru: 0.923, cnn: 0.913, winner: "GRU" },
  { metric: "Tiempo Entrenamiento (s)", lstm: 415.5, gru: 265.0, cnn: 169.3, winner: "CNN" },
  { metric: "Memoria GPU (MB)", lstm: 1248, gru: 876, cnn: 452, winner: "CNN" },
  { metric: "Tiempo Inferencia (ms)", lstm: 14.2, gru: 10.6, cnn: 4.8, winner: "CNN" },
];

export const anomalyHistory = [
  { id: 1, date: "2024-03-15", domain: "PhishTank", data: "http://secure-paypal-login.com/verify", model: "LSTM", confidence: 97.2, realLabel: "phishing", predicted: "phishing" },
  { id: 2, date: "2024-03-15", domain: "Finanzas", data: "TXN_9482_AD", model: "CNN", confidence: 98.5, realLabel: "fraude", predicted: "fraude" },
  { id: 3, date: "2024-03-14", domain: "Energía", data: "2024-03-14 14:30:00", model: "GRU", confidence: 89.3, realLabel: "anomalía", predicted: "anomalía" },
];
