// Mock data for the LSTM vs GRU anomaly detection dashboard

export const kpiData = {
  totalRecords: 24853,
  totalAnomalies: { phishtank: 1247, opsd: 389 },
  bestF1: { phishtank: { model: "LSTM", score: 0.964 }, opsd: { model: "GRU", score: 0.938 } },
  avgInferenceTime: 12.4, // ms
};

export const phishtankUrls = [
  { id: 1, url: "http://secure-paypal-login.com/verify", real: "phishing", lstm: "phishing", gru: "phishing", anomaly: true },
  { id: 2, url: "https://www.google.com/search?q=test", real: "legit", lstm: "legit", gru: "legit", anomaly: false },
  { id: 3, url: "http://amaz0n-security.net/update", real: "phishing", lstm: "phishing", gru: "legit", anomaly: true },
  { id: 4, url: "https://github.com/tensorflow/keras", real: "legit", lstm: "legit", gru: "legit", anomaly: false },
  { id: 5, url: "http://bank0famerica-login.xyz/auth", real: "phishing", lstm: "phishing", gru: "phishing", anomaly: true },
  { id: 6, url: "https://stackoverflow.com/questions", real: "legit", lstm: "legit", gru: "legit", anomaly: false },
  { id: 7, url: "http://micr0soft-update.tk/download", real: "phishing", lstm: "phishing", gru: "phishing", anomaly: true },
  { id: 8, url: "https://www.wikipedia.org/wiki/LSTM", real: "legit", lstm: "legit", gru: "phishing", anomaly: false },
  { id: 9, url: "http://app1e-id-verify.ml/secure", real: "phishing", lstm: "phishing", gru: "phishing", anomaly: true },
  { id: 10, url: "https://docs.python.org/3/library", real: "legit", lstm: "legit", gru: "legit", anomaly: false },
];

export const phishtankMetrics = {
  lstm: { precision: 0.957, recall: 0.971, f1: 0.964, auc: 0.983 },
  gru: { precision: 0.941, recall: 0.928, f1: 0.934, auc: 0.969 },
};

export const phishtankBarData = [
  { name: "Phishing Detectado", LSTM: 1198, GRU: 1156 },
  { name: "Legítimo Detectado", LSTM: 11423, GRU: 11389 },
  { name: "Falso Positivo", LSTM: 52, GRU: 87 },
  { name: "Falso Negativo", LSTM: 49, GRU: 91 },
];

export const confusionMatrix = {
  lstm: { tp: 1198, fp: 52, fn: 49, tn: 11423 },
  gru: { tp: 1156, fp: 87, fn: 91, tn: 11389 },
};

// Generate time series data for Open Power System
export const generateTimeSeriesData = () => {
  const data = [];
  const startDate = new Date("2024-01-01");
  for (let i = 0; i < 365; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    const base = 450 + Math.sin(i * 0.017) * 100 + Math.sin(i * 0.1) * 30;
    const noise = (Math.random() - 0.5) * 40;
    const actual = base + noise;
    const lstmPred = base + (Math.random() - 0.5) * 15;
    const gruPred = base + (Math.random() - 0.5) * 18;
    const isAnomaly = Math.random() < 0.03;
    const anomalyActual = isAnomaly ? actual + (Math.random() > 0.5 ? 1 : -1) * (120 + Math.random() * 80) : actual;

    data.push({
      date: date.toISOString().split("T")[0],
      actual: Math.round(anomalyActual * 10) / 10,
      lstm: Math.round(lstmPred * 10) / 10,
      gru: Math.round(gruPred * 10) / 10,
      anomaly: isAnomaly,
    });
  }
  return data;
};

export const opsdMetrics = {
  lstm: { rmse: 18.42, mae: 12.87 },
  gru: { rmse: 16.95, mae: 11.63 },
};

export const radarData = [
  { metric: "F1-Score", LSTM: 0.964, GRU: 0.934 },
  { metric: "AUC-ROC", LSTM: 0.983, GRU: 0.969 },
  { metric: "1-RMSE(norm)", LSTM: 0.82, GRU: 0.85 },
  { metric: "1-MAE(norm)", LSTM: 0.87, GRU: 0.88 },
  { metric: "Velocidad", LSTM: 0.72, GRU: 0.89 },
  { metric: "Eficiencia Mem.", LSTM: 0.65, GRU: 0.82 },
];

export const comparisonBarData = [
  { metric: "Precision (PhishTank)", LSTM: 0.957, GRU: 0.941 },
  { metric: "Recall (PhishTank)", LSTM: 0.971, GRU: 0.928 },
  { metric: "F1 (PhishTank)", LSTM: 0.964, GRU: 0.934 },
  { metric: "RMSE (OPSD)", LSTM: 18.42, GRU: 16.95 },
  { metric: "MAE (OPSD)", LSTM: 12.87, GRU: 11.63 },
];

export const scatterData = [
  { model: "LSTM-PhishTank", time: 342, accuracy: 0.964, domain: "PhishTank" },
  { model: "GRU-PhishTank", time: 218, accuracy: 0.934, domain: "PhishTank" },
  { model: "LSTM-OPSD", time: 489, accuracy: 0.891, domain: "OPSD" },
  { model: "GRU-OPSD", time: 312, accuracy: 0.903, domain: "OPSD" },
];

export const comparisonTable = [
  { metric: "Precision (PhishTank)", lstm: 0.957, gru: 0.941, winner: "LSTM" },
  { metric: "Recall (PhishTank)", lstm: 0.971, gru: 0.928, winner: "LSTM" },
  { metric: "F1-Score (PhishTank)", lstm: 0.964, gru: 0.934, winner: "LSTM" },
  { metric: "AUC-ROC (PhishTank)", lstm: 0.983, gru: 0.969, winner: "LSTM" },
  { metric: "RMSE (OPSD)", lstm: 18.42, gru: 16.95, winner: "GRU" },
  { metric: "MAE (OPSD)", lstm: 12.87, gru: 11.63, winner: "GRU" },
  { metric: "Tiempo Entrenamiento (s)", lstm: 415.5, gru: 265.0, winner: "GRU" },
  { metric: "Memoria GPU (MB)", lstm: 1248, gru: 876, winner: "GRU" },
  { metric: "Tiempo Inferencia (ms)", lstm: 14.2, gru: 10.6, winner: "GRU" },
];

export const anomalyHistory = [
  { id: 1, date: "2024-03-15", domain: "PhishTank", data: "http://secure-paypal-login.com/verify", model: "LSTM", confidence: 97.2, realLabel: "phishing", predicted: "phishing" },
  { id: 2, date: "2024-03-15", domain: "PhishTank", data: "http://amaz0n-security.net/update", model: "LSTM", confidence: 94.8, realLabel: "phishing", predicted: "phishing" },
  { id: 3, date: "2024-03-14", domain: "OPSD", data: "2024-03-14 14:30:00", model: "GRU", confidence: 89.3, realLabel: "anomalía", predicted: "anomalía" },
  { id: 4, date: "2024-03-14", domain: "PhishTank", data: "http://bank0famerica-login.xyz/auth", model: "GRU", confidence: 91.5, realLabel: "phishing", predicted: "phishing" },
  { id: 5, date: "2024-03-13", domain: "OPSD", data: "2024-03-13 02:15:00", model: "LSTM", confidence: 86.7, realLabel: "anomalía", predicted: "anomalía" },
  { id: 6, date: "2024-03-12", domain: "PhishTank", data: "http://micr0soft-update.tk/download", model: "LSTM", confidence: 98.1, realLabel: "phishing", predicted: "phishing" },
  { id: 7, date: "2024-03-12", domain: "OPSD", data: "2024-03-12 19:45:00", model: "GRU", confidence: 92.4, realLabel: "anomalía", predicted: "anomalía" },
  { id: 8, date: "2024-03-11", domain: "PhishTank", data: "http://app1e-id-verify.ml/secure", model: "LSTM", confidence: 96.3, realLabel: "phishing", predicted: "phishing" },
  { id: 9, date: "2024-03-10", domain: "OPSD", data: "2024-03-10 08:00:00", model: "LSTM", confidence: 83.9, realLabel: "anomalía", predicted: "normal" },
  { id: 10, date: "2024-03-09", domain: "OPSD", data: "2024-03-09 23:30:00", model: "GRU", confidence: 95.1, realLabel: "anomalía", predicted: "anomalía" },
];
