/**
 * detector.ts
 * Infiere automáticamente el dominio del dataset cargado
 * analizando nombres de columnas, tipos de datos y valores de muestra.
 */

import type { DataDomain } from "./dataStore";

// ─────────────────────────────────────────────────────────
// Diccionarios de palabras clave por dominio
// ─────────────────────────────────────────────────────────

const PHISHING_KEYWORDS = [
  "url", "domain", "phish", "link", "http", "https", "web",
  "site", "host", "spam", "email", "mail", "label", "malicious",
  "phishing", "legit", "legitimate", "clase", "tipo_url", "tipo",
  "category", "result", "status", "class",
];

const ENERGY_KEYWORDS = [
  "energia", "energy", "kwh", "kw", "wh", "watt",
  "consumo", "consume", "power", "potencia", "voltage", "voltaje",
  "corriente", "current", "sensor", "medicion", "medición",
  "temperatura", "temperature", "presion", "pressure",
  "timestamp", "time", "hora", "fecha", "date", "datetime",
  "reading", "measurement", "value", "valor",
];

const FINANCE_KEYWORDS = [
  "amount", "monto", "price", "precio", "transaction", "transaccion",
  "fraud", "fraude", "bank", "banco", "credit", "credito", "debit",
  "debito", "payment", "pago", "balance", "account", "cuenta",
  "transfer", "transferencia", "merchant", "tarjeta", "card",
  "dinero", "money", "ingreso", "gasto", "expense", "revenue",
];

// ─────────────────────────────────────────────────────────
// Función principal de detección
// ─────────────────────────────────────────────────────────

export function detectDomain(
  columns: string[],
  dataTypes: Record<string, string>,
  sampleData: Record<string, unknown>[]
): DataDomain {
  const lowerCols = columns.map((c) => c.toLowerCase());

  // Contar coincidencias de palabras clave por dominio en los nombres de columna
  const score = (keywords: string[]) =>
    lowerCols.reduce((acc, col) => {
      return acc + keywords.filter((kw) => col.includes(kw)).length;
    }, 0);

  let phishScore = score(PHISHING_KEYWORDS);
  let energyScore = score(ENERGY_KEYWORDS);
  let financeScore = score(FINANCE_KEYWORDS);

  // ── Bonus por tipo de dato ────────────────────────────────
  const colTypes = Object.values(dataTypes);
  const hasDate = colTypes.includes("fecha");
  const numericCount = colTypes.filter((t) => t === "número").length;
  const textCount = colTypes.filter((t) => t === "texto").length;

  if (hasDate) energyScore += 2;           // Fechas → series temporales
  if (textCount >= 2) phishScore += 2;     // Muchas columnas texto → URLs/phishing
  if (numericCount >= 3) financeScore += 1; // Muchas columnas numéricas → finanzas

  // ── Bonus por valores de muestra (URLs en texto) ──────────
  if (sampleData.length > 0) {
    const firstRow = sampleData[0];
    for (const val of Object.values(firstRow)) {
      const str = String(val).toLowerCase();
      if (str.startsWith("http") || str.includes(".com") || str.includes(".net") || str.includes(".org")) {
        phishScore += 3; // Valor parece URL real
        break;
      }
    }
  }

  // ── Nombre del archivo ────────────────────────────────────
  // (detectDomain recibe columnas; el nombre se evalúa en DataUpload antes de llamar)

  // ── Decisión final ────────────────────────────────────────
  const maxScore = Math.max(phishScore, energyScore, financeScore);

  // Si el mayor score es muy bajo (< 1), no se puede determinar
  if (maxScore < 1) return "general";

  if (phishScore === maxScore) return "phishing";
  if (energyScore === maxScore) return "energia";
  if (financeScore === maxScore) return "finanzas";

  return "general";
}

/** Bonus adicional basado en el nombre del archivo */
export function boostScoreByFilename(filename: string): Partial<Record<DataDomain, number>> {
  const f = filename.toLowerCase();
  const boost: Partial<Record<DataDomain, number>> = {};

  if (f.includes("phish") || f.includes("url") || f.includes("spam") || f.includes("email")) {
    boost["phishing"] = 5;
  }
  if (f.includes("energ") || f.includes("kwh") || f.includes("sensor") || f.includes("consum") || f.includes("time")) {
    boost["energia"] = 5;
  }
  if (f.includes("fraud") || f.includes("transac") || f.includes("financ") || f.includes("bank") || f.includes("credit")) {
    boost["finanzas"] = 5;
  }
  return boost;
}

/** Versión completa con bonus de nombre de archivo */
export function detectDomainFull(
  filename: string,
  columns: string[],
  dataTypes: Record<string, string>,
  sampleData: Record<string, unknown>[]
): DataDomain {
  const lowerCols = columns.map((c) => c.toLowerCase());

  const score = (keywords: string[]) =>
    lowerCols.reduce((acc, col) => {
      return acc + keywords.filter((kw) => col.includes(kw)).length;
    }, 0);

  let phishScore = score(PHISHING_KEYWORDS);
  let energyScore = score(ENERGY_KEYWORDS);
  let financeScore = score(FINANCE_KEYWORDS);

  // Tipos de dato
  const colTypes = Object.values(dataTypes);
  const hasDate = colTypes.includes("fecha");
  const numericCount = colTypes.filter((t) => t === "número").length;
  const textCount = colTypes.filter((t) => t === "texto").length;

  if (hasDate) energyScore += 2;
  if (textCount >= 2) phishScore += 2;
  if (numericCount >= 3) financeScore += 1;

  // Valores de muestra
  if (sampleData.length > 0) {
    for (const val of Object.values(sampleData[0])) {
      const str = String(val).toLowerCase();
      if (str.startsWith("http") || str.includes(".com") || str.includes(".net")) {
        phishScore += 3;
        break;
      }
    }
  }

  // Bonus por nombre de archivo
  const boost = boostScoreByFilename(filename);
  if (boost["phishing"]) phishScore += boost["phishing"]!;
  if (boost["energia"]) energyScore += boost["energia"]!;
  if (boost["finanzas"]) financeScore += boost["finanzas"]!;

  const maxScore = Math.max(phishScore, energyScore, financeScore);
  if (maxScore < 1) return "general";

  if (phishScore === maxScore) return "phishing";
  if (energyScore === maxScore) return "energia";
  if (financeScore === maxScore) return "finanzas";

  return "general";
}
