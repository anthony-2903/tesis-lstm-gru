import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { extname, join } from "node:path";

const root = process.cwd();
const sourceRoots = ["src", "docs", "backend/app"];
const sourceExts = new Set([".ts", ".tsx", ".css", ".md", ".py"]);
const generatedDirs = [
  "backend/app/__pycache__",
  "backend/app/cleaning/__pycache__",
  "backend/app/ingestion/__pycache__",
  "backend/app/training/__pycache__",
  "backend/app/xai/__pycache__",
];
const resultFiles = [
  "backend/storage/results/dashboard.json",
  "backend/storage/results/analysis.json",
  "backend/storage/results/comparison.json",
  "backend/storage/results/history.json",
  "backend/storage/results/xai.json",
];

const failures = [];

function walk(dir, visitor) {
  if (!existsSync(dir)) return;
  for (const entry of readdirSync(dir)) {
    const fullPath = join(dir, entry);
    const stat = statSync(fullPath);
    if (stat.isDirectory()) {
      if (entry === "node_modules" || entry === "dist" || entry === ".git") continue;
      walk(fullPath, visitor);
      continue;
    }
    visitor(fullPath);
  }
}

for (const dir of sourceRoots) {
  walk(join(root, dir), (file) => {
    if (!sourceExts.has(extname(file))) return;
    const text = readFileSync(file, "utf8");
    if (/[ÃÂ]|â[\u0080-\u00BF]/.test(text)) {
      failures.push(`Texto posiblemente mal codificado: ${file}`);
    }
  });
}

for (const dir of generatedDirs) {
  if (existsSync(join(root, dir))) {
    failures.push(`Directorio generado presente: ${dir}`);
  }
}

for (const file of resultFiles) {
  const fullPath = join(root, file);
  if (!existsSync(fullPath)) continue;
  try {
    JSON.parse(readFileSync(fullPath, "utf8"));
  } catch (error) {
    failures.push(`JSON inválido en ${file}: ${error.message}`);
  }
}

const dashboardPath = join(root, "backend/storage/results/dashboard.json");
if (existsSync(dashboardPath)) {
  const data = JSON.parse(readFileSync(dashboardPath, "utf8"));
  const dataset = data.dataset;
  if (!dataset || typeof dataset.originalRows !== "number" || !Array.isArray(dataset.columns)) {
    failures.push("dashboard.json no contiene dataset.originalRows y dataset.columns con la estructura esperada.");
  }
}

if (failures.length) {
  console.error("Verificación fallida:");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log("Verificación del proyecto completada sin errores.");
