import { useState, useCallback } from "react";
import { Upload, FileType, CheckCircle2, AlertCircle, BarChart2, Trash2 } from "lucide-react";
import { toast } from "sonner";
import Papa from "papaparse";
import * as XLSX from "xlsx";
import { useDataStore, type ParsedDataset, type DataDomain } from "@/lib/dataStore";
import { detectDomainFull } from "@/lib/detector";
import { motion, AnimatePresence } from "framer-motion";

// ─────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────

function inferType(values: unknown[]): string {
  const nonNull = values.filter((v) => v !== null && v !== undefined && v !== "");
  if (nonNull.length === 0) return "vacío";
  const asNums = nonNull.map((v) => Number(v));
  if (asNums.every((n) => !isNaN(n))) return "número";
  const dateRe = /^\d{4}-\d{2}-\d{2}|^\d{2}\/\d{2}\/\d{4}/;
  if (nonNull.every((v) => dateRe.test(String(v)))) return "fecha";
  return "texto";
}

function cleanRows(rows: Record<string, unknown>[]): Record<string, unknown>[] {
  return rows.filter((row) =>
    Object.values(row).every((v) => v !== null && v !== undefined && v !== "")
  );
}

function buildDataTypes(rows: Record<string, unknown>[], columns: string[]): Record<string, string> {
  const types: Record<string, string> = {};
  for (const col of columns) {
    types[col] = inferType(rows.map((r) => r[col]));
  }
  return types;
}

async function parseFile(file: File): Promise<Record<string, unknown>[]> {
  if (file.name.endsWith(".csv")) {
    return new Promise((resolve, reject) => {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        dynamicTyping: true,
        complete: (res) => resolve(res.data as Record<string, unknown>[]),
        error: reject,
      });
    });
  }
  const buffer = await file.arrayBuffer();
  const wb = XLSX.read(buffer, { type: "array" });
  const ws = wb.Sheets[wb.SheetNames[0]];
  return XLSX.utils.sheet_to_json<Record<string, unknown>>(ws, { defval: "" });
}

// Configuración visual por dominio
const DOMAIN_CONFIG: Record<DataDomain, { label: string; emoji: string; color: string }> = {
  phishing:  { label: "Phishing / URLs",              emoji: "🛡️", color: "bg-red-500/10 text-red-400 border-red-500/30" },
  energia:   { label: "Energía / Series Temporales",  emoji: "⚡", color: "bg-yellow-500/10 text-yellow-400 border-yellow-500/30" },
  finanzas:  { label: "Finanzas / Fraude",            emoji: "🏦", color: "bg-green-500/10 text-green-400 border-green-500/30" },
  general:   { label: "Dataset General",              emoji: "📊", color: "bg-primary/10 text-primary border-primary/30" },
};

// ─────────────────────────────────────────────────────────
// Componente principal
// ─────────────────────────────────────────────────────────

export function DataUpload() {
  const { dataset, setDataset, clearDataset } = useDataStore();
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dragOver, setDragOver] = useState(false);

  const processFile = useCallback(async (selectedFile: File) => {
    setIsProcessing(true);
    setUploadProgress(0);

    const progressInterval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 92) { clearInterval(progressInterval); return 92; }
        return Math.min(prev + Math.floor(Math.random() * 15) + 5, 92);
      });
    }, 120);

    try {
      const allRows = await parseFile(selectedFile);

      if (allRows.length === 0) {
        clearInterval(progressInterval);
        toast.error("El archivo está vacío o no tiene un formato reconocible.");
        setIsProcessing(false);
        return;
      }

      const columns = Object.keys(allRows[0]);
      const cleaned = cleanRows(allRows);
      const dataTypes = buildDataTypes(cleaned, columns);
      const domain = detectDomainFull(selectedFile.name, columns, dataTypes, cleaned.slice(0, 20));

      const ds: ParsedDataset = {
        filename: selectedFile.name,
        originalRows: allRows.length,
        cleanedRows: cleaned.length,
        rowsRemoved: allRows.length - cleaned.length,
        columns,
        dataTypes,
        sampleData: cleaned.slice(0, 10),
        allData: cleaned,
        domain,
      };

      clearInterval(progressInterval);
      setUploadProgress(100);
      await new Promise((resolve) => setTimeout(resolve, 350));

      setDataset(ds);
      const cfg = DOMAIN_CONFIG[domain];
      toast.success(`✅ "${selectedFile.name}" — ${cfg.emoji} ${cfg.label} — ${cleaned.length.toLocaleString("es-ES")} filas limpias.`);
    } catch (err) {
      clearInterval(progressInterval);
      console.error(err);
      toast.error("Error al procesar el archivo. Verifica que sea CSV o Excel válido.");
    } finally {
      setIsProcessing(false);
      setUploadProgress(0);
    }
  }, [setDataset]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) { setFile(selected); processFile(selected); }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) { setFile(dropped); processFile(dropped); }
  };

  const handleClear = () => {
    setFile(null);
    clearDataset();
    toast.info("Dataset eliminado del dashboard.");
  };

  return (
    <div className="space-y-6">

      {/* ── Zona de carga ── */}
      <div className="bg-card border rounded-xl p-6 shadow-sm">
        <h2 className="text-xl font-bold flex items-center gap-2 mb-4">
          <BarChart2 className="w-5 h-5 text-primary" />
          Carga de Dataset
        </h2>

        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-lg p-10 flex flex-col items-center justify-center transition-colors relative cursor-pointer
            ${dragOver ? "border-primary bg-primary/5" : "border-border bg-muted/20 hover:bg-muted/40"}
            ${isProcessing ? "pointer-events-none opacity-80" : ""}`}
        >
          <input
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleFileChange}
            disabled={isProcessing}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />

          <Upload className={`w-10 h-10 mb-3 transition-transform ${isProcessing ? "animate-pulse scale-110 text-primary" : "text-muted-foreground"}`} />

          <p className="text-sm font-medium text-muted-foreground">
            {isProcessing ? "Leyendo y procesando archivo…" : "Arrastra tu archivo aquí o haz clic para seleccionarlo"}
          </p>
          <p className="text-xs text-muted-foreground mt-1">CSV · XLSX · XLS — sin límite de tamaño</p>

          {/* Barra de progreso animada */}
          <AnimatePresence>
            {isProcessing && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="w-full max-w-sm mt-6 space-y-2"
              >
                <div className="flex justify-between items-center text-xs font-bold text-muted-foreground px-0.5">
                  <span>Analizando y limpiando dataset...</span>
                  <span className="text-primary tabular-nums">{uploadProgress}%</span>
                </div>
                <div className="w-full bg-border rounded-full h-2.5 overflow-hidden">
                  <motion.div
                    className="bg-primary h-full rounded-full"
                    initial={{ width: "0%" }}
                    animate={{ width: `${uploadProgress}%` }}
                    transition={{ ease: "easeOut", duration: 0.12 }}
                  />
                </div>
                <p className="text-[10px] text-muted-foreground text-center">
                  Detectando columnas · Inferiendo tipos · Eliminando nulos
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Archivo seleccionado */}
        {file && !isProcessing && (
          <div className="mt-4 flex items-center justify-between bg-muted/30 p-3 rounded-md border">
            <div className="flex items-center gap-3">
              <FileType className="w-7 h-7 text-blue-500 flex-shrink-0" />
              <div>
                <p className="text-sm font-semibold truncate max-w-[260px]">{file.name}</p>
                <p className="text-xs text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
            </div>
            <button
              onClick={handleClear}
              title="Eliminar dataset"
              className="text-destructive hover:bg-destructive/10 p-1.5 rounded-md transition-colors"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

      {/* ── Resultados de la carga ── */}
      {dataset && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-1 md:grid-cols-2 gap-6"
        >
          {/* Estadísticas */}
          <div className="bg-card border rounded-xl p-6 shadow-sm">
            <h3 className="text-lg font-bold flex items-center gap-2 mb-4">
              <CheckCircle2 className="w-5 h-5 text-green-500" />
              Estadísticas del Dataset
            </h3>

            <div className="space-y-3">
              {[
                { label: "Archivo",                    value: dataset.filename },
                { label: "Filas originales",           value: dataset.originalRows.toLocaleString("es-ES") },
                { label: "Filas limpias",              value: dataset.cleanedRows.toLocaleString("es-ES"), green: true },
                { label: "Filas con nulos (removidas)",value: dataset.rowsRemoved.toLocaleString("es-ES"), red: true },
                { label: "Columnas",                   value: dataset.columns.length },
              ].map(({ label, value, green, red }) => (
                <div key={label} className="flex justify-between items-center border-b pb-2 last:border-0 last:pb-0">
                  <span className="text-sm text-muted-foreground">{label}</span>
                  <span className={`font-mono text-sm font-semibold ${green ? "text-green-500" : red ? "text-destructive" : ""}`}>
                    {String(value)}
                  </span>
                </div>
              ))}
            </div>

            {/* Badge de dominio detectado */}
            {(() => {
              const cfg = DOMAIN_CONFIG[dataset.domain] ?? DOMAIN_CONFIG.general;
              return (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.2 }}
                  className={`mt-5 flex items-center gap-2 px-4 py-3 rounded-xl border text-sm font-bold tracking-wide ${cfg.color}`}
                >
                  <span className="text-base">{cfg.emoji}</span>
                  <div>
                    <p className="text-[10px] uppercase tracking-widest opacity-70 font-bold">Dominio detectado</p>
                    <p>{cfg.label}</p>
                  </div>
                </motion.div>
              );
            })()}
          </div>

          {/* Tipos de dato */}
          <div className="bg-card border rounded-xl p-6 shadow-sm">
            <h3 className="text-lg font-bold flex items-center gap-2 mb-4">
              <AlertCircle className="w-5 h-5 text-primary" />
              Estructura de Columnas
            </h3>
            <div className="max-h-64 overflow-y-auto pr-1 space-y-1">
              {Object.entries(dataset.dataTypes).map(([col, type]) => (
                <div key={col} className="flex justify-between items-center bg-muted/30 px-3 py-1.5 rounded text-xs">
                  <span className="truncate max-w-[180px] font-medium" title={col}>{col}</span>
                  <span className={`font-mono px-2 py-0.5 rounded border text-[10px]
                    ${type === "número"
                      ? "bg-blue-50 border-blue-200 text-blue-700 dark:bg-blue-900/20 dark:border-blue-800 dark:text-blue-300"
                      : type === "fecha"
                      ? "bg-purple-50 border-purple-200 text-purple-700 dark:bg-purple-900/20 dark:border-purple-800 dark:text-purple-300"
                      : "bg-muted border-border text-muted-foreground"}`}>
                    {type}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Muestra de datos */}
          <div className="bg-card border rounded-xl p-6 shadow-sm md:col-span-2 overflow-auto">
            <h3 className="text-lg font-bold mb-4">Vista Previa (primeras 10 filas)</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr>
                    {dataset.columns.map((col) => (
                      <th key={col} className="border border-border bg-muted px-3 py-1.5 text-left font-semibold text-muted-foreground whitespace-nowrap">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dataset.sampleData.map((row, i) => (
                    <tr key={i} className="hover:bg-muted/30 transition-colors">
                      {dataset.columns.map((col) => (
                        <td key={col} className="border border-border px-3 py-1.5 whitespace-nowrap max-w-[200px] truncate" title={String(row[col] ?? "")}>
                          {String(row[col] ?? "—")}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
