/**
 * dataStore.ts
 * Almacén global (Zustand) que mantiene los datos cargados desde el archivo CSV/Excel.
 * Todas las rutas del dashboard leen desde aquí en lugar de hacer peticiones al backend.
 */
import { create } from "zustand";

export interface ParsedDataset {
  filename: string;
  originalRows: number;
  cleanedRows: number;
  rowsRemoved: number;
  columns: string[];
  dataTypes: Record<string, string>;
  sampleData: Record<string, unknown>[];
  /** Todos los datos parseados (sin truncar) */
  allData: Record<string, unknown>[];
}

interface DataStore {
  dataset: ParsedDataset | null;
  setDataset: (ds: ParsedDataset) => void;
  clearDataset: () => void;
}

export const useDataStore = create<DataStore>((set) => ({
  dataset: null,
  setDataset: (ds) => set({ dataset: ds }),
  clearDataset: () => set({ dataset: null }),
}));
