import { createFileRoute } from "@tanstack/react-router";
import { DataUpload } from "@/components/DataUpload";

export const Route = createFileRoute("/upload")({
  head: () => ({
    meta: [
      { title: "Carga y Limpieza de Datos" },
      { name: "description", content: "Sube tus datasets para limpieza profunda y entrenamiento" },
    ],
  }),
  component: UploadPage,
});

function UploadPage() {
  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Gestión de Datos</h1>
        <p className="text-muted-foreground mt-2">
          Sube tus archivos CSV o Excel. El sistema realizará una validación rápida, y el backend de Python aplicará una limpieza profunda y ordenamiento para preparar el entrenamiento de los modelos LSTM, GRU, Transformer y TCN.
        </p>
      </div>

      <DataUpload />
    </div>
  );
}
