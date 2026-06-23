import { motion } from "framer-motion";
import { AlertCircle, Database, Loader2, RefreshCw } from "lucide-react";

interface BackendStateProps {
  title?: string;
  message?: string;
  error?: string | null;
  isLoading?: boolean;
  onRetry?: () => void;
}

export function BackendState({
  title,
  message,
  error,
  isLoading = false,
  onRetry,
}: BackendStateProps) {
  const Icon = isLoading ? Loader2 : error ? AlertCircle : Database;

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex min-h-[60vh] flex-col items-center justify-center gap-5 text-center"
    >
      <div className="flex h-20 w-20 items-center justify-center rounded-full border border-primary/20 bg-primary/10 text-primary">
        <Icon className={`h-10 w-10 ${isLoading ? "animate-spin" : ""}`} />
      </div>
      <div>
        <h2 className="text-2xl font-bold text-foreground">
          {title || (isLoading ? "Conectando con backend local" : "Backend local no disponible")}
        </h2>
        <p className="mx-auto mt-2 max-w-xl text-sm text-muted-foreground">
          {message ||
            error ||
            "Levanta el servicio local en http://localhost:8000/api para visualizar los resultados procesados."}
        </p>
      </div>
      {onRetry && !isLoading && (
        <button
          type="button"
          onClick={onRetry}
          className="inline-flex items-center gap-2 rounded-md bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground shadow-sm transition-colors hover:bg-primary/90"
        >
          <RefreshCw className="h-4 w-4" />
          Reintentar conexion
        </button>
      )}
    </motion.div>
  );
}
