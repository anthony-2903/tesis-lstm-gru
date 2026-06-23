import { useEffect, useState } from "react";

export function useApiData<T>(loader: () => Promise<T>) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const load = () => {
    setIsLoading(true);
    setError(null);
    loader()
      .then(setData)
      .catch((err) => {
        setError(err instanceof Error ? err.message : "No se pudo conectar con el backend local.");
        setData(null);
      })
      .finally(() => setIsLoading(false));
  };

  useEffect(() => {
    load();
  }, []);

  return { data, error, isLoading, reload: load };
}
