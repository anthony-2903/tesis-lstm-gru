import { useCallback, useEffect, useRef, useState } from "react";

const STATIC_DEPENDENCY = Symbol("static-api-loader");

export function useApiData<T>(loader: () => Promise<T>, dependency: unknown = STATIC_DEPENDENCY) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const loaderRef = useRef(loader);
  const requestRef = useRef(0);

  useEffect(() => {
    loaderRef.current = loader;
  }, [loader]);

  const load = useCallback(() => {
    // The dependency is an explicit reload key (for example, the active domain).
    void dependency;
    const requestId = ++requestRef.current;
    setIsLoading(true);
    setError(null);
    loaderRef.current()
      .then((value) => {
        if (requestRef.current === requestId) setData(value);
      })
      .catch((err) => {
        if (requestRef.current !== requestId) return;
        setError(err instanceof Error ? err.message : "No se pudo conectar con el backend.");
        setData(null);
      })
      .finally(() => {
        if (requestRef.current === requestId) setIsLoading(false);
      });
  }, [dependency]);

  useEffect(() => {
    load();
    return () => {
      requestRef.current += 1;
    };
  }, [load]);

  return { data, error, isLoading, reload: load };
}
