import { createFileRoute } from "@tanstack/react-router";
import { motion, useReducedMotion } from "framer-motion";
import {
  ArrowRight,
  BarChart3,
  Beaker,
  BrainCircuit,
  Check,
  ChevronRight,
  Database,
  FileCheck2,
  Fingerprint,
  FlaskConical,
  Gauge,
  GitCompareArrows,
  Landmark,
  Layers3,
  LineChart,
  LockKeyhole,
  Pause,
  Play,
  RotateCcw,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  TestTubeDiagonal,
  Waypoints,
  Workflow,
  Zap,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/metodologia")({
  head: () => ({
    meta: [
      { title: "Metodología — Pipeline reproducible de tesis" },
      {
        name: "description",
        content: "Recorrido animado desde la ingesta de datos hasta el sellado y la presentación de resultados.",
      },
    ],
  }),
  component: MethodologyPage,
});

type Tone = "blue" | "green" | "purple" | "teal";

interface RecipeStep {
  number: number;
  title: string;
  description: string;
  output: string;
  icon: LucideIcon;
  tone: Tone;
}

const PHASES: Array<{ title: string; subtitle: string; steps: RecipeStep[] }> = [
  {
    title: "Preparar los datos",
    subtitle: "De la fuente original a un conjunto auditable y sin fuga temporal.",
    steps: [
      {
        number: 1,
        title: "Ingesta",
        description: "Obtener datos desde fuentes verificadas y conservar su procedencia.",
        output: "Datos crudos",
        icon: Database,
        tone: "blue",
      },
      {
        number: 2,
        title: "Versionado",
        description: "Registrar fuente, fecha de descarga, versión y huella SHA-256.",
        output: "Linaje reproducible",
        icon: Fingerprint,
        tone: "blue",
      },
      {
        number: 3,
        title: "Limpieza",
        description: "Corregir tipos, duplicados, faltantes y valores inválidos sin alterar el objetivo.",
        output: "Dataset limpio",
        icon: FlaskConical,
        tone: "blue",
      },
      {
        number: 4,
        title: "Auditoría",
        description: "Validar calidad, trazabilidad, cobertura y el contrato científico del dominio.",
        output: "Puerta de calidad",
        icon: FileCheck2,
        tone: "blue",
      },
      {
        number: 5,
        title: "Partición temporal",
        description: "Usar 85% para desarrollo y reservar 15% como test final bloqueado.",
        output: "Test aislado",
        icon: LockKeyhole,
        tone: "blue",
      },
    ],
  },
  {
    title: "Entrenar y combinar",
    subtitle: "Todos los candidatos aprenden y compiten bajo exactamente el mismo protocolo.",
    steps: [
      {
        number: 6,
        title: "Secuencias",
        description: "Crear ventanas temporales usando únicamente información disponible en el pasado.",
        output: "Tensores de entrada",
        icon: Waypoints,
        tone: "green",
      },
      {
        number: 7,
        title: "Validación OOF",
        description: "Aplicar 5 folds temporales por 5 semillas y generar predicciones fuera de muestra.",
        output: "Matriz 5 × 5",
        icon: Layers3,
        tone: "green",
      },
      {
        number: 8,
        title: "Modelos base",
        description: "Entrenar LSTM, GRU, BiRNN, TCN y Transformer con checkpoints verificables.",
        output: "5 predictores",
        icon: BrainCircuit,
        tone: "green",
      },
      {
        number: 9,
        title: "Meta-features",
        description: "Reunir predicciones, probabilidades, residuos y desacuerdo entre modelos.",
        output: "Tabla para el meta-modelo",
        icon: Workflow,
        tone: "purple",
      },
      {
        number: 10,
        title: "Stacking",
        description: "Entrenar el meta-modelo solo con información OOF, evitando contaminación y fuga.",
        output: "Candidato Stacking",
        icon: Beaker,
        tone: "purple",
      },
    ],
  },
  {
    title: "Validar y servir",
    subtitle: "La evidencia decide el ganador; después se explican y sellan los resultados.",
    steps: [
      {
        number: 11,
        title: "Comparación justa",
        description: "Comparar modelos con los mismos folds, semillas, muestras y métrica primaria.",
        output: "Ranking comparable",
        icon: GitCompareArrows,
        tone: "teal",
      },
      {
        number: 12,
        title: "Revalidación",
        description: "Ejecutar ablación, bootstrap con IC 95% y una comparación independiente.",
        output: "Evidencia estadística",
        icon: TestTubeDiagonal,
        tone: "teal",
      },
      {
        number: 13,
        title: "XAI",
        description: "Explicar la contribución de cada modelo y las variables usadas por el Stacking.",
        output: "Importancias explicables",
        icon: ScanSearch,
        tone: "teal",
      },
      {
        number: 14,
        title: "Sellado",
        description: "Congelar configuración, hashes, métricas, modelos y artefactos anteriores al test.",
        output: "Evidencia inmutable",
        icon: ShieldCheck,
        tone: "teal",
      },
      {
        number: 15,
        title: "Dashboard",
        description: "Presentar tablas, gráficas, anomalías, explicaciones y conclusiones honestas.",
        output: "Resultado comunicable",
        icon: BarChart3,
        tone: "teal",
      },
    ],
  },
];

const ALL_STEPS = PHASES.flatMap((phase) => phase.steps);

const TONE_STYLES: Record<Tone, { card: string; badge: string; icon: string; glow: string }> = {
  blue: {
    card: "border-sky-500/25 bg-sky-500/[0.04]",
    badge: "bg-sky-500 text-white",
    icon: "bg-sky-500/10 text-sky-600 dark:text-sky-300",
    glow: "shadow-[0_0_0_2px_rgba(14,165,233,0.2),0_14px_35px_rgba(14,165,233,0.14)]",
  },
  green: {
    card: "border-emerald-500/25 bg-emerald-500/[0.04]",
    badge: "bg-emerald-500 text-white",
    icon: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-300",
    glow: "shadow-[0_0_0_2px_rgba(16,185,129,0.2),0_14px_35px_rgba(16,185,129,0.14)]",
  },
  purple: {
    card: "border-violet-500/25 bg-violet-500/[0.04]",
    badge: "bg-violet-500 text-white",
    icon: "bg-violet-500/10 text-violet-600 dark:text-violet-300",
    glow: "shadow-[0_0_0_2px_rgba(139,92,246,0.2),0_14px_35px_rgba(139,92,246,0.14)]",
  },
  teal: {
    card: "border-cyan-600/25 bg-cyan-600/[0.04]",
    badge: "bg-cyan-600 text-white",
    icon: "bg-cyan-600/10 text-cyan-700 dark:text-cyan-300",
    glow: "shadow-[0_0_0_2px_rgba(8,145,178,0.2),0_14px_35px_rgba(8,145,178,0.14)]",
  },
};

function MethodologyPage() {
  const reduceMotion = useReducedMotion();
  const [activeStep, setActiveStep] = useState(1);
  const [isPlaying, setIsPlaying] = useState(true);

  useEffect(() => {
    if (reduceMotion) setIsPlaying(false);
  }, [reduceMotion]);

  useEffect(() => {
    if (!isPlaying || reduceMotion) return;
    const timer = window.setInterval(() => {
      setActiveStep((current) => (current >= ALL_STEPS.length ? 1 : current + 1));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [isPlaying, reduceMotion]);

  const current = ALL_STEPS[activeStep - 1];
  const progress = ((activeStep - 1) / (ALL_STEPS.length - 1)) * 100;

  const restart = () => {
    setActiveStep(1);
    setIsPlaying(!reduceMotion);
  };

  return (
    <div className="dashboard-page space-y-6 pb-10">
      <header className="relative overflow-hidden rounded-xl border border-border bg-card p-5 shadow-sm sm:p-7">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(139,92,246,0.13),transparent_35%),radial-gradient(circle_at_bottom_left,rgba(14,165,233,0.10),transparent_40%)]" />
        <div className="relative flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
          <div className="max-w-3xl">
            <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-3 py-1 text-[11px] font-bold uppercase tracking-[0.16em] text-primary">
              <Sparkles className="h-3.5 w-3.5" /> Receta científica reproducible
            </div>
            <h1 className="text-3xl font-black tracking-tight text-foreground sm:text-4xl">
              Metodología del pipeline de tesis
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground sm:text-base">
              Sigue cada paquete de datos desde su fuente hasta el dashboard. El recorrido reproduce el orden real del proceso y mantiene el conjunto de prueba final fuera del entrenamiento.
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setIsPlaying((value) => !value)}
              disabled={Boolean(reduceMotion)}
              className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-xs font-bold text-primary-foreground shadow-sm transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              {isPlaying ? "Pausar recorrido" : "Reproducir recorrido"}
            </button>
            <button
              type="button"
              onClick={restart}
              className="inline-flex items-center gap-2 rounded-md border border-border bg-background px-4 py-2 text-xs font-bold text-foreground transition hover:bg-muted"
            >
              <RotateCcw className="h-4 w-4" /> Reiniciar
            </button>
          </div>
        </div>
      </header>

      <section className="grid gap-3 lg:grid-cols-[1fr_1.7fr]">
        <div className="rounded-xl border border-border bg-card p-5 shadow-sm">
          <div className="flex items-center gap-2">
            <FlaskConical className="h-5 w-5 text-primary" />
            <h2 className="text-sm font-black uppercase tracking-[0.12em] text-foreground">Ingredientes</h2>
          </div>
          <div className="mt-4 grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
            <DomainIngredient icon={ShieldCheck} title="Ciberseguridad" detail="URLs y señales de phishing" color="text-sky-600" />
            <DomainIngredient icon={Zap} title="Energía eléctrica" detail="Series temporales" color="text-emerald-600" />
            <DomainIngredient icon={Landmark} title="Finanzas" detail="Índices Soberanos del MEF" color="text-amber-600" />
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card p-5 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-muted-foreground">Unidad actual</p>
              <p className="mt-1 text-lg font-black text-foreground">
                Paso {activeStep}: {current.title}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">Salida: {current.output}</p>
            </div>
            <div className="rounded-lg border border-primary/20 bg-primary/5 px-3 py-2 text-right">
              <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Avance de la receta</p>
              <p className="text-xl font-black text-primary">{Math.round(((activeStep) / ALL_STEPS.length) * 100)}%</p>
            </div>
          </div>

          <div className="relative mt-7 h-3 rounded-full bg-muted">
            <motion.div
              className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-sky-500 via-violet-500 to-cyan-600"
              animate={{ width: `${progress}%` }}
              transition={{ duration: reduceMotion ? 0 : 0.55, ease: "easeInOut" }}
            />
            <motion.div
              className="absolute top-1/2 z-10 flex h-8 w-8 -translate-y-1/2 items-center justify-center rounded-full border-2 border-background bg-primary text-primary-foreground shadow-lg"
              animate={{ left: `calc(${progress}% - 1rem)` }}
              transition={{ duration: reduceMotion ? 0 : 0.55, ease: "easeInOut" }}
            >
              <Database className="h-4 w-4" />
            </motion.div>
          </div>
          <div className="mt-5 flex justify-between text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
            <span>Datos crudos</span>
            <span>Entrenamiento</span>
            <span>Evidencia</span>
            <span>Dashboard</span>
          </div>
        </div>
      </section>

      {PHASES.map((phase, phaseIndex) => (
        <motion.section
          key={phase.title}
          initial={reduceMotion ? false : { opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.15 }}
          transition={{ duration: 0.45, delay: phaseIndex * 0.08 }}
          className="rounded-xl border border-border bg-card p-4 shadow-sm sm:p-5"
        >
          <div className="mb-4 flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-[10px] font-black uppercase tracking-[0.18em] text-primary">Fase {phaseIndex + 1}</p>
              <h2 className="mt-1 text-xl font-black text-foreground">{phase.title}</h2>
            </div>
            <p className="max-w-xl text-xs leading-5 text-muted-foreground sm:text-right">{phase.subtitle}</p>
          </div>

          <div className="grid gap-3 lg:grid-cols-5">
            {phase.steps.map((step, index) => (
              <StepCard
                key={step.number}
                step={step}
                active={activeStep === step.number}
                completed={activeStep > step.number}
                showConnector={index < phase.steps.length - 1}
                reduceMotion={Boolean(reduceMotion)}
                onSelect={() => {
                  setActiveStep(step.number);
                  setIsPlaying(false);
                }}
              />
            ))}
          </div>
        </motion.section>
      ))}

      <section className="grid gap-4 xl:grid-cols-[1.25fr_0.9fr_1fr]">
        <div className="rounded-xl border border-border bg-card p-5 shadow-sm">
          <div className="flex items-center gap-2">
            <Gauge className="h-5 w-5 text-primary" />
            <h2 className="text-sm font-black text-foreground">Métricas de salida</h2>
          </div>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <MetricRecipe title="Phishing" detail="PR-AUC · ROC-AUC · F1 · MCC" icon={ShieldCheck} />
            <MetricRecipe title="Energía y Finanzas" detail="RMSE · MAE · R²" icon={LineChart} />
          </div>
        </div>

        <div className="relative overflow-hidden rounded-xl border border-red-500/30 bg-red-500/[0.05] p-5 shadow-sm">
          <motion.div
            animate={reduceMotion ? undefined : { y: [0, -4, 0] }}
            transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
            className="flex h-12 w-12 items-center justify-center rounded-xl bg-red-500/10 text-red-600"
          >
            <LockKeyhole className="h-6 w-6" />
          </motion.div>
          <h2 className="mt-4 text-sm font-black uppercase tracking-wide text-red-700 dark:text-red-300">Test final bloqueado</h2>
          <p className="mt-2 text-xs leading-5 text-muted-foreground">
            No participa en entrenamiento, selección, umbrales, Stacking ni XAI. Solo puede abrirse con autorización explícita.
          </p>
        </div>

        <div className="rounded-xl border border-amber-500/30 bg-amber-500/[0.05] p-5 shadow-sm">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-amber-500/10 text-amber-600">
            <Sparkles className="h-6 w-6" />
          </div>
          <h2 className="mt-4 text-sm font-black text-foreground">La evidencia decide</h2>
          <p className="mt-2 text-xs leading-5 text-muted-foreground">
            Stacking compite bajo las mismas reglas, pero no está forzado a ganar. El mejor resultado es el que respaldan las métricas y la prueba independiente.
          </p>
        </div>
      </section>
    </div>
  );
}

function DomainIngredient({ icon: Icon, title, detail, color }: { icon: LucideIcon; title: string; detail: string; color: string }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-border bg-background/70 p-3">
      <div className={cn("flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-muted", color)}>
        <Icon className="h-5 w-5" />
      </div>
      <div>
        <p className="text-xs font-black text-foreground">{title}</p>
        <p className="mt-0.5 text-[11px] text-muted-foreground">{detail}</p>
      </div>
    </div>
  );
}

function StepCard({
  step,
  active,
  completed,
  showConnector,
  reduceMotion,
  onSelect,
}: {
  step: RecipeStep;
  active: boolean;
  completed: boolean;
  showConnector: boolean;
  reduceMotion: boolean;
  onSelect: () => void;
}) {
  const styles = TONE_STYLES[step.tone];
  const Icon = step.icon;

  return (
    <div className="relative min-w-0">
      <motion.button
        type="button"
        onClick={onSelect}
        aria-current={active ? "step" : undefined}
        animate={active && !reduceMotion ? { y: [0, -5, 0] } : { y: 0 }}
        transition={active && !reduceMotion ? { duration: 1.4, repeat: Infinity, ease: "easeInOut" } : { duration: 0.2 }}
        className={cn(
          "relative flex h-full min-h-56 w-full flex-col overflow-hidden rounded-xl border p-4 text-left transition duration-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary",
          styles.card,
          active && styles.glow,
          !active && "hover:-translate-y-1 hover:border-primary/35 hover:shadow-md",
        )}
      >
        <div className="flex items-center justify-between gap-3">
          <span className={cn("flex h-8 min-w-8 items-center justify-center rounded-full text-xs font-black shadow-sm", styles.badge)}>
            {step.number}
          </span>
          <span className={cn("flex h-10 w-10 items-center justify-center rounded-lg", styles.icon)}>
            <Icon className="h-5 w-5" />
          </span>
        </div>
        <h3 className="mt-4 text-sm font-black uppercase tracking-wide text-foreground">{step.title}</h3>
        <p className="mt-2 flex-1 text-xs leading-5 text-muted-foreground">{step.description}</p>
        <div className="mt-4 flex items-center gap-2 border-t border-border/70 pt-3 text-[10px] font-bold uppercase tracking-wide text-foreground">
          {completed ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <ArrowRight className="h-3.5 w-3.5 text-primary" />}
          {step.output}
        </div>

        {active && (
          <motion.div
            initial={{ x: "-100%" }}
            animate={{ x: "310%" }}
            transition={{ duration: 1.15, repeat: Infinity, ease: "linear" }}
            className="absolute bottom-0 left-0 h-1 w-1/3 rounded-full bg-primary"
          />
        )}
      </motion.button>
      {showConnector && (
        <div className="pointer-events-none absolute -right-3 top-1/2 z-10 hidden -translate-y-1/2 items-center lg:flex">
          <ChevronRight className="h-5 w-5 text-muted-foreground/60" />
        </div>
      )}
    </div>
  );
}

function MetricRecipe({ title, detail, icon: Icon }: { title: string; detail: string; icon: LucideIcon }) {
  return (
    <div className="rounded-lg border border-border bg-background/70 p-3">
      <div className="flex items-center gap-2">
        <Icon className="h-4 w-4 text-primary" />
        <p className="text-xs font-black text-foreground">{title}</p>
      </div>
      <p className="mt-2 text-[11px] font-bold tracking-wide text-muted-foreground">{detail}</p>
    </div>
  );
}
