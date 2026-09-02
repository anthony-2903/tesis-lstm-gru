import { createFileRoute } from "@tanstack/react-router";
import { motion, useReducedMotion } from "framer-motion";
import {
  ArrowRight,
  BarChart3,
  Beaker,
  Boxes,
  BrainCircuit,
  Cable,
  Check,
  ChevronRight,
  Cog,
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
  MonitorCog,
  Pause,
  Play,
  Radio,
  RotateCcw,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  ServerCog,
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

const PIPELINE_BLUEPRINT = new URL("../../docs/images/metodologia-pipeline-tesis.png", import.meta.url).href;

interface MethodStep {
  number: number;
  title: string;
  description: string;
  output: string;
  icon: LucideIcon;
  tone: Tone;
}

const PHASES: Array<{ title: string; subtitle: string; steps: MethodStep[] }> = [
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
              <Sparkles className="h-3.5 w-3.5" /> Protocolo científico reproducible
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

      <AnimatedPipeline
        activeStep={activeStep}
        isPlaying={isPlaying}
        reduceMotion={Boolean(reduceMotion)}
      />

      <section className="grid gap-3 lg:grid-cols-[1fr_1.7fr]">
        <div className="rounded-xl border border-border bg-card p-5 shadow-sm">
          <div className="flex items-center gap-2">
            <FlaskConical className="h-5 w-5 text-primary" />
            <h2 className="text-sm font-black uppercase tracking-[0.12em] text-foreground">Dominios de aplicación</h2>
          </div>
          <div className="mt-4 grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
            <DomainSource icon={ShieldCheck} title="Ciberseguridad" detail="URLs y señales de phishing" color="text-sky-600" />
            <DomainSource icon={Zap} title="Energía eléctrica" detail="Series temporales" color="text-emerald-600" />
            <DomainSource icon={Landmark} title="Finanzas" detail="Índices Soberanos del MEF" color="text-amber-600" />
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
              <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Avance metodológico</p>
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
            <MetricSummary title="Phishing" detail="PR-AUC · ROC-AUC · F1 · MCC" icon={ShieldCheck} />
            <MetricSummary title="Energía y Finanzas" detail="RMSE · MAE · R²" icon={LineChart} />
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

const PIPELINE_PROCESSORS = [
  { label: "Ingesta", detail: "Recepción", icon: Database, steps: [1, 2] },
  { label: "Limpieza", detail: "Filtrado", icon: FlaskConical, steps: [3, 4] },
  { label: "Partición", detail: "85 / 15", icon: LockKeyhole, steps: [5] },
  { label: "Secuencias", detail: "Ventanas", icon: Waypoints, steps: [6, 7] },
];

const BASE_MODELS = ["LSTM", "GRU", "BiRNN", "TCN", "Transformer"];

function AnimatedPipeline({
  activeStep,
  isPlaying,
  reduceMotion,
}: {
  activeStep: number;
  isPlaying: boolean;
  reduceMotion: boolean;
}) {
  const running = isPlaying && !reduceMotion;
  const [blueprintReady, setBlueprintReady] = useState(false);
  useEffect(() => setBlueprintReady(true), []);
  const pipelineStage =
    activeStep <= 2 ? "receiving" :
      activeStep <= 5 ? "cleaning" :
        activeStep <= 7 ? "sequencing" :
          activeStep === 8 ? "models" :
            activeStep <= 10 ? "stacking" :
              activeStep <= 14 ? "validation" : "dashboard";

  return (
    <>
      <section className="relative overflow-hidden rounded-2xl border border-slate-700 bg-slate-950 text-slate-100 shadow-[0_24px_80px_rgba(2,6,23,0.28)]">
        <div className="pointer-events-none absolute inset-0 opacity-35 [background-image:linear-gradient(rgba(56,189,248,0.13)_1px,transparent_1px),linear-gradient(90deg,rgba(56,189,248,0.13)_1px,transparent_1px)] [background-size:28px_28px]" />
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_48%_32%,rgba(14,165,233,0.18),transparent_34%),radial-gradient(circle_at_75%_75%,rgba(139,92,246,0.16),transparent_30%)]" />

        <div className="relative flex flex-col gap-3 border-b border-slate-700/80 bg-slate-900/85 px-4 py-4 backdrop-blur sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <div className="flex items-center gap-3">
            <motion.div
              animate={running ? { scale: [1, 1.08, 1], boxShadow: ["0 0 0 rgba(34,211,238,0)", "0 0 28px rgba(34,211,238,.45)", "0 0 0 rgba(34,211,238,0)"] } : undefined}
              transition={{ duration: 2, repeat: Infinity }}
              className="flex h-11 w-11 items-center justify-center rounded-xl border border-cyan-400/30 bg-cyan-400/10 text-cyan-300"
            >
              <Workflow className="h-6 w-6" />
            </motion.div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-black uppercase tracking-[0.16em] text-white">Flujo metodológico animado</h2>
                <span className="relative flex h-2.5 w-2.5">
                  {running && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-70" />}
                  <span className={cn("relative inline-flex h-2.5 w-2.5 rounded-full", running ? "bg-emerald-400" : "bg-amber-400")} />
                </span>
              </div>
              <p className="mt-1 text-xs text-slate-400">Representación explicativa de cómo avanzan los datos entre las etapas.</p>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2 text-center">
            <Telemetry label="Animación" value={running ? "ACTIVA" : "PAUSADA"} active={running} />
            <Telemetry label="Unidad" value={`PASO ${String(activeStep).padStart(2, "0")}`} />
            <Telemetry label="Test final" value="AISLADO" danger />
          </div>
        </div>

        <div className="relative p-4 sm:p-6">
          <div className="pointer-events-none absolute left-8 right-8 top-[48%] hidden h-1 overflow-hidden rounded-full bg-slate-700 xl:block">
            <motion.div
              className="h-full w-1/4 bg-gradient-to-r from-transparent via-cyan-300 to-transparent"
              animate={running ? { x: ["-100%", "500%"] } : { x: "170%" }}
              transition={{ duration: 3.2, repeat: Infinity, ease: "linear" }}
            />
          </div>

          {Array.from({ length: 7 }, (_, index) => (
            <motion.span
              key={`pipeline-packet-${index}`}
              className="pointer-events-none absolute top-[47.15%] z-20 hidden h-3 w-5 rounded-sm border border-cyan-200/80 bg-cyan-300 shadow-[0_0_14px_rgba(34,211,238,.9)] xl:block"
              initial={{ left: "2%", opacity: 0 }}
              animate={running ? { left: ["2%", "96%"], opacity: [0, 1, 1, 0] } : { left: `${10 + index * 12}%`, opacity: 0.45 }}
              transition={{ duration: 9, repeat: Infinity, delay: index * 1.15, ease: "linear" }}
            />
          ))}

          <div className="relative z-10 grid gap-5 xl:grid-cols-[220px_minmax(0,1fr)_220px]">
            <div className="space-y-3">
              <StationLabel icon={Boxes} title="Fuentes de datos" subtitle="Entradas verificadas" />
              <SourceHopper
                icon={ShieldCheck}
                title="Ciberseguridad"
                detail="URLs y señales"
                color="cyan"
                running={running}
                active={pipelineStage === "receiving"}
              />
              <SourceHopper
                icon={Zap}
                title="Energía"
                detail="Lecturas horarias"
                color="emerald"
                running={running}
                active={pipelineStage === "receiving"}
              />
              <SourceHopper
                icon={Landmark}
                title="Finanzas"
                detail="Índices del MEF"
                color="amber"
                running={running}
                active={pipelineStage === "receiving"}
              />
              <div className="rounded-xl border border-dashed border-slate-600 bg-slate-900/60 px-3 py-2 text-[10px] leading-4 text-slate-400">
                Cada fuente conserva fecha, versión, procedencia y huella digital.
              </div>
            </div>

            <div className="relative overflow-hidden rounded-2xl border border-slate-600/80 bg-slate-900/75 p-4 shadow-inner sm:p-5">
              <div className="pointer-events-none absolute inset-x-0 top-0 h-16 bg-gradient-to-b from-cyan-400/10 to-transparent" />
              <div className="relative flex items-center justify-between gap-3">
                <StationLabel icon={ServerCog} title="Núcleo de procesamiento" subtitle={`Etapa ilustrada: ${pipelineStage}`} />
                <div className="flex items-center gap-1.5">
                  {[0, 1, 2].map((item) => (
                    <motion.span
                      key={item}
                      animate={running ? { opacity: [0.25, 1, 0.25] } : undefined}
                      transition={{ duration: 1.2, repeat: Infinity, delay: item * 0.25 }}
                      className="h-2 w-2 rounded-full bg-cyan-300"
                    />
                  ))}
                </div>
              </div>

              <div className="relative mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                {PIPELINE_PROCESSORS.map((processor) => (
                  <ProcessorMachine
                    key={processor.label}
                    {...processor}
                    active={processor.steps.includes(activeStep)}
                    running={running}
                  />
                ))}
              </div>

              <Conveyor running={running} />

              <div className="mt-4 rounded-2xl border border-violet-400/25 bg-violet-400/[0.06] p-4">
                <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex items-center gap-2">
                    <BrainCircuit className="h-5 w-5 text-violet-300" />
                    <div>
                      <p className="text-xs font-black uppercase tracking-wider text-white">Sala de redes neuronales</p>
                      <p className="text-[10px] text-slate-400">Cinco líneas trabajan en paralelo con los mismos folds.</p>
                    </div>
                  </div>
                  <span className="rounded-full border border-violet-300/20 bg-violet-300/10 px-2 py-1 text-[9px] font-black text-violet-200">5 × 5 × 5</span>
                </div>

                <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-5">
                  {BASE_MODELS.map((model, index) => (
                    <ModelChamber
                      key={model}
                      model={model}
                      index={index}
                      running={running}
                      active={pipelineStage === "models" || pipelineStage === "stacking"}
                    />
                  ))}
                </div>

                <div className="relative mt-4 flex items-center justify-center">
                  <div className="absolute left-[10%] right-[10%] top-1/2 h-px bg-violet-300/25" />
                  {Array.from({ length: 5 }, (_, index) => (
                    <motion.span
                      key={`model-signal-${index}`}
                      className="absolute left-[10%] h-2 w-2 rounded-full bg-violet-300 shadow-[0_0_10px_rgba(196,181,253,.9)]"
                      animate={running ? { left: ["10%", "88%"], opacity: [0, 1, 0] } : undefined}
                      transition={{ duration: 2.8, repeat: Infinity, delay: index * 0.45, ease: "linear" }}
                    />
                  ))}
                  <motion.div
                    animate={running && pipelineStage === "stacking" ? { scale: [1, 1.07, 1] } : undefined}
                    transition={{ duration: 1.4, repeat: Infinity }}
                    className={cn(
                      "relative z-10 flex min-w-44 items-center justify-center gap-3 rounded-xl border px-4 py-3",
                      pipelineStage === "stacking"
                        ? "border-fuchsia-300/60 bg-fuchsia-400/20 shadow-[0_0_28px_rgba(217,70,239,.3)]"
                        : "border-violet-300/25 bg-slate-900",
                    )}
                  >
                    <motion.div animate={running ? { rotate: 360 } : undefined} transition={{ duration: 7, repeat: Infinity, ease: "linear" }}>
                      <Cog className="h-6 w-6 text-fuchsia-300" />
                    </motion.div>
                    <div>
                      <p className="text-xs font-black text-white">META-MODELO STACKING</p>
                      <p className="text-[9px] text-violet-200">Integra meta-features OOF</p>
                    </div>
                  </motion.div>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <StationLabel icon={MonitorCog} title="Resultados" subtitle="Evaluación y evidencia" />
              <OutputMonitor
                icon={GitCompareArrows}
                title="Comparación"
                value="Ranking justo"
                active={pipelineStage === "validation"}
                running={running}
              />
              <OutputMonitor
                icon={ScanSearch}
                title="Explicabilidad"
                value="XAI + residuos"
                active={pipelineStage === "validation"}
                running={running}
              />
              <OutputMonitor
                icon={BarChart3}
                title="Dashboard"
                value="Tablas y gráficas"
                active={pipelineStage === "dashboard"}
                running={running}
              />

              <div className="relative overflow-hidden rounded-xl border border-red-400/35 bg-red-950/40 p-3">
                <div className="absolute inset-y-0 left-0 w-1 bg-red-400" />
                <div className="flex items-center gap-3">
                  <motion.div animate={running ? { y: [0, -3, 0] } : undefined} transition={{ duration: 2, repeat: Infinity }}>
                    <LockKeyhole className="h-6 w-6 text-red-300" />
                  </motion.div>
                  <div>
                    <p className="text-[10px] font-black uppercase tracking-wider text-red-200">Bóveda del test final</p>
                    <p className="mt-1 text-[10px] text-red-100/65">Sin conexión con la línea de entrenamiento.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="relative z-10 mt-5 flex flex-col gap-3 rounded-xl border border-slate-700 bg-slate-900/75 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider text-slate-300">
              <Radio className={cn("h-4 w-4", running ? "text-emerald-300" : "text-amber-300")} />
              Etapa ilustrada: {ALL_STEPS[activeStep - 1].title} → {ALL_STEPS[activeStep - 1].output}
            </div>
            <div className="flex flex-wrap gap-2 text-[9px] font-bold text-slate-400">
              <span className="rounded bg-slate-800 px-2 py-1">SIN FUGA TEMPORAL</span>
              <span className="rounded bg-slate-800 px-2 py-1">CHECKPOINTS ACTIVOS</span>
              <span className="rounded bg-slate-800 px-2 py-1">TRAZABILIDAD SHA-256</span>
            </div>
          </div>
        </div>
      </section>

      <section className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
        <div className="flex flex-col gap-2 border-b border-border px-4 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <div>
            <p className="text-[10px] font-black uppercase tracking-[0.18em] text-primary">Plano general</p>
            <h2 className="mt-1 text-lg font-black text-foreground">Secuencia metodológica completa</h2>
          </div>
          <p className="max-w-xl text-xs leading-5 text-muted-foreground sm:text-right">
            La animación superior explica el recorrido conceptual de los datos; no representa una ejecución ni telemetría real del backend.
          </p>
        </div>
        <div className="relative bg-[#f7f1e5] p-2 sm:p-4">
          {blueprintReady ? (
            <img
              src={PIPELINE_BLUEPRINT}
              alt="Infografía detallada de los quince pasos del pipeline de tesis"
              className="h-auto w-full rounded-xl border border-amber-900/15 shadow-[0_18px_45px_rgba(71,48,18,.16)]"
            />
          ) : (
            <div className="aspect-[16/10] w-full animate-pulse rounded-xl bg-amber-100" aria-label="Cargando plano visual" />
          )}
          <motion.div
            animate={running ? { x: ["-120%", "520%"] } : undefined}
            transition={{ duration: 7, repeat: Infinity, ease: "linear" }}
            className="pointer-events-none absolute inset-y-4 left-0 w-1/5 bg-gradient-to-r from-transparent via-white/35 to-transparent blur-sm"
          />
        </div>
      </section>
    </>
  );
}

function Telemetry({ label, value, active = false, danger = false }: { label: string; value: string; active?: boolean; danger?: boolean }) {
  return (
    <div className="min-w-20 rounded-lg border border-slate-700 bg-slate-950/75 px-2 py-1.5">
      <p className="text-[8px] font-bold uppercase tracking-wider text-slate-500">{label}</p>
      <p className={cn("mt-0.5 font-data text-[10px] font-black", danger ? "text-red-300" : active ? "text-emerald-300" : "text-cyan-200")}>{value}</p>
    </div>
  );
}

function StationLabel({ icon: Icon, title, subtitle }: { icon: LucideIcon; title: string; subtitle: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-slate-600 bg-slate-800 text-cyan-300">
        <Icon className="h-4 w-4" />
      </div>
      <div>
        <p className="text-[10px] font-black uppercase tracking-wider text-white">{title}</p>
        <p className="text-[9px] text-slate-400">{subtitle}</p>
      </div>
    </div>
  );
}

function SourceHopper({
  icon: Icon,
  title,
  detail,
  color,
  running,
  active,
}: {
  icon: LucideIcon;
  title: string;
  detail: string;
  color: "cyan" | "emerald" | "amber";
  running: boolean;
  active: boolean;
}) {
  const tones = {
    cyan: "border-cyan-400/35 bg-cyan-400/10 text-cyan-300",
    emerald: "border-emerald-400/35 bg-emerald-400/10 text-emerald-300",
    amber: "border-amber-400/35 bg-amber-400/10 text-amber-300",
  };
  return (
    <motion.div
      animate={active && running ? { x: [0, 3, 0] } : undefined}
      transition={{ duration: 1.6, repeat: Infinity }}
      className={cn("relative overflow-hidden rounded-xl border p-3", tones[color], active && "shadow-[0_0_22px_rgba(34,211,238,.16)]")}
    >
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-slate-950/45">
          <Icon className="h-5 w-5" />
        </div>
        <div>
          <p className="text-xs font-black text-white">{title}</p>
          <p className="mt-0.5 text-[10px] opacity-75">{detail}</p>
        </div>
      </div>
      <div className="mt-3 flex gap-1">
        {Array.from({ length: 8 }, (_, index) => (
          <motion.span
            key={index}
            animate={running ? { opacity: [0.2, 1, 0.2], y: [0, -2, 0] } : undefined}
            transition={{ duration: 1.4, repeat: Infinity, delay: index * 0.12 }}
            className="h-1 flex-1 rounded-full bg-current"
          />
        ))}
      </div>
    </motion.div>
  );
}

function ProcessorMachine({
  label,
  detail,
  icon: Icon,
  active,
  running,
}: {
  label: string;
  detail: string;
  icon: LucideIcon;
  steps: number[];
  active: boolean;
  running: boolean;
}) {
  return (
    <motion.div
      animate={active && running ? { y: [0, -4, 0] } : undefined}
      transition={{ duration: 1.35, repeat: Infinity }}
      className={cn(
        "relative overflow-hidden rounded-xl border bg-slate-950/70 p-3",
        active ? "border-cyan-300/70 shadow-[0_0_24px_rgba(34,211,238,.2)]" : "border-slate-700",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <Icon className={cn("h-5 w-5", active ? "text-cyan-300" : "text-slate-500")} />
        <motion.div animate={running ? { rotate: 360 } : undefined} transition={{ duration: 8, repeat: Infinity, ease: "linear" }}>
          <Cog className="h-4 w-4 text-slate-600" />
        </motion.div>
      </div>
      <p className="mt-3 text-[10px] font-black uppercase tracking-wider text-white">{label}</p>
      <p className="mt-1 text-[9px] text-slate-500">{detail}</p>
      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-slate-800">
        <motion.div
          className="h-full w-2/5 rounded-full bg-cyan-300"
          animate={running ? { x: ["-100%", "280%"] } : undefined}
          transition={{ duration: 2.1, repeat: Infinity, ease: "linear" }}
        />
      </div>
    </motion.div>
  );
}

function Conveyor({ running }: { running: boolean }) {
  return (
    <div className="relative mt-4 h-9 overflow-hidden rounded-lg border border-slate-700 bg-slate-950">
      <div className="absolute inset-x-0 top-2 h-3 border-y border-slate-600 bg-slate-800" />
      <div className="absolute inset-x-2 bottom-1 flex justify-between">
        {Array.from({ length: 14 }, (_, index) => (
          <motion.span
            key={index}
            animate={running ? { rotate: 360 } : undefined}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            className="h-2.5 w-2.5 rounded-full border border-slate-500 bg-slate-700"
          />
        ))}
      </div>
      {Array.from({ length: 5 }, (_, index) => (
        <motion.span
          key={index}
          className="absolute top-1.5 h-4 w-6 rounded border border-emerald-200/70 bg-emerald-300/90 shadow-[0_0_10px_rgba(110,231,183,.65)]"
          animate={running ? { left: ["-8%", "104%"] } : { left: `${12 + index * 18}%` }}
          transition={{ duration: 6, repeat: Infinity, delay: index * 1.05, ease: "linear" }}
        />
      ))}
    </div>
  );
}

function ModelChamber({ model, index, running, active }: { model: string; index: number; running: boolean; active: boolean }) {
  return (
    <div className={cn("relative overflow-hidden rounded-xl border bg-slate-950/70 px-2 py-3 text-center", active ? "border-violet-300/55" : "border-slate-700")}>
      <div className="relative mx-auto flex h-9 w-9 items-center justify-center rounded-full border border-violet-300/30 bg-violet-300/10">
        <BrainCircuit className="h-4 w-4 text-violet-200" />
        <motion.span
          animate={running ? { scale: [0.8, 1.45], opacity: [0.7, 0] } : undefined}
          transition={{ duration: 1.8, repeat: Infinity, delay: index * 0.2 }}
          className="absolute inset-0 rounded-full border border-violet-300"
        />
      </div>
      <p className="mt-2 truncate text-[9px] font-black text-white">{model}</p>
      <div className="mt-2 flex h-4 items-end justify-center gap-0.5">
        {[35, 75, 50, 90, 60].map((height, barIndex) => (
          <motion.span
            key={barIndex}
            animate={running ? { height: [`${Math.max(18, height - 25)}%`, `${height}%`, `${Math.max(18, height - 25)}%`] } : undefined}
            transition={{ duration: 1.1, repeat: Infinity, delay: index * 0.12 + barIndex * 0.08 }}
            className="w-1 rounded-t bg-violet-300/80"
            style={{ height: `${height}%` }}
          />
        ))}
      </div>
    </div>
  );
}

function OutputMonitor({
  icon: Icon,
  title,
  value,
  active,
  running,
}: {
  icon: LucideIcon;
  title: string;
  value: string;
  active: boolean;
  running: boolean;
}) {
  return (
    <div className={cn("relative overflow-hidden rounded-xl border bg-slate-900/80 p-3", active ? "border-emerald-300/60 shadow-[0_0_22px_rgba(110,231,183,.14)]" : "border-slate-700")}>
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-slate-950 text-emerald-300">
          <Icon className="h-4 w-4" />
        </div>
        <div>
          <p className="text-[10px] font-black uppercase tracking-wider text-white">{title}</p>
          <p className="mt-0.5 text-[9px] text-slate-400">{value}</p>
        </div>
      </div>
      <div className="mt-3 flex gap-1">
        {[45, 70, 38, 82, 62, 92].map((height, index) => (
          <motion.span
            key={index}
            animate={running ? { opacity: [0.25, 1, 0.25] } : undefined}
            transition={{ duration: 1.5, repeat: Infinity, delay: index * 0.14 }}
            className="h-1.5 flex-1 rounded bg-emerald-300"
            style={{ opacity: height / 100 }}
          />
        ))}
      </div>
    </div>
  );
}

function DomainSource({ icon: Icon, title, detail, color }: { icon: LucideIcon; title: string; detail: string; color: string }) {
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
  step: MethodStep;
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

function MetricSummary({ title, detail, icon: Icon }: { title: string; detail: string; icon: LucideIcon }) {
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
