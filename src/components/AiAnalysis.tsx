import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Sparkles, Send, Loader2, ChevronRight, CheckCircle2 } from "lucide-react";
import { generateAiAnalysis } from "@/lib/ai-analyst";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

interface AiAnalysisProps {
  type: "general" | "phishtank" | "opsd";
}

export function AiAnalysis({ type }: AiAnalysisProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [displayedText, setDisplayedText] = useState("");
  const [index, setIndex] = useState(0);

  const startAnalysis = async () => {
    setIsAnalyzing(true);
    setAnalysis(null);
    setDisplayedText("");
    setIndex(0);
    
    try {
      const result = await generateAiAnalysis(type);
      setAnalysis(result);
    } catch (error) {
      console.error("Error generating analysis", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  useEffect(() => {
    if (analysis && index < analysis.length) {
      const timeout = setTimeout(() => {
        setDisplayedText((prev) => prev + analysis[index]);
        setIndex((prev) => prev + 1);
      }, 5); // Typing speed
      return () => clearTimeout(timeout);
    }
  }, [analysis, index]);

  // Simple Markdown-to-JSX parser for basics used in ai-analyst.ts
  const formatText = (text: string) => {
    return text.split("\n").map((line, i) => {
      if (line.startsWith("###")) {
        return (
          <h3 key={i} className="text-lg font-bold text-primary mt-6 mb-2 flex items-center gap-2">
            <Sparkles className="h-4 w-4" />
            {line.replace("###", "").trim()}
          </h3>
        );
      }
      if (line.startsWith("-")) {
        return (
          <li key={i} className="ml-4 text-sm text-muted-foreground list-none flex gap-2 items-start mt-1">
            <ChevronRight className="h-4 w-4 mt-0.5 text-primary/50 shrink-0" />
            <span>{parseBold(line.replace("-", "").trim())}</span>
          </li>
        );
      }
      if (line.match(/^\d+\./)) {
        return (
          <div key={i} className="ml-4 text-sm text-foreground flex gap-2 items-start mt-2 font-medium">
            <span className="text-primary">{line.split(".")[0]}.</span>
            <span>{parseBold(line.split(".").slice(1).join(".").trim())}</span>
          </div>
        );
      }
      if (line.trim() === "") return <div key={i} className="h-2" />;
      return <p key={i} className="text-sm text-foreground leading-relaxed mt-1">{parseBold(line)}</p>;
    });
  };

  const parseBold = (text: string) => {
    const parts = text.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={i} className="text-primary font-semibold">{part.slice(2, -2)}</strong>;
      }
      return part;
    });
  };

  return (
    <Card className="relative overflow-hidden border-primary/20 bg-card/50 backdrop-blur-xl shadow-2xl">
      <div className="absolute top-0 right-0 p-8 opacity-5">
        <Brain className="h-32 w-32 text-primary" />
      </div>
      
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <CardTitle className="text-xl flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-primary/10 text-primary">
                <Brain className="h-5 w-5" />
              </div>
              Asistente de Análisis IA
            </CardTitle>
            <CardDescription>
              Interpretación experta de resultados del modelo {type === 'general' ? 'completo' : type}
            </CardDescription>
          </div>
          {!analysis && !isAnalyzing && (
            <Button 
              onClick={startAnalysis}
              className="bg-primary hover:bg-primary/90 text-primary-foreground shadow-[0_0_20px_rgba(var(--primary),0.3)] transition-all hover:scale-105"
            >
              <Send className="mr-2 h-4 w-4" />
              Generar Análisis
            </Button>
          )}
        </div>
      </CardHeader>

      <CardContent className="min-h-[100px] relative">
        <AnimatePresence mode="wait">
          {isAnalyzing ? (
            <motion.div 
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center py-12 space-y-4"
            >
              <Loader2 className="h-8 w-8 text-primary animate-spin" />
              <div className="text-sm text-muted-foreground animate-pulse font-medium">
                Sintetizando métricas de {type}...
              </div>
            </motion.div>
          ) : analysis ? (
            <motion.div 
              key="content"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="prose prose-invert max-w-none"
            >
              <div className="bg-muted/30 rounded-xl p-6 border border-border/50">
                {formatText(displayedText)}
                {index < analysis.length && (
                  <motion.span
                    animate={{ opacity: [1, 0] }}
                    transition={{ repeat: Infinity, duration: 0.8 }}
                    className="inline-block w-2 h-4 bg-primary ml-1 translate-y-0.5"
                  />
                )}
                {index >= analysis.length && (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mt-8 pt-4 border-t border-border/50 flex items-center justify-between"
                  >
                    <div className="flex items-center gap-2 text-xs text-success font-medium">
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      Análisis finalizado — Basado en datos del modelo
                    </div>
                    <Button variant="ghost" size="sm" onClick={startAnalysis} className="text-xs h-8">
                      Recalcular
                    </Button>
                  </motion.div>
                )}
              </div>
            </motion.div>
          ) : (
            <motion.div 
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center justify-center py-10 border-2 border-dashed border-border/50 rounded-xl"
            >
              <Sparkles className="h-10 w-10 text-muted-foreground/30 mb-3" />
              <p className="text-sm text-muted-foreground">Presiona el botón para iniciar el análisis profundo por IA</p>
            </motion.div>
          )}
        </AnimatePresence>
      </CardContent>
    </Card>
  );
}
