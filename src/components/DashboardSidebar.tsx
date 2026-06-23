import { Link, useLocation } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Home,
  Database,
  GitCompare,
  History,
  Lightbulb,
  Brain,
  BrainCircuit,
  X,
  ChevronLeft,
  ChevronRight,
  Sun,
  Moon,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { title: "Resumen", path: "/", icon: Home },
  { title: "Datos", path: "/upload", icon: Database },
  { title: "Análisis", path: "/analysis", icon: Database },
  { title: "Comparativa", path: "/comparison", icon: GitCompare },
  { title: "XAI SHAP", path: "/xai", icon: BrainCircuit },
  { title: "Historial", path: "/history", icon: History },
  { title: "Selector", path: "/selector", icon: Lightbulb },
];

interface DashboardSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
}

export function DashboardSidebar({ 
  isOpen, 
  onClose, 
  isCollapsed, 
  onToggleCollapse 
}: DashboardSidebarProps) {
  const location = useLocation();

  const [isDark, setIsDark] = useState(() => {
    if (typeof document !== "undefined") {
      return document.documentElement.classList.contains("dark") || localStorage.getItem("theme") === "dark";
    }
    return false;
  });

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [isDark]);

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 z-50 flex h-screen flex-col border-r border-border bg-sidebar shadow-sm transition-all duration-300 ease-in-out md:translate-x-0",
        isOpen ? "translate-x-0" : "-translate-x-full",
        isCollapsed ? "w-20" : "w-64"
      )}
    >
      <div className={cn(
        "flex items-center border-b border-border/50 mb-4 bg-muted/5 transition-all duration-300",
        isCollapsed ? "px-4 py-8 justify-center" : "px-6 py-8 justify-between"
      )}>
        <div className="flex items-center gap-3 overflow-hidden">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary border border-primary/20">
            <Brain className="h-5 w-5" />
          </div>
          {!isCollapsed && (
            <motion.div 
              initial={{ opacity: 0, x: -10 }} 
              animate={{ opacity: 1, x: 0 }}
              className="flex flex-col whitespace-nowrap"
            >
              <h1 className="text-[10px] font-bold text-foreground tracking-tighter uppercase">
                LSTM | GRU | BRNN | TRF | TCN
              </h1>
              <p className="text-[9px] text-muted-foreground font-medium uppercase tracking-tighter">
                Plataforma de Investigación
              </p>
            </motion.div>
          )}
        </div>
        {!isCollapsed && (
          <button
            onClick={onClose}
            className="rounded-md p-1.5 text-muted-foreground hover:bg-muted md:hidden"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      <nav className="flex-1 px-4 py-2 space-y-1.5 overflow-y-auto overflow-x-hidden">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={cn(
                "flex items-center rounded-md px-3 py-2.5 text-[13px] font-semibold transition-all duration-200 group",
                isCollapsed ? "justify-center" : "gap-3",
                isActive
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:bg-muted/50 hover:text-foreground",
              )}
            >
              <item.icon
                className={cn("h-4 w-4 shrink-0", isActive ? "text-primary-foreground" : "text-muted-foreground/70 group-hover:text-foreground")}
              />
              {!isCollapsed && (
                <motion.span 
                  initial={{ opacity: 0 }} 
                  animate={{ opacity: 1 }}
                  className="whitespace-nowrap"
                >
                  {item.title}
                </motion.span>
              )}
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-border px-4 py-4 flex flex-col items-center gap-4">
        {!isCollapsed && (
          <p className="text-[10px] text-muted-foreground text-center whitespace-nowrap">
            Tesis de Ingeniería — 2026
          </p>
        )}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsDark(!isDark)}
            className="flex h-8 w-8 items-center justify-center rounded-full border border-border bg-background text-muted-foreground hover:text-foreground hover:border-primary/50 transition-all shadow-sm"
            title="Alternar tema claro/oscuro"
          >
            {isDark ? <Sun className="h-4 w-4 text-warning" /> : <Moon className="h-4 w-4" />}
          </button>
          <button
            onClick={onToggleCollapse}
            className="hidden md:flex h-8 w-8 items-center justify-center rounded-full border border-border bg-background text-muted-foreground hover:text-foreground hover:border-primary/50 transition-all shadow-sm"
            title="Contraer barra lateral"
          >
            {isCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </button>
        </div>
      </div>
    </aside>
  );
}
