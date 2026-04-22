import { Link, useLocation } from "@tanstack/react-router";
import {
  Home,
  Database,
  GitCompare,
  History,
  Lightbulb,
  Brain,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { title: "Resumen", path: "/", icon: Home },
  { title: "Análisis", path: "/analysis", icon: Database },
  { title: "Comparativa", path: "/comparison", icon: GitCompare },
  { title: "Historial", path: "/history", icon: History },
  { title: "Selector", path: "/selector", icon: Lightbulb },
];

interface DashboardSidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export function DashboardSidebar({ isOpen, onClose }: DashboardSidebarProps) {
  const location = useLocation();

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 z-50 flex h-screen w-64 flex-col border-r border-border bg-sidebar shadow-sm transition-transform duration-300 ease-in-out md:translate-x-0",
        isOpen ? "translate-x-0" : "-translate-x-full",
      )}
    >
      <div className="flex items-center justify-between px-6 py-8 border-b border-border/50 mb-4 bg-muted/5">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary border border-primary/20">
            <Brain className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-xs font-bold text-foreground tracking-widest uppercase">
              LSTM <span className="text-primary">vs</span> GRU
            </h1>
            <p className="text-[9px] text-muted-foreground font-medium uppercase tracking-tighter">
              Plataforma de Investigación
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="rounded-md p-1.5 text-muted-foreground hover:bg-muted md:hidden"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <nav className="flex-1 px-4 py-2 space-y-1.5 overflow-y-auto">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2.5 text-[13px] font-semibold transition-all duration-200",
                isActive
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:bg-muted/50 hover:text-foreground",
              )}
            >
              <item.icon
                className={cn("h-4 w-4", isActive ? "text-primary-foreground" : "text-muted-foreground/70")}
              />
              <span>{item.title}</span>
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-border px-4 py-4">
        <p className="text-[10px] text-muted-foreground text-center">
          Tesis de Ingeniería — 2026
        </p>
      </div>
    </aside>
  );
}
