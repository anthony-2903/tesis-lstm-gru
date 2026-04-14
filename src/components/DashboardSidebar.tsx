import { Link, useLocation } from "@tanstack/react-router";
import {
  Home,
  Database,
  GitCompare,
  History,
  Lightbulb,
  Brain,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { title: "Resumen", path: "/", icon: Home },
  { title: "Análisis", path: "/analysis", icon: Database },
  { title: "Comparativa", path: "/comparison", icon: GitCompare },
  { title: "Historial", path: "/history", icon: History },
  { title: "Selector", path: "/selector", icon: Lightbulb },
];

export function DashboardSidebar() {
  const location = useLocation();

  return (
    <aside className="fixed left-0 top-0 z-40 flex h-screen w-64 flex-col border-r border-border bg-sidebar shadow-sm">
      <div className="flex items-center gap-3 px-6 py-8 border-b border-border/50 mb-4 bg-muted/5">
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

      <nav className="flex-1 px-4 py-2 space-y-1.5">
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
