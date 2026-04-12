import { Link, useLocation } from "@tanstack/react-router";
import { Home, Database, GitCompare, History, Lightbulb, Brain } from "lucide-react";
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
    <aside className="fixed left-0 top-0 z-40 flex h-screen w-64 flex-col border-r border-border bg-sidebar">
      <div className="flex items-center gap-3 px-6 py-6">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl gradient-cyan">
          <Brain className="h-5 w-5 text-cyan-foreground" />
        </div>
        <div>
          <h1 className="text-sm font-bold text-foreground tracking-tight">LSTM vs GRU</h1>
          <p className="text-[10px] text-muted-foreground">Detección de Anomalías</p>
        </div>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200",
                isActive
                  ? "bg-primary/15 text-primary card-glow"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
            >
              <item.icon className={cn("h-4 w-4", isActive && "text-primary")} />
              <span>{item.title}</span>
            </Link>
          );
        })}
      </nav>

      <div className="border-t border-border px-4 py-4">
        <p className="text-[10px] text-muted-foreground text-center">
          Tesis de Ingeniería — 2024
        </p>
      </div>
    </aside>
  );
}
