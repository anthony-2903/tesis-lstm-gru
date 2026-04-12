import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import type { LucideIcon } from "lucide-react";

interface KpiCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  variant?: "cyan" | "violet" | "default";
  delay?: number;
}

export function KpiCard({ title, value, subtitle, icon: Icon, variant = "default", delay = 0 }: KpiCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      className={cn(
        "rounded-xl border border-border bg-card p-5 transition-shadow duration-300",
        variant === "cyan" && "card-glow",
        variant === "violet" && "card-glow-violet"
      )}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{title}</p>
          <p className="mt-2 text-2xl font-bold font-data text-foreground">{value}</p>
          {subtitle && <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>}
        </div>
        <div className={cn(
          "flex h-10 w-10 items-center justify-center rounded-lg",
          variant === "cyan" && "gradient-cyan",
          variant === "violet" && "gradient-violet",
          variant === "default" && "bg-muted"
        )}>
          <Icon className={cn(
            "h-5 w-5",
            variant === "cyan" && "text-cyan-foreground",
            variant === "violet" && "text-violet-foreground",
            variant === "default" && "text-muted-foreground"
          )} />
        </div>
      </div>
    </motion.div>
  );
}
