import { Activity, Landmark, Shield } from "lucide-react";
import type { DomainId } from "@/lib/api";

export const DOMAIN_OPTIONS: {
  id: DomainId;
  title: string;
  shortTitle: string;
  source: string;
  description: string;
  aiType: "phishtank" | "energia" | "finanzas";
  icon: typeof Shield;
}[] = [
  {
    id: "phishing",
    title: "Phishing",
    shortTitle: "Phishing",
    source: "PhishTank",
    description: "URLs verificadas, rasgos de texto y clasificacion de enlaces maliciosos.",
    aiType: "phishtank",
    icon: Shield,
  },
  {
    id: "energia",
    title: "Energia",
    shortTitle: "Energia",
    source: "Open Power System Data",
    description: "Series temporales de consumo/generación y detección de desviaciones.",
    aiType: "energia",
    icon: Activity,
  },
  {
    id: "finanzas",
    title: "Finanzas publicas",
    shortTitle: "Finanzas",
    source: "MEF Datos Abiertos",
    description: "Indicadores de brechas y operadores Invierte.pe con limpieza tabular.",
    aiType: "finanzas",
    icon: Landmark,
  },
];

export const getDomainOption = (domain: DomainId) => DOMAIN_OPTIONS.find((item) => item.id === domain) || DOMAIN_OPTIONS[0];

export const getInitialDomain = (): DomainId => {
  if (typeof window === "undefined") return "phishing";
  const query = new URLSearchParams(window.location.search);
  const value = query.get("domain");
  if (value === "energia" || value === "finanzas" || value === "phishing") return value;
  if (value === "phishtank") return "phishing";
  return "phishing";
};
