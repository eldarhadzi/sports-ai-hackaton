import type { MetricKey, Profile, Slot } from "./types";

// Validated pair on the light surface (dataviz validator: CVD ΔE 23.8, contrast ≥ 3:1).
export const AKA_C = "#D9483B";
export const AO_C = "#2F6FD6";
export const NEUTRAL = "#9ca3af";
export const GRID = "#e8e8e6";
export const INK = "#1f2328";
export const MUTED = "#6b7280";
export const SLOT_COLOR: Record<Slot, string> = { aka: AKA_C, ao: AO_C };

// Definitions from HANDOFF.md section 2.
export const METRICS: Record<MetricKey, { name: string; unit: string; help: string }> = {
  strike_rate: {
    name: "Strike rate",
    unit: "/min",
    help: "Strikes (punches + kicks) per minute of live time. A strike is a wrist or ankle speed peak ≥ 5 torso/s aimed at the opponent, at 1.2–6 torso distance.",
  },
  initiative_share: {
    name: "Initiative",
    unit: "",
    help: "Exchanges started ÷ exchanges with a known initiator. The initiator is whoever advanced or struck first (looking 0.8 s back).",
  },
  range_control: {
    name: "Range control",
    unit: "",
    help: "Share of moving frames where this athlete presses forward more than the opponent.",
  },
  tempo: {
    name: "Tempo",
    unit: "/10 s",
    help: "(Strikes + advances) per 10 s of live time. An advance is moving toward the opponent > 1.5 torso/s for ≥ 0.2 s.",
  },
};
export const METRIC_KEYS = Object.keys(METRICS) as MetricKey[];

export const PCT_METRICS: MetricKey[] = ["initiative_share", "range_control"];

export function fmt(k: MetricKey, v: number | null | undefined): string {
  if (v == null) return "—";
  return PCT_METRICS.includes(k) ? `${Math.round(v * 100)}%` : v.toFixed(1);
}
export const fmtUnit = (k: MetricKey, v: number | null | undefined) =>
  v == null ? "—" : fmt(k, v) + METRICS[k].unit;

// HANDOFF.md section 7, items 1-4: always visible. The cohort line is computed from the data.
export function limitations(cohortN?: number): string[] {
  const c =
    cohortN == null
      ? "Cohorts are small, so percentiles and the gap are illustrative only."
      : `Cohort = ${cohortN} athlete${cohortN === 1 ? "" : "s"}, so percentiles and the gap are ${cohortN < 5 ? "illustrative only" : "still rough"}.`;
  return [
    "Strike counts are not validated against a human count. Treat them as estimates.",
    "Kicks are often missed at 360p: the pose model loses the athlete mid-kick. Punch vs kick comes from the limb that struck.",
    c,
    "standing_pct is a placeholder (bout points share).",
  ];
}

export function slotOf(p: Profile): Slot | null {
  const s = new Set(p.clips.map((c) => c.slot));
  return s.size === 1 ? ([...s][0] as Slot) : null;
}
export const colorOf = (p: Profile) => {
  const s = slotOf(p);
  return s ? SLOT_COLOR[s] : NEUTRAL;
};

export const shortName = (label: string) => label.replace(/\s*\((AKA|AO)\)\s*$/, "");

export const clipName = (clip: string) => {
  const m = clip.match(/^kumite_(\d+)/);
  return m ? `Clip ${m[1]}` : clip;
};

export const AGE_GROUPS: Record<string, string> = { senior_f: "Senior female", senior_m: "Senior male" };
export const ageGroup = (g: string) => AGE_GROUPS[g] ?? g.replace(/_/g, " ");

export const limbName = (l: string) => l.replace(/_/g, " ");
