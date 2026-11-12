// Schemas from HANDOFF.md section 4. Field names are the pipeline's own; do not rename.

export type Slot = "aka" | "ao";
export type MetricKey = "strike_rate" | "initiative_share" | "range_control" | "tempo";
export type Metrics = Record<MetricKey, number | null>;
export type Confidence = "high" | "medium" | "low";

export interface Counts {
  strikes: number;
  punches: number;
  kicks: number;
  advances: number;
  exchanges_initiated: number;
  exchanges_total?: number;
  reached?: number; // strikes whose tip got within reach_thr of the opponent's head/shoulders/hips
  reach_known?: number;
  knee_raises?: number;
}

export interface TimelinePoint {
  t: number;
  initiative: number | null; // null = paused / idle → draw as a gap
  strike: boolean;
  clip?: string;
}

export interface StrikeEvent {
  frame: number;
  t: number;
  type: "punch" | "kick";
  limb: string;
  speed: number;
  clip?: string;
  clip_t?: number; // seconds into that clip (profiles.json); use it to seek the video
  reach?: number | null; // torso units to the opponent's body; a 2D proxy, not contact
  reached?: boolean | null;
}

export interface KneeRaise {
  frame: number;
  t: number;
  duration_s: number;
  clip?: string;
  clip_t?: number;
}

export interface ProfileClip {
  clip: string;
  slot: Slot;
  offset: number;
  duration_s: number;
  active_s: number;
  metrics: Metrics;
  counts: Counts;
  reliability: number;
}

export interface Profile {
  id: string;
  label: string;
  age_group: string;
  belt: string;
  n_bouts: number;
  reliability: number;
  metrics: Metrics;
  skill_pct: number | null;
  standing_pct: number | null;
  gap: number | null;
  confidence: Confidence;
  cohort_n: number;
  timeline: TimelinePoint[];
  metric_pct: Partial<Record<MetricKey, number | null>>;
  standing_basis?: string;
  n_clips: number;
  active_s: number;
  points_for: number;
  points_against: number;
  counts: Counts;
  events: StrikeEvent[];
  knee_raises?: KneeRaise[];
  clips: ProfileClip[];
}

export interface Exchange {
  t0: number;
  t1: number;
  n_strikes: number;
  initiator: Slot | null;
}

export interface ClipAthlete {
  reliability: number;
  coverage: number;
  identity_known: number;
  keypoint_quality: number;
  metrics: Metrics;
  counts: Counts;
  events: StrikeEvent[];
  knee_raises?: KneeRaise[];
  timeline: TimelinePoint[];
}

export interface Features {
  clip: string;
  T: number;
  fps: number;
  duration_s: number;
  active_s: number;
  paused_s: number;
  idle_s: number;
  exchanges: Exchange[];
  athletes: Record<Slot, ClipAthlete>;
  config: Record<string, unknown>;
}

export interface Report {
  athlete_id: string;
  model?: string;
  headline?: string;
  summary?: string;
  strengths?: string[];
  weaknesses?: string[];
  focus?: string[];
  drills?: string[];
  caveats?: string[];
}

export interface ClipInfo {
  clip: string;
  duration_s: number;
  athletes: Partial<Record<Slot, string>>;
  has_features: boolean;
  has_video: boolean;
}

export interface BoutSide {
  athlete_id: string;
  label: string;
  name: string;
  belt: string;
  points: number;
}

export interface Bout {
  bout_id: string;
  age_group: string;
  clips: string[];
  sides: Partial<Record<Slot, BoutSide>>;
  title: string;
  subtitle: string;
}
