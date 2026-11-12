import {
  CartesianGrid,
  ComposedChart,
  LabelList,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import {
  AKA_C,
  AO_C,
  GRID,
  METRICS,
  METRIC_KEYS,
  MUTED,
  NEUTRAL,
  SLOT_COLOR,
  clipName,
  colorOf,
  fmtUnit,
  limbName,
  shortName,
} from "../lib/constants";
import type { Features, MetricKey, Profile, Slot, StrikeEvent, TimelinePoint } from "../lib/types";

const AXIS = { stroke: GRID, tick: { fill: MUTED, fontSize: 12 }, tickLine: false } as const;

/** Value of the (0.5 s) initiative timeline at time t, linearly interpolated; null inside a pause. */
function initiativeAt(tl: TimelinePoint[], t: number): number | null {
  let i = 0;
  while (i < tl.length - 1 && tl[i + 1].t <= t) i++;
  const a = tl[i], b = tl[i + 1];
  if (!a || a.initiative == null) return null;
  if (!b || b.initiative == null || b.t === a.t) return a.t === t ? a.initiative : null;
  return a.initiative + ((t - a.t) / (b.t - a.t)) * (b.initiative - a.initiative);
}

/** Triangle = punch, diamond = kick (shape is the secondary encoding next to color). */
function StrikeMark({ cx, cy, color, type, size = 7 }: { cx: number; cy: number; color: string; type: string; size?: number }) {
  if (!Number.isFinite(cx) || !Number.isFinite(cy)) return null;
  const s = size;
  const d =
    type === "kick"
      ? `M${cx},${cy - s} L${cx + s},${cy} L${cx},${cy + s} L${cx - s},${cy} Z`
      : `M${cx},${cy - s} L${cx + s},${cy + s * 0.75} L${cx - s},${cy + s * 0.75} Z`;
  return <path d={d} fill={color} stroke="#fff" strokeWidth={2} style={{ cursor: "pointer" }} />;
}

export function StrikeLegend({ items }: { items: { color: string; label: string; line?: boolean }[] }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.label} className="legend-item">
          {it.line ? (
            <svg width="18" height="10" aria-hidden>
              <line x1="0" y1="5" x2="18" y2="5" stroke={it.color} strokeWidth="2" />
            </svg>
          ) : (
            <span className="swatch" style={{ background: it.color }} />
          )}
          {it.label}
        </span>
      ))}
      <span className="legend-item">
        <svg width="14" height="14" aria-hidden>
          <path d="M7,1 L13,11.5 L1,11.5 Z" fill={MUTED} />
        </svg>
        punch
      </span>
      <span className="legend-item">
        <svg width="14" height="14" aria-hidden>
          <path d="M7,0 L14,7 L7,14 L0,7 Z" fill={MUTED} />
        </svg>
        kick
      </span>
    </div>
  );
}

function TipBox({ children }: { children: React.ReactNode }) {
  return <div className="chart-tip">{children}</div>;
}

// ------------------------------------------------------------------ athlete initiative timeline
type TlRow = { t: number; initiative: number | null; strikeY?: number; ev?: StrikeEvent };

export function InitiativeTimeline({ p, onPick }: { p: Profile; onPick?: (e: StrikeEvent) => void }) {
  const color = colorOf(p);
  const rows: TlRow[] = [
    ...p.timeline.map((x) => ({ t: x.t, initiative: x.initiative })),
    ...p.events.map((e) => {
      const v = initiativeAt(p.timeline, e.t);
      return { t: e.t, initiative: v, strikeY: v ?? 0.5, ev: e };
    }),
  ].sort((a, b) => a.t - b.t);
  const end = p.clips.length ? Math.max(...p.clips.map((c) => c.offset + c.duration_s)) : undefined;

  return (
    <div>
      <StrikeLegend items={[{ color, label: `${shortName(p.label)} initiative`, line: true }]} />
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={rows} margin={{ top: 22, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="t" type="number" domain={[0, end ?? "dataMax"]} {...AXIS} tickFormatter={(v) => `${v}s`} />
          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...AXIS} width={36} />
          <ReferenceLine y={0.5} stroke={NEUTRAL} strokeDasharray="3 3" />
          {p.clips.map((c, i) => (
            <ReferenceLine
              key={c.clip}
              x={c.offset}
              stroke={i === 0 ? "transparent" : NEUTRAL}
              label={{ value: clipName(c.clip), position: "insideTopLeft", fill: MUTED, fontSize: 11, dy: -18 }}
            />
          ))}
          <Line dataKey="initiative" stroke={color} strokeWidth={2} dot={false} connectNulls={false} isAnimationActive={false} />
          <Scatter
            dataKey="strikeY"
            isAnimationActive={false}
            onClick={(d: { payload?: TlRow }) => d.payload?.ev && onPick?.(d.payload.ev)}
            shape={(props: { cx?: number; cy?: number; payload?: TlRow }) =>
              props.payload?.ev ? (
                <StrikeMark cx={props.cx ?? NaN} cy={props.cy ?? NaN} color={color} type={props.payload.ev.type} />
              ) : (
                <g />
              )
            }
          />
          <Tooltip
            cursor={{ stroke: NEUTRAL }}
            content={({ active, payload }) => {
              const r = active && (payload?.[0]?.payload as TlRow | undefined);
              if (!r) return null;
              return (
                <TipBox>
                  <b>{r.t.toFixed(1)} s</b>
                  {r.ev?.clip && ` · ${clipName(r.ev.clip)} @ ${r.ev.clip_t?.toFixed(1)} s`}
                  <br />
                  initiative {r.initiative == null ? "— (paused / idle)" : r.initiative.toFixed(2)}
                  {r.ev && (
                    <>
                      <br />
                      {r.ev.type} ({limbName(r.ev.limb)}) · click to watch
                    </>
                  )}
                </TipBox>
              );
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

// ------------------------------------------------------------------ bout chart (time-aligned with the video)
type BoutRow = { t: number; initiative: number | null; akaY?: number; aoY?: number; ev?: StrikeEvent; slot?: Slot };

export function BoutChart({
  f,
  names,
  currentTime,
  onSeek,
}: {
  f: Features;
  names: Record<Slot, string>;
  currentTime: number;
  onSeek: (t: number) => void;
}) {
  const tl = f.athletes.aka.timeline;
  const rows: BoutRow[] = [
    ...tl.map((x) => ({ t: x.t, initiative: x.initiative })),
    ...(["aka", "ao"] as Slot[]).flatMap((slot) =>
      f.athletes[slot].events.map((e) => ({
        t: e.t,
        initiative: initiativeAt(tl, e.t),
        [slot === "aka" ? "akaY" : "aoY"]: slot === "aka" ? 1.06 : -0.06,
        ev: e,
        slot,
      })),
    ),
  ].sort((a, b) => a.t - b.t);

  // Stagger the initiator labels of exchanges that are close together, so they don't overprint.
  const minGap = f.duration_s * 0.05;
  const levels: number[] = [];
  f.exchanges.forEach((x, i) => {
    const prev = f.exchanges[i - 1];
    levels.push(prev && x.t0 - prev.t0 < minGap ? 1 - levels[i - 1] : 0);
  });

  const shape = (slot: Slot) => (props: { cx?: number; cy?: number; payload?: BoutRow }) =>
    props.payload?.slot === slot && props.payload.ev ? (
      <StrikeMark cx={props.cx ?? NaN} cy={props.cy ?? NaN} color={SLOT_COLOR[slot]} type={props.payload.ev.type} size={6} />
    ) : (
      <g />
    );

  return (
    <div>
      <StrikeLegend
        items={[
          { color: AKA_C, label: `${names.aka} strikes` },
          { color: AO_C, label: `${names.ao} strikes` },
          { color: AKA_C, label: `${names.aka} initiative`, line: true },
        ]}
      />
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart
          data={rows}
          margin={{ top: 30, right: 12, bottom: 4, left: 0 }}
          onClick={(s) => {
            const t = Number(s?.activeLabel);
            if (Number.isFinite(t)) onSeek(t);
          }}
          style={{ cursor: "pointer" }}
        >
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="t" type="number" domain={[0, f.duration_s]} {...AXIS} tickFormatter={(v) => `${v}s`} />
          <YAxis domain={[-0.12, 1.12]} ticks={[0, 0.5, 1]} {...AXIS} width={36} />
          {f.exchanges.map((x, i) => (
            <ReferenceArea
              key={i}
              x1={x.t0 - 0.2}
              x2={x.t1 + 0.2}
              y1={-0.12}
              y2={1.12}
              fill={GRID}
              fillOpacity={0.9}
              label={{
                value: x.initiator ? names[x.initiator] : "?",
                position: "insideTop",
                fill: x.initiator ? SLOT_COLOR[x.initiator] : MUTED,
                fontSize: 10,
                fontWeight: 600,
                dy: levels[i] ? -14 : -28,
              }}
            />
          ))}
          <ReferenceLine y={0.5} stroke={NEUTRAL} strokeDasharray="3 3" />
          <Line dataKey="initiative" stroke={AKA_C} strokeWidth={2} dot={false} connectNulls={false} isAnimationActive={false} />
          <Scatter dataKey="akaY" isAnimationActive={false} shape={shape("aka")} />
          <Scatter dataKey="aoY" isAnimationActive={false} shape={shape("ao")} />
          <ReferenceLine x={currentTime} stroke={MUTED} strokeWidth={1.5} />
          <Tooltip
            cursor={{ stroke: NEUTRAL }}
            content={({ active, payload }) => {
              const r = active && (payload?.[0]?.payload as BoutRow | undefined);
              if (!r) return null;
              const ex = f.exchanges.find((x) => r.t >= x.t0 - 0.2 && r.t <= x.t1 + 0.2);
              return (
                <TipBox>
                  <b>{r.t.toFixed(1)} s</b> · click to seek
                  <br />
                  {r.initiative == null
                    ? "paused / idle"
                    : r.initiative >= 0.5
                      ? `${names.aka} pressing (${r.initiative.toFixed(2)})`
                      : `${names.ao} pressing (${r.initiative.toFixed(2)})`}
                  {r.ev && r.slot && (
                    <>
                      <br />
                      {names[r.slot]}: {r.ev.type} ({limbName(r.ev.limb)})
                    </>
                  )}
                  {ex && (
                    <>
                      <br />
                      exchange, started by {ex.initiator ? names[ex.initiator] : "unknown"}
                    </>
                  )}
                </TipBox>
              );
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="caption">
        Above 0.5 = {names.aka} presses, below = {names.ao}. Click to seek the video.
      </p>
    </div>
  );
}

// ------------------------------------------------------------------ match overview: one small chart per metric
export function MetricCompare({ athletes }: { athletes: Profile[] }) {
  return (
    <div className="metric-grid">
      {METRIC_KEYS.map((k) => {
        const max = Math.max(...athletes.map((p) => p.metrics[k] ?? 0), 1e-9);
        const isShare = k === "initiative_share" || k === "range_control";
        const scale = isShare ? 1 : max;
        return (
          <div key={k} className="metric-panel">
            <div className="metric-panel-title" title={METRICS[k].help}>
              {METRICS[k].name}
              {METRICS[k].unit && <span className="muted"> {METRICS[k].unit}</span>}
            </div>
            {athletes.map((p) => {
              const v = p.metrics[k];
              return (
                <div key={p.id} className="hbar-row" title={`${p.label}: ${fmtUnit(k, v)}`}>
                  <span className="hbar-name">{shortName(p.label)}</span>
                  <span className="hbar-track">
                    <span
                      className="hbar"
                      style={{ width: `${v == null ? 0 : Math.max(1, (v / scale) * 100)}%`, background: colorOf(p) }}
                    />
                  </span>
                  <span className="hbar-value">{fmtUnit(k, v)}</span>
                </div>
              );
            })}
            {isShare && <div className="hbar-scale">bar = share of 100%</div>}
          </div>
        );
      })}
    </div>
  );
}

// ------------------------------------------------------------------ athlete vs cohort (percentile per metric)
export function CohortDots({ p, group }: { p: Profile; group: Profile[] }) {
  return (
    <div className="dots">
      {METRIC_KEYS.map((k: MetricKey) => (
        <div key={k} className="dots-row">
          <span className="dots-name">{METRICS[k].name}</span>
          <span className="dots-track">
            {[0, 25, 50, 75, 100].map((x) => (
              <span key={x} className="dots-tick" style={{ left: `${x}%` }} />
            ))}
            {[...group]
              .sort((a) => (a.id === p.id ? 1 : -1))
              .map((q) => {
                const pct = q.metric_pct[k];
                if (pct == null) return null;
                const me = q.id === p.id;
                return (
                  <span
                    key={q.id}
                    className={"dot" + (me ? " me" : "")}
                    style={{ left: `${pct}%`, background: me ? colorOf(q) : NEUTRAL }}
                    title={`${q.label}: ${fmtUnit(k, q.metrics[k])} (P${pct})`}
                  />
                );
              })}
          </span>
        </div>
      ))}
      <div className="dots-row axis">
        <span className="dots-name" />
        <span className="dots-track plain">
          {[0, 25, 50, 75, 100].map((x) => (
            <span key={x} className="dots-label" style={{ left: `${x}%` }}>
              P{x}
            </span>
          ))}
        </span>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ scouting: skill vs standing
const CONF_R = { high: 11, medium: 8, low: 6 } as const;

export function ScoutingScatter({ group, onPick }: { group: Profile[]; onPick: (p: Profile) => void }) {
  const pts = group.filter((p) => p.skill_pct != null && p.standing_pct != null);
  return (
    <ResponsiveContainer width="100%" height={420}>
      <ScatterChart margin={{ top: 24, right: 24, bottom: 28, left: 8 }}>
        <CartesianGrid stroke={GRID} />
        <XAxis
          dataKey="standing_pct"
          type="number"
          domain={[0, 100]}
          ticks={[0, 25, 50, 75, 100]}
          {...AXIS}
          label={{ value: "standing percentile (result)", position: "insideBottom", dy: 20, fill: MUTED, fontSize: 12 }}
        />
        <YAxis
          dataKey="skill_pct"
          type="number"
          domain={[0, 100]}
          ticks={[0, 25, 50, 75, 100]}
          {...AXIS}
          width={44}
          label={{ value: "skill percentile (video)", angle: -90, position: "insideLeft", dx: 4, dy: 70, fill: MUTED, fontSize: 12 }}
        />
        <ZAxis range={[1, 1]} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]} stroke={NEUTRAL} strokeDasharray="4 4" />
        {pts.map((p) => (
          <Scatter
            key={p.id}
            data={[{ ...p, short: shortName(p.label) }]}
            isAnimationActive={false}
            onClick={() => onPick(p)}
            shape={(s: { cx?: number; cy?: number }) => (
              <circle cx={s.cx} cy={s.cy} r={CONF_R[p.confidence]} fill={colorOf(p)} stroke="#fff" strokeWidth={2} style={{ cursor: "pointer" }} />
            )}
          >
            <LabelList dataKey="short" position="top" offset={14} fill="#1f2328" fontSize={12} fontWeight={600} />
          </Scatter>
        ))}
        <Tooltip
          cursor={false}
          content={({ active, payload }) => {
            const p = active && (payload?.[0]?.payload as Profile | undefined);
            if (!p) return null;
            return (
              <TipBox>
                <b>{p.label}</b>
                <br />
                skill P{p.skill_pct} · standing P{p.standing_pct}
                <br />
                gap {p.gap != null && p.gap > 0 ? "+" : ""}
                {p.gap} · {p.confidence} confidence
              </TipBox>
            );
          }}
        />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
