import { useMemo, useState } from "react";
import { Navigate, useNavigate, useParams } from "react-router-dom";
import { useBouts, useClips, useProfiles, useReports, videoUrl } from "../lib/api";
import { CoachReport, ConfidenceBadge, MetricTile, QueriesState, ReachedTag, StatTile } from "../components/common";
import { CohortDots, InitiativeTimeline } from "../components/charts";
import VideoPlayer, { type SeekRequest } from "../components/VideoPlayer";
import { METRIC_KEYS, ageGroup, clipName, colorOf, limbName } from "../lib/constants";
import { downloadReport } from "../lib/report";
import type { StrikeEvent } from "../lib/types";

export default function AthleteCard() {
  const { run = "", id } = useParams();
  const nav = useNavigate();
  const profiles = useProfiles(run);
  const reports = useReports(run);
  const bouts = useBouts(run);
  const clips = useClips(run);
  const [picked, setPicked] = useState<{ ev: StrikeEvent; n: number } | null>(null);
  // Any change of :id (select, browser Back/Forward) drops the previous athlete's picked strike.
  const [shownId, setShownId] = useState(id);
  if (shownId !== id) {
    setShownId(id);
    setPicked(null);
  }

  const p = profiles.data?.find((x) => x.id === id);
  const evKey = (e: StrikeEvent) => `${e.clip}|${e.frame}`;
  const current = picked?.ev ?? p?.events[0];
  const seek: SeekRequest | null = useMemo(
    () => (picked ? { t: Math.max(0, (picked.ev.clip_t ?? 0) - 1), n: picked.n } : null),
    [picked],
  );

  if (profiles.data && !p) {
    const first = profiles.data[0];
    return first ? <Navigate to={`/r/${run}/athlete/${first.id}`} replace /> : <div className="notice">No athletes.</div>;
  }
  const pick = (ev: StrikeEvent) => setPicked((old) => ({ ev, n: (old?.n ?? 0) + 1 }));

  return (
    <div className="page">
      <QueriesState qs={[profiles, reports, bouts, clips]}>
        {() => {
          const prof = p!;
          const group = profiles.data!.filter((q) => q.age_group === prof.age_group);
          const report = reports.data![prof.id];
          const bout = bouts.data!.find((b) => Object.values(b.sides).some((s) => s?.athlete_id === prof.id));
          const title = bout ? `${bout.title}, ${bout.subtitle}` : "Kumite analysis";
          return (
            <>
              <header className="athlete-head" style={{ borderLeftColor: colorOf(prof) }}>
                <div>
                  <div className="row gap">
                    <select aria-label="Athlete" value={prof.id} onChange={(e) => { setPicked(null); nav(`/r/${run}/athlete/${e.target.value}`); }}>
                      {profiles.data!.map((q) => (
                        <option key={q.id} value={q.id}>
                          {q.label}
                        </option>
                      ))}
                    </select>
                    <ConfidenceBadge c={prof.confidence} />
                  </div>
                  <h1>{prof.label}</h1>
                  <p className="muted">
                    {ageGroup(prof.age_group)} · {prof.belt} belt · {prof.n_bouts} bout(s), {prof.n_clips} clip(s) ·{" "}
                    {prof.active_s} s of live fighting · reliability {prof.reliability.toFixed(2)} · cohort n ={" "}
                    {prof.cohort_n}
                  </p>
                </div>
                <button className="btn" onClick={() => downloadReport(prof, report, title)}>
                  Download report
                </button>
              </header>

              <div className="tiles">
                {METRIC_KEYS.map((k) => (
                  <MetricTile key={k} k={k} value={prof.metrics[k]} pct={prof.metric_pct[k] ?? null} />
                ))}
                <StatTile
                  label="Skill"
                  value={prof.skill_pct == null ? "—" : `P${prof.skill_pct}`}
                  sub="mean metric percentile"
                  help="Percentile of the mean metric percentile within the cohort (same age group)."
                />
                <StatTile
                  label="Gap"
                  value={prof.gap == null ? "—" : `${prof.gap > 0 ? "+" : ""}${prof.gap}`}
                  sub={`standing P${prof.standing_pct ?? "—"} (placeholder)`}
                  help="skill_pct − standing_pct. Positive = fights better than the result shows. standing_pct is a placeholder: bout points share."
                />
              </div>

              <div className="split">
                <div className="col">
                  <section className="card">
                    <h2>Initiative over time</h2>
                    <InitiativeTimeline p={prof} onPick={pick} />
                  </section>
                  <section className="card">
                    <h2>Against the cohort</h2>
                    <CohortDots p={prof} group={group} />
                  </section>
                  <section className="card">
                    <h2>Strikes</h2>
                    {current ? (
                      <>
                        {clips.data!.find((c) => c.clip === current.clip)?.has_video ? (
                          <VideoPlayer src={videoUrl(run, current.clip!)} seek={seek} />
                        ) : (
                          <div className="notice">Annotated video not found for this clip.</div>
                        )}
                        <div className="jump-list" role="list">
                          {prof.events.map((e) => (
                            <button
                              key={evKey(e)}
                              role="listitem"
                              className={"jump" + (current && evKey(current) === evKey(e) ? " active" : "")}
                              onClick={() => pick(e)}
                            >
                              <span className="mono">{clipName(e.clip ?? "")} @ {e.clip_t?.toFixed(1)} s</span>
                              <span>
                                {e.reached && <ReachedTag />} {e.type} <span className="muted">({limbName(e.limb)})</span>
                              </span>
                            </button>
                          ))}
                        </div>
                      </>
                    ) : (
                      <p className="muted">No strikes detected.</p>
                    )}
                    <p className="caption">
                      ~{prof.counts.strikes} strikes ({prof.counts.punches} punches, {prof.counts.kicks} kicks) ·{" "}
                      {prof.counts.advances} forward attacks
                      {prof.counts.reached != null && <> · {prof.counts.reached} reached the opponent</>}
                    </p>
                  </section>
                </div>
                <div className="col">
                  <section className="card">
                    <h2>Coach report</h2>
                    <CoachReport report={report} runHint={run} />
                  </section>
                  <details className="card">
                    <summary>Raw JSON (team format)</summary>
                    <pre className="json">
                      {JSON.stringify(
                        {
                          ...Object.fromEntries(
                            (["id", "age_group", "belt", "n_bouts", "reliability", "metrics", "skill_pct", "standing_pct", "gap", "confidence", "cohort_n"] as const).map(
                              (k) => [k, prof[k]],
                            ),
                          ),
                          timeline: [...prof.timeline.slice(0, 3), `… ${prof.timeline.length} points`],
                        },
                        null,
                        2,
                      )}
                    </pre>
                  </details>
                </div>
              </div>
            </>
          );
        }}
      </QueriesState>
    </div>
  );
}
