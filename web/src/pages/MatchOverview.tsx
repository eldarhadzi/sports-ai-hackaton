import { Link, useParams } from "react-router-dom";
import { useBouts, useClips, useProfiles, useReports } from "../lib/api";
import { QueriesState, Limitations } from "../components/common";
import { MetricCompare } from "../components/charts";
import { SLOT_COLOR, clipName, shortName } from "../lib/constants";
import type { Slot } from "../lib/types";

export default function MatchOverview() {
  const { run = "" } = useParams();
  const profiles = useProfiles(run);
  const bouts = useBouts(run);
  const reports = useReports(run);
  const clips = useClips(run);

  return (
    <div className="page">
      <QueriesState qs={[profiles, bouts, reports, clips]}>
        {() => {
          const byId = Object.fromEntries(profiles.data!.map((p) => [p.id, p]));
          if (!bouts.data!.length) return <div className="notice">No bouts in data/roster.csv for this run.</div>;
          return bouts.data!.map((b) => {
            const sides = (["aka", "ao"] as Slot[]).filter((s) => b.sides[s]);
            const athletes = sides.map((s) => byId[b.sides[s]!.athlete_id]).filter(Boolean);
            return (
              <section key={b.bout_id} className="bout">
                <header className="match-head">
                  <div>
                    <h1>
                      {b.title}, {b.subtitle}
                    </h1>
                    <p className="muted">
                      {b.clips.length} clip(s) analyzed · run <code>{run}</code>
                    </p>
                  </div>
                  <div className="scoreboard" aria-label={`Final score ${sides.map((s) => `${b.sides[s]!.name} ${b.sides[s]!.points}`).join(", ")}`}>
                    {sides.map((s, i) => {
                      const name = (
                        <span className="score-name" style={{ borderColor: SLOT_COLOR[s] }}>
                          {b.sides[s]!.name}
                          <small>{s.toUpperCase()}</small>
                        </span>
                      );
                      const num = <span className="score-num">{b.sides[s]!.points}</span>;
                      return i === 0 ? (
                        <span key={s} className="score-side">{name}{num}</span>
                      ) : (
                        <span key={s} className="score-side">
                          <span className="score-sep">–</span>
                          {num}
                          {name}
                        </span>
                      );
                    })}
                  </div>
                </header>

                <h2>Metrics side by side</h2>
                <MetricCompare athletes={athletes} />
                <p className="caption">
                  Strike counts are estimates (not checked against a human count). Hover a metric name for its
                  definition.
                </p>

                <h2>Coach headlines</h2>
                <div className="headline-grid">
                  {athletes.map((p) => {
                    const r = reports.data![p.id];
                    const slot = sides.find((s) => b.sides[s]!.athlete_id === p.id)!;
                    return (
                      <Link key={p.id} to={`/r/${run}/athlete/${p.id}`} className="headline-card" style={{ borderTopColor: SLOT_COLOR[slot] }}>
                        <div className="headline-who">{shortName(p.label)}</div>
                        <div className="headline-text">{r?.headline ?? "No coach report yet."}</div>
                        <span className="link-more">Open athlete card →</span>
                      </Link>
                    );
                  })}
                </div>

                <h2>Clips</h2>
                <div className="clip-list">
                  {clips.data!
                    .filter((c) => b.clips.includes(c.clip))
                    .map((c) => (
                      <Link key={c.clip} to={`/r/${run}/bout/${c.clip}`} className="clip-card">
                        <b>{clipName(c.clip)}</b>
                        <span className="muted">{c.duration_s?.toFixed(1)} s</span>
                        {!c.has_video && <span className="muted">no video</span>}
                        <span className="link-more">Watch →</span>
                      </Link>
                    ))}
                </div>
              </section>
            );
          });
        }}
      </QueriesState>
      <Limitations />
    </div>
  );
}
