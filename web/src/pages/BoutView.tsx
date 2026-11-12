import { useCallback, useMemo, useState } from "react";
import { Navigate, NavLink, useParams } from "react-router-dom";
import { useClips, useFeatures, useProfiles, videoUrl } from "../lib/api";
import { QueriesState, QueryState, ReachedTag } from "../components/common";
import { BoutChart } from "../components/charts";
import VideoPlayer, { type SeekRequest } from "../components/VideoPlayer";
import { METRICS, METRIC_KEYS, SLOT_COLOR, clipName, fmtUnit, limbName, shortName } from "../lib/constants";
import type { Slot } from "../lib/types";

export default function BoutView() {
  const { run = "", clip } = useParams();
  const clips = useClips(run);
  const profiles = useProfiles(run);
  const features = useFeatures(run, clip);
  const [time, setTime] = useState(0);
  const [seek, setSeek] = useState<SeekRequest | null>(null);
  const onTime = useCallback((t: number) => setTime(t), []);
  const seekTo = useCallback((t: number) => setSeek((s) => ({ t: Math.max(0, t), n: (s?.n ?? 0) + 1 })), []);

  const names = useMemo(() => {
    const out: Record<Slot, string> = { aka: "AKA", ao: "AO" };
    for (const p of profiles.data ?? []) for (const c of p.clips) if (c.clip === clip) out[c.slot] = shortName(p.label);
    return out;
  }, [profiles.data, clip]);

  if (clips.data && !clip && clips.data[0]) return <Navigate to={`/r/${run}/bout/${clips.data[0].clip}`} replace />;
  const info = clips.data?.find((c) => c.clip === clip);

  return (
    <div className="page">
      <QueriesState qs={[clips, profiles]}>
        {() => (
          <>
            <header className="page-head">
              <h1>Bout view</h1>
              <nav className="tabs" aria-label="Clips">
                {clips.data!.map((c) => (
                  <NavLink
                    key={c.clip}
                    to={`/r/${run}/bout/${c.clip}`}
                    className={({ isActive }) => "tab" + (isActive ? " active" : "")}
                    onClick={() => { setSeek(null); setTime(0); }}
                  >
                    {clipName(c.clip)}
                  </NavLink>
                ))}
              </nav>
            </header>
            <QueryState q={features}>
              {(f) => {
                const events = (["aka", "ao"] as Slot[])
                  .flatMap((s) => f.athletes[s].events.map((e) => ({ ...e, slot: s })))
                  .sort((a, b) => a.t - b.t);
                const knees = (["aka", "ao"] as Slot[])
                  .flatMap((s) => (f.athletes[s].knee_raises ?? []).map((k) => ({ ...k, slot: s })))
                  .sort((a, b) => a.t - b.t);
                return (
                  <>
                    <p className="muted">
                      {f.duration_s.toFixed(1)} s clip · {f.active_s.toFixed(1)} s live · {f.paused_s.toFixed(1)} s score
                      graphic · {f.idle_s.toFixed(1)} s idle · {f.exchanges.length} exchanges
                    </p>
                    <div className="split even">
                      <section className="card">
                        {info?.has_video ? (
                          <VideoPlayer key={f.clip} src={videoUrl(run, f.clip)} seek={seek} onTime={onTime} />
                        ) : (
                          <div className="notice">Annotated video not found for this clip.</div>
                        )}
                      </section>
                      <section className="card">
                        <BoutChart f={f} names={names} currentTime={time} onSeek={(t) => seekTo(t - 0.5)} />
                      </section>
                    </div>

                    <div className="split even">
                      <section className="card">
                        <h2>This clip</h2>
                        <div className="table-wrap">
                          <table>
                            <thead>
                              <tr>
                                <th />
                                {METRIC_KEYS.map((k) => (
                                  <th key={k} className="num" title={METRICS[k].help}>
                                    {METRICS[k].name}
                                  </th>
                                ))}
                                <th className="num">Strikes</th>
                                <th className="num">Reliability</th>
                              </tr>
                            </thead>
                            <tbody>
                              {(["aka", "ao"] as Slot[]).map((s) => (
                                <tr key={s}>
                                  <td>
                                    <span className="swatch" style={{ background: SLOT_COLOR[s] }} /> {names[s]}
                                  </td>
                                  {METRIC_KEYS.map((k) => (
                                    <td key={k} className="num">
                                      {fmtUnit(k, f.athletes[s].metrics[k])}
                                    </td>
                                  ))}
                                  <td className="num">~{f.athletes[s].counts.strikes}</td>
                                  <td className="num">{f.athletes[s].reliability.toFixed(2)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                        <h2>Exchanges</h2>
                        {f.exchanges.length ? (
                          <div className="jump-list">
                            {f.exchanges.map((x, i) => (
                              <button key={i} className="jump" onClick={() => seekTo(x.t0 - 1)}>
                                <span className="mono">
                                  {x.t0.toFixed(1)}–{x.t1.toFixed(1)} s
                                </span>
                                <span>
                                  started by{" "}
                                  {x.initiator ? (
                                    <b style={{ color: SLOT_COLOR[x.initiator] }}>{names[x.initiator]}</b>
                                  ) : (
                                    "unknown"
                                  )}{" "}
                                  <span className="muted">· {x.n_strikes} strike(s)</span>
                                </span>
                              </button>
                            ))}
                          </div>
                        ) : (
                          <p className="muted">No exchanges in this clip.</p>
                        )}
                      </section>
                      <section className="card">
                        <h2>Jump to a strike</h2>
                        {events.length ? (
                          <div className="jump-list tall">
                            {events.map((e) => (
                              <button key={`${e.slot}-${e.frame}`} className="jump" onClick={() => seekTo(e.t - 1)}>
                                <span className="mono">{e.t.toFixed(1)} s</span>
                                <span>
                                  {e.reached && <ReachedTag />}{" "}
                                  <span className="swatch" style={{ background: SLOT_COLOR[e.slot] }} /> {names[e.slot]}{" "}
                                  {e.type} <span className="muted">({limbName(e.limb)})</span>
                                </span>
                              </button>
                            ))}
                          </div>
                        ) : (
                          <p className="muted">No strikes detected in this clip.</p>
                        )}
                        <h2>Knee raises without a strike</h2>
                        {knees.length ? (
                          <div className="jump-list">
                            {knees.map((k) => (
                              <button key={`${k.slot}-${k.frame}`} className="jump" onClick={() => seekTo(k.t - 1)}>
                                <span className="mono">{k.t.toFixed(1)} s</span>
                                <span>
                                  <span className="swatch" style={{ background: SLOT_COLOR[k.slot] }} /> {names[k.slot]} knee
                                  raise <span className="muted">(possible feint, {k.duration_s.toFixed(2)} s)</span>
                                </span>
                              </button>
                            ))}
                          </div>
                        ) : (
                          <p className="muted small">None detected in this clip.</p>
                        )}
                      </section>
                    </div>
                  </>
                );
              }}
            </QueryState>
          </>
        )}
      </QueriesState>
    </div>
  );
}
