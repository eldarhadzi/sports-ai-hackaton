import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useProfiles } from "../lib/api";
import { ConfidenceBadge, QueryState } from "../components/common";
import { ScoutingScatter } from "../components/charts";
import { METRICS, METRIC_KEYS, ageGroup, colorOf, fmtUnit } from "../lib/constants";

export default function ScoutingBoard() {
  const { run = "" } = useParams();
  const nav = useNavigate();
  const profiles = useProfiles(run);
  const [group, setGroup] = useState<string | null>(null);

  return (
    <div className="page">
      <header className="page-head">
        <h1>Scouting board</h1>
      </header>
      <QueryState q={profiles}>
        {(all) => {
          const groups = [...new Set(all.map((p) => p.age_group))].sort();
          const g = group && groups.includes(group) ? group : groups[0];
          const rows = all.filter((p) => p.age_group === g).sort((a, b) => (b.gap ?? -999) - (a.gap ?? -999));
          return (
            <>
              <p className="muted">
                Above the diagonal = the video shows more than the result does (positive gap). Dot size = confidence.
                <code>standing_pct</code> is currently bout points share (a placeholder).
              </p>
              {groups.length > 1 && (
                <div className="tabs" role="tablist">
                  {groups.map((x) => (
                    <button key={x} role="tab" aria-selected={x === g} className={"tab" + (x === g ? " active" : "")} onClick={() => setGroup(x)}>
                      {ageGroup(x)}
                    </button>
                  ))}
                </div>
              )}
              <div className="stack">
                <section className="card">
                  <ScoutingScatter group={rows} onPick={(p) => nav(`/r/${run}/athlete/${p.id}`)} />
                </section>
                <section className="card">
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Athlete</th>
                          <th className="num">Skill</th>
                          <th className="num">Standing</th>
                          <th className="num">Gap</th>
                          {METRIC_KEYS.map((k) => (
                            <th key={k} className="num" title={METRICS[k].help}>
                              {METRICS[k].name}
                            </th>
                          ))}
                          <th>Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map((p) => (
                          <tr key={p.id} className="clickable" onClick={() => nav(`/r/${run}/athlete/${p.id}`)}>
                            <td>
                              <span className="swatch" style={{ background: colorOf(p) }} /> {p.label}
                              <div className="muted small">{p.belt} belt · {p.n_bouts} bout(s)</div>
                            </td>
                            <td className="num">{p.skill_pct == null ? "—" : `P${p.skill_pct}`}</td>
                            <td className="num">{p.standing_pct == null ? "—" : `P${p.standing_pct}`}</td>
                            <td className="num">{p.gap == null ? "—" : `${p.gap > 0 ? "+" : ""}${p.gap}`}</td>
                            {METRIC_KEYS.map((k) => (
                              <td key={k} className="num">
                                {fmtUnit(k, p.metrics[k])}
                              </td>
                            ))}
                            <td>
                              <ConfidenceBadge c={p.confidence} />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="caption">
                    Cohort {ageGroup(g)}: n = {rows.length}. With n &lt; 5, percentiles move in big steps, so add
                    more athletes before reading them as rankings.
                  </p>
                </section>
              </div>
            </>
          );
        }}
      </QueryState>
    </div>
  );
}
