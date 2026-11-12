// Single-page HTML report for a coach to keep (HANDOFF.md section 10, item 4). Fully self-contained:
// inline CSS and an inline SVG timeline, no scripts, no external requests. Prints to one page.
import { limitations, METRICS, METRIC_KEYS, ageGroup, clipName, colorOf, fmtUnit, limbName } from "./constants";
import type { Profile, Report } from "./types";

const esc = (s: unknown) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

function timelineSvg(p: Profile, W = 680, H = 150): string {
  const pad = { l: 30, r: 8, t: 16, b: 20 };
  const end = Math.max(...p.clips.map((c) => c.offset + c.duration_s), ...p.timeline.map((x) => x.t), 1);
  const x = (t: number) => pad.l + (t / end) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - v) * (H - pad.t - pad.b);
  const color = colorOf(p);

  const segs: string[] = [];
  let cur: string[] = [];
  for (const pt of p.timeline) {
    if (pt.initiative == null) {
      if (cur.length > 1) segs.push(cur.join(" "));
      cur = [];
    } else cur.push(`${x(pt.t).toFixed(1)},${y(pt.initiative).toFixed(1)}`);
  }
  if (cur.length > 1) segs.push(cur.join(" "));

  const marks = p.events
    .map((e) => {
      const cx = x(e.t), cy = y(1.0) - 6, s = 4.5;
      return e.type === "kick"
        ? `<path d="M${cx},${cy - s} L${cx + s},${cy} L${cx},${cy + s} L${cx - s},${cy} Z" fill="${color}"/>`
        : `<path d="M${cx},${cy - s} L${cx + s},${cy + s * 0.75} L${cx - s},${cy + s * 0.75} Z" fill="${color}"/>`;
    })
    .join("");
  const bounds = p.clips
    .map(
      (c, i) =>
        (i ? `<line x1="${x(c.offset)}" x2="${x(c.offset)}" y1="${pad.t}" y2="${H - pad.b}" stroke="#9ca3af"/>` : "") +
        `<text x="${x(c.offset) + 3}" y="${H - 6}" font-size="10" fill="#6b7280">${esc(clipName(c.clip))}</text>`,
    )
    .join("");
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Initiative over time">
  ${[0, 0.5, 1].map((v) => `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y(v)}" y2="${y(v)}" stroke="#e8e8e6" ${v === 0.5 ? 'stroke-dasharray="3 3" stroke="#9ca3af"' : ""}/><text x="${pad.l - 4}" y="${y(v) + 3}" font-size="10" text-anchor="end" fill="#6b7280">${v}</text>`).join("")}
  ${bounds}
  ${segs.map((s) => `<polyline points="${s}" fill="none" stroke="${color}" stroke-width="2"/>`).join("")}
  ${marks}
</svg>`;
}

export function reportHtml(p: Profile, r: Report | undefined, title: string): string {
  const list = (items?: string[]) => (items?.length ? `<ul>${items.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>` : "<p>—</p>");
  const rows = METRIC_KEYS.map(
    (k) =>
      `<tr><td>${esc(METRICS[k].name)}</td><td class="num">${esc(fmtUnit(k, p.metrics[k]))}</td><td class="num">${p.metric_pct[k] == null ? "—" : "P" + p.metric_pct[k]}</td><td class="def">${esc(METRICS[k].help)}</td></tr>`,
  ).join("");
  const clipRows = p.clips
    .map(
      (c) =>
        `<tr><td>${esc(clipName(c.clip))}</td><td class="num">${c.active_s.toFixed(1)} s</td>${METRIC_KEYS.map((k) => `<td class="num">${esc(fmtUnit(k, c.metrics[k]))}</td>`).join("")}<td class="num">${c.counts.strikes}</td></tr>`,
    )
    .join("");
  const today = new Date().toISOString().slice(0, 10);

  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${esc(p.label)} · coach report</title>
<style>
  @page { size: A4; margin: 12mm; }
  body { font: 13px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color: #1f2328; background: #fff; max-width: 760px; margin: 24px auto; padding: 0 16px; }
  h1 { font-size: 22px; margin: 0; } h2 { font-size: 14px; margin: 18px 0 6px; text-transform: uppercase; letter-spacing: .04em; color: #6b7280; }
  .bar { height: 4px; background: ${colorOf(p)}; border-radius: 2px; margin: 8px 0 12px; }
  .meta { color: #6b7280; } .headline { font-size: 16px; font-weight: 600; margin: 8px 0 4px; }
  table { border-collapse: collapse; width: 100%; } th, td { text-align: left; padding: 4px 6px; border-bottom: 1px solid #e8e8e6; vertical-align: top; }
  th { font-size: 11px; color: #6b7280; font-weight: 600; } .num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .def { color: #6b7280; font-size: 11px; } .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  ul { margin: 4px 0; padding-left: 18px; } li { margin: 2px 0; }
  .limits { margin-top: 16px; padding: 8px 12px; border: 1px solid #e8e8e6; border-radius: 6px; background: #fafaf8; font-size: 11px; color: #4b5563; }
  .small { font-size: 11px; color: #6b7280; }
</style></head><body>
<h1>${esc(p.label)}</h1>
<div class="meta">${esc(title)} · ${esc(ageGroup(p.age_group))} · ${esc(p.belt)} belt · ${p.n_bouts} bout(s), ${p.n_clips} clip(s), ${p.active_s} s live · reliability ${p.reliability.toFixed(2)} · ${esc(p.confidence)} confidence · cohort n = ${p.cohort_n}</div>
<div class="bar"></div>
${r?.headline ? `<div class="headline">${esc(r.headline)}</div>` : ""}
${r?.summary ? `<p>${esc(r.summary)}</p>` : ""}

<h2>Metrics</h2>
<table><tr><th>Metric</th><th class="num">Value</th><th class="num">Cohort</th><th>Definition</th></tr>${rows}
<tr><td>Skill / standing / gap</td><td class="num">P${p.skill_pct ?? "—"} / P${p.standing_pct ?? "—"}</td><td class="num">${p.gap == null ? "—" : (p.gap > 0 ? "+" : "") + p.gap}</td><td class="def">gap = skill_pct − standing_pct. standing = ${esc(p.standing_basis ?? "bout points share")}.</td></tr></table>
<p class="small">${p.counts.strikes} strikes detected (${p.counts.punches} punches, ${p.counts.kicks} kicks, estimates)${p.counts.reached != null ? `, ${p.counts.reached} clearly reached the opponent` : ""}${p.counts.knee_raises ? ` · ${p.counts.knee_raises} knee raises without a strike (possible feints)` : ""} · ${p.counts.advances} forward attacks · ${p.counts.exchanges_initiated} exchanges initiated · score ${p.points_for}–${p.points_against}</p>

<h2>Per clip</h2>
<table><tr><th>Clip</th><th class="num">Live</th>${METRIC_KEYS.map((k) => `<th class="num">${esc(METRICS[k].name)}</th>`).join("")}<th class="num">Strikes</th></tr>${clipRows}</table>

<h2>Initiative over time</h2>
${timelineSvg(p)}
<p class="small">Above 0.5 = pressing forward. Clips back to back; gaps = paused or idle. Markers at the top: ▲ punch, ◆ kick.</p>
${p.events.length ? `<p class="small">Strikes: ${p.events.map((e) => `${esc(clipName(e.clip ?? ""))} ${e.clip_t?.toFixed(1)} s ${esc(e.type)} (${esc(limbName(e.limb))})${e.reached ? " reached" : ""}`).join(" · ")}</p>` : ""}

${
  r
    ? `<h2>Coach report</h2>
<div class="cols"><div><b>Strengths</b>${list(r.strengths)}</div><div><b>Weaknesses</b>${list(r.weaknesses)}</div></div>
<b>Focus next</b>${list(r.focus)}
${r.drills?.length ? `<b>Drills</b>${list(r.drills)}` : ""}
${r.caveats?.length ? `<p class="small">Caveats: ${esc(r.caveats.join(" "))}</p>` : ""}
<p class="small">Written by ${esc(r.model ?? "an LLM")} from the metrics only (no video was sent).</p>`
    : `<p class="small">No coach report available for this athlete.</p>`
}

<div class="limits"><b>Known limitations</b><ol>${limitations(p.cohort_n).map((l) => `<li>${esc(l)}</li>`).join("")}</ol></div>
<p class="small">Generated ${today} by the Kumite Analysis dashboard. All processing local.</p>
</body></html>`;
}

export function downloadReport(p: Profile, r: Report | undefined, title: string) {
  const blob = new Blob([reportHtml(p, r, title)], { type: "text/html;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${p.id}_report.html`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
