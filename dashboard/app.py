"""Karate match analysis dashboard.

    .venv\\Scripts\\streamlit run dashboard/app.py

Reads data/<run>/profiles.json (scripts/build_profiles.py), data/<run>/features/<clip>.json
(src/analysis/features.py), data/<run>/reports/<athlete>.json (coach reports) and the annotated
videos in videos/output/<run>/.
"""
import json
import subprocess
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
AKA_C, AO_C = "#D9483B", "#2F6FD6"   # validated pair (light surface)
INK, MUTED, GRID, NEUTRAL = "#1f2328", "#6b7280", "#e8e8e6", "#9ca3af"
METRICS = {
    "strike_rate": ("Strike rate", "/min", "Punches + kicks per minute of live fighting"),
    "initiative_share": ("Initiative", "", "Share of exchanges this athlete started"),
    "range_control": ("Range control", "", "Share of moving time this athlete dictated distance (pressing forward)"),
    "tempo": ("Tempo", "/10 s", "Strikes + forward attacks per 10 s of live fighting"),
}
CONF_BADGE = {"high": "🟢 high", "medium": "🟡 medium", "low": "🔴 low"}

st.set_page_config(page_title="Kumite Analysis", page_icon="🥋", layout="wide")


# ------------------------------------------------------------------ data
def runs():
    return sorted([p.parent.name for p in DATA.glob("*/profiles.json")], reverse=True)


@st.cache_data
def load_json(path: str, mtime: float):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jload(p: Path):
    return load_json(str(p), p.stat().st_mtime) if p.exists() else None


def web_video(run: str, clip: str):
    """Annotated videos are mp4v (not browser-playable); transcode once to H.264."""
    src = ROOT / "videos" / "output" / run / f"{clip}_analyzed.mp4"
    dst = src.with_name(f"{clip}_web.mp4")
    if src.exists() and (not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime):
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)
    return dst if dst.exists() else None


def color_of(p):
    slots = {c["slot"] for c in p.get("clips", [])}
    return AKA_C if slots == {"aka"} else AO_C if slots == {"ao"} else NEUTRAL


def fmt(k, v):
    if v is None:
        return "—"
    return f"{v:.0%}" if k in ("initiative_share", "range_control") else f"{v:.1f}"


def style(fig, h=320):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=30, b=10),
                      font=dict(color=INK, size=13), hoverlabel=dict(bgcolor="white"),
                      legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


# ------------------------------------------------------------------ pages
def scouting_board(profiles):
    st.subheader("Scouting board")
    st.caption("Above the diagonal = the video shows more than the result does (positive gap). "
               "`standing_pct` is currently **bout points share** (placeholder definition).")
    groups = sorted({p["age_group"] for p in profiles})
    g = st.radio("Cohort", groups, horizontal=True) if len(groups) > 1 else groups[0]
    group = [p for p in profiles if p["age_group"] == g]

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color=NEUTRAL, dash="dot", width=1))
        for p in group:
            if p["skill_pct"] is None or p["standing_pct"] is None:
                continue
            fig.add_trace(go.Scatter(
                x=[p["standing_pct"]], y=[p["skill_pct"]], mode="markers+text", name=p["label"],
                text=[p["label"]], textposition="top center", textfont=dict(color=INK),
                marker=dict(size=10 + 10 * {"high": 2, "medium": 1, "low": 0}[p["confidence"]],
                            color=color_of(p), line=dict(color="white", width=2)),
                hovertemplate=f"<b>{p['label']}</b><br>skill P%{{y}} · standing P%{{x}}<br>gap {p['gap']:+d}"
                              f"<br>confidence {p['confidence']}<extra></extra>"))
        fig.update_xaxes(title="standing percentile (result)", range=[-5, 105])
        fig.update_yaxes(title="skill percentile (video)", range=[-5, 110])
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 420), width="stretch")
    with c2:
        df = pd.DataFrame([{
            "athlete": p["label"], "belt": p["belt"], "bouts": p["n_bouts"],
            "skill %": p["skill_pct"], "standing %": p["standing_pct"], "gap": p["gap"],
            "confidence": p["confidence"], "reliability": p["reliability"],
            **{METRICS[k][0]: fmt(k, p["metrics"][k]) for k in METRICS}} for p in group])
        st.dataframe(df.sort_values("gap", ascending=False), hide_index=True, width="stretch")
        st.caption(f"Cohort `{g}`: n = {len(group)}. With n < 5, percentiles move in big steps — "
                   "add more athletes before reading them as rankings.")


def metric_dots(p, group):
    """One row per metric: cohort athletes as grey dots, this athlete highlighted (raw values)."""
    fig = go.Figure()
    names = [METRICS[k][0] for k in METRICS]
    for q in group:
        me = q["id"] == p["id"]
        xs = [q["metric_pct"].get(k) for k in METRICS]
        fig.add_trace(go.Scatter(
            x=xs, y=names, mode="markers", name=q["label"], showlegend=me,
            marker=dict(size=16 if me else 10, color=color_of(q) if me else NEUTRAL, line=dict(color="white", width=2)),
            customdata=[fmt(k, q["metrics"][k]) for k in METRICS],
            hovertemplate=f"<b>{q['label']}</b><br>%{{y}}: %{{customdata}} (P%{{x}})<extra></extra>"))
    fig.update_xaxes(title="cohort percentile", range=[-5, 105])
    fig.update_yaxes(autorange="reversed")
    return style(fig, 260)


def timeline_fig(p):
    tl = pd.DataFrame(p["timeline"])
    ev = pd.DataFrame(p["events"])
    c = color_of(p)
    fig = go.Figure()
    fig.add_hline(y=0.5, line=dict(color=NEUTRAL, width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=tl["t"], y=tl["initiative"], mode="lines", name="initiative",
                             line=dict(color=c, width=2), connectgaps=False,
                             hovertemplate="t=%{x:.1f}s · initiative %{y:.2f}<extra></extra>"))
    if len(ev):
        y = [tl.loc[(tl["t"] - t).abs().idxmin(), "initiative"] for t in ev["t"]]
        y = [0.5 if v is None or pd.isna(v) else v for v in y]
        fig.add_trace(go.Scatter(
            x=ev["t"], y=y, mode="markers", name="strike",
            marker=dict(symbol=["triangle-up" if t == "punch" else "diamond" for t in ev["type"]],
                        size=12, color=c, line=dict(color="white", width=2)),
            customdata=ev[["type", "limb", "clip", "clip_t"]].values,
            hovertemplate="%{customdata[0]} (%{customdata[1]})<br>%{customdata[2]} @ %{customdata[3]:.1f}s<extra></extra>"))
    for cl in p.get("clips", [])[1:]:
        fig.add_vline(x=cl["offset"], line=dict(color=GRID, width=2))
    fig.update_yaxes(title="initiative (0.5 = even)", range=[-0.02, 1.05])
    fig.update_xaxes(title="time (s, clips back to back) — gaps = paused / idle")
    return style(fig, 300)


def coach_panel(run, p):
    rep = jload(DATA / run / "reports" / f"{p['id']}.json")
    st.markdown("#### 🧠 Coach report")
    if not rep:
        st.info("No report yet — run `python scripts/coach_report.py --run " + run + "`.")
        return
    st.markdown(f"**{rep.get('headline', '')}**")
    if rep.get("summary"):
        st.write(rep["summary"])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**✅ Strengths**")
        for s in rep.get("strengths", []):
            st.markdown(f"- {s}")
    with c2:
        st.markdown("**⚠️ Weaknesses**")
        for s in rep.get("weaknesses", []):
            st.markdown(f"- {s}")
    st.markdown("**🎯 Focus next**")
    for s in rep.get("focus", []):
        st.markdown(f"- {s}")
    if rep.get("drills"):
        with st.expander("Suggested drills"):
            for s in rep["drills"]:
                st.markdown(f"- {s}")
    if rep.get("caveats"):
        st.caption("Caveats: " + " ".join(rep["caveats"]))
    st.caption(f"Generated by {rep.get('model', 'LLM')} from the metrics above only (no video sent).")


def athlete_card(run, profiles):
    labels = {p["label"]: p for p in profiles}
    p = labels[st.selectbox("Athlete", list(labels))]
    group = [q for q in profiles if q["age_group"] == p["age_group"]]
    st.markdown(f"### {p['label']}  ·  `{p['age_group']}` · {p['belt']} belt")
    st.caption(f"{p['n_bouts']} bout(s), {p['n_clips']} clip(s), {p['active_s']} s of live fighting · "
               f"reliability {p['reliability']:.2f} · confidence {CONF_BADGE[p['confidence']]} · cohort n = {p['cohort_n']}")

    cols = st.columns(6)
    for col, k in zip(cols, METRICS):
        name, unit, help_ = METRICS[k]
        pct = p["metric_pct"].get(k)
        col.metric(name, fmt(k, p["metrics"][k]) + unit, None if pct is None else f"P{pct} in cohort",
                   delta_color="off", help=help_)
    cols[4].metric("Skill", f"P{p['skill_pct']}", help="Percentile of the mean metric percentile within the cohort")
    cols[5].metric("Gap", f"{p['gap']:+d}" if p["gap"] is not None else "—",
                   help="skill_pct − standing_pct. Positive = fights better than the result shows.")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("#### Initiative over time")
        st.plotly_chart(timeline_fig(p), width="stretch")
        st.markdown("#### Against the cohort")
        st.plotly_chart(metric_dots(p, group), width="stretch")
    with c2:
        coach_panel(run, p)
        st.markdown("#### Strikes")
        ev = p["events"]
        if ev:
            pick = st.selectbox("Jump to", [f"{e['clip']} @ {e['clip_t']:.1f}s — {e['type']} ({e['limb']})" for e in ev])
            e = ev[[f"{e['clip']} @ {e['clip_t']:.1f}s — {e['type']} ({e['limb']})" for e in ev].index(pick)]
            vid = web_video(run, e["clip"])
            if vid:
                st.video(str(vid), start_time=max(0, int(e["clip_t"] - 1)))
        st.caption(f"{p['counts']['strikes']} strikes ({p['counts']['punches']} punches, {p['counts']['kicks']} kicks) · "
                   f"{p['counts']['advances']} forward attacks · punch/kick split is unreliable at 360p.")
    with st.expander("Raw JSON (team format)"):
        st.json({k: p[k] for k in ["id", "age_group", "belt", "n_bouts", "reliability", "metrics", "skill_pct",
                                   "standing_pct", "gap", "confidence", "cohort_n"]}
                | {"timeline": p["timeline"][:5] + [{"…": f"{len(p['timeline'])} points"}]})


def bout_view(run, profiles):
    clips = sorted({c["clip"] for p in profiles for c in p.get("clips", [])})
    clip = st.selectbox("Clip", clips)
    f = jload(DATA / run / "features" / f"{clip}.json")
    if not f:
        st.warning("No features for this clip.")
        return
    who = {c["slot"]: p["label"] for p in profiles for c in p.get("clips", []) if c["clip"] == clip}
    st.caption(f"{f['duration_s']} s clip · {f['active_s']} s live · {f['paused_s']} s score graphic · {f['idle_s']} s idle "
               f"· {len(f['exchanges'])} exchanges")
    c1, c2 = st.columns([1, 1])
    with c1:
        vid = web_video(run, clip)
        if vid:
            st.video(str(vid))
        else:
            st.info("Annotated video not found.")
    with c2:
        tl = pd.DataFrame(f["athletes"]["aka"]["timeline"])
        fig = go.Figure()
        for x in f["exchanges"]:
            fig.add_vrect(x0=x["t0"] - 0.2, x1=x["t1"] + 0.2, fillcolor=GRID, opacity=0.6, line_width=0, layer="below")
        fig.add_hline(y=0.5, line=dict(color=NEUTRAL, width=1, dash="dot"))
        fig.add_trace(go.Scatter(x=tl["t"], y=tl["initiative"], mode="lines", name=f"{who.get('aka', 'AKA')} pressing ↑",
                                 line=dict(color=AKA_C, width=2), hovertemplate="t=%{x:.1f}s · %{y:.2f}<extra></extra>"))
        for slot, c, y in (("aka", AKA_C, 1.02), ("ao", AO_C, -0.02)):
            ev = f["athletes"][slot]["events"]
            if ev:
                fig.add_trace(go.Scatter(
                    x=[e["t"] for e in ev], y=[y] * len(ev), mode="markers", name=f"{who.get(slot, slot)} strikes",
                    marker=dict(symbol=["triangle-up" if e["type"] == "punch" else "diamond" for e in ev], size=12, color=c,
                                line=dict(color="white", width=2)),
                    hovertemplate="%{x:.1f}s<extra></extra>"))
        fig.update_yaxes(title=f"↑ {who.get('aka', 'AKA')} presses · {who.get('ao', 'AO')} presses ↓", range=[-0.08, 1.08])
        fig.update_xaxes(title="time (s) — grey bands = exchanges", range=[0, f["duration_s"]])
        st.plotly_chart(style(fig, 380), width="stretch")
        rows = [{"": who.get(s, s), **{METRICS[k][0]: fmt(k, f["athletes"][s]["metrics"][k]) for k in METRICS},
                 "strikes": f["athletes"][s]["counts"]["strikes"], "reliability": f["athletes"][s]["reliability"]}
                for s in ("aka", "ao")]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(f["exchanges"]).rename(columns={"t0": "start s", "t1": "end s"}),
                     hide_index=True, width="stretch")


# ------------------------------------------------------------------ main
all_runs = runs()
st.sidebar.title("🥋 Kumite Analysis")
if not all_runs:
    st.warning("No data/<run>/profiles.json yet. Run the pipeline, features and scripts/build_profiles.py.")
    st.stop()
run = st.sidebar.selectbox("Run", all_runs)
page = st.sidebar.radio("View", ["Athlete card", "Bout view", "Scouting board"])
profiles = jload(DATA / run / "profiles.json")
st.sidebar.caption("Pose: YOLO11-pose + ByteTrack · metrics from keypoints, normalized by body size · "
                   "all processing local; the LLM only sees aggregated numbers.")
if page == "Scouting board":
    scouting_board(profiles)
elif page == "Athlete card":
    athlete_card(run, profiles)
else:
    bout_view(run, profiles)
