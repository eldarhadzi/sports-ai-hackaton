import { Navigate, NavLink, Outlet, Route, Routes, useNavigate, useParams } from "react-router-dom";
import { useRuns } from "./lib/api";
import { Limitations, QueryState } from "./components/common";
import MatchOverview from "./pages/MatchOverview";
import AthleteCard from "./pages/AthleteCard";
import BoutView from "./pages/BoutView";
import ScoutingBoard from "./pages/ScoutingBoard";

function FirstRun() {
  const runs = useRuns();
  return (
    <QueryState q={runs}>
      {(list) =>
        list.length ? (
          <Navigate to={`/r/${list[0]}`} replace />
        ) : (
          <div className="page">
            <div className="notice">
              No <code>data/&lt;run&gt;/profiles.json</code> yet. Run the pipeline and{" "}
              <code>scripts/build_profiles.py</code> first.
            </div>
          </div>
        )
      }
    </QueryState>
  );
}

function Layout() {
  const { run = "" } = useParams();
  const runs = useRuns();
  const nav = useNavigate();
  const link = (to: string, label: string, end = false) => (
    <NavLink to={`/r/${run}${to}`} end={end} className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}>
      {label}
    </NavLink>
  );
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span aria-hidden>🥋</span> Kumite Analysis
        </div>
        {runs.data && runs.data.length > 1 && (
          <label className="field">
            <span>Run</span>
            <select value={run} onChange={(e) => nav(`/r/${e.target.value}`)}>
              {runs.data.map((r) => (
                <option key={r}>{r}</option>
              ))}
            </select>
          </label>
        )}
        <nav className="nav">
          {link("", "Match overview", true)}
          {link("/athlete", "Athlete card")}
          {link("/bout", "Bout view")}
          {link("/scouting", "Scouting board")}
        </nav>
        <Limitations compact />
        <p className="fineprint">
          Pose: YOLO11-pose + ByteTrack. Metrics come from keypoints, normalized by body size. All processing is
          local; the LLM only sees aggregated numbers.
        </p>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<FirstRun />} />
      <Route path="/r/:run" element={<Layout />}>
        <Route index element={<MatchOverview />} />
        <Route path="athlete/:id?" element={<AthleteCard />} />
        <Route path="bout/:clip?" element={<BoutView />} />
        <Route path="scouting" element={<ScoutingBoard />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
