/* ============================================================================
   Veridian Studio — app shell, routing, global state, sidebar.
   Exposes: App, VDCtx
   ========================================================================== */
const VDCtx = React.createContext(null);

function StorageBadge() {
  const s = window.VD.storage;
  return (
    <div style={{ padding: "10px 12px", borderTop: "1px solid var(--line)", display: "flex", gap: 9, alignItems: "center" }}>
      <div style={{ width: 28, height: 28, borderRadius: 7, background: "var(--bg-2)", border: "1px solid var(--line-2)", display: "grid", placeItems: "center", color: "#34d39a", flexShrink: 0 }}>
        <Icon name="database" size={15} />
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 11.5, fontWeight: 600, display: "flex", alignItems: "center", gap: 6 }}>
          {s.engine}
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#34d39a", boxShadow: "0 0 6px #34d39a" }} />
        </div>
        <div className="mono" style={{ fontSize: 10, color: "var(--tx-2)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.host}</div>
        <div className="mono" style={{ fontSize: 9.5, color: "var(--tx-3)", marginTop: 1 }}>api: {(window.VERIDIAN_CONFIG && window.VERIDIAN_CONFIG.backend) || "mock"} · {(window.VERIDIAN_CONFIG && window.VERIDIAN_CONFIG.baseUrl) || ""}</div>
      </div>
    </div>
  );
}

function NavItem({ icon, label, active, onClick, badge }) {
  return (
    <button onClick={onClick} style={{
      display: "flex", alignItems: "center", gap: 10, width: "100%",
      padding: "8px 11px", borderRadius: 7, textAlign: "left",
      color: active ? "var(--tx-0)" : "var(--tx-1)",
      background: active ? "var(--accent-dim)" : "transparent",
      fontWeight: active ? 600 : 500, fontSize: 13, position: "relative",
      transition: "background .12s, color .12s",
    }}
      onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = "var(--bg-2)"; }}
      onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = "transparent"; }}>
      {active && <span style={{ position: "absolute", left: -1, top: 7, bottom: 7, width: 3, borderRadius: 3, background: "var(--accent)" }} />}
      <Icon name={icon} size={17} stroke={active ? 1.9 : 1.6} />
      <span style={{ flex: 1 }}>{label}</span>
      {badge != null && <span className="chip tnum" style={{ padding: "1px 7px", fontSize: 10.5, borderColor: "var(--line)" }}>{badge}</span>}
    </button>
  );
}

function Sidebar() {
  const ctx = React.useContext(VDCtx);
  const { route, go } = ctx;
  const totalConflicts = window.VD.datasets.reduce((s, d) => s + d.conflicts, 0);
  return (
    <aside style={{ width: "var(--sidebar-w)", flexShrink: 0, background: "var(--bg-1)", borderRight: "1px solid var(--line)", display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "16px 14px 14px", display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ width: 30, height: 30, borderRadius: 8, background: "linear-gradient(135deg,var(--accent),#8aa0ff)", display: "grid", placeItems: "center", color: "#07090d", boxShadow: "0 4px 14px rgba(91,124,250,.4)" }}>
          <Icon name="box" size={17} stroke={2} />
        </div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 14.5, letterSpacing: "-0.01em" }}>Veridian</div>
          <div style={{ fontSize: 10, color: "var(--tx-2)", letterSpacing: "0.06em", textTransform: "uppercase" }}>Studio</div>
        </div>
      </div>

      <div style={{ padding: "4px 10px", display: "flex", flexDirection: "column", gap: 2 }}>
        <div style={{ fontSize: 10, color: "var(--tx-3)", textTransform: "uppercase", letterSpacing: "0.08em", padding: "8px 11px 4px", fontWeight: 600 }}>Workspace</div>
        <NavItem icon="folder" label="Datasets" active={route.name === "datasets" || route.name === "grid" || route.name === "canvas" || route.name === "audio"} onClick={() => go({ name: "datasets" })} />
        <NavItem icon="upload" label="New source" active={route.name === "ingest"} onClick={() => go({ name: "ingest" })} />
        <NavItem icon="cpu" label="Models & Versions" active={route.name === "models"} onClick={() => go({ name: "models" })} />
        <NavItem icon="trend" label="Regression review" active={route.name === "regression"} onClick={() => go({ name: "regression" })} />
        <NavItem icon="dag" label="Lineage" active={route.name === "lineage"} onClick={() => go({ name: "lineage" })} />
        <NavItem icon="alert" label="Review Queue" active={route.name === "queue"} onClick={() => go({ name: "queue" })} badge={totalConflicts} />
      </div>

      <div style={{ padding: "4px 10px", display: "flex", flexDirection: "column", gap: 2, marginTop: 4 }}>
        <div style={{ fontSize: 10, color: "var(--tx-3)", textTransform: "uppercase", letterSpacing: "0.08em", padding: "8px 11px 4px", fontWeight: 600 }}>Datasets</div>
        {window.VD.datasets.map((d) => (
          <button key={d.id} onClick={() => go({ name: "grid", datasetId: d.id })}
            style={{ display: "flex", alignItems: "center", gap: 9, padding: "7px 11px", borderRadius: 7, textAlign: "left", color: route.datasetId === d.id && route.name !== "datasets" ? "var(--tx-0)" : "var(--tx-1)", background: route.datasetId === d.id && route.name !== "datasets" ? "var(--bg-2)" : "transparent", fontSize: 12.5 }}
            onMouseEnter={(e) => { e.currentTarget.style.background = "var(--bg-2)"; }}
            onMouseLeave={(e) => { if (!(route.datasetId === d.id && route.name !== "datasets")) e.currentTarget.style.background = "transparent"; }}>
            <span style={{ width: 7, height: 7, borderRadius: 2, background: vhelp.modelColor(d.models[0] === "fd" ? "fd" : d.models[0]), flexShrink: 0 }} />
            <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.name.split(" — ")[0]}</span>
            <span className="mono tnum" style={{ fontSize: 10.5, color: "var(--tx-3)" }}>{d.count}</span>
          </button>
        ))}
      </div>

      <div style={{ flex: 1 }} />
      <StorageBadge />
    </aside>
  );
}

function LoadingScreen({ label }) {
  return (
    <div style={{ display: "grid", placeItems: "center", height: "100vh", background: "var(--bg-0)", color: "var(--tx-2)" }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12 }}>
        <div className="spin" style={{ width: 30, height: 30, borderRadius: "50%", border: "3px solid var(--line-2)", borderTopColor: "var(--accent)" }} />
        <div style={{ fontSize: 13 }}>{label || "Loading…"}</div>
      </div>
    </div>
  );
}

function App() {
  const [route, setRoute] = React.useState({ name: "datasets" });
  const [reviews, setReviews] = React.useState({});           // objId -> {verdict, box, ts}
  const [t, setTweak] = useTweaks(window.TWEAK_DEFAULTS);
  const histRef = React.useRef([]);
  const rest = (window.VERIDIAN_CONFIG && window.VERIDIAN_CONFIG.backend) === "rest";
  const [hydrated, setHydrated] = React.useState(!rest);      // mock: ready immediately
  const [dataVer, bumpData] = React.useReducer((x) => x + 1, 0);  // re-render when VD fills

  const go = React.useCallback((r) => {
    setRoute((prev) => { histRef.current.push(prev); return r; });
  }, []);
  const back = React.useCallback(() => {
    const prev = histRef.current.pop();
    if (prev) setRoute(prev);
  }, []);
  const setReview = React.useCallback((id, patch) => {
    setReviews((prev) => ({ ...prev, [id]: { ...(prev[id] || {}), ...patch, ts: Date.now() } }));
    // persist through the backend seam (optimistic — UI already updated)
    try { window.VeridianAPI && window.VeridianAPI.saveReview(id, patch); } catch (e) { /* offline: keep optimistic state */ }
  }, []);

  // hydrate the client cache from the backend on boot (no-op in mock mode)
  React.useEffect(() => {
    let alive = true;
    (async () => {
      try { if (window.VeridianAPI && window.VeridianAPI.hydrate) await window.VeridianAPI.hydrate(); }
      catch (e) { console.warn("hydrate failed", e); }
      if (alive) setHydrated(true);
    })();
    return () => { alive = false; };
  }, []);

  // lazily load a dataset's images / a single image's detections on navigation
  React.useEffect(() => {
    if (!hydrated || !rest) return;
    const { name, datasetId, imageId } = route;
    if (!datasetId || !(name === "grid" || name === "canvas" || name === "audio")) return;
    let alive = true;
    (async () => {
      try {
        if (window.VD.ensureDataset) await window.VD.ensureDataset(datasetId);
        if (name === "canvas" && imageId && window.VD.ensureImage) await window.VD.ensureImage(datasetId, imageId);
      } catch (e) { console.warn("ensure failed", e); }
      if (alive) bumpData();
    })();
    return () => { alive = false; };
  }, [route, hydrated, rest]);

  const ctx = { route, go, back, reviews, setReview, t, setTweak };

  const Screen = {
    datasets: window.DatasetsScreen,
    grid: window.GridScreen,
    canvas: window.CanvasScreen,
    models: window.ModelsScreen,
    ingest: window.IngestScreen,
    queue: window.QueueScreen,
    audio: window.AudioScreen,
    regression: window.RegressionScreen,
    lineage: window.LineageScreen,
  }[route.name] || window.DatasetsScreen;

  const immersive = route.name === "canvas" || route.name === "audio";

  // readiness gate (REST only): the synchronous screens need their data present
  const ds = route.datasetId ? window.VD.getDataset(route.datasetId) : null;
  let ready = true;
  if (rest) {
    if (!hydrated) ready = false;
    else if ((route.name === "grid" || route.name === "audio") && route.datasetId) ready = !!(ds && ds._loaded);
    else if (route.name === "canvas" && route.datasetId) {
      const im = ds && ds.images ? ds.images.find((x) => x.id === route.imageId) : null;
      ready = !!(ds && ds._loaded && im && im._loaded);
    }
  }

  if (!ready) {
    return (
      <VDCtx.Provider value={ctx}>
        {!hydrated ? <LoadingScreen label="Connecting to Veridian…" /> : <LoadingScreen label="Loading dataset…" />}
      </VDCtx.Provider>
    );
  }

  return (
    <VDCtx.Provider value={ctx}>
      <div data-ver={dataVer} style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
        {!immersive && <Sidebar />}
        <main style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", overflow: "hidden", background: "var(--bg-0)" }}>
          <ScreenBoundary key={route.name + (route.datasetId || "") + (route.imageId || "")}>
            <Screen />
          </ScreenBoundary>
        </main>
      </div>
      <window.VeridianTweaks t={t} setTweak={setTweak} />
    </VDCtx.Provider>
  );
}

// Catch render errors in a screen and show them instead of blanking the app.
class ScreenBoundary extends React.Component {
  constructor(props) { super(props); this.state = { err: null }; }
  static getDerivedStateFromError(err) { return { err }; }
  componentDidCatch(err, info) { console.error("[veridian] screen render error", err, info); }
  render() {
    if (!this.state.err) return this.props.children;
    const e = this.state.err;
    return (
      <div className="scroll" style={{ padding: 24, color: "var(--tx-0)" }}>
        <div style={{ fontWeight: 700, color: "var(--conflict)", marginBottom: 8, fontSize: 14 }}>This screen hit an error.</div>
        <pre className="mono" style={{ fontSize: 11.5, whiteSpace: "pre-wrap", color: "var(--tx-1)", background: "var(--bg-1)", border: "1px solid var(--line)", borderRadius: 8, padding: 12, maxHeight: "60vh", overflow: "auto" }}>{String((e && e.stack) || e)}</pre>
        <button className="btn" style={{ marginTop: 12 }} onClick={() => location.reload()}>Reload</button>
      </div>
    );
  }
}

Object.assign(window, { App, VDCtx });
