/* ============================================================================
   Veridian Studio — Provenance / lineage DAG (idea #7).
   Source → frames/augmentations → model runs → verified GT → exports.
   Click a node to trace its full upstream+downstream lineage.
   Exposes: LineageScreen
   ========================================================================== */

const LIN_NODES = [
  { id: "s1", col: 0, type: "source", label: "ring_cam_driveway.mp4", sub: "184s · 1080p", meta: "video source · 368 frames sampled" },
  { id: "s2", col: 0, type: "source", label: "hwy_onramp_dashcam.mp4", sub: "240s · 1080p", meta: "video source · 480 frames sampled" },
  { id: "s3", col: 0, type: "source", label: "lobby_entrance.mov", sub: "96s · 720p", meta: "video source · 192 frames sampled" },
  { id: "s4", col: 0, type: "source", label: "forklift_aisle22.mp4", sub: "312s · 1440p", meta: "video source · 312 frames sampled" },
  { id: "a1", col: 0, type: "aug", label: "distance-sweep", sub: "5→30 ft · synthetic", meta: "ZoeDepth + SAM + LaMa augmentation" },

  { id: "d_street", col: 1, type: "dataset", label: "Street Scenes", sub: "28 frames", meta: "vision dataset · amod", ds: "street_scenes" },
  { id: "d_access", col: 1, type: "dataset", label: "Access — Faces", sub: "22 frames", meta: "vision dataset · fd + fr", ds: "access_faces" },
  { id: "d_qr", col: 1, type: "dataset", label: "Warehouse — QR", sub: "18 frames", meta: "vision dataset · qrcode", ds: "warehouse_qr" },
  { id: "d_dist", col: 1, type: "dataset", label: "Distance Eval", sub: "16 frames", meta: "vision dataset · multi", ds: "distance_eval" },

  { id: "r_amod", col: 2, type: "run", model: "amod", label: "AMOD v8.2.4", sub: "run: baseline", meta: "prediction layer · sha 9f3a…" },
  { id: "r_fd", col: 2, type: "run", model: "fd", label: "FD v8.1.4", sub: "run: baseline", meta: "prediction layer · sha 2c81…" },
  { id: "r_fr", col: 2, type: "run", model: "fr", label: "FR v8.1.4", sub: "run: baseline", meta: "prediction layer · sha b07e…" },
  { id: "r_qr", col: 2, type: "run", model: "qrcode", label: "QR v1.4.4", sub: "run: baseline", meta: "prediction layer · sha 41dd…" },

  { id: "v_street", col: 3, type: "verified", label: "Verified GT · Street", sub: "human-approved", meta: "ground-truth layer · 24 reviewers' edits" },
  { id: "v_access", col: 3, type: "verified", label: "Verified GT · Access", sub: "human-approved", meta: "ground-truth layer · golden set" },

  { id: "e_wide", col: 4, type: "export", label: "ground_truth_wide.parquet", sub: "exported 2026-05-30", meta: "downstream training set" },
  { id: "e_dist", col: 4, type: "export", label: "distance_eval.parquet", sub: "exported 2026-06-02", meta: "robustness eval set" },
];
const LIN_EDGES = [
  ["s1", "d_street"], ["s2", "d_street"], ["s3", "d_access"], ["s4", "d_qr"], ["a1", "d_dist"], ["s2", "a1"],
  ["d_street", "r_amod"], ["d_access", "r_fd"], ["d_access", "r_fr"], ["d_qr", "r_qr"], ["d_dist", "r_amod"], ["d_dist", "r_qr"],
  ["r_amod", "v_street"], ["r_fd", "v_access"], ["r_fr", "v_access"],
  ["v_street", "e_wide"], ["v_access", "e_wide"], ["r_amod", "e_dist"], ["d_dist", "e_dist"],
];

const LIN_STYLE = {
  source: { color: "var(--m-fd)", icon: "film" }, aug: { color: "var(--m-qr)", icon: "ruler" },
  dataset: { color: "var(--accent)", icon: "folder" }, run: { color: "var(--m-amod)", icon: "cpu" },
  verified: { color: "var(--match)", icon: "shield" }, export: { color: "var(--gt)", icon: "download" },
};

const COL_X = [30, 280, 540, 800, 1030];
const COL_LABELS = ["Sources", "Datasets (frames)", "Model runs", "Verified ground-truth", "Exports"];
const NODE_W = 196, NODE_H = 52, ROW_H = 74;

// Map a server lineage node ({id,type,label,meta}) to the layout shape the
// screen renders (col by type, model for run-node color, ds for the open-button).
const _LIN_COL = { source: 0, aug: 0, dataset: 1, run: 2, verified: 3, export: 4 };
function normalizeLineage(data) {
  const nodes = (data && data.nodes ? data.nodes : []).map((n) => {
    let model, ds;
    if (n.type === "run") { const p = String(n.id).split(":"); model = p[p.length - 1]; }
    if (n.type === "dataset" && String(n.id).startsWith("d:")) ds = String(n.id).slice(2);
    return { id: n.id, type: n.type, label: n.label, sub: n.meta || "", meta: n.meta || "",
             col: _LIN_COL[n.type] != null ? _LIN_COL[n.type] : 1, model, ds };
  });
  return { nodes, edges: (data && data.edges) || [] };
}

function LineageScreen() {
  const ctx = React.useContext(VDCtx);
  const [sel, setSel] = React.useState(null);
  const isRest = (window.VERIDIAN_CONFIG || {}).backend === "rest";
  const [data, setData] = React.useState({ nodes: LIN_NODES, edges: LIN_EDGES });

  React.useEffect(() => {
    let alive = true;
    if (isRest && window.VeridianAPI && window.VeridianAPI.getLineage) {
      window.VeridianAPI.getLineage()
        .then((d) => { if (alive && d && (d.nodes || []).length) setData(normalizeLineage(d)); })
        .catch((e) => console.warn("[veridian] lineage load failed", e));
    }
    return () => { alive = false; };
  }, [isRest]);

  const { nodes, H } = React.useMemo(() => {
    const cols = {}; data.nodes.forEach((n) => { (cols[n.col] = cols[n.col] || []).push(n); });
    const counts = Object.values(cols).map((c) => c.length);
    const maxCount = counts.length ? Math.max(...counts) : 0;
    const H = maxCount * ROW_H + 30;
    const mid = H / 2;
    const placed = {};
    Object.entries(cols).forEach(([col, list]) => {
      list.forEach((n, i) => {
        const y = mid + (i - (list.length - 1) / 2) * ROW_H - NODE_H / 2;
        placed[n.id] = { ...n, x: COL_X[col], y };
      });
    });
    return { nodes: placed, H };
  }, [data]);

  // adjacency for lineage highlight
  const lineage = React.useMemo(() => {
    if (!sel) return null;
    const up = {}, down = {};
    data.edges.forEach(([a, b]) => { (down[a] = down[a] || []).push(b); (up[b] = up[b] || []).push(a); });
    const set = new Set([sel]);
    const walk = (id, map) => { (map[id] || []).forEach((n) => { if (!set.has(n)) { set.add(n); walk(n, map); } }); };
    walk(sel, up); walk(sel, down);
    return set;
  }, [sel, data]);

  const edgeOn = (a, b) => !lineage || (lineage.has(a) && lineage.has(b));
  const nodeOn = (id) => !lineage || lineage.has(id);
  const selNode = sel ? nodes[sel] : null;
  const width = COL_X[4] + NODE_W + 30;

  return (
    <>
      <TopBar crumbs={[{ label: "Lineage" }]}>
        {sel && <button className="btn sm ghost" onClick={() => setSel(null)}><Icon name="x" size={13} />Clear trace</button>}
        <span className="chip" style={{ borderColor: "var(--line-2)" }}><Icon name="database" size={12} style={{ color: "var(--match)" }} />content-addressed · {window.VD.storage.engine}</span>
      </TopBar>

      <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
        <div className="scroll" style={{ flex: 1, padding: 0, overflow: "auto" }}>
          {/* column headers */}
          <div style={{ position: "relative", width, minWidth: "100%", height: 38, borderBottom: "1px solid var(--line)", background: "var(--bg-1)" }}>
            {COL_LABELS.map((l, i) => <div key={i} style={{ position: "absolute", left: COL_X[i], top: 11, width: NODE_W, fontSize: 10.5, fontWeight: 600, color: "var(--tx-3)", textTransform: "uppercase", letterSpacing: ".06em" }}>{l}</div>)}
          </div>

          <div style={{ position: "relative", width, minWidth: "100%", height: H, background: "radial-gradient(circle at 30% 20%, rgba(91,124,250,.04), transparent 60%)" }}>
            {/* edges */}
            <svg width={width} height={H} style={{ position: "absolute", inset: 0 }}>
              {data.edges.map(([a, b], i) => {
                const na = nodes[a], nb = nodes[b]; if (!na || !nb) return null;
                const x0 = na.x + NODE_W, y0 = na.y + NODE_H / 2, x1 = nb.x, y1 = nb.y + NODE_H / 2;
                const mx = (x0 + x1) / 2;
                const on = edgeOn(a, b);
                return <path key={i} d={`M ${x0} ${y0} C ${mx} ${y0}, ${mx} ${y1}, ${x1} ${y1}`} fill="none" stroke={on ? "var(--accent)" : "var(--line)"} strokeWidth={on && lineage ? 2 : 1.3} opacity={on ? (lineage ? 0.9 : 0.4) : 0.12} />;
              })}
            </svg>
            {/* nodes */}
            {Object.values(nodes).map((n) => {
              const st = LIN_STYLE[n.type];
              const color = n.type === "run" ? vhelp.modelColor(n.model) : st.color;
              const on = nodeOn(n.id);
              const isSel = sel === n.id;
              return (
                <button key={n.id} onClick={() => setSel(isSel ? null : n.id)}
                  style={{ position: "absolute", left: n.x, top: n.y, width: NODE_W, height: NODE_H, background: isSel ? "var(--bg-3)" : "var(--bg-1)", border: "1px solid " + (isSel ? color : "var(--line-2)"), borderLeft: "3px solid " + color, borderRadius: 8, display: "flex", alignItems: "center", gap: 9, padding: "0 11px", textAlign: "left", opacity: on ? 1 : 0.28, transition: "opacity .15s, border-color .15s", boxShadow: isSel ? "var(--shadow)" : "none" }}>
                  <div style={{ width: 26, height: 26, borderRadius: 6, background: color + "22", display: "grid", placeItems: "center", color, flexShrink: 0 }}><Icon name={st.icon} size={15} /></div>
                  <div style={{ minWidth: 0 }}>
                    <div className={n.type === "source" || n.type === "export" ? "mono" : ""} style={{ fontSize: n.type === "source" || n.type === "export" ? 11 : 12.5, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{n.label}</div>
                    <div style={{ fontSize: 10, color: "var(--tx-2)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{n.sub}</div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* detail */}
        <div style={{ width: 280, flexShrink: 0, borderLeft: "1px solid var(--line)", background: "var(--bg-1)", padding: 16 }}>
          {selNode ? (
            <>
              <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 12 }}>
                <div style={{ width: 32, height: 32, borderRadius: 8, background: (selNode.type === "run" ? vhelp.modelColor(selNode.model) : LIN_STYLE[selNode.type].color) + "22", display: "grid", placeItems: "center", color: selNode.type === "run" ? vhelp.modelColor(selNode.model) : LIN_STYLE[selNode.type].color }}><Icon name={LIN_STYLE[selNode.type].icon} size={17} /></div>
                <div style={{ minWidth: 0 }}><div style={{ fontSize: 13, fontWeight: 700, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{selNode.label}</div><div style={{ fontSize: 10.5, color: "var(--tx-2)", textTransform: "capitalize" }}>{selNode.type}</div></div>
              </div>
              <div style={{ fontSize: 12, color: "var(--tx-1)", lineHeight: 1.5, marginBottom: 14 }}>{selNode.meta}</div>
              <div style={{ fontSize: 11, color: "var(--tx-2)", marginBottom: 8 }}>Tracing <b style={{ color: "var(--accent-2)" }}>{lineage.size}</b> connected nodes — full upstream provenance and downstream impact.</div>
              {selNode.ds && <button className="btn primary sm" style={{ width: "100%", justifyContent: "center" }} onClick={() => ctx.go({ name: "grid", datasetId: selNode.ds })}>Open dataset<Icon name="arrowR" size={13} /></button>}
              {selNode.type === "run" && <button className="btn sm" style={{ width: "100%", justifyContent: "center", marginTop: 8 }} onClick={() => ctx.go({ name: "models" })}><Icon name="cpu" size={13} />View in model manager</button>}
            </>
          ) : (
            <div style={{ color: "var(--tx-2)", fontSize: 12.5, lineHeight: 1.55 }}>
              <Icon name="dag" size={22} style={{ color: "var(--tx-3)", marginBottom: 8 }} />
              <div>Click any node to trace its lineage — every source, run, and human edit that produced it, and everything downstream that depends on it.</div>
              <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 7 }}>
                {Object.entries(LIN_STYLE).map(([k, v]) => <span key={k} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11.5, textTransform: "capitalize" }}><span style={{ width: 10, height: 10, borderRadius: 3, background: v.color }} />{k}</span>)}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

window.LineageScreen = LineageScreen;
window.__LIN_NODES = LIN_NODES;
window.__LIN_EDGES = LIN_EDGES;
