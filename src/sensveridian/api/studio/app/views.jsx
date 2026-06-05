/* ============================================================================
   Veridian Studio — extra dataset views.
   EmbeddingMap (#3): 2D projection of detections, conflicts cluster.
   TracksView (#4): temporal track timelines + flicker detection + propagate.
   Exposes: EmbeddingMap, TracksView
   ========================================================================== */

function hashf(str) { let h = 2166136261; for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); } return (h >>> 0) / 4294967296; }
const CLASS_HUE = { car: 220, truck: 255, person: 150, bicycle: 180, traffic_light: 40, stop_sign: 0, face: 190, qr: 285 };
function classColor(cls) { const h = CLASS_HUE[cls] != null ? CLASS_HUE[cls] : (hashf(cls) * 360); return `hsl(${h} 70% 62%)`; }

/* ----------------------------- EMBEDDING MAP ----------------------------- */
function EmbeddingMap({ dataset, onOpen }) {
  const [colorBy, setColorBy] = React.useState("agreement");
  const [sel, setSel] = React.useState([]); // indices
  const [hover, setHover] = React.useState(null);

  const pts = React.useMemo(() => {
    const classes = [...new Set(dataset.images.flatMap((im) => im.objects.map((o) => o.cls)))];
    const anchors = {};
    classes.forEach((c, i) => { const a = (i / classes.length) * Math.PI * 2; anchors[c] = [50 + Math.cos(a) * 26, 46 + Math.sin(a) * 26]; });
    const out = [];
    dataset.images.forEach((im) => im.objects.forEach((o) => {
      const r1 = hashf(o.id), r2 = hashf(o.id + "y");
      const conflict = o.state !== "match";
      const [ax, ay] = anchors[o.cls] || [50, 46];
      const spread = conflict ? 13 : 7;
      let x = ax + (r1 - 0.5) * spread * 2;
      let y = ay + (r2 - 0.5) * spread * 2;
      if (conflict) { x = x * 0.7 + 50 * 0.3; y = y * 0.6 + 80 * 0.4; } // pull conflicts toward failure basin
      out.push({ im, o, x: Math.max(4, Math.min(96, x)), y: Math.max(6, Math.min(94, y)), conflict });
    }));
    return out;
  }, [dataset]);

  const colorOf = (p) => {
    if (colorBy === "agreement") return p.conflict ? "var(--conflict)" : "var(--match)";
    if (colorBy === "model") return vhelp.modelColor(p.o.model);
    return classColor(p.o.cls);
  };

  const clickPoint = (i) => {
    const p = pts[i];
    const near = pts.map((q, j) => ({ j, d: Math.hypot(q.x - p.x, q.y - p.y) })).filter((q) => q.d < 11).map((q) => q.j);
    setSel(near);
  };
  const selImages = [...new Map(sel.map((i) => [pts[i].im.id, pts[i].im])).values()];
  const conflictPts = pts.filter((p) => p.conflict).length;

  return (
    <div style={{ display: "flex", gap: 16, height: "100%" }}>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 12 }}>
          <div style={{ fontSize: 12.5, color: "var(--tx-2)" }}>Each point is a detection. Conflicts gravitate to the lower <b style={{ color: "var(--conflict)" }}>failure basin</b> — click a cluster to triage similar errors together.</div>
          <div style={{ flex: 1 }} />
          <span style={{ fontSize: 11, color: "var(--tx-2)" }}>color</span>
          <div style={{ display: "flex", background: "var(--bg-3)", borderRadius: 7, padding: 2, border: "1px solid var(--line-2)" }}>
            {[["agreement", "Agreement"], ["class", "Class"], ["model", "Model"]].map(([id, lab]) => (
              <button key={id} onClick={() => setColorBy(id)} className="btn sm" style={{ background: colorBy === id ? "var(--bg-1)" : "transparent", border: "none", color: colorBy === id ? "var(--tx-0)" : "var(--tx-2)" }}>{lab}</button>
            ))}
          </div>
        </div>
        <div className="card" style={{ flex: 1, position: "relative", background: "var(--bg-canvas)", overflow: "hidden" }}>
          {/* failure basin */}
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
            <defs><radialGradient id="basin" cx="50%" cy="80%" r="42%"><stop offset="0%" stopColor="var(--conflict)" stopOpacity="0.16" /><stop offset="100%" stopColor="var(--conflict)" stopOpacity="0" /></radialGradient></defs>
            <rect x="0" y="0" width="100" height="100" fill="url(#basin)" />
          </svg>
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
            onClick={() => setSel([])}>
            {pts.map((p, i) => {
              const on = sel.includes(i);
              return <circle key={i} cx={p.x} cy={p.y} r={on ? 1.5 : hover === i ? 1.4 : 1.0} fill={colorOf(p)} opacity={sel.length && !on ? 0.22 : 0.92}
                stroke={on ? "var(--tx-0)" : "none"} strokeWidth={on ? 0.4 : 0} vectorEffect="non-scaling-stroke" style={{ cursor: "pointer" }}
                onClick={(e) => { e.stopPropagation(); clickPoint(i); }} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} />;
            })}
          </svg>
          {hover != null && (
            <div style={{ position: "absolute", left: `calc(${pts[hover].x}% + 8px)`, top: `calc(${pts[hover].y}% + 8px)`, pointerEvents: "none", zIndex: 5 }}>
              <div className="card" style={{ width: 120, overflow: "hidden", boxShadow: "var(--shadow-pop)" }}>
                <div style={{ aspectRatio: "1280/800", position: "relative" }}><Scene image={pts[hover].im} style={{ width: "100%", height: "100%" }} /></div>
                <div style={{ padding: 6, fontSize: 10 }}><b>{window.VD.CLASSES[pts[hover].o.cls].label}</b> · {vhelp.stateLabel(pts[hover].o.state)}</div>
              </div>
            </div>
          )}
          <div style={{ position: "absolute", bottom: 10, left: 12, fontSize: 10.5, color: "var(--conflict)", fontWeight: 600 }}>failure basin · {conflictPts} conflicts</div>
        </div>
      </div>

      {/* cluster panel */}
      <div style={{ width: 250, flexShrink: 0, display: "flex", flexDirection: "column" }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 10 }}>{sel.length ? `Cluster · ${sel.length} detections` : "Select a cluster"}</div>
        {sel.length === 0 ? (
          <div style={{ fontSize: 11.5, color: "var(--tx-2)", lineHeight: 1.5 }}>Click any point to grab its neighborhood. Tight conflict clusters usually share a root cause — a lighting condition, a class, a range.</div>
        ) : (
          <>
            <button className="btn sm primary" style={{ justifyContent: "center", marginBottom: 10 }} onClick={() => selImages[0] && onOpen(selImages[0])}><Icon name="filter" size={13} />Triage cluster ({selImages.length} frames)</button>
            <div className="scroll" style={{ flex: 1, display: "flex", flexDirection: "column", gap: 7 }}>
              {selImages.slice(0, 20).map((im) => (
                <button key={im.id} onClick={() => onOpen(im)} className="card" style={{ padding: 6, display: "flex", gap: 8, alignItems: "center", textAlign: "left" }}>
                  <div style={{ width: 44, height: 28, borderRadius: 4, overflow: "hidden", flexShrink: 0 }}><Scene image={im} style={{ width: "100%", height: "100%" }} /></div>
                  <span className="mono" style={{ fontSize: 10.5, color: "var(--tx-2)", flex: 1 }}>{vhelp.shortId(im.id, 8)}</span>
                  <span className="tnum" style={{ fontSize: 10.5, color: im.conflicts ? "var(--conflict)" : "var(--match)" }}>{im.conflicts || "ok"}</span>
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/* ----------------------------- TRACKS VIEW ------------------------------- */
function TracksView({ dataset, onOpen }) {
  const [propagated, setPropagated] = React.useState({});
  const [toast, setToast] = React.useState(null);

  const tracks = React.useMemo(() => {
    const classes = [...new Set(dataset.images.flatMap((im) => im.objects.map((o) => o.cls)))];
    const frames = dataset.images;
    const out = [];
    classes.forEach((cls) => {
      // up to 2 track lanes per class
      const lanes = cls === "qr" || cls === "stop_sign" ? 1 : 2;
      for (let ln = 0; ln < lanes; ln++) {
        const cells = frames.map((im, fi) => {
          const objs = im.objects.filter((o) => o.cls === cls);
          const o = objs[ln];
          if (!o) return { status: "absent", im };
          const st = o.state === "match" ? "present" : o.state === "miss" ? "missed" : o.state === "fp" ? "fp" : "present";
          return { status: st, im, o };
        });
        // only keep tracks that actually appear
        if (cells.some((c) => c.status !== "absent")) {
          // flicker = present -> missed -> present transitions
          let flicker = 0;
          const seq = cells.map((c) => c.status);
          for (let i = 1; i < seq.length - 1; i++) if (seq[i] === "missed" && seq[i - 1] === "present" && seq.slice(i + 1).includes("present")) flicker++;
          out.push({ id: cls + "_" + ln, cls, cells, flicker, missed: seq.filter((s) => s === "missed").length });
        }
      }
    });
    return out.sort((a, b) => b.flicker - a.flicker);
  }, [dataset]);

  const flickering = tracks.filter((t) => t.flicker > 0).length;
  const propagate = (id) => { setPropagated((p) => ({ ...p, [id]: true })); const t = tracks.find((x) => x.id === id); setToast(`Propagated ${t.cls} track across ${t.missed} gap${t.missed !== 1 ? "s" : ""} — interpolated boxes queued for review.`); setTimeout(() => setToast(null), 3200); };

  const cellColor = { present: "var(--pred)", missed: "var(--conflict)", fp: "var(--gt)", absent: "transparent" };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 14 }}>
        <div style={{ fontSize: 12.5, color: "var(--tx-2)" }}>Detections as tracks across frames. <b style={{ color: "var(--conflict)" }}>Flicker</b> (present→missed→present) almost always means a model error, not a real disappearance — propagate to fill the gap.</div>
        <div style={{ flex: 1 }} />
        <span className="chip" style={{ borderColor: flickering ? "var(--conflict)" : "var(--line)", color: flickering ? "var(--conflict)" : "var(--tx-2)" }}><Icon name="alert" size={12} />{flickering} flickering track{flickering !== 1 ? "s" : ""}</span>
      </div>

      {/* legend */}
      <div style={{ display: "flex", gap: 14, marginBottom: 10, fontSize: 11, color: "var(--tx-2)" }}>
        {[["present", "detected"], ["missed", "missed (FN)"], ["fp", "false positive"]].map(([k, l]) => <span key={k} style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 11, height: 11, borderRadius: 3, background: cellColor[k] }} />{l}</span>)}
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 11, height: 11, borderRadius: 3, border: "1.5px dashed var(--pred)" }} />interpolated</span>
      </div>

      <div className="card scroll" style={{ flex: 1, padding: 0 }}>
        <div style={{ minWidth: dataset.images.length * 26 + 200 }}>
          {/* frame header */}
          <div style={{ display: "flex", position: "sticky", top: 0, background: "var(--bg-1)", borderBottom: "1px solid var(--line)", zIndex: 2 }}>
            <div style={{ width: 200, flexShrink: 0, padding: "8px 12px", fontSize: 10.5, color: "var(--tx-3)", textTransform: "uppercase", letterSpacing: ".06em", fontWeight: 600 }}>Track</div>
            <div style={{ display: "flex" }}>{dataset.images.map((im, i) => <div key={i} style={{ width: 26, textAlign: "center", fontSize: 9.5, color: "var(--tx-3)", padding: "8px 0" }} className="tnum">{i + 1}</div>)}</div>
          </div>
          {tracks.map((t) => {
            const isProp = propagated[t.id];
            return (
              <div key={t.id} style={{ display: "flex", alignItems: "center", borderBottom: "1px solid var(--line)" }}>
                <div style={{ width: 200, flexShrink: 0, padding: "7px 12px", display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ width: 9, height: 9, borderRadius: 2, background: classColor(t.cls), flexShrink: 0 }} />
                  <span style={{ fontSize: 12, fontWeight: 500, flex: 1 }}>{window.VD.CLASSES[t.cls].label}</span>
                  {t.flicker > 0 && !isProp && <span className="chip" style={{ padding: "0 6px", fontSize: 9.5, borderColor: "var(--conflict)", color: "var(--conflict)" }}>flicker {t.flicker}</span>}
                  {t.flicker > 0 && (isProp ? <span className="chip" style={{ padding: "0 6px", fontSize: 9.5, borderColor: "var(--match)", color: "var(--match)" }}>fixed</span> : <button className="btn sm" style={{ padding: "2px 7px", fontSize: 10 }} onClick={() => propagate(t.id)} title="Interpolate across gaps"><Icon name="spark" size={11} />fill</button>)}
                </div>
                <div style={{ display: "flex" }}>
                  {t.cells.map((c, i) => {
                    const interp = isProp && c.status === "missed";
                    return <button key={i} onClick={() => c.im && onOpen(c.im)} title={c.status} style={{ width: 26, height: 26, display: "grid", placeItems: "center", background: "transparent", cursor: c.status === "absent" ? "default" : "pointer" }}>
                      <span style={{ width: 16, height: 16, borderRadius: 4, background: interp ? "transparent" : cellColor[c.status], border: interp ? "1.5px dashed var(--pred)" : "none", opacity: c.status === "absent" ? 0.12 : 1, ...(c.status === "absent" ? { background: "var(--bg-3)" } : {}) }} />
                    </button>;
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {toast && <div className="pop" style={{ position: "fixed", bottom: 24, left: "50%", transform: "translateX(-50%)", background: "var(--bg-2)", border: "1px solid var(--match)", borderRadius: 10, padding: "11px 16px", fontSize: 12.5, display: "flex", alignItems: "center", gap: 9, boxShadow: "var(--shadow-pop)", zIndex: 50 }}><Icon name="check" size={15} style={{ color: "var(--match)" }} />{toast}</div>}
    </div>
  );
}

Object.assign(window, { EmbeddingMap, TracksView });
