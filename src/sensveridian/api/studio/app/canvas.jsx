/* ============================================================================
   Veridian Studio — Annotation canvas (HERO).
   Screen orchestration + zoomable viewport + pred/GT overlay + distance sweep
   + keyboard triage. Right panel/filmstrip live in inspector.jsx.
   Exposes: CanvasScreen
   ========================================================================== */

const IMG_AR = 0.625; // 800/1280

/* derive a distance-swept copy of the objects: boxes shrink with range and
   weak predictions drop out (model degradation at distance). */
function computeSweep(image, deltaFt) {
  if (!deltaFt) return image.objects;
  const s = image.d0_ft / (image.d0_ft + deltaFt); // pinhole scale
  const cx = 0.5, cy = 0.42;
  const scaleBox = (b) => b && [cx + (b[0] - cx) * s, cy + (b[1] - cy) * s, b[2] * s, b[3] * s];
  return image.objects.map((o) => {
    const pred = scaleBox(o.pred);
    const gt = scaleBox(o.gt);
    let state = o.state, conf = o.conf;
    if (pred) {
      const pxH = pred[3] * image.h * 1; // projected height in px
      conf = +Math.max(0.05, o.conf * (0.55 + 0.45 * s)).toFixed(3); // confidence decays with range
      if (pxH < 14 || conf < 0.28) { state = o.gt ? "miss" : "fp"; }
    }
    const dropped = pred && (pred[3] * image.h < 14 || conf < 0.28);
    return { ...o, pred: dropped ? null : pred, gt, conf, state: dropped && o.gt ? "miss" : state, iou: dropped ? 0 : o.iou, _swept: !!deltaFt };
  });
}

function CanvasScreen() {
  const ctx = React.useContext(VDCtx);
  const { t, setTweak, reviews, setReview } = ctx;
  const d = window.VD.getDataset(ctx.route.datasetId);
  const idx = d.images.findIndex((im) => im.id === ctx.route.imageId);
  const [curIdx, setCurIdx] = React.useState(idx < 0 ? 0 : idx);
  const image = d.images[curIdx];

  const [selectedId, setSelectedId] = React.useState(null);
  const [hoverId, setHoverId] = React.useState(null);
  const [tool, setTool] = React.useState("select");
  const [view, setView] = React.useState({ scale: 1, tx: 0, ty: 0 });
  const [delta, setDelta] = React.useState(0);
  const [drawing, setDrawing] = React.useState(null);
  const [extraBoxes, setExtraBoxes] = React.useState([]);
  const vpRef = React.useRef(null);

  const maxRange = 24;
  const objects = React.useMemo(() => computeSweep(image, delta), [image, delta]);

  // visible objects after layer + confidence filtering
  const conf = t.confThreshold;
  const visObjects = objects.filter((o) => t.models[o.model] !== false);

  const conflictObjs = visObjects.filter((o) => o.state !== "match");
  const liveAgreement = (() => {
    const matched = visObjects.filter((o) => o.state === "match").length;
    const c = visObjects.filter((o) => o.state !== "match").length;
    return matched + c === 0 ? 1 : matched / (matched + c);
  })();

  const goImg = React.useCallback((ni) => {
    const n = (ni + d.images.length) % d.images.length;
    setCurIdx(n); setSelectedId(null); setDelta(0); setExtraBoxes([]);
    setView({ scale: 1, tx: 0, ty: 0 });
  }, [d]);

  const selectableIds = visObjects.map((o) => o.id);
  const stepSel = (dir) => {
    if (!selectableIds.length) return;
    const i = selectableIds.indexOf(selectedId);
    const n = i < 0 ? 0 : (i + dir + selectableIds.length) % selectableIds.length;
    setSelectedId(selectableIds[n]);
  };

  // keyboard triage
  React.useEffect(() => {
    const h = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      const k = e.key.toLowerCase();
      if (k === "j" || e.key === "ArrowDown") { e.preventDefault(); stepSel(1); }
      else if (k === "k" || e.key === "ArrowUp") { e.preventDefault(); stepSel(-1); }
      else if (e.key === "ArrowRight") { e.preventDefault(); goImg(curIdx + 1); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); goImg(curIdx - 1); }
      else if (k === "a" && selectedId) setReview(selectedId, { verdict: "accepted" });
      else if (k === "x" && selectedId) setReview(selectedId, { verdict: "rejected" });
      else if (k === "e" && selectedId) { setTool("select"); }
      else if (k === "v") setTool("select");
      else if (k === "b") setTool("box");
      else if (k === "p") setTool("polygon");
      else if (k === "h") setTool("pan");
      else if (k === "escape") setSelectedId(null);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [selectedId, curIdx, selectableIds.join(",")]);

  // zoom on wheel
  const onWheel = (e) => {
    if (!e.ctrlKey && !e.metaKey && Math.abs(e.deltaY) < 50 && tool !== "pan") { /* allow */ }
    e.preventDefault();
    const f = e.deltaY < 0 ? 1.12 : 0.89;
    setView((v) => ({ ...v, scale: Math.max(1, Math.min(6, v.scale * f)) }));
  };

  // pan / draw drag
  const onDown = (e) => {
    if (tool === "pan" || e.button === 1 || e.altKey) {
      const start = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty };
      const mv = (ev) => setView((v) => ({ ...v, tx: start.tx + (ev.clientX - start.x), ty: start.ty + (ev.clientY - start.y) }));
      const up = () => { window.removeEventListener("mousemove", mv); window.removeEventListener("mouseup", up); };
      window.addEventListener("mousemove", mv); window.addEventListener("mouseup", up);
      return;
    }
    if (tool === "box") {
      const rect = vpRef.current.getBoundingClientRect();
      const nx = (e.clientX - rect.left) / rect.width;
      const ny = (e.clientY - rect.top) / rect.height;
      setDrawing({ x0: nx, y0: ny, x1: nx, y1: ny });
      const mv = (ev) => {
        const x1 = (ev.clientX - rect.left) / rect.width, y1 = (ev.clientY - rect.top) / rect.height;
        setDrawing((dn) => dn && { ...dn, x1, y1 });
      };
      const up = () => {
        window.removeEventListener("mousemove", mv); window.removeEventListener("mouseup", up);
        setDrawing((dn) => {
          if (dn) {
            const x = Math.min(dn.x0, dn.x1), y = Math.min(dn.y0, dn.y1);
            const w = Math.abs(dn.x1 - dn.x0), hh = Math.abs(dn.y1 - dn.y0);
            if (w > 0.02 && hh > 0.02) {
              const nb = { id: "new_" + Date.now(), cls: "car", model: d.models[0], gt: [x, y, w, hh], pred: null, conf: 0, state: "miss", iou: 0, _new: true };
              setExtraBoxes((b) => [...b, nb]);
              setSelectedId(nb.id);
            }
          }
          return null;
        });
        setTool("select");
      };
      window.addEventListener("mousemove", mv); window.addEventListener("mouseup", up);
    } else {
      setSelectedId(null);
    }
  };

  const allObjects = [...visObjects, ...extraBoxes];
  const selObj = allObjects.find((o) => o.id === selectedId);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "var(--bg-canvas)" }}>
      {/* top bar */}
      <header style={{ height: 50, flexShrink: 0, borderBottom: "1px solid var(--line)", display: "flex", alignItems: "center", padding: "0 14px", gap: 12, background: "var(--bg-1)" }}>
        <button className="btn ghost sm" onClick={() => ctx.go({ name: "grid", datasetId: d.id })}><Icon name="chevL" size={15} />Grid</button>
        <div style={{ width: 1, height: 22, background: "var(--line)" }} />
        <div style={{ display: "flex", flexDirection: "column", lineHeight: 1.25, minWidth: 0 }}>
          <span style={{ fontWeight: 600, fontSize: 13 }}>{d.name}</span>
          <span className="mono" style={{ fontSize: 10.5, color: "var(--tx-2)" }}>{vhelp.shortId(image.id, 16)}{image.augmented ? " · aug" : ""}</span>
        </div>
        <div style={{ flex: 1 }} />
        <RunCompare image={image} />
        <div style={{ width: 1, height: 22, background: "var(--line)" }} />
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <button className="btn ghost icon" onClick={() => goImg(curIdx - 1)} title="Previous (←)"><Icon name="chevL" size={16} /></button>
          <span className="tnum mono" style={{ fontSize: 12, color: "var(--tx-1)", minWidth: 54, textAlign: "center" }}>{curIdx + 1} / {d.images.length}</span>
          <button className="btn ghost icon" onClick={() => goImg(curIdx + 1)} title="Next (→)"><Icon name="chevR" size={16} /></button>
        </div>
        <button className="btn sm" onClick={() => setReview("img:" + image.id, { verdict: image.status === "flagged" ? null : "flagged" })} style={{ color: reviews["img:" + image.id]?.verdict === "flagged" ? "var(--gt)" : undefined }}>
          <Icon name="flag" size={14} />Flag
        </button>
        <button className="btn primary sm" onClick={() => { allObjects.forEach((o) => setReview(o.id, { verdict: "accepted" })); }}><Icon name="check" size={14} />Verify frame</button>
      </header>

      {/* body */}
      <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
        <window.Filmstrip dataset={d} curIdx={curIdx} onPick={goImg} reviews={reviews} />

        {/* center viewport */}
        <div style={{ flex: 1, position: "relative", overflow: "hidden", minWidth: 0 }}>
          <div ref={vpRef} onWheel={onWheel} onMouseDown={onDown}
            style={{ position: "absolute", inset: 0, cursor: tool === "pan" ? "grab" : tool === "box" ? "crosshair" : "default", display: "grid", placeItems: "center", padding: 24 }}>
            <div style={{ position: "relative", width: `min(100%, calc((100vh - 200px) * ${(image.w && image.h ? image.w / image.h : 1.6).toFixed(4)}))`, aspectRatio: `${image.w || 1280}/${image.h || 800}`, transform: `translate(${view.tx}px,${view.ty}px) scale(${view.scale})`, transformOrigin: "center", boxShadow: "0 12px 60px rgba(0,0,0,.6)", borderRadius: 4, overflow: "hidden", outline: "1px solid var(--line-2)" }}>
              <Scene image={image} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} />
              <BoxLayer objects={allObjects} t={t} conf={conf} selectedId={selectedId} hoverId={hoverId}
                onSelect={setSelectedId} onHover={setHoverId} reviews={reviews} drawing={drawing} />
            </div>
          </div>

          {/* floating toolbar */}
          <div style={{ position: "absolute", top: 14, left: 14, display: "flex", flexDirection: "column", gap: 4, background: "rgba(15,19,26,.92)", border: "1px solid var(--line-2)", borderRadius: 10, padding: 5, backdropFilter: "blur(6px)" }}>
            {[["select", "cursor", "Select (V)"], ["box", "box", "Draw box (B)"], ["polygon", "polygon", "Polygon mask (P)"], ["pan", "hand", "Pan (H)"]].map(([id, ic, tip]) => (
              <button key={id} title={tip} onClick={() => setTool(id)} className="btn icon" style={{ background: tool === id ? "var(--accent-dim)" : "transparent", border: "1px solid " + (tool === id ? "var(--accent)" : "transparent"), color: tool === id ? "var(--accent-2)" : "var(--tx-1)" }}>
                <Icon name={ic} size={17} />
              </button>
            ))}
            <div style={{ height: 1, background: "var(--line)", margin: "2px 3px" }} />
            <button className="btn icon" title="Zoom in" onClick={() => setView((v) => ({ ...v, scale: Math.min(6, v.scale * 1.2) }))} style={{ background: "transparent", color: "var(--tx-1)" }}><Icon name="zoomIn" size={17} /></button>
            <button className="btn icon" title="Zoom out" onClick={() => setView((v) => ({ ...v, scale: Math.max(1, v.scale / 1.2) }))} style={{ background: "transparent", color: "var(--tx-1)" }}><Icon name="zoomOut" size={17} /></button>
            <button className="btn icon" title="Reset view" onClick={() => setView({ scale: 1, tx: 0, ty: 0 })} style={{ background: "transparent", color: "var(--tx-1)" }}><Icon name="maximize" size={16} /></button>
          </div>

          {/* legend */}
          <div style={{ position: "absolute", top: 14, right: 14, display: "flex", gap: 12, background: "rgba(15,19,26,.92)", border: "1px solid var(--line-2)", borderRadius: 9, padding: "7px 12px", backdropFilter: "blur(6px)", fontSize: 11.5 }}>
            <Legend c="var(--pred)" label="Prediction" dash={false} />
            <Legend c="var(--gt)" label="Ground-truth" dash />
            <div style={{ width: 1, background: "var(--line)" }} />
            <Legend c="var(--match)" label="Agree" />
            <Legend c="var(--conflict)" label="Conflict" />
          </div>

          {/* distance sweep + confidence dock */}
          <div style={{ position: "absolute", left: 14, right: 14, bottom: 14, display: "flex", gap: 12, alignItems: "stretch" }}>
            <div style={{ flex: 1, background: "rgba(15,19,26,.94)", border: "1px solid var(--line-2)", borderRadius: 10, padding: "10px 14px", backdropFilter: "blur(6px)", display: "flex", alignItems: "center", gap: 14 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 7, color: "var(--accent-2)" }}>
                <Icon name="ruler" size={16} /><span style={{ fontSize: 12, fontWeight: 600, color: "var(--tx-0)" }}>Distance sweep</span>
              </div>
              <input type="range" min={0} max={maxRange} step={0.5} value={delta} onChange={(e) => setDelta(+e.target.value)} style={{ flex: 1, accentColor: "var(--accent)" }} />
              <span className="mono tnum" style={{ fontSize: 12, color: "var(--tx-0)", minWidth: 116, textAlign: "right" }}>
                {(image.d0_ft + delta).toFixed(1)} ft <span style={{ color: "var(--tx-3)" }}>(d₀ {image.d0_ft})</span>
              </span>
              {delta > 0 && <button className="btn sm ghost" onClick={() => setDelta(0)}><Icon name="x" size={12} />reset</button>}
            </div>
            <div style={{ background: "rgba(15,19,26,.94)", border: "1px solid var(--line-2)", borderRadius: 10, padding: "10px 14px", backdropFilter: "blur(6px)", display: "flex", alignItems: "center", gap: 12, minWidth: 250 }}>
              <span style={{ fontSize: 12, color: "var(--tx-2)" }}>conf ≥</span>
              <input type="range" min={0} max={0.95} step={0.01} value={conf} onChange={(e) => setTweak("confThreshold", +e.target.value)} style={{ flex: 1, accentColor: "var(--pred)" }} />
              <span className="mono tnum" style={{ fontSize: 12, color: "var(--pred)", minWidth: 34 }}>{conf.toFixed(2)}</span>
            </div>
          </div>

          {/* live agreement pill */}
          <div style={{ position: "absolute", left: "50%", top: 14, transform: "translateX(-50%)", display: "flex", gap: 10, alignItems: "center", background: "rgba(15,19,26,.92)", border: "1px solid var(--line-2)", borderRadius: 999, padding: "5px 14px", backdropFilter: "blur(6px)", fontSize: 12 }}>
            <span style={{ color: "var(--tx-2)" }}>frame agreement</span>
            <span className="tnum" style={{ fontWeight: 700, color: liveAgreement >= 0.85 ? "var(--match)" : liveAgreement >= 0.6 ? "var(--gt)" : "var(--conflict)" }}>{vhelp.pct(liveAgreement)}</span>
            {conflictObjs.length > 0 && <span className="chip" style={{ borderColor: "var(--conflict)", color: "var(--conflict)", padding: "0 8px" }}>{conflictObjs.length} to review</span>}
          </div>
        </div>

        <window.Inspector image={image} objects={allObjects} selObj={selObj} selectedId={selectedId}
          onSelect={setSelectedId} onHover={setHoverId} reviews={reviews} setReview={setReview}
          t={t} setTweak={setTweak} dataset={d} delta={delta} />
      </div>
    </div>
  );
}

function Legend({ c, label, dash }) {
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--tx-1)" }}>
      <svg width="16" height="10"><rect x="1" y="1" width="14" height="8" fill="none" stroke={c} strokeWidth="1.6" strokeDasharray={dash ? "3 2" : "0"} /></svg>{label}
    </span>
  );
}

function RunCompare({ image }) {
  const ctx = React.useContext(VDCtx);
  const d = window.VD.getDataset(ctx.route.datasetId);
  const m = window.VD.models.find((x) => x.id === ((d && d.models && d.models[0]) || "amod"));
  const [open, setOpen] = React.useState(false);
  // tolerate a model with a single (or zero) version history
  if (!m || !m.versions || !m.versions.length) return null;
  const cur = m.versions[0], cmp = m.versions[1] || m.versions[0];
  return (
    <div style={{ position: "relative" }}>
      <button className="btn sm" onClick={() => setOpen((o) => !o)}>
        <Icon name="compare" size={14} /><span className="mono">v{cur.version}</span>
        <span style={{ color: "var(--tx-3)" }}>vs</span><span className="mono" style={{ color: "var(--tx-2)" }}>v{cmp.version}</span>
        <Icon name="chevD" size={13} />
      </button>
      {open && (
        <div className="card pop" style={{ position: "absolute", top: 38, right: 0, width: 270, zIndex: 30, boxShadow: "var(--shadow-pop)", padding: 12 }}>
          <div style={{ fontSize: 11, color: "var(--tx-2)", marginBottom: 8 }}>Comparing two runs of <b style={{ color: "var(--tx-0)" }}>{m.short}</b> on this frame. Overlay shows the active version; switch baseline below.</div>
          {m.versions.slice(0, 4).map((v, i) => (
            <div key={v.version} style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 8px", borderRadius: 6, background: i === 0 ? "var(--accent-dim)" : "transparent" }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: i === 0 ? "var(--accent)" : "var(--tx-3)" }} />
              <span className="mono" style={{ fontSize: 12, flex: 1 }}>v{v.version}</span>
              <span className="mono" style={{ fontSize: 10.5, color: "var(--tx-2)" }}>{v.date}</span>
              <span className="tnum" style={{ fontSize: 11, color: "var(--match)" }}>{vhelp.pct(v.metrics.agreement)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function BoxLayer({ objects, t, conf, selectedId, hoverId, onSelect, onHover, reviews, drawing }) {
  const SY = IMG_AR;
  return (
    <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
      {/* SVG: masks, qr points, conflict connectors */}
      <svg viewBox={`0 0 1 ${SY}`} preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", overflow: "visible" }}>
        {objects.map((o) => {
          const out = [];
          if (t.showMasks && o.mask && o.pred && o.conf >= conf) {
            out.push(<polygon key={o.id + "m"} points={o.mask.map((p) => `${p[0]},${(p[1] * SY).toFixed(4)}`).join(" ")}
              fill="var(--pred-soft)" stroke="var(--pred)" strokeWidth="0.002" vectorEffect="non-scaling-stroke" opacity="0.85" />);
          }
          // connector between pred and gt centers when both exist (diff vector)
          if (o.pred && o.gt && t.showPred && t.showGT && o.state !== "match" && o.conf >= conf) {
            const pc = [o.pred[0] + o.pred[2] / 2, (o.pred[1] + o.pred[3] / 2) * SY];
            const gc = [o.gt[0] + o.gt[2] / 2, (o.gt[1] + o.gt[3] / 2) * SY];
            out.push(<line key={o.id + "c"} x1={pc[0]} y1={pc[1]} x2={gc[0]} y2={gc[1]} stroke="var(--conflict)" strokeWidth="0.0015" strokeDasharray="0.006 0.004" vectorEffect="non-scaling-stroke" opacity="0.8" />);
          }
          // qr quad points
          if (o.cls === "qr" && o.pred && t.showPred && o.conf >= conf) {
            const b = o.pred;
            const corners = [[b[0], b[1]], [b[0] + b[2], b[1]], [b[0] + b[2], b[1] + b[3]], [b[0], b[1] + b[3]]];
            corners.forEach((c, ci) => out.push(<rect key={o.id + "q" + ci} x={c[0] - 0.006} y={c[1] * SY - 0.006 * SY} width="0.012" height={0.012 * SY} fill="var(--pred)" stroke="#07090d" strokeWidth="0.001" vectorEffect="non-scaling-stroke" />));
          }
          return out;
        })}
      </svg>

      {/* HTML box divs (raw normalized %) */}
      {objects.map((o) => {
        const verdict = reviews[o.id]?.verdict;
        const isSel = o.id === selectedId, isHov = o.id === hoverId;
        const conflict = o.state !== "match";
        const showPred = t.showPred && o.pred && o.conf >= conf;
        const showGT = t.showGT && o.gt;
        const els = [];
        const pct = (b) => ({ left: b[0] * 100 + "%", top: b[1] * 100 + "%", width: b[2] * 100 + "%", height: b[3] * 100 + "%" });

        if (showGT) {
          els.push(<div key="gt" style={{ position: "absolute", ...pct(o.gt), border: "1.5px dashed var(--gt)", borderRadius: 2, boxSizing: "border-box", opacity: isSel || isHov ? 1 : 0.85 }} />);
        }
        if (showPred) {
          els.push(<div key="pred" onMouseDown={(e) => { e.stopPropagation(); onSelect(o.id); }} onMouseEnter={() => onHover(o.id)} onMouseLeave={() => onHover(null)}
            style={{ position: "absolute", ...pct(o.pred), border: `2px solid var(--pred)`, borderRadius: 2, boxSizing: "border-box", pointerEvents: "auto", cursor: "pointer", boxShadow: conflict ? "0 0 0 1px var(--conflict), 0 0 14px var(--conflict-soft)" : isSel ? "0 0 0 2px var(--accent)" : "none", background: isSel ? "rgba(91,124,250,.08)" : isHov ? "rgba(34,211,238,.06)" : "transparent" }} />);
        }
        // a single hit/label anchor box (prefer pred, else gt)
        const anchor = o.pred || o.gt;
        if (!anchor) return null;
        if ((showPred || showGT) && t.showLabels) {
          const col = o.pred ? "var(--pred)" : "var(--gt)";
          const label = window.VD.CLASSES[o.cls].label + (o.pred ? " " + o.conf.toFixed(2) : "");
          els.push(
            <div key="lbl" onMouseDown={(e) => { e.stopPropagation(); onSelect(o.id); }}
              style={{ position: "absolute", left: anchor[0] * 100 + "%", top: anchor[1] * 100 + "%", transform: "translateY(-100%)", pointerEvents: "auto", cursor: "pointer", display: "flex", gap: 0, alignItems: "stretch", whiteSpace: "nowrap" }}>
              <span className="mono" style={{ fontSize: 10, fontWeight: 600, padding: "1px 5px", background: col, color: "#07090d", borderRadius: "3px 0 0 0", lineHeight: 1.5 }}>{label}</span>
              {conflict && <span style={{ fontSize: 9.5, fontWeight: 700, padding: "1px 5px", background: "var(--conflict)", color: "#07090d", lineHeight: 1.6 }}>{vhelp.stateShort(o.state)}</span>}
              {verdict === "accepted" && <span style={{ fontSize: 9.5, padding: "1px 4px", background: "var(--match)", color: "#07090d", display: "grid", placeItems: "center" }}>✓</span>}
              {verdict === "rejected" && <span style={{ fontSize: 9.5, padding: "1px 4px", background: "var(--tx-2)", color: "#07090d", display: "grid", placeItems: "center" }}>✕</span>}
            </div>
          );
        }
        // selection handles
        if (isSel) {
          const b = o.pred || o.gt;
          [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]].forEach(([hx, hy], hi) => {
            els.push(<div key={"h" + hi} style={{ position: "absolute", left: `calc(${(b[0] + b[2] * hx) * 100}% - 4px)`, top: `calc(${(b[1] + b[3] * hy) * 100}% - 4px)`, width: 8, height: 8, background: "var(--accent)", border: "1.5px solid #07090d", borderRadius: 2, pointerEvents: "auto", cursor: "nwse-resize" }} />);
          });
        }
        return <React.Fragment key={o.id}>{els}</React.Fragment>;
      })}

      {/* in-progress draw */}
      {drawing && (
        <div style={{ position: "absolute", left: Math.min(drawing.x0, drawing.x1) * 100 + "%", top: Math.min(drawing.y0, drawing.y1) * 100 + "%", width: Math.abs(drawing.x1 - drawing.x0) * 100 + "%", height: Math.abs(drawing.y1 - drawing.y0) * 100 + "%", border: "2px dashed var(--accent)", background: "var(--accent-dim)", borderRadius: 2 }} />
      )}
    </div>
  );
}

window.CanvasScreen = CanvasScreen;
