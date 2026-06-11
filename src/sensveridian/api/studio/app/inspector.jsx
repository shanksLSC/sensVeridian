/* ============================================================================
   Veridian Studio — canvas side panels: Filmstrip (left) + Inspector (right).
   Exposes: Filmstrip, Inspector
   ========================================================================== */

function Filmstrip({ dataset, curIdx, onPick, reviews }) {
  const [filter, setFilter] = React.useState("all");
  const imgs = dataset.images.filter((im) => filter === "all" ? true : filter === "conflict" ? im.conflicts > 0 : im.status === filter);
  return (
    <div style={{ width: 158, flexShrink: 0, borderRight: "1px solid var(--line)", background: "var(--bg-1)", display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "8px 8px 6px", display: "flex", gap: 4, borderBottom: "1px solid var(--line)" }}>
        {[["all", "All"], ["conflict", "⚠"], ["unreviewed", "•"]].map(([id, lab]) => (
          <button key={id} onClick={() => setFilter(id)} className="btn sm" style={{ flex: 1, padding: "4px 0", justifyContent: "center", background: filter === id ? "var(--bg-3)" : "transparent", border: "1px solid " + (filter === id ? "var(--line-2)" : "transparent"), color: filter === id ? "var(--tx-0)" : "var(--tx-2)" }}>{lab}</button>
        ))}
      </div>
      <div className="scroll" style={{ flex: 1, padding: 8, display: "flex", flexDirection: "column", gap: 7 }}>
        {imgs.map((im) => {
          const active = im.idx === curIdx;
          return (
            <button key={im.id} onClick={() => onPick(im.idx)} style={{ position: "relative", borderRadius: 6, overflow: "hidden", border: "2px solid " + (active ? "var(--accent)" : "transparent"), padding: 0, background: "var(--bg-canvas)" }}>
              <div style={{ position: "relative", aspectRatio: "1280/800" }}>
                <Scene image={im} style={{ width: "100%", height: "100%", display: "block", opacity: active ? 1 : 0.78 }} />
                <svg viewBox="0 0 1 0.625" preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
                  {im.objects.map((o, i) => { const b = o.gt || o.pred; if (!b) return null; return <rect key={i} x={b[0]} y={b[1] * 0.625} width={b[2]} height={b[3] * 0.625} fill="none" stroke={o.state === "match" ? "var(--match)" : "var(--conflict)"} strokeWidth="0.004" vectorEffect="non-scaling-stroke" opacity="0.9" />; })}
                </svg>
              </div>
              <div style={{ position: "absolute", top: 3, left: 3, fontSize: 9.5, fontWeight: 700, fontFamily: "var(--mono)", color: "var(--tx-1)", background: "rgba(7,9,13,.7)", padding: "0 4px", borderRadius: 3 }}>{im.idx + 1}</div>
              {im.conflicts > 0 && <div style={{ position: "absolute", top: 3, right: 3, fontSize: 9.5, fontWeight: 700, color: "#07090d", background: "var(--conflict)", padding: "0 4px", borderRadius: 3 }}>{im.conflicts}</div>}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function ToggleRow({ label, on, onClick, color }) {
  return (
    <button onClick={onClick} style={{ display: "flex", alignItems: "center", gap: 9, width: "100%", padding: "6px 8px", borderRadius: 6, background: "transparent" }}
      onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-2)")} onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}>
      <span style={{ width: 30, height: 17, borderRadius: 999, background: on ? "var(--accent)" : "var(--bg-3)", border: "1px solid " + (on ? "var(--accent)" : "var(--line-2)"), position: "relative", flexShrink: 0, transition: "background .12s" }}>
        <span style={{ position: "absolute", top: 1, left: on ? 14 : 1, width: 13, height: 13, borderRadius: "50%", background: on ? "#07090d" : "var(--tx-2)", transition: "left .12s" }} />
      </span>
      {color && <span style={{ width: 10, height: 10, borderRadius: 2, background: color, flexShrink: 0 }} />}
      <span style={{ fontSize: 12.5, color: on ? "var(--tx-0)" : "var(--tx-2)" }}>{label}</span>
    </button>
  );
}

function CoordRow({ label, box, color }) {
  if (!box) return <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}><span style={{ color }}>{label}</span><span className="mono" style={{ color: "var(--tx-3)" }}>— none —</span></div>;
  const xyxy = [box[0], box[1], box[0] + box[2], box[1] + box[3]].map((v) => v.toFixed(3));
  return (
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, alignItems: "center" }}>
      <span style={{ color, fontWeight: 600 }}>{label}</span>
      <span className="mono tnum" style={{ color: "var(--tx-1)", fontSize: 10.5 }}>[{xyxy.join(", ")}]</span>
    </div>
  );
}

function DetRow({ o, selected, onSelect, onHover, reviews, setReview, dataset, t }) {
  const verdict = reviews[o.id]?.verdict;
  const auto = !verdict && t && t.autoAccept && o.state === "match" && o.conf >= t.trustThreshold;
  const conflict = o.state !== "match";
  const stCol = conflict ? "var(--conflict)" : "var(--match)";
  const m = window.VD.models.find((x) => x.id === o.model);
  const ver = m?.versions[0];
  return (
    <div style={{ borderBottom: "1px solid var(--line)", background: selected ? "var(--bg-2)" : "transparent" }}>
      <button onClick={() => onSelect(o.id)} onMouseEnter={() => onHover(o.id)} onMouseLeave={() => onHover(null)}
        style={{ display: "flex", alignItems: "center", gap: 9, width: "100%", padding: "8px 12px", textAlign: "left" }}>
        <span style={{ width: 8, height: 8, borderRadius: "50%", background: stCol, flexShrink: 0, boxShadow: conflict ? "0 0 6px var(--conflict)" : "none" }} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 12.5, fontWeight: 500, color: "var(--tx-0)", display: "flex", alignItems: "center", gap: 6 }}>
            {window.VD.CLASSES[o.cls].label}
            {o._new && <span className="chip" style={{ padding: "0 5px", fontSize: 9, borderColor: "var(--accent)", color: "var(--accent-2)" }}>new</span>}
          </div>
          <div className="mono" style={{ fontSize: 10, color: "var(--tx-2)" }}>{vhelp.stateLabel(o.state)} · IoU {o.iou.toFixed(2)}</div>
        </div>
        {o.pred && <span className="mono tnum" style={{ fontSize: 11, color: o.conf >= 0.5 ? "var(--pred)" : "var(--gt)", fontWeight: 600 }}>{o.conf.toFixed(2)}</span>}
        {verdict === "accepted" && <Icon name="check" size={14} style={{ color: "var(--match)" }} />}
        {verdict === "rejected" && <Icon name="x" size={14} style={{ color: "var(--tx-2)" }} />}
        {auto && <span title="auto-accepted (confidence-gated)" style={{ display: "flex", alignItems: "center", gap: 2, fontSize: 9.5, color: "var(--pred)", fontWeight: 700 }}><Icon name="spark" size={11} />auto</span>}
      </button>

      {selected && (
        <div className="fade" style={{ padding: "4px 12px 12px" }}>
          <div style={{ background: "var(--bg-1)", border: "1px solid var(--line)", borderRadius: 8, padding: 10, display: "flex", flexDirection: "column", gap: 7 }}>
            <CoordRow label="Prediction" box={o.pred} color="var(--pred)" />
            <CoordRow label="Ground-truth" box={o.gt} color="var(--gt)" />
            {o.identity && (
              <div style={{ borderTop: "1px solid var(--line)", paddingTop: 7, display: "flex", flexDirection: "column", gap: 4 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}><span style={{ color: "var(--gt)", fontWeight: 600 }}>Identity (GT)</span><span style={{ color: "var(--tx-0)" }}>{o.identity.gt} <span className="mono" style={{ color: "var(--tx-3)" }}>{o.identity.person_id}</span></span></div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}><span style={{ color: "var(--pred)", fontWeight: 600 }}>Match (FR)</span><span style={{ color: o.identity.pred === o.identity.gt ? "var(--match)" : "var(--conflict)" }}>{o.identity.pred || "no match"} · sim {o.identity.sim.toFixed(2)}</span></div>
              </div>
            )}
            {o.decoded && (
              <div style={{ borderTop: "1px solid var(--line)", paddingTop: 7, display: "flex", flexDirection: "column", gap: 4 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}><span style={{ color: "var(--gt)", fontWeight: 600 }}>Decoded (GT)</span><span className="mono" style={{ color: "var(--tx-0)" }}>{o.decoded.gt}</span></div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}><span style={{ color: "var(--pred)", fontWeight: 600 }}>Decoded (pred)</span><span className="mono" style={{ color: o.decoded.pred === o.decoded.gt ? "var(--match)" : "var(--conflict)" }}>{o.decoded.pred || "—"}</span></div>
              </div>
            )}
            <div style={{ borderTop: "1px solid var(--line)", paddingTop: 7, display: "flex", alignItems: "center", gap: 6, fontSize: 10.5, color: "var(--tx-2)" }}>
              <Icon name="cpu" size={12} style={{ color: vhelp.modelColor(o.model) }} />
              <span>{m?.short} v{ver?.version}</span>
              <span className="mono" style={{ color: "var(--tx-3)" }}>· {ver?.weights_sha.slice(0, 10)}</span>
            </div>
            <div style={{ display: "flex", gap: 6, marginTop: 2 }}>
              <button className="btn sm" style={{ flex: 1, justifyContent: "center", background: verdict === "accepted" ? "var(--match)" : "var(--bg-3)", color: verdict === "accepted" ? "#07090d" : "var(--tx-0)", borderColor: verdict === "accepted" ? "var(--match)" : "var(--line-2)" }} onClick={() => setReview(o.id, { verdict: "accepted" })}><Icon name="check" size={13} />Accept <span className="kbd" style={{ marginLeft: 2 }}>A</span></button>
              <button className="btn sm" style={{ flex: 1, justifyContent: "center", background: verdict === "rejected" ? "var(--conflict)" : "var(--bg-3)", color: verdict === "rejected" ? "#07090d" : "var(--tx-0)", borderColor: verdict === "rejected" ? "var(--conflict)" : "var(--line-2)" }} onClick={() => setReview(o.id, { verdict: "rejected" })}><Icon name="x" size={13} />Reject <span className="kbd" style={{ marginLeft: 2 }}>X</span></button>
              <button className="btn sm icon" title="Edit box (E)" onClick={() => setReview(o.id, { verdict: "edited" })}><Icon name="edit" size={13} /></button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Inspector({ image, objects, selObj, selectedId, onSelect, onHover, reviews, setReview, t, setTweak, dataset, delta }) {
  const counts = { match: 0, conflict: 0 };
  objects.forEach((o) => (o.state === "match" ? counts.match++ : counts.conflict++));
  const reviewedN = objects.filter((o) => reviews[o.id] || (t.autoAccept && o.state === "match" && o.conf >= t.trustThreshold)).length;
  const autoN = objects.filter((o) => !reviews[o.id] && t.autoAccept && o.state === "match" && o.conf >= t.trustThreshold).length;
  const toggleModel = (id) => setTweak("models", { ...t.models, [id]: t.models[id] === false ? true : false });

  return (
    <div style={{ width: 332, flexShrink: 0, borderLeft: "1px solid var(--line)", background: "var(--bg-1)", display: "flex", flexDirection: "column" }}>
      {/* header */}
      <div style={{ padding: "12px 14px", borderBottom: "1px solid var(--line)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
          <span style={{ fontSize: 13, fontWeight: 700, flex: 1 }}>Detections</span>
          <span className="chip" style={{ borderColor: "transparent", background: "var(--match-soft)", color: "var(--match)" }}>{counts.match} agree</span>
          {counts.conflict > 0 && <span className="chip" style={{ borderColor: "transparent", background: "var(--conflict-soft)", color: "var(--conflict)" }}>{counts.conflict} conflict</span>}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10.5, color: "var(--tx-2)", marginBottom: 4 }}>
          <span>{reviewedN} / {objects.length} reviewed{delta ? " · swept" : ""}</span>
          <span className="mono">{vhelp.shortId(image.id, 12)}</span>
        </div>
        <AgreeBar value={objects.length ? reviewedN / objects.length : 0} height={4} />
        {autoN > 0 && <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 6, fontSize: 10.5, color: "var(--pred)", background: "var(--pred-soft)", borderRadius: 6, padding: "5px 8px" }}><Icon name="spark" size={12} />{autoN} auto-accepted ≥ {t.trustThreshold.toFixed(2)} — review the rest</div>}
      </div>

      {/* layers */}
      <div style={{ padding: "10px 8px", borderBottom: "1px solid var(--line)" }}>
        <div style={{ fontSize: 10, color: "var(--tx-3)", textTransform: "uppercase", letterSpacing: ".07em", fontWeight: 600, padding: "0 6px 4px" }}>Layers</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr" }}>
          <ToggleRow label="Prediction" on={t.showPred} color="var(--pred)" onClick={() => setTweak("showPred", !t.showPred)} />
          <ToggleRow label="Ground-truth" on={t.showGT} color="var(--gt)" onClick={() => setTweak("showGT", !t.showGT)} />
          <ToggleRow label="Masks" on={t.showMasks} onClick={() => setTweak("showMasks", !t.showMasks)} />
          <ToggleRow label="Labels" on={t.showLabels} onClick={() => setTweak("showLabels", !t.showLabels)} />
        </div>
        <div style={{ fontSize: 10, color: "var(--tx-3)", textTransform: "uppercase", letterSpacing: ".07em", fontWeight: 600, padding: "8px 6px 6px" }}>Models</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, padding: "0 6px" }}>
          {dataset.models.map((id) => {
            const m = window.VD.models.find((x) => x.id === id); const on = t.models[id] !== false;
            return (
              <button key={id} onClick={() => toggleModel(id)} className="chip" style={{ cursor: "pointer", borderColor: on ? vhelp.modelColor(id) : "var(--line)", background: on ? "var(--bg-2)" : "transparent", opacity: on ? 1 : 0.5 }}>
                <span className="dot" style={{ background: vhelp.modelColor(id) }} />{m ? m.short : id}
                <Icon name={on ? "eye" : "eyeOff"} size={11} style={{ color: "var(--tx-2)" }} />
              </button>
            );
          })}
        </div>
      </div>

      {/* detection list */}
      <div className="scroll" style={{ flex: 1 }}>
        {objects.length === 0 && <div style={{ padding: 30, textAlign: "center", color: "var(--tx-2)", fontSize: 12 }}>No detections in visible layers.</div>}
        {objects.map((o) => (
          <DetRow key={o.id} o={o} selected={o.id === selectedId} onSelect={onSelect} onHover={onHover} reviews={reviews} setReview={setReview} dataset={dataset} t={t} />
        ))}
      </div>

      {/* keyboard hints */}
      <div style={{ padding: "9px 12px", borderTop: "1px solid var(--line)", display: "flex", flexWrap: "wrap", gap: "5px 12px", fontSize: 10.5, color: "var(--tx-2)" }}>
        {[["J / K", "detection"], ["← →", "frame"], ["A", "accept"], ["X", "reject"], ["E", "edit"], ["B", "box"]].map(([k, l]) => (
          <span key={k} style={{ display: "flex", alignItems: "center", gap: 5 }}><span className="kbd">{k}</span>{l}</span>
        ))}
      </div>
    </div>
  );
}

Object.assign(window, { Filmstrip, Inspector });
