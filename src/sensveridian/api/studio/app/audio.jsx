/* ============================================================================
   Veridian Studio — Audio modality (idea #5).
   AudioGrid: clip list for an audio dataset.
   AudioScreen: waveform + label-track timeline verify hero.
   Exposes: AudioGrid, AudioScreen
   ========================================================================== */

const AUDIO_COL = { speech: "#5b7cfa", music: "#c084fc", siren: "#fb5e7e", alarm: "#f5a524", keyword: "#34d39a", noise: "#6b7686", silence: "#3a4150" };
function segColor(label) { return AUDIO_COL[label] || "var(--tx-2)"; }
function fmtT(s) { const m = Math.floor(s / 60); const sec = (s % 60).toFixed(1); return m + ":" + (sec < 10 ? "0" : "") + sec; }

function Waveform({ wave, style, color = "var(--tx-2)", height = 120 }) {
  return (
    <svg viewBox={`0 0 ${wave.length} 100`} preserveAspectRatio="none" style={{ width: "100%", height, display: "block", ...style }}>
      {wave.map((a, i) => <rect key={i} x={i + 0.15} y={50 - a * 48} width={0.7} height={a * 96} fill={color} rx={0.3} />)}
    </svg>
  );
}

function MiniWave({ wave, segments }) {
  return (
    <div style={{ position: "relative", height: 38, background: "var(--bg-canvas)" }}>
      <Waveform wave={wave} color="var(--tx-3)" height={38} />
      <svg viewBox="0 0 1 1" preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
        {segments.map((s, i) => <rect key={i} x={s.start} y={0.78} width={s.end - s.start} height={0.18} fill={s.state === "match" ? "var(--match)" : "var(--conflict)"} opacity="0.9" />)}
      </svg>
    </div>
  );
}

function AudioGrid({ dataset }) {
  const ctx = React.useContext(VDCtx);
  const [filter, setFilter] = React.useState("all");
  const clips = dataset.clips.filter((c) => filter === "all" ? true : filter === "conflict" ? c.conflicts > 0 : c.status === filter);
  const open = (c) => ctx.go({ name: "audio", datasetId: dataset.id, clipId: c.id });
  return (
    <>
      <TopBar crumbs={[{ label: "Datasets", onClick: () => ctx.go({ name: "datasets" }) }, { label: dataset.name }]}>
        <ModelChip id="aed" size="sm" />
      </TopBar>
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "12px 18px", borderBottom: "1px solid var(--line)", background: "var(--bg-1)" }}>
        {[["all", "All"], ["conflict", "Disagreements"], ["unreviewed", "Unreviewed"], ["verified", "Verified"], ["flagged", "Flagged"]].map(([id, lab]) => {
          const n = dataset.clips.filter((c) => id === "all" ? true : id === "conflict" ? c.conflicts > 0 : c.status === id).length;
          return <button key={id} onClick={() => setFilter(id)} className="btn sm" style={{ background: filter === id ? "var(--bg-3)" : "transparent", border: "1px solid " + (filter === id ? "var(--line-2)" : "transparent"), color: filter === id ? "var(--tx-0)" : "var(--tx-1)" }}>{lab}<span className="tnum mono" style={{ fontSize: 10, color: "var(--tx-3)" }}>{n}</span></button>;
        })}
      </div>
      <div className="scroll" style={{ flex: 1, padding: 18 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 10, maxWidth: 920, margin: "0 auto" }}>
          {clips.map((c) => (
            <button key={c.id} onClick={() => open(c)} className="card" style={{ padding: 0, overflow: "hidden", textAlign: "left", display: "flex", alignItems: "stretch" }}
              onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--line-2)")} onMouseLeave={(e) => (e.currentTarget.style.borderColor = "var(--line)")}>
              <div style={{ width: 52, flexShrink: 0, background: "var(--bg-2)", display: "grid", placeItems: "center", color: "var(--m-aed)" }}><Icon name="audio" size={20} /></div>
              <div style={{ flex: 1, minWidth: 0, padding: "10px 14px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span className="mono" style={{ fontSize: 12.5, fontWeight: 600 }}>{c.name}</span>
                  <span className="mono" style={{ fontSize: 10.5, color: "var(--tx-2)" }}>{fmtT(c.dur)}</span>
                  <div style={{ flex: 1 }} />
                  {c.conflicts > 0 ? <span className="chip" style={{ borderColor: "var(--conflict)", color: "var(--conflict)" }}>{c.conflicts} conflict{c.conflicts > 1 ? "s" : ""}</span> : <span className="chip" style={{ borderColor: "var(--match)", color: "var(--match)" }}>clean</span>}
                  <span style={{ width: 7, height: 7, borderRadius: "50%", background: { verified: "var(--match)", flagged: "var(--gt)", unreviewed: "var(--tx-3)" }[c.status] }} />
                </div>
                <div style={{ marginTop: 8, borderRadius: 6, overflow: "hidden" }}><MiniWave wave={c.wave} segments={c.segments} /></div>
              </div>
            </button>
          ))}
        </div>
      </div>
    </>
  );
}

function AudioScreen() {
  const ctx = React.useContext(VDCtx);
  const { reviews, setReview } = ctx;
  const d = window.VD.getDataset(ctx.route.datasetId);
  const idx0 = d.clips.findIndex((c) => c.id === ctx.route.clipId);
  const [ci, setCi] = React.useState(idx0 < 0 ? 0 : idx0);
  const clip = d.clips[ci];
  const [sel, setSel] = React.useState(null);
  const [pos, setPos] = React.useState(0); // 0..1 playhead
  const [playing, setPlaying] = React.useState(false);
  const laneRef = React.useRef(null);

  React.useEffect(() => {
    if (!playing) return;
    const iv = setInterval(() => setPos((p) => { const np = p + 0.7 / clip.dur; if (np >= 1) { setPlaying(false); return 1; } return np; }), 70);
    return () => clearInterval(iv);
  }, [playing, clip.dur]);

  const goClip = (n) => { const k = (n + d.clips.length) % d.clips.length; setCi(k); setSel(null); setPos(0); setPlaying(false); };
  const segIds = clip.segments.map((s) => s.id);
  const stepSeg = (dir) => { if (!segIds.length) return; const i = segIds.indexOf(sel); const n = i < 0 ? 0 : (i + dir + segIds.length) % segIds.length; setSel(segIds[n]); const s = clip.segments[n]; setPos((s.start + s.end) / 2); };

  React.useEffect(() => {
    const h = (e) => {
      if (e.target.tagName === "INPUT") return;
      const k = e.key.toLowerCase();
      if (e.key === " ") { e.preventDefault(); setPlaying((p) => !p); }
      else if (k === "j" || e.key === "ArrowDown") { e.preventDefault(); stepSeg(1); }
      else if (k === "k" || e.key === "ArrowUp") { e.preventDefault(); stepSeg(-1); }
      else if (e.key === "ArrowRight") goClip(ci + 1);
      else if (e.key === "ArrowLeft") goClip(ci - 1);
      else if (k === "a" && sel) setReview(sel, { verdict: "accepted" });
      else if (k === "x" && sel) setReview(sel, { verdict: "rejected" });
      else if (k === "escape") setSel(null);
    };
    window.addEventListener("keydown", h); return () => window.removeEventListener("keydown", h);
  }, [sel, ci, segIds.join(",")]);

  const onLaneClick = (e) => { const r = laneRef.current.getBoundingClientRect(); setPos((e.clientX - r.left) / r.width); };
  const matched = clip.segments.filter((s) => s.state === "match").length;
  const agreement = clip.segments.length ? matched / clip.segments.length : 1;

  const Seg = ({ s, lane }) => {
    const box = lane === "gt" ? s.gt : s.pred;
    if (lane === "gt" && !s.gt) return null;
    if (lane === "pred" && !s.pred) return null;
    const conflict = s.state !== "match";
    const col = lane === "gt" ? "var(--gt)" : "var(--pred)";
    const isSel = sel === s.id;
    return (
      <div onClick={(e) => { e.stopPropagation(); setSel(s.id); }} title={box}
        style={{ position: "absolute", left: s.start * 100 + "%", width: (s.end - s.start) * 100 + "%", top: 3, bottom: 3, background: col + "26", border: "1.5px solid " + col, borderRadius: 5, display: "flex", alignItems: "center", padding: "0 6px", cursor: "pointer", boxShadow: isSel ? "0 0 0 2px var(--accent)" : conflict ? "0 0 0 1px var(--conflict)" : "none", overflow: "hidden" }}>
        <span style={{ fontSize: 10.5, fontWeight: 600, color: "var(--tx-0)", whiteSpace: "nowrap" }}>{box}{s.keyword ? " " + s.keyword : ""}</span>
        {conflict && lane === "pred" && <span style={{ marginLeft: 4, fontSize: 9, fontWeight: 700, color: "var(--conflict)" }}>{vhelp.stateShort(s.state)}</span>}
      </div>
    );
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "var(--bg-canvas)" }}>
      <header style={{ height: 50, flexShrink: 0, borderBottom: "1px solid var(--line)", display: "flex", alignItems: "center", padding: "0 14px", gap: 12, background: "var(--bg-1)" }}>
        <button className="btn ghost sm" onClick={() => ctx.go({ name: "grid", datasetId: d.id })}><Icon name="chevL" size={15} />Clips</button>
        <div style={{ width: 1, height: 22, background: "var(--line)" }} />
        <Icon name="audio" size={16} style={{ color: "var(--m-aed)" }} />
        <span className="mono" style={{ fontSize: 13, fontWeight: 600 }}>{clip.name}</span>
        <div style={{ flex: 1 }} />
        <span className="chip" style={{ borderColor: "var(--line-2)" }}><Icon name="cpu" size={12} style={{ color: "var(--m-aed)" }} />AED v{window.VD.models.find((m) => m.id === "aed").versions[0].version}</span>
        <div style={{ width: 1, height: 22, background: "var(--line)" }} />
        <button className="btn ghost icon" onClick={() => goClip(ci - 1)}><Icon name="chevL" size={16} /></button>
        <span className="tnum mono" style={{ fontSize: 12, color: "var(--tx-1)", minWidth: 54, textAlign: "center" }}>{ci + 1} / {d.clips.length}</span>
        <button className="btn ghost icon" onClick={() => goClip(ci + 1)}><Icon name="chevR" size={16} /></button>
        <button className="btn primary sm" onClick={() => clip.segments.forEach((s) => setReview(s.id, { verdict: "accepted" }))}><Icon name="check" size={14} />Verify clip</button>
      </header>

      <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
        {/* center: waveform + lanes */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", padding: 24, minWidth: 0, gap: 14 }}>
          <div className="card" style={{ padding: "16px 18px", background: "var(--bg-1)" }}>
            {/* transport */}
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
              <button className="btn icon" onClick={() => setPlaying((p) => !p)} style={{ width: 38, height: 38, borderRadius: "50%", background: "var(--accent)", color: "#07090d" }}>
                {playing ? <svg width="14" height="14" viewBox="0 0 14 14"><rect x="2" y="1" width="3.5" height="12" fill="currentColor" /><rect x="8.5" y="1" width="3.5" height="12" fill="currentColor" /></svg> : <svg width="14" height="14" viewBox="0 0 14 14"><path d="M3 1l10 6-10 6z" fill="currentColor" /></svg>}
              </button>
              <span className="mono tnum" style={{ fontSize: 13, color: "var(--tx-0)" }}>{fmtT(pos * clip.dur)} <span style={{ color: "var(--tx-3)" }}>/ {fmtT(clip.dur)}</span></span>
              <div style={{ flex: 1 }} />
              <span style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 11, color: "var(--tx-2)" }}>
                <span style={{ display: "flex", alignItems: "center", gap: 5 }}><svg width="16" height="9"><rect x="1" y="1" width="14" height="7" fill="none" stroke="var(--pred)" strokeWidth="1.5" /></svg>pred</span>
                <span style={{ display: "flex", alignItems: "center", gap: 5 }}><svg width="16" height="9"><rect x="1" y="1" width="14" height="7" fill="none" stroke="var(--gt)" strokeWidth="1.5" /></svg>GT</span>
              </span>
            </div>

            {/* waveform + playhead + lanes */}
            <div ref={laneRef} onClick={onLaneClick} style={{ position: "relative", cursor: "text" }}>
              <div style={{ background: "var(--bg-canvas)", borderRadius: 8, padding: "10px 0", border: "1px solid var(--line)" }}>
                <Waveform wave={clip.wave} color="var(--tx-2)" height={130} />
              </div>
              {/* lanes */}
              <div style={{ marginTop: 10 }}>
                <LaneLabel label="Ground-truth" color="var(--gt)" />
                <div style={{ position: "relative", height: 34, background: "var(--bg-2)", borderRadius: 7, marginTop: 4 }}>{clip.segments.map((s) => <Seg key={s.id + "g"} s={s} lane="gt" />)}</div>
                <div style={{ marginTop: 8 }}><LaneLabel label="Prediction (AED)" color="var(--pred)" /></div>
                <div style={{ position: "relative", height: 34, background: "var(--bg-2)", borderRadius: 7, marginTop: 4 }}>{clip.segments.map((s) => <Seg key={s.id + "p"} s={s} lane="pred" />)}</div>
              </div>
              {/* playhead spanning everything */}
              <div style={{ position: "absolute", top: 0, bottom: 0, left: pos * 100 + "%", width: 2, background: "var(--accent-2)", boxShadow: "0 0 8px var(--accent-2)", pointerEvents: "none" }}>
                <div style={{ position: "absolute", top: -4, left: -4, width: 10, height: 10, borderRadius: "50%", background: "var(--accent-2)" }} />
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 14 }}>
              <span style={{ fontSize: 12, color: "var(--tx-2)" }}>clip agreement</span>
              <span className="tnum" style={{ fontWeight: 700, color: agreement >= 0.85 ? "var(--match)" : agreement >= 0.6 ? "var(--gt)" : "var(--conflict)" }}>{vhelp.pct(agreement)}</span>
              <div style={{ flex: 1 }} />
              <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>{[...new Set(clip.segments.map((s) => s.gt || s.pred))].filter(Boolean).map((l) => <span key={l} className="chip" style={{ borderColor: "var(--line)" }}><span className="dot" style={{ background: segColor(l) }} />{l}</span>)}</div>
            </div>
          </div>
        </div>

        {/* right: segment list */}
        <div style={{ width: 332, flexShrink: 0, borderLeft: "1px solid var(--line)", background: "var(--bg-1)", display: "flex", flexDirection: "column" }}>
          <div style={{ padding: "12px 14px", borderBottom: "1px solid var(--line)", display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontWeight: 700, fontSize: 13, flex: 1 }}>Segments</span>
            <span className="chip" style={{ borderColor: "transparent", background: "var(--match-soft)", color: "var(--match)" }}>{matched} agree</span>
            {clip.conflicts > 0 && <span className="chip" style={{ borderColor: "transparent", background: "var(--conflict-soft)", color: "var(--conflict)" }}>{clip.conflicts}</span>}
          </div>
          <div className="scroll" style={{ flex: 1 }}>
            {clip.segments.map((s) => {
              const verdict = reviews[s.id]?.verdict;
              const conflict = s.state !== "match";
              return (
                <div key={s.id} style={{ borderBottom: "1px solid var(--line)", background: sel === s.id ? "var(--bg-2)" : "transparent" }}>
                  <button onClick={() => { setSel(s.id); setPos((s.start + s.end) / 2); }} style={{ display: "flex", alignItems: "center", gap: 9, width: "100%", padding: "9px 12px", textAlign: "left" }}>
                    <span style={{ width: 8, height: 8, borderRadius: 2, background: segColor(s.gt || s.pred), flexShrink: 0 }} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 12.5, fontWeight: 500 }}>{s.gt || s.pred}{s.keyword ? <span style={{ color: "var(--match)" }}> {s.keyword}</span> : ""}</div>
                      <div className="mono" style={{ fontSize: 10, color: "var(--tx-2)" }}>{fmtT(s.start * clip.dur)}–{fmtT(s.end * clip.dur)} · {vhelp.stateLabel(s.state)}</div>
                    </div>
                    {s.pred && <span className="mono tnum" style={{ fontSize: 11, color: conflict ? "var(--conflict)" : "var(--pred)", fontWeight: 600 }}>{s.conf.toFixed(2)}</span>}
                    {verdict === "accepted" && <Icon name="check" size={14} style={{ color: "var(--match)" }} />}
                    {verdict === "rejected" && <Icon name="x" size={14} style={{ color: "var(--tx-2)" }} />}
                  </button>
                  {sel === s.id && (
                    <div className="fade" style={{ padding: "0 12px 12px" }}>
                      <div style={{ background: "var(--bg-1)", border: "1px solid var(--line)", borderRadius: 8, padding: 10, display: "flex", flexDirection: "column", gap: 6 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}><span style={{ color: "var(--gt)", fontWeight: 600 }}>GT label</span><span>{s.gt || "—"}</span></div>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}><span style={{ color: "var(--pred)", fontWeight: 600 }}>Predicted</span><span style={{ color: s.pred === s.gt ? "var(--match)" : "var(--conflict)" }}>{s.pred || "no detection"} · {s.conf.toFixed(2)}</span></div>
                        <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
                          <button className="btn sm" style={{ flex: 1, justifyContent: "center", background: verdict === "accepted" ? "var(--match)" : "var(--bg-3)", color: verdict === "accepted" ? "#07090d" : "var(--tx-0)" }} onClick={() => setReview(s.id, { verdict: "accepted" })}><Icon name="check" size={13} />Accept</button>
                          <button className="btn sm" style={{ flex: 1, justifyContent: "center", background: verdict === "rejected" ? "var(--conflict)" : "var(--bg-3)", color: verdict === "rejected" ? "#07090d" : "var(--tx-0)" }} onClick={() => setReview(s.id, { verdict: "rejected" })}><Icon name="x" size={13} />Reject</button>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          <div style={{ padding: "9px 12px", borderTop: "1px solid var(--line)", display: "flex", flexWrap: "wrap", gap: "5px 12px", fontSize: 10.5, color: "var(--tx-2)" }}>
            {[["Space", "play"], ["J / K", "segment"], ["← →", "clip"], ["A", "accept"], ["X", "reject"]].map(([k, l]) => <span key={k} style={{ display: "flex", alignItems: "center", gap: 5 }}><span className="kbd">{k}</span>{l}</span>)}
          </div>
        </div>
      </div>
    </div>
  );
}

function LaneLabel({ label, color }) {
  return <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10.5, color: "var(--tx-2)" }}><span style={{ width: 8, height: 8, borderRadius: 2, background: color }} />{label}</div>;
}

Object.assign(window, { AudioGrid, AudioScreen });
