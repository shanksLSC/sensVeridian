/* ============================================================================
   Veridian Studio — Regression review queue (idea #1).
   Register/compare a new model version and review ONLY the frames where it now
   disagrees with verified ground-truth: regressions (was right, now wrong) and
   fixes (was wrong, now right).
   Exposes: RegressionScreen
   ========================================================================== */

function hashr(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return (h >>> 0) / 4294967296; }

function buildFlips(model, vA, vB) {
  // datasets this model labels
  const dss = window.VD.datasets.filter((d) => d.kind === "vision" && d.models.includes(model.id));
  const flips = [];
  dss.forEach((d) => d.images.forEach((im) => im.objects.forEach((o) => {
    if (o.model !== model.id) return;
    const r = hashr(o.id + vB.version);
    if (r > 0.16) return; // only ~16% of detections flip between versions
    // direction: improvements more likely (newer version better), but some regressions
    const regress = hashr(o.id + vB.version + "d") < 0.4;
    const confA = +(0.5 + hashr(o.id + vA.version) * 0.45).toFixed(2);
    const confB = +(0.5 + hashr(o.id + vB.version + "c") * 0.48).toFixed(2);
    flips.push({ d, im, o, regress, confA, confB });
  })));
  return flips;
}

function RegressionScreen() {
  const ctx = React.useContext(VDCtx);
  const { reviews, setReview } = ctx;
  const isRest = (window.VERIDIAN_CONFIG || {}).backend === "rest";
  const [modelId, setModelId] = React.useState(ctx.route.modelId || "amod");
  const model = window.VD.models.find((m) => m.id === modelId) || window.VD.models[0];
  // real models may carry a single version -> guard versions[1]
  const versions = (model.versions && model.versions.length)
    ? model.versions : [{ version: "current", date: "", metrics: {}, weights_sha: "" }];
  const defB = versions[0];
  const defA = versions[1] || versions[0];
  const [aV, setAV] = React.useState(defA.version);
  const [bV, setBV] = React.useState(defB.version);
  const [filter, setFilter] = React.useState("regress");

  React.useEffect(() => { setAV(defA.version); setBV(defB.version); }, [modelId]);
  const vA = versions.find((v) => v.version === aV) || defA;
  const vB = versions.find((v) => v.version === bV) || defB;
  const ag = (v) => (v && v.metrics && typeof v.metrics.agreement === "number") ? v.metrics.agreement : 0;
  const dAg = ag(vB) - ag(vA);

  // Flips come from the backend (real runs vs GT) in REST mode; mock otherwise.
  const [flips, setFlips] = React.useState([]);
  React.useEffect(() => {
    let alive = true;
    if (isRest && window.VeridianAPI && window.VeridianAPI.getRegressions) {
      window.VeridianAPI.getRegressions(model.id, aV, bV)
        .then((rows) => {
          if (!alive) return;
          setFlips((rows || []).map((r) => {
            const d = window.VD.getDataset(r.datasetId) || { id: r.datasetId, name: r.datasetId };
            const im = (d.images || []).find((x) => x.id === r.imageId) || { id: r.imageId, datasetId: r.datasetId };
            return { d, im, o: { id: r.detId, cls: r.cls, gt: null, pred: null },
                     regress: !!r.regress, confA: r.confA, confB: r.confB };
          }));
        })
        .catch((e) => { console.warn("[veridian] regressions load failed", e); if (alive) setFlips([]); });
    } else {
      setFlips(buildFlips(model, vA, vB));
    }
    return () => { alive = false; };
  }, [modelId, aV, bV, isRest]);

  const regressions = flips.filter((f) => f.regress);
  const fixes = flips.filter((f) => !f.regress);
  const shown = filter === "regress" ? regressions : filter === "fix" ? fixes : flips;

  return (
    <>
      <TopBar crumbs={[{ label: "Regression review" }]}>
        <select value={modelId} onChange={(e) => setModelId(e.target.value)} style={{ background: "var(--bg-3)", border: "1px solid var(--line-2)", borderRadius: 6, color: "var(--tx-0)", padding: "6px 9px", fontSize: 12.5 }}>
          {window.VD.models.filter((m) => m.id !== "fr").map((m) => <option key={m.id} value={m.id}>{m.short} — {m.display_name}</option>)}
        </select>
      </TopBar>

      <div className="scroll" style={{ flex: 1, padding: "20px 24px" }}>
        <div style={{ maxWidth: 980, margin: "0 auto" }}>
          <div style={{ fontSize: 13, color: "var(--tx-1)", marginBottom: 16, maxWidth: 680 }}>
            Diffing two runs of <b style={{ color: "var(--tx-0)" }}>{model.short}</b> against your <b style={{ color: "var(--match)" }}>verified ground-truth</b>. You only see detections whose correctness <b>flipped</b> — not the whole set.
          </div>

          {/* version pickers + summary */}
          <div className="card" style={{ padding: 16, marginBottom: 18 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
              <span style={{ fontSize: 11.5, color: "var(--tx-2)" }}>baseline</span>
              <select value={aV} onChange={(e) => setAV(e.target.value)} className="mono" style={{ background: "var(--bg-3)", border: "1px solid var(--line-2)", borderRadius: 6, color: "var(--tx-0)", padding: "6px 9px", fontSize: 12 }}>{model.versions.map((v) => <option key={v.version} value={v.version}>v{v.version} · {v.date}</option>)}</select>
              <Icon name="arrowR" size={15} style={{ color: "var(--tx-3)" }} />
              <span style={{ fontSize: 11.5, color: "var(--tx-2)" }}>candidate</span>
              <select value={bV} onChange={(e) => setBV(e.target.value)} className="mono" style={{ background: "var(--bg-3)", border: "1px solid var(--accent)", borderRadius: 6, color: "var(--tx-0)", padding: "6px 9px", fontSize: 12 }}>{model.versions.map((v) => <option key={v.version} value={v.version}>v{v.version} · {v.date}</option>)}</select>
              <div style={{ flex: 1 }} />
              <span className="mono" style={{ fontSize: 11, color: "var(--tx-2)" }}>{(vB.weights_sha || "").slice(0, 12)}…</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
              <FlipStat icon="trend" label="Net agreement Δ" v={(dAg >= 0 ? "+" : "") + (dAg * 100).toFixed(1) + "%"} c={dAg >= 0 ? "var(--match)" : "var(--conflict)"} />
              <FlipStat icon="alert" label="Regressions" v={regressions.length} c="var(--conflict)" sub="was right → now wrong" />
              <FlipStat icon="spark" label="Fixes" v={fixes.length} c="var(--match)" sub="was wrong → now right" />
              <FlipStat icon="check" label="Stable" v={"—"} c="var(--tx-2)" sub="unchanged, hidden" />
            </div>
          </div>

          {/* filter */}
          <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
            {[["regress", "Regressions", regressions.length, "var(--conflict)"], ["fix", "Fixes", fixes.length, "var(--match)"], ["all", "All flips", flips.length, "var(--tx-2)"]].map(([id, lab, n, c]) => (
              <button key={id} onClick={() => setFilter(id)} className="btn sm" style={{ background: filter === id ? "var(--bg-3)" : "transparent", border: "1px solid " + (filter === id ? "var(--line-2)" : "transparent"), color: filter === id ? "var(--tx-0)" : "var(--tx-1)" }}>
                <span style={{ width: 7, height: 7, borderRadius: "50%", background: c }} />{lab}<span className="tnum mono" style={{ fontSize: 10.5, color: "var(--tx-3)" }}>{n}</span>
              </button>
            ))}
          </div>

          {/* flip list */}
          <div className="card" style={{ overflow: "hidden" }}>
            {shown.length === 0 && <div style={{ padding: 40, textAlign: "center", color: "var(--tx-2)", fontSize: 13 }}>No {filter === "regress" ? "regressions" : filter === "fix" ? "fixes" : "flips"} between these versions. 🎉</div>}
            {shown.slice(0, 50).map((f, i) => {
              const rev = reviews["flip:" + f.o.id];
              return (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 14, padding: "10px 14px", borderBottom: i < Math.min(49, shown.length - 1) ? "1px solid var(--line)" : "none" }}>
                  <button onClick={() => ctx.go({ name: "canvas", datasetId: f.d.id, imageId: f.im.id })} style={{ width: 70, height: 44, borderRadius: 5, overflow: "hidden", position: "relative", flexShrink: 0, border: "1px solid var(--line)" }}>
                    <Scene image={f.im} style={{ width: "100%", height: "100%" }} />
                    <svg viewBox="0 0 1 0.625" preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>{(() => { const b = f.o.gt || f.o.pred; return b ? <rect x={b[0]} y={b[1] * 0.625} width={b[2]} height={b[3] * 0.625} fill="none" stroke={f.regress ? "var(--conflict)" : "var(--match)"} strokeWidth="0.008" vectorEffect="non-scaling-stroke" /> : null; })()}</svg>
                  </button>
                  <div style={{ width: 140, minWidth: 0 }}>
                    <div style={{ fontSize: 12.5, fontWeight: 500 }}>{window.VD.CLASSES[f.o.cls].label}</div>
                    <div className="mono" style={{ fontSize: 10.5, color: "var(--tx-2)" }}>{f.d.name.split(" — ")[0]}</div>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, flex: 1 }}>
                    <span className="chip" style={{ borderColor: "transparent", background: f.regress ? "var(--match-soft)" : "var(--conflict-soft)", color: f.regress ? "var(--match)" : "var(--conflict)" }}>v{vA.version}: {f.regress ? "correct" : "wrong"}</span>
                    <Icon name="arrowR" size={13} style={{ color: "var(--tx-3)" }} />
                    <span className="chip" style={{ borderColor: "transparent", background: f.regress ? "var(--conflict-soft)" : "var(--match-soft)", color: f.regress ? "var(--conflict)" : "var(--match)", fontWeight: 600 }}>v{vB.version}: {f.regress ? "wrong" : "correct"}</span>
                    <span className="mono tnum" style={{ fontSize: 10.5, color: "var(--tx-3)" }}>conf {f.confA}→{f.confB}</span>
                  </div>
                  {f.regress ? (
                    <button className="btn sm" style={{ color: rev?.verdict === "flagged" ? "var(--gt)" : undefined }} onClick={() => setReview("flip:" + f.o.id, { verdict: "flagged" })}><Icon name="flag" size={13} />Flag</button>
                  ) : (
                    <button className="btn sm" style={{ background: rev?.verdict === "accepted" ? "var(--match)" : "var(--bg-3)", color: rev?.verdict === "accepted" ? "#07090d" : "var(--tx-0)" }} onClick={() => setReview("flip:" + f.o.id, { verdict: "accepted" })}><Icon name="check" size={13} />Accept fix</button>
                  )}
                </div>
              );
            })}
          </div>

          <div style={{ display: "flex", justifyContent: "center", gap: 10, marginTop: 18 }}>
            <button className="btn" onClick={() => ctx.go({ name: "models" })}><Icon name="cpu" size={15} />Model manager</button>
            <button className="btn primary"><Icon name="check" size={15} />Promote v{vB.version} to current</button>
          </div>
        </div>
      </div>
    </>
  );
}

function FlipStat({ icon, label, v, c, sub }) {
  return (
    <div style={{ background: "var(--bg-2)", border: "1px solid var(--line)", borderRadius: 9, padding: "11px 13px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 7, color: c, marginBottom: 7 }}><Icon name={icon} size={14} /><span style={{ fontSize: 11, color: "var(--tx-2)", fontWeight: 500 }}>{label}</span></div>
      <div className="tnum" style={{ fontSize: 22, fontWeight: 700, color: c, letterSpacing: "-0.02em" }}>{v}</div>
      {sub && <div style={{ fontSize: 10, color: "var(--tx-3)", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

window.RegressionScreen = RegressionScreen;
window.__buildFlips = buildFlips;
