/* ============================================================================
   Veridian Studio — Model & version manager: timeline, weights hashes,
   A/B metric diff. Exposes: ModelsScreen
   ========================================================================== */

const METRIC_KEYS = [
  { k: "mAP", label: "mAP@.5" },
  { k: "precision", label: "Precision" },
  { k: "recall", label: "Recall" },
  { k: "f1", label: "F1" },
  { k: "agreement", label: "GT agreement" },
];

function ModelsScreen() {
  const ctx = React.useContext(VDCtx);
  const [sel, setSel] = React.useState(window.VD.models[0].id);
  const m = window.VD.models.find((x) => x.id === sel);
  const [a, setA] = React.useState(m.versions[1].version);
  const [b, setB] = React.useState(m.versions[0].version);

  React.useEffect(() => { setA(m.versions[1].version); setB(m.versions[0].version); }, [sel]);

  const vA = m.versions.find((v) => v.version === a) || m.versions[1];
  const vB = m.versions.find((v) => v.version === b) || m.versions[0];

  return (
    <>
      <TopBar crumbs={[{ label: "Models & Versions" }]}>
        <span className="chip" style={{ borderColor: "var(--line-2)" }}><Icon name="database" size={12} style={{ color: "var(--match)" }} />synced from {window.VD.storage.engine}</span>
        <button className="btn"><Icon name="plus" size={15} />Register weights</button>
      </TopBar>

      <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
        {/* model list */}
        <div style={{ width: 268, flexShrink: 0, borderRight: "1px solid var(--line)", background: "var(--bg-1)", padding: 10, display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ fontSize: 10, color: "var(--tx-3)", textTransform: "uppercase", letterSpacing: ".07em", fontWeight: 600, padding: "4px 6px" }}>Oracle models</div>
          {window.VD.models.map((mm) => {
            const cur = mm.versions[0];
            const active = mm.id === sel;
            return (
              <button key={mm.id} onClick={() => setSel(mm.id)} className="card" style={{ padding: 12, textAlign: "left", borderColor: active ? vhelp.modelColor(mm.id) : "var(--line)", background: active ? "var(--bg-2)" : "var(--bg-1)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ width: 9, height: 9, borderRadius: 2, background: vhelp.modelColor(mm.id) }} />
                  <span style={{ fontWeight: 700, fontSize: 13 }}>{mm.short}</span>
                  <span className="mono" style={{ fontSize: 10.5, color: "var(--tx-2)" }}>v{cur.version}</span>
                  <div style={{ flex: 1 }} />
                  {mm.depends_on && <span className="chip" style={{ padding: "0 6px", fontSize: 9.5, borderColor: "var(--line)" }}>needs {mm.depends_on.toUpperCase()}</span>}
                </div>
                <div style={{ fontSize: 11, color: "var(--tx-2)", marginTop: 5 }}>{mm.display_name}</div>
                <div style={{ display: "flex", gap: 12, marginTop: 8 }}>
                  <Mini label="agreement" v={cur.metrics.agreement} />
                  <Mini label="mAP" v={cur.metrics.mAP} />
                  <span style={{ flex: 1 }} />
                  <span className="mono" style={{ fontSize: 10, color: "var(--tx-3)", alignSelf: "flex-end" }}>{mm.versions.length} versions</span>
                </div>
              </button>
            );
          })}
        </div>

        {/* detail */}
        <div className="scroll" style={{ flex: 1, padding: "20px 24px" }}>
          <div style={{ maxWidth: 940, margin: "0 auto" }}>
            {/* model header */}
            <div className="card" style={{ padding: "16px 18px", marginBottom: 18 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <div style={{ width: 40, height: 40, borderRadius: 10, background: vhelp.modelColor(m.id) + "22", display: "grid", placeItems: "center", color: vhelp.modelColor(m.id) }}><Icon name="cpu" size={22} /></div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 700, fontSize: 17 }}>{m.display_name}</div>
                  <div className="mono" style={{ fontSize: 11.5, color: "var(--tx-2)" }}>{m.weights_path}</div>
                </div>
                <div style={{ display: "flex", gap: 22 }}>
                  {[["input", m.input], ["classes", m.classes], ["versions", m.versions.length]].map(([l, v]) => (
                    <div key={l} style={{ textAlign: "right" }}>
                      <div className="mono tnum" style={{ fontSize: 15, fontWeight: 600 }}>{v}</div>
                      <div style={{ fontSize: 10, color: "var(--tx-2)" }}>{l}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* A/B compare */}
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "var(--tx-2)", textTransform: "uppercase", letterSpacing: ".06em" }}>Compare versions</div>
              <div style={{ flex: 1 }} />
              <button className="btn sm" onClick={() => ctx.go({ name: "regression", modelId: m.id })}><Icon name="trend" size={13} />Review regressions vs verified GT</button>
            </div>
            <ABCompare model={m} a={vA} b={vB} setA={setA} setB={setB} />

            {/* timeline */}
            <div style={{ fontSize: 12, fontWeight: 600, color: "var(--tx-2)", textTransform: "uppercase", letterSpacing: ".06em", margin: "22px 0 10px" }}>Version history</div>
            <div style={{ position: "relative", paddingLeft: 26 }}>
              <div style={{ position: "absolute", left: 7, top: 8, bottom: 8, width: 2, background: "var(--line)" }} />
              {m.versions.map((v, i) => (
                <VersionNode key={v.version} v={v} first={i === 0} isA={v.version === a} isB={v.version === b} onA={() => setA(v.version)} onB={() => setB(v.version)} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function Mini({ label, v }) {
  return (
    <div>
      <div className="tnum" style={{ fontSize: 13, fontWeight: 700, color: "var(--match)" }}>{vhelp.pct(v)}</div>
      <div style={{ fontSize: 9.5, color: "var(--tx-2)" }}>{label}</div>
    </div>
  );
}

function ABCompare({ model, a, b, setA, setB }) {
  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
        <span className="chip" style={{ borderColor: "var(--tx-2)", color: "var(--tx-1)" }}>A · v{a.version} <span className="mono" style={{ color: "var(--tx-3)" }}>{a.date}</span></span>
        <Icon name="arrowR" size={14} style={{ color: "var(--tx-3)" }} />
        <span className="chip" style={{ borderColor: "var(--accent)", color: "var(--accent-2)" }}>B · v{b.version} <span className="mono" style={{ color: "var(--tx-3)" }}>{b.date}</span></span>
        <div style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: "var(--tx-2)" }}>Δ = B − A on shared eval set</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10 }}>
        {METRIC_KEYS.map(({ k, label }) => {
          const av = a.metrics[k], bv = b.metrics[k], d = bv - av;
          const up = d >= 0;
          return (
            <div key={k} style={{ background: "var(--bg-2)", border: "1px solid var(--line)", borderRadius: 9, padding: "11px 12px" }}>
              <div style={{ fontSize: 10.5, color: "var(--tx-2)", marginBottom: 7 }}>{label}</div>
              <div className="tnum" style={{ fontSize: 19, fontWeight: 700, letterSpacing: "-0.02em" }}>{vhelp.pct1(bv)}</div>
              <div style={{ display: "flex", alignItems: "center", gap: 4, marginTop: 5 }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: Math.abs(d) < 0.001 ? "var(--tx-3)" : up ? "var(--match)" : "var(--conflict)", display: "flex", alignItems: "center", gap: 2 }}>
                  {Math.abs(d) < 0.001 ? "±0" : (up ? "▲ " : "▼ ") + (Math.abs(d) * 100).toFixed(1)}
                </span>
                <span className="mono" style={{ fontSize: 9.5, color: "var(--tx-3)" }}>from {vhelp.pct1(av)}</span>
              </div>
              <div className="meter" style={{ height: 4, marginTop: 8 }}><i style={{ width: bv * 100 + "%", background: "var(--accent)" }} /></div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function VersionNode({ v, first, isA, isB, onA, onB }) {
  const [copied, setCopied] = React.useState(false);
  return (
    <div style={{ position: "relative", marginBottom: 12 }}>
      <div style={{ position: "absolute", left: -23, top: 16, width: 14, height: 14, borderRadius: "50%", background: first ? "var(--accent)" : "var(--bg-1)", border: "2px solid " + (first ? "var(--accent)" : "var(--line-2)"), boxShadow: first ? "0 0 10px var(--accent)" : "none" }} />
      <div className="card" style={{ padding: "12px 14px", borderColor: isA || isB ? (isB ? "var(--accent)" : "var(--tx-2)") : "var(--line)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className="mono" style={{ fontSize: 14, fontWeight: 700 }}>v{v.version}</span>
          {first && <span className="chip" style={{ borderColor: "var(--accent)", color: "var(--accent-2)", padding: "1px 8px" }}><span className="dot" style={{ background: "var(--accent)" }} />current</span>}
          <span className="chip" style={{ borderColor: "var(--line)", color: "var(--tx-2)" }}>{v.notes}</span>
          <div style={{ flex: 1 }} />
          <span className="mono" style={{ fontSize: 11, color: "var(--tx-2)" }}><Icon name="clock" size={11} style={{ verticalAlign: "-1px" }} /> {v.date}</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 9 }}>
          <Icon name="branch" size={13} style={{ color: "var(--tx-3)" }} />
          <button onClick={() => { setCopied(true); setTimeout(() => setCopied(false), 1000); }}
            className="mono" style={{ fontSize: 11, color: "var(--tx-2)", background: "var(--bg-2)", border: "1px solid var(--line)", borderRadius: 5, padding: "3px 8px", display: "flex", alignItems: "center", gap: 6 }}>
            {copied ? "copied" : v.weights_sha.slice(0, 24) + "…"}
            <Icon name={copied ? "check" : "link"} size={11} style={{ color: copied ? "var(--match)" : "var(--tx-3)" }} />
          </button>
          <div style={{ flex: 1 }} />
          <div style={{ display: "flex", gap: 14 }}>
            {METRIC_KEYS.slice(0, 4).map(({ k, label }) => (
              <div key={k} style={{ textAlign: "right" }}>
                <div className="tnum" style={{ fontSize: 12, fontWeight: 600 }}>{vhelp.pct(v.metrics[k])}</div>
                <div style={{ fontSize: 9, color: "var(--tx-3)" }}>{label}</div>
              </div>
            ))}
          </div>
          <div style={{ display: "flex", gap: 4, marginLeft: 6 }}>
            <button className="btn sm" style={{ padding: "3px 9px", background: isA ? "var(--tx-2)" : "var(--bg-3)", color: isA ? "#07090d" : "var(--tx-1)", fontWeight: 700 }} onClick={onA}>A</button>
            <button className="btn sm" style={{ padding: "3px 9px", background: isB ? "var(--accent)" : "var(--bg-3)", color: isB ? "#07090d" : "var(--tx-1)", fontWeight: 700 }} onClick={onB}>B</button>
          </div>
        </div>
      </div>
    </div>
  );
}

window.ModelsScreen = ModelsScreen;
