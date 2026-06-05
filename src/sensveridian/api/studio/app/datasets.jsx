/* ============================================================================
   Veridian Studio — Datasets screen + Import modal (shared) + small primitives.
   Exposes: DatasetsScreen, ImportModal, TopBar, AgreeBar, ModelChip, StatusDot
   ========================================================================== */

function TopBar({ children, crumbs }) {
  const ctx = React.useContext(VDCtx);
  return (
    <header style={{ height: 52, flexShrink: 0, borderBottom: "1px solid var(--line)", display: "flex", alignItems: "center", padding: "0 18px", gap: 12, background: "var(--bg-1)" }}>
      {crumbs && (
        <nav style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 13, minWidth: 0 }}>
          {crumbs.map((c, i) => (
            <React.Fragment key={i}>
              {i > 0 && <Icon name="chevR" size={13} style={{ color: "var(--tx-3)" }} />}
              {c.onClick ? (
                <button onClick={c.onClick} style={{ color: "var(--tx-1)", fontWeight: 500 }}
                  onMouseEnter={(e) => (e.currentTarget.style.color = "var(--tx-0)")}
                  onMouseLeave={(e) => (e.currentTarget.style.color = "var(--tx-1)")}>{c.label}</button>
              ) : (
                <span style={{ color: "var(--tx-0)", fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.label}</span>
              )}
            </React.Fragment>
          ))}
        </nav>
      )}
      <div style={{ flex: 1 }} />
      {children}
    </header>
  );
}

function ModelChip({ id, size = "md" }) {
  const m = window.VD.models.find((x) => x.id === id);
  if (!m) return null;
  const c = vhelp.modelColor(id);
  return (
    <span className="chip" style={{ borderColor: "transparent", background: "var(--bg-2)", fontSize: size === "sm" ? 10.5 : 11.5, fontWeight: 600 }}>
      <span className="dot" style={{ background: c, boxShadow: `0 0 6px ${c}` }} />
      <span style={{ color: "var(--tx-0)" }}>{m.short}</span>
    </span>
  );
}

function AgreeBar({ value, height = 6 }) {
  const pct = Math.round(value * 100);
  const col = value >= 0.85 ? "var(--match)" : value >= 0.65 ? "var(--gt)" : "var(--conflict)";
  return (
    <div className="meter" style={{ height }}>
      <i style={{ width: pct + "%", background: col, boxShadow: `0 0 8px ${col}66` }} />
    </div>
  );
}

function StatusDot({ status }) {
  const map = { verified: ["var(--match)", "Verified"], flagged: ["var(--conflict)", "Flagged"], unreviewed: ["var(--tx-3)", "Unreviewed"] };
  const [c, label] = map[status] || map.unreviewed;
  return <span className="chip" style={{ borderColor: "var(--line)" }}><span className="dot" style={{ background: c }} />{label}</span>;
}

/* ---- Import modal -------------------------------------------------------- */
function ImportModal({ dataset, onClose }) {
  const [fmt, setFmt] = React.useState("yolo");
  const [stage, setStage] = React.useState("pick"); // pick -> map -> done
  const fmtDef = window.VD.importFormats.find((f) => f.id === fmt);
  const sampleFiles = {
    yolo: ["img_0001.txt", "img_0002.txt", "img_0003.txt", "…", "labels.names"],
    coco: ["instances_val.json"],
    csv: ["ground_truth.csv"],
  }[fmt];
  return (
    <div onClick={onClose} style={{ position: "fixed", inset: 0, background: "rgba(4,6,10,.66)", backdropFilter: "blur(3px)", zIndex: 80, display: "grid", placeItems: "center" }} className="fade">
      <div onClick={(e) => e.stopPropagation()} className="card pop" style={{ width: 560, maxWidth: "92vw", boxShadow: "var(--shadow-pop)" }}>
        <div style={{ padding: "16px 20px", borderBottom: "1px solid var(--line)", display: "flex", alignItems: "center", gap: 10 }}>
          <Icon name="upload" size={18} style={{ color: "var(--accent)" }} />
          <div style={{ fontWeight: 700, fontSize: 15 }}>Load ground-truth labels</div>
          <div style={{ flex: 1 }} />
          <button className="btn ghost icon" onClick={onClose}><Icon name="x" size={16} /></button>
        </div>

        <div style={{ padding: 20 }}>
          <div style={{ fontSize: 12, color: "var(--tx-2)", marginBottom: 10 }}>Importing into <b style={{ color: "var(--tx-0)" }}>{dataset.name}</b> · {dataset.count} images. Labels are matched to images by filename / image_id and become the editable ground-truth layer.</div>

          <div style={{ fontSize: 11, fontWeight: 600, color: "var(--tx-2)", textTransform: "uppercase", letterSpacing: ".06em", marginBottom: 8 }}>Format</div>
          <div style={{ display: "flex", gap: 8, marginBottom: 18 }}>
            {window.VD.importFormats.map((f) => (
              <button key={f.id} onClick={() => setFmt(f.id)} style={{
                flex: 1, padding: "11px 12px", borderRadius: 9, textAlign: "left",
                border: "1px solid " + (fmt === f.id ? "var(--accent)" : "var(--line-2)"),
                background: fmt === f.id ? "var(--accent-dim)" : "var(--bg-2)",
              }}>
                <div style={{ fontWeight: 600, fontSize: 13 }}>{f.label}</div>
                <div className="mono" style={{ fontSize: 10, color: "var(--tx-2)" }}>{f.ext}</div>
              </button>
            ))}
          </div>

          <div style={{ border: "1.5px dashed var(--line-2)", borderRadius: 10, padding: "22px 16px", textAlign: "center", background: "var(--bg-2)" }}>
            <Icon name="upload" size={24} style={{ color: "var(--tx-2)" }} />
            <div style={{ fontSize: 13, fontWeight: 500, marginTop: 6 }}>Drop {fmtDef.label} files or a folder</div>
            <div style={{ fontSize: 11, color: "var(--tx-2)", marginTop: 2 }}>{fmtDef.note}</div>
          </div>

          <div style={{ marginTop: 14, background: "var(--bg-2)", border: "1px solid var(--line)", borderRadius: 8, padding: 12 }}>
            <div style={{ fontSize: 11, color: "var(--tx-2)", marginBottom: 6 }}>Detected files</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {sampleFiles.map((f, i) => (
                <span key={i} className="chip mono" style={{ fontSize: 10.5, borderColor: "var(--line)", background: "var(--bg-1)" }}>{f}</span>
              ))}
            </div>
          </div>
        </div>

        <div style={{ padding: "13px 20px", borderTop: "1px solid var(--line)", display: "flex", alignItems: "center", gap: 10 }}>
          <div className="mono" style={{ fontSize: 11, color: "var(--tx-2)" }}>→ ground_truth layer</div>
          <div style={{ flex: 1 }} />
          <button className="btn ghost" onClick={onClose}>Cancel</button>
          <button className="btn primary" onClick={onClose}><Icon name="check" size={15} />Import & match</button>
        </div>
      </div>
    </div>
  );
}

/* ---- Datasets screen ----------------------------------------------------- */
function DatasetsScreen() {
  const ctx = React.useContext(VDCtx);
  const [importing, setImporting] = React.useState(null);
  const ds = window.VD.datasets;
  const totals = {
    images: ds.reduce((s, d) => s + d.count, 0),
    conflicts: ds.reduce((s, d) => s + d.conflicts, 0),
    reviewed: ds.reduce((s, d) => s + d.reviewed, 0),
  };

  return (
    <>
      <TopBar crumbs={[{ label: "Datasets" }]}>
        <button className="btn" onClick={() => setImporting(ds[0])}><Icon name="upload" size={15} />Import labels</button>
        <button className="btn primary" onClick={() => ctx.go({ name: "ingest" })}><Icon name="plus" size={15} />New ingest run</button>
      </TopBar>

      <div className="scroll" style={{ flex: 1, padding: "22px 26px" }}>
        <div style={{ maxWidth: 1180, margin: "0 auto" }}>
          {/* summary strip */}
          <div style={{ display: "flex", gap: 14, marginBottom: 22 }}>
            {[
              { k: "Datasets", v: ds.length, icon: "folder" },
              { k: "Images cached", v: totals.images, icon: "image" },
              { k: "Open conflicts", v: totals.conflicts, icon: "alert", c: "var(--conflict)" },
              { k: "Verified", v: totals.reviewed, icon: "check", c: "var(--match)" },
            ].map((s) => (
              <div key={s.k} className="card" style={{ flex: 1, padding: "14px 16px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, color: s.c || "var(--tx-2)", marginBottom: 8 }}>
                  <Icon name={s.icon} size={15} /><span style={{ fontSize: 11.5, color: "var(--tx-2)", fontWeight: 500 }}>{s.k}</span>
                </div>
                <div className="tnum" style={{ fontSize: 26, fontWeight: 700, letterSpacing: "-0.02em" }}>{s.v}</div>
              </div>
            ))}
          </div>

          <div style={{ fontSize: 12, fontWeight: 600, color: "var(--tx-2)", textTransform: "uppercase", letterSpacing: ".06em", marginBottom: 12 }}>All datasets</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 16 }}>
            {ds.map((d) => (
              <DatasetCard key={d.id} d={d} onOpen={() => ctx.go({ name: "grid", datasetId: d.id })} onImport={() => setImporting(d)} />
            ))}
          </div>
        </div>
      </div>
      {importing && <ImportModal dataset={importing} onClose={() => setImporting(null)} />}
    </>
  );
}

function DatasetCard({ d, onOpen, onImport }) {
  const thumbs = d.images.slice(0, 4);
  const reviewPct = d.reviewed / d.count;
  return (
    <div className="card" style={{ overflow: "hidden", cursor: "pointer", transition: "border-color .14s, transform .1s" }}
      onClick={onOpen}
      onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--line-2)"; e.currentTarget.style.transform = "translateY(-2px)"; }}
      onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--line)"; e.currentTarget.style.transform = "none"; }}>
      <div style={{ display: "flex", gap: 2, height: 116, background: "var(--bg-canvas)" }}>
        {thumbs.map((im, i) => (
          <div key={i} style={{ flex: 1, position: "relative", overflow: "hidden" }}>
            <Scene image={im} style={{ width: "100%", height: "100%", display: "block" }} />
          </div>
        ))}
      </div>
      <div style={{ padding: "14px 16px" }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontWeight: 700, fontSize: 15, letterSpacing: "-0.01em" }}>{d.name}</div>
            <div style={{ fontSize: 12, color: "var(--tx-2)", marginTop: 2 }}>{d.desc}</div>
          </div>
          <div style={{ display: "flex", gap: 5 }}>{d.models.map((m) => <ModelChip key={m} id={m} size="sm" />)}</div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 18, marginTop: 14 }}>
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--tx-2)", marginBottom: 5 }}>
              <span>Pred ↔ GT agreement</span>
              <span className="tnum" style={{ color: "var(--tx-0)", fontWeight: 600 }}>{vhelp.pct(d.agreement)}</span>
            </div>
            <AgreeBar value={d.agreement} />
          </div>
          <div style={{ textAlign: "right" }}>
            <div className="tnum" style={{ fontSize: 13, fontWeight: 600, color: d.conflicts ? "var(--conflict)" : "var(--match)" }}>{d.conflicts}</div>
            <div style={{ fontSize: 10.5, color: "var(--tx-2)" }}>conflicts</div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 14, paddingTop: 13, borderTop: "1px solid var(--line)" }}>
          <span className="mono" style={{ fontSize: 11, color: "var(--tx-2)" }}>{d.count} imgs · run:{d.runId}</span>
          <div style={{ flex: 1 }} />
          <span className="tnum" style={{ fontSize: 11, color: "var(--tx-2)" }}>{Math.round(reviewPct * 100)}% reviewed</span>
          <button className="btn sm ghost" onClick={(e) => { e.stopPropagation(); onImport(); }}><Icon name="upload" size={13} />Labels</button>
          <button className="btn sm" onClick={(e) => { e.stopPropagation(); onOpen(); }}>Open<Icon name="arrowR" size={13} /></button>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { DatasetsScreen, ImportModal, TopBar, AgreeBar, ModelChip, StatusDot });
