/* ============================================================================
   Veridian Studio — Import labeled data flow (Sources -> ground-truth layer).
   Formats (COCO/YOLO/CSV/VOC, audio segments), CSV column-mapper, and a
   label-lint health check (idea #6). Media-only staging variant too.
   Exposes: window.ImportFlow (receives ProcessingView/DoneView/DropZone props)
   ========================================================================== */

const VISION_FORMATS = [
  { id: "coco", label: "COCO JSON", ext: ".json", files: ["instances_val.json"], note: "images / annotations / categories" },
  { id: "yolo", label: "YOLO txt", ext: ".txt", files: ["000001.txt", "000002.txt", "000003.txt", "…", "classes.txt"], note: "class cx cy w h (normalized)" },
  { id: "csv", label: "CSV / parquet", ext: ".csv", files: ["ground_truth.csv"], note: "one row per box", mapper: true },
  { id: "voc", label: "Pascal VOC", ext: ".xml", files: ["000001.xml", "000002.xml", "…"], note: "per-image XML" },
];
const AUDIO_FORMATS = [
  { id: "audacity", label: "Audacity labels", ext: ".txt", files: ["clip_001.txt", "clip_002.txt", "…"], note: "start \\t end \\t label" },
  { id: "rttm", label: "RTTM", ext: ".rttm", files: ["segments.rttm"], note: "diarization segments" },
  { id: "csv", label: "CSV", ext: ".csv", files: ["segments.csv"], note: "start,end,label,score", mapper: true },
];

const CSV_COLUMNS = ["image_id", "x_min", "y_min", "x_max", "y_max", "label", "score", "frame_ts"];
const CSV_TARGETS = [
  { id: "image", label: "Image key", auto: "image_id" },
  { id: "x1", label: "X min", auto: "x_min" },
  { id: "y1", label: "Y min", auto: "y_min" },
  { id: "x2", label: "X max", auto: "x_max" },
  { id: "y2", label: "Y max", auto: "y_max" },
  { id: "cls", label: "Class", auto: "label" },
  { id: "conf", label: "Score (opt)", auto: "score" },
];

// deterministic-ish lint report for a parsed label set
function lintReport(modality, total) {
  if (modality === "audio") {
    return {
      images: total.clips, boxes: total.segs,
      checks: [
        { sev: "ok", label: "Segments parsed", n: total.segs, note: `across ${total.clips} clips` },
        { sev: "warn", label: "Overlapping segments", n: 4, note: "same track, time overlap > 0" },
        { sev: "warn", label: "Zero-length segments", n: 2, note: "start == end" },
        { sev: "error", label: "Out-of-range timestamps", n: 1, note: "end beyond clip duration" },
        { sev: "info", label: "Label balance", n: null, note: "speech 61% · music 18% · other 21%" },
      ],
    };
  }
  return {
    images: total.images, boxes: total.boxes,
    checks: [
      { sev: "ok", label: "Boxes parsed", n: total.boxes, note: `across ${total.images} images` },
      { sev: "warn", label: "Boxes clipped to image bounds", n: 7, note: "coords outside [0,1]" },
      { sev: "warn", label: "Duplicate / overlapping (IoU > .95)", n: 12, note: "likely double-labelled" },
      { sev: "warn", label: "Sub-8px boxes", n: 23, note: "below trainable size" },
      { sev: "error", label: "Label files without an image", n: 3, note: "unmatched — will skip" },
      { sev: "info", label: "Class imbalance", n: null, note: "car 72% · person 19% · other 9%" },
    ],
  };
}
function healthScore(report) {
  const errs = report.checks.filter((c) => c.sev === "error").reduce((s, c) => s + (c.n || 0), 0);
  const warns = report.checks.filter((c) => c.sev === "warn").reduce((s, c) => s + (c.n || 0), 0);
  return Math.max(72, Math.round(100 - errs * 4 - warns * 0.4));
}

function ImportFlow({ mode, onExit, ProcessingView, DoneView, DropZone }) {
  const ctx = React.useContext(VDCtx);
  const mediaOnly = mode === "media";
  const [stage, setStage] = React.useState("compose");
  const [modality, setModality] = React.useState("vision");
  const [fmt, setFmt] = React.useState("coco");
  const [dropped, setDropped] = React.useState(true);
  const [dsName, setDsName] = React.useState(mediaOnly ? "staged_media" : "imported_eval_set");
  const [mapping, setMapping] = React.useState(() => Object.fromEntries(CSV_TARGETS.map((t) => [t.id, t.auto])));
  const [compareModel, setCompareModel] = React.useState(null);
  const createdRef = React.useRef(null);

  const formats = modality === "audio" ? AUDIO_FORMATS : VISION_FORMATS;
  const fmtDef = formats.find((f) => f.id === fmt) || formats[0];
  const totals = modality === "audio" ? { clips: 14, segs: 286 } : { images: 820, boxes: 2410 };
  const report = React.useMemo(() => lintReport(modality, totals), [modality]);
  const health = healthScore(report);

  React.useEffect(() => { if (!formats.some((f) => f.id === fmt)) setFmt(formats[0].id); }, [modality]);

  const finish = () => {
    // create through the backend seam; mock writes the cache, rest POSTs /datasets:import
    const spec = modality === "audio"
      ? { kind: "audio", name: dsName, desc: "Imported audio segments (ground-truth layer)." }
      : { kind: "vision", name: dsName, desc: mediaOnly ? "Staged media (no labels yet)." : "Imported ground-truth labels.", models: compareModel ? [compareModel] : ["amod"], palette: "dusk", n: mediaOnly ? 18 : 24, format: fmt, mapping: fmtDef.mapper ? mapping : undefined, compareRun: compareModel || undefined };
    Promise.resolve(window.VeridianAPI.createDatasetFromImport(spec)).then((res) => {
      createdRef.current = window.VD.getDataset(res.datasetId);
      setStage("done");
    });
  };

  if (stage === "processing") {
    const stages = mediaOnly
      ? [{ label: "Reading media", icon: "image" }, { label: "Hash + dedup", icon: "link" }, { label: "Writing to Postgres", icon: "database" }]
      : modality === "audio"
        ? [{ label: "Decoding audio", icon: "audio" }, { label: "Parsing segments", icon: "wave" }, { label: "Validating", icon: "shield" }, { label: "Writing to Postgres", icon: "database" }]
        : [{ label: "Reading media", icon: "image" }, { label: "Parsing labels", icon: "tag" }, { label: "Validating", icon: "shield" }, ...(compareModel ? [{ label: "Compare run", icon: "cpu" }] : []), { label: "Writing to Postgres", icon: "database" }];
    const unit = modality === "audio" ? "segments" : mediaOnly ? "files" : "labels";
    const total = modality === "audio" ? totals.segs : mediaOnly ? 540 : totals.boxes;
    const log = [
      "» connecting to veridian-db.lattice.internal (PostgreSQL)",
      `» reading ${fmtDef.label} from drop`,
      ...(mediaOnly ? ["» hashing media (sha-256)"] : [`» parsing ${total.toLocaleString()} ${unit}`, "» running label health checks", `✓ health ${health}/100 — ${report.checks.filter((c) => c.sev === "error").length} blocking issue(s) auto-skipped`]),
      ...(compareModel ? [`» running ${compareModel.toUpperCase()} to diff against imported GT`] : []),
      "» content hashing + dedup",
      "» writing ground-truth layer + image rows",
      "✓ import complete",
    ];
    return <ProcessingView title={mediaOnly ? "Staging media" : "Importing labels"} subtitle={mediaOnly ? "Hashing and registering media — attach labels later." : "Parsing, validating, and committing your ground-truth layer."}
      stages={stages} models={compareModel ? [compareModel] : []} modelStage={3} total={total} unit={unit} logLines={log} onDone={finish} />;
  }
  if (stage === "done") {
    const ds = createdRef.current;
    const card = {
      icon: modality === "audio" ? "audio" : mediaOnly ? "image" : "tag", color: "var(--m-fr)",
      title: ds.name, badge: "created",
      sub: mediaOnly ? `media staged · ${ds.count} items` : (modality === "audio" ? `ground-truth layer · ${totals.segs} segments` : `ground-truth layer · ${totals.boxes.toLocaleString()} boxes`),
      stats: mediaOnly
        ? [{ v: ds.count, l: "items", c: "var(--match)" }, { v: "0", l: "layers", c: "var(--tx-2)" }]
        : [{ v: (modality === "audio" ? totals.segs : totals.boxes).toLocaleString(), l: modality === "audio" ? "segments" : "boxes", c: "var(--match)" }, { v: health + "/100", l: "health", c: health > 90 ? "var(--match)" : "var(--gt)" }, ...(compareModel ? [{ v: compareModel.toUpperCase(), l: "compare run", c: "var(--pred)" }] : [])],
      openDsId: ds.id,
    };
    return <DoneView title={mediaOnly ? "Media staged" : "Import complete"} subtitle={mediaOnly ? "Your media is registered and ready to label." : "Your ground-truth layer is in — open it to verify or attach a model run."} cards={[card]} onReset={() => setStage("compose")} onExit={onExit} go={ctx.go} />;
  }

  return (
    <>
      <TopBar crumbs={[{ label: "New source", onClick: onExit }, { label: mediaOnly ? "Upload media" : "Import labeled data" }]}>
        <button className="btn primary" onClick={() => setStage("processing")}><Icon name={mediaOnly ? "upload" : "check"} size={15} />{mediaOnly ? "Stage media" : "Import & validate"}</button>
      </TopBar>

      <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
        {/* left: format + upload + mapper */}
        <div className="scroll" style={{ flex: 1, padding: "20px 24px" }}>
          <div style={{ maxWidth: 700, margin: "0 auto", display: "flex", flexDirection: "column", gap: 18 }}>
            {/* modality */}
            <div>
              <SectionLabel>Modality</SectionLabel>
              <div style={{ display: "flex", gap: 8 }}>
                {[["vision", "image", "Vision (images / video frames)"], ["audio", "audio", "Audio (waveform + segments)"]].map(([id, ic, lab]) => (
                  <button key={id} onClick={() => setModality(id)} style={{ flex: 1, display: "flex", alignItems: "center", gap: 9, padding: "11px 13px", borderRadius: 9, border: "1px solid " + (modality === id ? "var(--accent)" : "var(--line-2)"), background: modality === id ? "var(--accent-dim)" : "var(--bg-1)" }}>
                    <Icon name={ic} size={18} style={{ color: modality === id ? "var(--accent-2)" : "var(--tx-2)" }} /><span style={{ fontSize: 13, fontWeight: 600 }}>{lab}</span>
                  </button>
                ))}
              </div>
            </div>

            {!mediaOnly && (
              <div>
                <SectionLabel>Label format</SectionLabel>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {formats.map((f) => (
                    <button key={f.id} onClick={() => setFmt(f.id)} style={{ padding: "9px 13px", borderRadius: 8, textAlign: "left", border: "1px solid " + (fmt === f.id ? "var(--accent)" : "var(--line-2)"), background: fmt === f.id ? "var(--accent-dim)" : "var(--bg-2)", minWidth: 120 }}>
                      <div style={{ fontWeight: 600, fontSize: 12.5 }}>{f.label}</div><div className="mono" style={{ fontSize: 10, color: "var(--tx-2)" }}>{f.ext} · {f.note}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div>
              <SectionLabel>{mediaOnly ? "Media" : "Files"}</SectionLabel>
              <DropZone onBrowse={() => setDropped(true)} onSamples={() => setDropped(true)} big={!dropped} label={mediaOnly ? "Drop images / video / audio" : `Drop ${fmtDef.label} + media`} sample="use sample set" />
              {dropped && (
                <div style={{ marginTop: 12, background: "var(--bg-2)", border: "1px solid var(--line)", borderRadius: 8, padding: 12 }}>
                  <div style={{ fontSize: 11, color: "var(--tx-2)", marginBottom: 7 }}>Detected</div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {(mediaOnly ? ["frame_0001.png", "frame_0002.png", "…", "+540 files"] : fmtDef.files).map((f, i) => <span key={i} className="chip mono" style={{ fontSize: 10.5, borderColor: "var(--line)", background: "var(--bg-1)" }}>{f}</span>)}
                    {!mediaOnly && <span className="chip mono" style={{ fontSize: 10.5, borderColor: "var(--line)", background: "var(--bg-1)" }}>{modality === "audio" ? "14 wav" : "820 images"}</span>}
                  </div>
                </div>
              )}
            </div>

            {/* CSV column mapper */}
            {!mediaOnly && fmtDef.mapper && dropped && <CsvMapper mapping={mapping} setMapping={setMapping} modality={modality} />}

            {/* label preview */}
            {!mediaOnly && dropped && modality === "vision" && <LabelPreview />}
          </div>
        </div>

        {/* right: health check + target */}
        <div style={{ width: 332, flexShrink: 0, borderLeft: "1px solid var(--line)", background: "var(--bg-1)", display: "flex", flexDirection: "column" }}>
          {!mediaOnly && (
            <div style={{ padding: 16, borderBottom: "1px solid var(--line)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                <Icon name="shield" size={16} style={{ color: health > 90 ? "var(--match)" : "var(--gt)" }} />
                <span style={{ fontWeight: 700, fontSize: 14 }}>Label health check</span>
                <div style={{ flex: 1 }} />
                <span className="tnum" style={{ fontWeight: 700, fontSize: 16, color: health > 90 ? "var(--match)" : "var(--gt)" }}>{health}<span style={{ fontSize: 11, color: "var(--tx-3)" }}>/100</span></span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                {report.checks.map((c, i) => {
                  const col = c.sev === "error" ? "var(--conflict)" : c.sev === "warn" ? "var(--gt)" : c.sev === "ok" ? "var(--match)" : "var(--tx-2)";
                  return (
                    <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 9, padding: "7px 9px", background: "var(--bg-2)", borderRadius: 7, border: "1px solid var(--line)" }}>
                      <Icon name={c.sev === "error" ? "alert" : c.sev === "warn" ? "alert" : c.sev === "ok" ? "check" : "dot"} size={14} style={{ color: col, marginTop: 1, flexShrink: 0 }} />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 12, fontWeight: 500, display: "flex", gap: 6 }}>{c.label}{c.n != null && <span className="tnum" style={{ color: col, fontWeight: 700 }}>{c.n.toLocaleString()}</span>}</div>
                        <div style={{ fontSize: 10.5, color: "var(--tx-2)" }}>{c.note}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div className="scroll" style={{ flex: 1, padding: 16, display: "flex", flexDirection: "column", gap: 16 }}>
            <div>
              <SectionLabel>New dataset</SectionLabel>
              <input value={dsName} onChange={(e) => setDsName(e.target.value)} className="mono" style={{ width: "100%", background: "var(--bg-2)", border: "1px solid var(--line-2)", borderRadius: 7, color: "var(--tx-0)", fontSize: 12.5, padding: "9px 11px", outline: "none" }} />
              <div style={{ fontSize: 10.5, color: "var(--tx-2)", marginTop: 6, display: "flex", alignItems: "center", gap: 6 }}><Icon name="layers" size={12} style={{ color: "var(--m-fr)" }} />produces a <b style={{ color: "var(--tx-1)" }}>ground-truth</b> layer</div>
            </div>

            {!mediaOnly && modality === "vision" && (
              <div>
                <SectionLabel>Attach a compare run (optional)</SectionLabel>
                <div style={{ fontSize: 11, color: "var(--tx-2)", marginBottom: 8 }}>Also run a model so the canvas can diff predictions against your imported ground-truth right away.</div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {["amod", "fd", "qrcode"].map((m) => (
                    <button key={m} onClick={() => setCompareModel(compareModel === m ? null : m)} className="chip" style={{ cursor: "pointer", borderColor: compareModel === m ? vhelp.modelColor(m) : "var(--line-2)", background: compareModel === m ? "var(--bg-3)" : "transparent" }}><span className="dot" style={{ background: vhelp.modelColor(m) }} />{window.VD.models.find((x) => x.id === m).short}{compareModel === m && <Icon name="check" size={11} />}</button>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div style={{ padding: 16, borderTop: "1px solid var(--line)" }}>
            <button className="btn primary" onClick={() => setStage("processing")} style={{ width: "100%", justifyContent: "center" }}><Icon name={mediaOnly ? "upload" : "check"} size={15} />{mediaOnly ? "Stage media" : "Import & validate"}</button>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 10, fontSize: 10.5, color: "var(--tx-3)", justifyContent: "center" }}><Icon name="database" size={12} style={{ color: "var(--match)" }} />writes to {window.VD.storage.engine}</div>
          </div>
        </div>
      </div>
    </>
  );
}

function SectionLabel({ children }) {
  return <div style={{ fontSize: 11, fontWeight: 600, color: "var(--tx-2)", textTransform: "uppercase", letterSpacing: ".06em", marginBottom: 8 }}>{children}</div>;
}

function CsvMapper({ mapping, setMapping, modality }) {
  return (
    <div className="card" style={{ padding: 14 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <Icon name="sliders" size={15} style={{ color: "var(--accent-2)" }} />
        <span style={{ fontWeight: 600, fontSize: 13 }}>Map columns</span>
        <span className="chip" style={{ borderColor: "var(--line)", color: "var(--tx-2)", fontSize: 10 }}>auto-detected</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 9 }}>
        {CSV_TARGETS.map((t) => (
          <div key={t.id} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 11.5, color: "var(--tx-1)", width: 78, flexShrink: 0 }}>{t.label}</span>
            <select value={mapping[t.id]} onChange={(e) => setMapping((m) => ({ ...m, [t.id]: e.target.value }))} style={{ flex: 1, background: "var(--bg-2)", border: "1px solid " + (mapping[t.id] === t.auto ? "var(--line-2)" : "var(--accent)"), borderRadius: 6, color: "var(--tx-0)", padding: "5px 7px", fontSize: 11.5 }}>
              <option value="">—</option>
              {CSV_COLUMNS.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 10, fontSize: 11, color: "var(--tx-2)", display: "flex", alignItems: "center", gap: 6 }}><Icon name="check" size={12} style={{ color: "var(--match)" }} />Coordinates read as <b style={{ color: "var(--tx-1)" }}>xyxy (absolute)</b> · 7 columns mapped</div>
    </div>
  );
}

function LabelPreview() {
  const sample = window.VD.datasets.find((d) => d.kind === "vision").images[2];
  return (
    <div>
      <SectionLabel>Preview · imported labels on a sample frame</SectionLabel>
      <div className="card" style={{ overflow: "hidden", maxWidth: 360 }}>
        <div style={{ position: "relative", aspectRatio: "1280/800", background: "var(--bg-canvas)" }}>
          <Scene image={sample} style={{ width: "100%", height: "100%" }} />
          <svg viewBox="0 0 1 0.625" preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
            {sample.objects.map((o, i) => { const b = o.gt || o.pred; if (!b) return null; return <rect key={i} x={b[0]} y={b[1] * 0.625} width={b[2]} height={b[3] * 0.625} fill="none" stroke="var(--gt)" strokeWidth="0.004" strokeDasharray="0.01 0.006" vectorEffect="non-scaling-stroke" />; })}
          </svg>
          <div style={{ position: "absolute", top: 8, left: 8 }} className="chip"><span className="dot" style={{ background: "var(--gt)" }} />ground-truth</div>
        </div>
      </div>
    </div>
  );
}

window.ImportFlow = ImportFlow;
