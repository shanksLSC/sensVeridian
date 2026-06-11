/* ============================================================================
   Veridian Studio — Image grid screen: filter queues, disagreement heatmap,
   thumbnail overlays. Exposes: GridScreen, QueueScreen
   ========================================================================== */

function MiniOverlay({ image }) {
  // small pred/gt box overlay for thumbnails
  return (
    <svg viewBox="0 0 1 0.625" preserveAspectRatio="none" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
      {(image.objects || []).map((o, i) => {
        const conflict = o.state !== "match";
        const b = o.gt || o.pred;
        if (!b) return null;
        const col = conflict ? "var(--conflict)" : "var(--match)";
        return <rect key={i} x={b[0]} y={b[1]} width={b[2]} height={b[3]} fill="none" stroke={col} strokeWidth="0.004" vectorEffect="non-scaling-stroke" opacity={conflict ? 0.95 : 0.7} />;
      })}
    </svg>
  );
}

const FILTERS = [
  { id: "all", label: "All", test: () => true },
  { id: "conflict", label: "Disagreements", test: (im) => im.conflicts > 0, c: "var(--conflict)" },
  { id: "miss", label: "Misses (FN)", test: (im) => im.objects.some((o) => o.state === "miss"), c: "var(--conflict)" },
  { id: "fp", label: "False positives", test: (im) => im.objects.some((o) => o.state === "fp"), c: "var(--conflict)" },
  { id: "unreviewed", label: "Unreviewed", test: (im) => im.status === "unreviewed" },
  { id: "verified", label: "Verified", test: (im) => im.status === "verified", c: "var(--match)" },
  { id: "flagged", label: "Flagged", test: (im) => im.status === "flagged", c: "var(--gt)" },
];

function GridScreen() {
  const ctx = React.useContext(VDCtx);
  const d = window.VD.getDataset(ctx.route.datasetId);
  if (d.kind === "audio") return <window.AudioGrid dataset={d} />;
  const [filter, setFilter] = React.useState("all");
  const [view, setView] = React.useState("grid");
  const [sort, setSort] = React.useState("conflicts");
  const [importing, setImporting] = React.useState(false);

  const images = React.useMemo(() => {
    let arr = d.images.filter(FILTERS.find((f) => f.id === filter).test);
    if (sort === "conflicts") arr = [...arr].sort((a, b) => b.conflicts - a.conflicts);
    else if (sort === "agreement") arr = [...arr].sort((a, b) => a.agreement - b.agreement);
    else arr = [...arr].sort((a, b) => a.idx - b.idx);
    return arr;
  }, [d, filter, sort]);

  const open = (im) => ctx.go({ name: "canvas", datasetId: d.id, imageId: im.id });

  return (
    <>
      <TopBar crumbs={[{ label: "Datasets", onClick: () => ctx.go({ name: "datasets" }) }, { label: d.name }]}>
        <div style={{ display: "flex", gap: 5, marginRight: 8 }}>{d.models.map((m) => <ModelChip key={m} id={m} size="sm" />)}</div>
        <button className="btn" onClick={() => setImporting(true)}><Icon name="upload" size={15} />Import labels</button>
      </TopBar>

      {/* control bar */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 18px", borderBottom: "1px solid var(--line)", background: "var(--bg-1)", flexWrap: "wrap" }}>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {FILTERS.map((f) => {
            const n = d.images.filter(f.test).length;
            const active = filter === f.id;
            return (
              <button key={f.id} onClick={() => setFilter(f.id)} style={{
                display: "flex", alignItems: "center", gap: 7, padding: "6px 11px", borderRadius: 7, fontSize: 12.5, fontWeight: 500,
                border: "1px solid " + (active ? "var(--line-2)" : "transparent"),
                background: active ? "var(--bg-3)" : "transparent", color: active ? "var(--tx-0)" : "var(--tx-1)",
              }}>
                {f.c && <span style={{ width: 7, height: 7, borderRadius: "50%", background: f.c }} />}
                {f.label}
                <span className="tnum mono" style={{ fontSize: 10.5, color: "var(--tx-3)" }}>{n}</span>
              </button>
            );
          })}
        </div>
        <div style={{ flex: 1 }} />
        <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--tx-2)" }}>
          <Icon name="filter" size={13} />Sort
          <select value={sort} onChange={(e) => setSort(e.target.value)} style={{ background: "var(--bg-3)", border: "1px solid var(--line-2)", borderRadius: 6, color: "var(--tx-0)", padding: "5px 8px", fontSize: 12 }}>
            <option value="conflicts">Most conflicts</option>
            <option value="agreement">Lowest agreement</option>
            <option value="index">Capture order</option>
          </select>
        </div>
        <div style={{ display: "flex", background: "var(--bg-3)", borderRadius: 7, padding: 2, border: "1px solid var(--line-2)" }}>
          {[["grid", "grid", "Grid"], ["heatmap", "gauge", "Heatmap"], ["map", "scatter", "Map"], ["tracks", "film", "Tracks"]].map(([v, ic, lab]) => (
            <button key={v} onClick={() => setView(v)} className="btn sm" style={{ background: view === v ? "var(--bg-1)" : "transparent", border: "none", padding: "5px 9px", color: view === v ? "var(--tx-0)" : "var(--tx-2)" }}>
              <Icon name={ic} size={14} />{lab}
            </button>
          ))}
        </div>
      </div>

      {(view === "map" || view === "tracks") ? (
        <div style={{ flex: 1, padding: "18px 22px", minHeight: 0, overflow: "hidden" }}>
          {view === "map" ? <window.EmbeddingMap dataset={d} onOpen={open} /> : <window.TracksView dataset={d} onOpen={open} />}
        </div>
      ) : (
      <div className="scroll" style={{ flex: 1, padding: view === "grid" ? "18px" : "20px 22px" }}>
        {window.EvalPanel && <window.EvalPanel dataset={d} />}
        {view === "grid" ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(232px, 1fr))", gap: 14 }}>
            {images.map((im) => <Thumb key={im.id} image={im} onClick={() => open(im)} />)}
          </div>
        ) : (
          <Heatmap dataset={d} images={images} onOpen={open} />
        )}
        {images.length === 0 && <div style={{ textAlign: "center", color: "var(--tx-2)", padding: 60 }}>No images match this filter.</div>}
      </div>
      )}
      {importing && <ImportModal dataset={d} onClose={() => setImporting(false)} />}
    </>
  );
}

function Thumb({ image, onClick }) {
  const [hover, setHover] = React.useState(false);
  return (
    <div onClick={onClick} onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}
      className="card" style={{ overflow: "hidden", cursor: "pointer", borderColor: hover ? "var(--line-2)" : "var(--line)" }}>
      <div style={{ position: "relative", aspectRatio: "1280/800", background: "var(--bg-canvas)" }}>
        <Scene image={image} style={{ width: "100%", height: "100%", display: "block" }} />
        <MiniOverlay image={image} />
        <div style={{ position: "absolute", top: 8, left: 8, display: "flex", gap: 5 }}>
          {image.augmented && <span className="chip" style={{ background: "rgba(7,9,13,.7)", borderColor: "var(--line-2)", fontSize: 10, padding: "1px 7px" }}><Icon name="ruler" size={10} />aug</span>}
        </div>
        <div style={{ position: "absolute", top: 8, right: 8 }}>
          <span className="chip" style={{ background: "rgba(7,9,13,.72)", borderColor: image.conflicts ? "var(--conflict)" : "var(--line-2)", fontSize: 10.5, padding: "2px 8px", color: image.conflicts ? "var(--conflict)" : "var(--match)", fontWeight: 600 }}>
            {image.conflicts ? `${image.conflicts} conflict${image.conflicts > 1 ? "s" : ""}` : "clean"}
          </span>
        </div>
        <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 3, background: "var(--bg-3)" }}>
          <div style={{ height: "100%", width: (image.agreement * 100) + "%", background: image.agreement >= 0.85 ? "var(--match)" : image.agreement >= 0.65 ? "var(--gt)" : "var(--conflict)" }} />
        </div>
      </div>
      <div style={{ padding: "9px 11px", display: "flex", alignItems: "center", gap: 8 }}>
        <span className="mono" style={{ fontSize: 11, color: "var(--tx-2)", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{vhelp.shortId(image.id, 10)}</span>
        <span style={{ width: 7, height: 7, borderRadius: "50%", background: { verified: "var(--match)", flagged: "var(--gt)", unreviewed: "var(--tx-3)" }[image.status] }} title={image.status} />
        <span className="tnum" style={{ fontSize: 11, color: "var(--tx-1)", fontWeight: 600 }}>{vhelp.pct(image.agreement)}</span>
      </div>
    </div>
  );
}

function Heatmap({ dataset, images, onOpen }) {
  const [tip, setTip] = React.useState(null);
  const cellColor = (im) => {
    const a = im.agreement;
    if (im.objects.every((o) => o.state === "match")) return "var(--match)";
    const c = a >= 0.85 ? "var(--match)" : a >= 0.6 ? "var(--gt)" : "var(--conflict)";
    return c;
  };
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 16, fontSize: 12, color: "var(--tx-2)" }}>
        <span>Each cell = one image, sized by detections, colored by agreement.</span>
        <div style={{ flex: 1 }} />
        {[["var(--match)", "≥85%"], ["var(--gt)", "60–85%"], ["var(--conflict)", "<60%"]].map(([c, l]) => (
          <span key={l} style={{ display: "flex", alignItems: "center", gap: 6 }}><span style={{ width: 11, height: 11, borderRadius: 3, background: c }} />{l}</span>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(54px, 1fr))", gap: 6, position: "relative" }}>
        {images.map((im) => {
          const opacity = 0.4 + Math.min(1, im.objects.length / 6) * 0.6;
          return (
            <button key={im.id} onClick={() => onOpen(im)}
              onMouseEnter={(e) => setTip({ im, x: e.currentTarget.offsetLeft, y: e.currentTarget.offsetTop })}
              onMouseLeave={() => setTip(null)}
              style={{ aspectRatio: "1", borderRadius: 6, background: cellColor(im), opacity, position: "relative", border: "1px solid rgba(255,255,255,.06)", transition: "transform .1s" }}
              onMouseOver={(e) => (e.currentTarget.style.transform = "scale(1.08)")}
              onMouseOut={(e) => (e.currentTarget.style.transform = "none")}>
              <span className="tnum" style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", fontSize: 11, fontWeight: 700, color: "rgba(7,9,13,.8)" }}>{im.objects.length}</span>
              {im.conflicts > 0 && <span style={{ position: "absolute", top: 3, right: 3, fontSize: 9, fontWeight: 700, color: "rgba(7,9,13,.85)" }}>{im.conflicts}</span>}
            </button>
          );
        })}
      </div>
      {tip && (
        <div style={{ position: "absolute", left: tip.x, top: tip.y + 64, pointerEvents: "none", zIndex: 20 }}>
          <div className="card pop" style={{ width: 150, boxShadow: "var(--shadow-pop)", overflow: "hidden" }}>
            <div style={{ aspectRatio: "1280/800", position: "relative" }}><Scene image={tip.im} style={{ width: "100%", height: "100%" }} /><MiniOverlay image={tip.im} /></div>
            <div style={{ padding: 8 }}>
              <div className="mono" style={{ fontSize: 10, color: "var(--tx-2)" }}>{vhelp.shortId(tip.im.id, 12)}</div>
              <div style={{ fontSize: 11, marginTop: 3 }}>{vhelp.pct(tip.im.agreement)} agree · {tip.im.conflicts} conflicts</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ---- Review queue (cross-dataset) ---------------------------------------- */
function QueueScreen() {
  const ctx = React.useContext(VDCtx);
  const isRest = (window.VERIDIAN_CONFIG || {}).backend === "rest";
  // Local aggregation (mock / fallback): conflicts across already-loaded datasets.
  const localRows = () => {
    const out = [];
    const reviews = ctx.reviews || {};
    window.VD.datasets.forEach((d) => (d.images || []).forEach((im) => (im.objects || []).forEach((o) => {
      if (o.state !== "match" && !reviews[o.id]) out.push({ d, im, o });
    })));
    return out;
  };
  const [rows, setRows] = React.useState(localRows);
  React.useEffect(() => {
    let alive = true;
    if (isRest && window.VeridianAPI && window.VeridianAPI.getQueue) {
      window.VeridianAPI.getQueue(200)
        .then((data) => {
          if (!alive || !data) return;  // null -> keep local fallback (mock)
          setRows(data.map((r) => ({
            d: window.VD.getDataset(r.datasetId) || { id: r.datasetId, name: r.datasetName || r.datasetId },
            im: { id: r.imageId, datasetId: r.datasetId, objects: [], src: `/api/v1/datasets/${r.datasetId}/images/${r.imageId}/raw` },
            o: { id: r.detId, cls: r.cls, state: r.state, conf: r.conf || 0, iou: r.iou || 0 },
          })));
        })
        .catch((e) => console.warn("[veridian] queue load failed", e));
    }
    return () => { alive = false; };
  }, [isRest]);
  rows.sort((a, b) => a.o.conf - b.o.conf);
  return (
    <>
      <TopBar crumbs={[{ label: "Review Queue" }]}>
        <span className="chip" style={{ borderColor: "var(--conflict)", color: "var(--conflict)" }}><Icon name="alert" size={12} />{rows.length} conflicts</span>
      </TopBar>
      <div className="scroll" style={{ flex: 1, padding: "18px 22px" }}>
        <div style={{ maxWidth: 1100, margin: "0 auto" }}>
          <div style={{ fontSize: 12.5, color: "var(--tx-2)", marginBottom: 14 }}>Every prediction↔ground-truth disagreement across all datasets, lowest-confidence first. Triage from here or open the frame.</div>
          <div className="card" style={{ overflow: "hidden" }}>
            {rows.slice(0, 60).map((r, i) => {
              const rev = ctx.reviews[r.o.id];
              return (
                <button key={i} onClick={() => ctx.go({ name: "canvas", datasetId: r.d.id, imageId: r.im.id })}
                  style={{ display: "flex", alignItems: "center", gap: 14, width: "100%", textAlign: "left", padding: "10px 14px", borderBottom: i < 59 ? "1px solid var(--line)" : "none", background: "transparent" }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-2)")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}>
                  <div style={{ width: 64, height: 40, borderRadius: 5, overflow: "hidden", position: "relative", flexShrink: 0, border: "1px solid var(--line)" }}>
                    <Scene image={r.im} style={{ width: "100%", height: "100%" }} /><MiniOverlay image={r.im} />
                  </div>
                  <span style={{ width: 120, fontSize: 12, color: "var(--tx-1)" }}>{r.d.name.split(" — ")[0]}</span>
                  <span className="chip" style={{ borderColor: "transparent", background: vhelp.stateColor(r.o.state) + "22", color: vhelp.stateColor(r.o.state), fontWeight: 600 }}>{vhelp.stateLabel(r.o.state)}</span>
                  <span style={{ fontSize: 12.5, color: "var(--tx-0)", flex: 1 }}>{window.VD.CLASSES[r.o.cls].label}</span>
                  <span className="mono tnum" style={{ fontSize: 11.5, color: "var(--tx-2)" }}>conf {r.o.conf.toFixed(2)} · IoU {r.o.iou.toFixed(2)}</span>
                  {rev && <span className="chip" style={{ borderColor: "var(--match)", color: "var(--match)" }}><Icon name="check" size={11} />{rev.verdict}</span>}
                  <Icon name="chevR" size={15} style={{ color: "var(--tx-3)" }} />
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </>
  );
}

Object.assign(window, { GridScreen, QueueScreen });
