/* ============================================================================
   Veridian Studio — shared UI: icons, procedural scene, helpers.
   Exposes to window: Icon, Scene, vhelp
   ========================================================================== */
const { useState, useRef, useEffect, useMemo, useCallback } = React;

/* ---- icon set (stroke, 24x24) -------------------------------------------- */
const ICONS = {
  grid: "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z",
  layers: "M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5M3 17l9 5 9-5",
  box: "M3 7l9-4 9 4v10l-9 4-9-4zM3 7l9 4 9-4M12 11v10",
  tag: "M4 4h7l9 9-7 7-9-9zM8 8h.01",
  check: "M5 12l5 5L20 6",
  x: "M6 6l12 12M18 6L6 18",
  edit: "M4 20h4l11-11-4-4L4 16zM13 6l4 4",
  flag: "M5 21V4M5 4c3-2 6 2 9 0s4-1 5-1v9c-1 0-2-1-5 1s-6-2-9 0",
  search: "M11 4a7 7 0 100 14 7 7 0 000-14zM20 20l-4-4",
  chevR: "M9 6l6 6-6 6",
  chevL: "M15 6l-6 6 6 6",
  chevD: "M6 9l6 6 6-6",
  database: "M12 3c4.4 0 8 1.3 8 3s-3.6 3-8 3-8-1.3-8-3 3.6-3 8-3zM4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6",
  history: "M3 12a9 9 0 109-9 9 9 0 00-7 3.5M3 4v4h4M12 8v4l3 2",
  sliders: "M4 6h10M18 6h2M4 12h2M10 12h10M4 18h7M15 18h5M14 4v4M6 10v4M11 16v4",
  gauge: "M12 13l4-4M3 18a9 9 0 1118 0",
  compare: "M12 3v18M5 7l-3 4 3 4M19 7l3 4-3 4",
  image: "M3 4h18v16H3zM3 16l5-5 4 4 3-3 6 6",
  folder: "M3 6h6l2 2h10v11H3z",
  keyboard: "M3 6h18v12H3zM6 9h.01M10 9h.01M14 9h.01M18 9h.01M6 13h.01M18 13h.01M9 13h6",
  zoomIn: "M11 4a7 7 0 100 14 7 7 0 000-14zM20 20l-4-4M11 8v6M8 11h6",
  zoomOut: "M11 4a7 7 0 100 14 7 7 0 000-14zM20 20l-4-4M8 11h6",
  hand: "M8 13V5a1.5 1.5 0 013 0v6M11 11V4a1.5 1.5 0 013 0v7M14 11V6a1.5 1.5 0 013 0v8c0 3-2 6-5 6h-2c-2 0-3-1-4-3l-2-4a1.5 1.5 0 012-2l2 2",
  cursor: "M5 3l7 17 2-7 7-2z",
  polygon: "M12 3l8 5-3 10H7L4 8z",
  eye: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zM12 9a3 3 0 100 6 3 3 0 000-6z",
  eyeOff: "M3 3l18 18M10 6.5A9.8 9.8 0 0112 6c6 0 10 6 10 6a14 14 0 01-3 3.5M6 7C3.5 8.7 2 12 2 12s4 6 10 6a9 9 0 003.5-.7",
  arrowR: "M5 12h14M13 6l6 6-6 6",
  plus: "M12 5v14M5 12h14",
  upload: "M12 16V4M7 9l5-5 5 5M5 20h14",
  branch: "M6 4v12M6 16a3 3 0 100 4 3 3 0 000-4zM6 4a3 3 0 100-1M18 8a3 3 0 100-4 3 3 0 000 4zM18 8c0 5-6 4-6 8",
  clock: "M12 3a9 9 0 100 18 9 9 0 000-18zM12 7v5l3 2",
  cpu: "M9 9h6v6H9zM4 9H2M4 15H2M22 9h-2M22 15h-2M9 4V2M15 4V2M9 22v-2M15 22v-2M4 4h16v16H4z",
  filter: "M3 5h18l-7 8v6l-4-2v-4z",
  maximize: "M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5",
  ruler: "M3 8l5-5 13 13-5 5zM8 6l2 2M11 9l2 2M14 12l2 2",
  dot: "M12 12h.01",
  qr: "M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h2v2h-2zM18 14h2v2h-2zM14 18h2v2h-2zM18 18h2v2h-2z",
  face: "M12 3a9 9 0 100 18 9 9 0 000-18zM9 10h.01M15 10h.01M9 15c1 1 5 1 6 0",
  car: "M5 11l2-5h10l2 5M3 11h18v5H3zM6 16v2M18 16v2M6 13h.01M18 13h.01",
  download: "M12 4v12M7 11l5 5 5-5M5 20h14",
  sparkle: "M12 3l2 6 6 2-6 2-2 6-2-6-6-2 6-2z",
  link: "M9 15l6-6M10 7l1-1a4 4 0 016 6l-1 1M14 17l-1 1a4 4 0 01-6-6l1-1",
  alert: "M12 3l9 16H3zM12 10v4M12 17h.01",
  settings: "M12 9a3 3 0 100 6 3 3 0 000-6zM19 12a7 7 0 00-.1-1l2-1.5-2-3.5-2.4 1a7 7 0 00-1.7-1L14.5 2h-5l-.3 2.5a7 7 0 00-1.7 1l-2.4-1-2 3.5L3 11a7 7 0 000 2l-2 1.5 2 3.5 2.4-1a7 7 0 001.7 1l.3 2.5h5l.3-2.5a7 7 0 001.7-1l2.4 1 2-3.5L19 13a7 7 0 000-1z",
  wave: "M2 12h2l2-7 3 16 3-22 3 19 2-6h3",
  scatter: "M3 3v18h18M7 15a1.4 1.4 0 100 2.8 1.4 1.4 0 000-2.8zM11 9a1.4 1.4 0 100 2.8A1.4 1.4 0 0011 9zM16 13a1.4 1.4 0 100 2.8 1.4 1.4 0 000-2.8zM18 6a1.4 1.4 0 100 2.8A1.4 1.4 0 0018 6z",
  dag: "M5 4a2 2 0 100 4 2 2 0 000-4zM19 4a2 2 0 100 4 2 2 0 000-4zM12 16a2 2 0 100 4 2 2 0 000-4zM6 8l5 8M18 8l-5 8M7 6h10",
  shield: "M12 3l8 3v6c0 4-3.5 7-8 9-4.5-2-8-5-8-9V6z",
  audio: "M3 10v4h4l5 5V5L7 10zM16 9a4 4 0 010 6M18.5 6.5a8 8 0 010 11",
  film: "M3 4h18v16H3zM3 8h18M3 16h18M7 4v16M17 4v16",
  spark: "M3 17l5-5 4 4 8-9",
  trend: "M3 17l6-6 4 4 8-8M21 7v5h-5",
};
function Icon({ name, size = 16, stroke = 1.6, fill = false, style, className }) {
  const d = ICONS[name];
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round"
      style={style} className={className} aria-hidden="true">
      <path d={d} fill={fill ? "currentColor" : "none"} stroke={fill ? "none" : "currentColor"} />
    </svg>
  );
}

/* ---- helpers ------------------------------------------------------------- */
const vhelp = {
  pct: (v) => (v == null ? "—" : Math.round(v * 100) + "%"),
  pct1: (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%"),
  stateColor: (s) => ({ match: "var(--match)", miss: "var(--conflict)", fp: "var(--conflict)", mismatch: "var(--conflict)", low_conf: "var(--gt)" }[s] || "var(--tx-2)"),
  stateLabel: (s) => ({ match: "Agree", miss: "Missed (FN)", fp: "False positive", mismatch: "Class mismatch", low_conf: "Low confidence" }[s] || s),
  stateShort: (s) => ({ match: "OK", miss: "FN", fp: "FP", mismatch: "≠", low_conf: "low" }[s] || s),
  modelColor: (m) => ({ amod: "var(--m-amod)", qrcode: "var(--m-qr)", fd: "var(--m-fd)", fr: "var(--m-fr)" }[m] || "var(--accent)"),
  shortId: (id, n = 8) => (id ? id.slice(0, n) : ""),
};

/* ---- procedural scene (silhouettes under the boxes) ---------------------- */
const PALETTES = {
  dusk:      { sky: ["#1a2740", "#2d2236", "#3a2a2e"], ground: "#0c1018", haze: "#3b4a66", sun: "#c97b5a" },
  indoor:    { sky: ["#1c222e", "#222634", "#191d27"], ground: "#10141c", haze: "#39414f", sun: "#6b7384" },
  warehouse: { sky: ["#222018", "#2a2620", "#1c1a14"], ground: "#14120d", haze: "#4a4332", sun: "#b89a5e" },
};

function silhouette(o, key) {
  const b = o.gt || o.pred; if (!b) return null;
  const SY = 0.625;
  const x = b[0], y = b[1] * SY, w = b[2], h = b[3] * SY;
  const cx = x + w / 2;
  const g = window.VD.CLASSES[o.cls].glyph;
  const dark = "#05070b";
  const edge = "rgba(150,170,200,0.10)";
  const common = { fill: dark, stroke: edge, strokeWidth: 0.0015, vectorEffect: "non-scaling-stroke" };
  if (g === "person") {
    return (<g key={key} opacity="0.96">
      <ellipse cx={cx} cy={y + h * 0.11} rx={w * 0.26} ry={h * 0.1} {...common} />
      <path d={`M ${x + w * 0.2} ${y + h} L ${x + w * 0.28} ${y + h * 0.24} Q ${cx} ${y + h * 0.16} ${x + w * 0.72} ${y + h * 0.24} L ${x + w * 0.8} ${y + h} Z`} {...common} />
    </g>);
  }
  if (g === "face") {
    return (<g key={key} opacity="0.96">
      <ellipse cx={cx} cy={y + h * 0.5} rx={w * 0.42} ry={h * 0.46} fill="#0b0f16" stroke={edge} strokeWidth="0.001" vectorEffect="non-scaling-stroke" />
      <ellipse cx={cx} cy={y + h * 0.62} rx={w * 0.28} ry={h * 0.2} fill="#10151e" />
    </g>);
  }
  if (g === "car" || g === "truck") {
    const cabH = g === "truck" ? 0.6 : 0.48;
    return (<g key={key} opacity="0.97">
      <path d={`M ${x} ${y + h} L ${x} ${y + h * 0.55} Q ${x} ${y + h * 0.45} ${x + w * 0.1} ${y + h * 0.45} L ${x + w * 0.24} ${y + h * cabH} Q ${x + w * 0.28} ${y + h * 0.15} ${x + w * 0.5} ${y + h * 0.15} L ${x + w * 0.74} ${y + h * 0.18} Q ${x + w * 0.82} ${y + h * 0.2} ${x + w * 0.88} ${y + h * 0.48} L ${x + w} ${y + h * 0.55} L ${x + w} ${y + h} Z`} {...common} />
      <circle cx={x + w * 0.26} cy={y + h} r={w * 0.1} fill="#02040a" />
      <circle cx={x + w * 0.74} cy={y + h} r={w * 0.1} fill="#02040a" />
      <rect x={x + w * 0.3} y={y + h * 0.24} width={w * 0.36} height={h * 0.22} rx={w * 0.02} fill="#141b28" opacity="0.7" />
    </g>);
  }
  if (g === "bike") {
    return (<g key={key} opacity="0.92" stroke="#0a0e15" strokeWidth="0.006" fill="none" vectorEffect="non-scaling-stroke">
      <circle cx={x + w * 0.22} cy={y + h * 0.72} r={h * 0.26} />
      <circle cx={x + w * 0.78} cy={y + h * 0.72} r={h * 0.26} />
      <path d={`M ${x + w * 0.22} ${y + h * 0.72} L ${x + w * 0.5} ${y + h * 0.72} L ${x + w * 0.4} ${y + h * 0.34} L ${x + w * 0.78} ${y + h * 0.72} M ${x + w * 0.4} ${y + h * 0.34} L ${x + w * 0.55} ${y + h * 0.34}`} />
    </g>);
  }
  if (g === "qr") {
    return <QRGlyph key={key} x={x} y={y} w={w} h={h} seed={o.id} />;
  }
  if (g === "sign") {
    const r = Math.min(w, h) / 2;
    const pts = Array.from({ length: 8 }, (_, i) => {
      const a = (Math.PI / 8) + (i * Math.PI) / 4;
      return `${cx + Math.cos(a) * r},${y + h / 2 + Math.sin(a) * r}`;
    }).join(" ");
    return <polygon key={key} points={pts} fill="#3a1418" stroke="rgba(255,120,120,0.25)" strokeWidth="0.001" vectorEffect="non-scaling-stroke" />;
  }
  if (g === "tlight") {
    return (<g key={key}>
      <rect x={x + w * 0.3} y={y} width={w * 0.4} height={h} rx={w * 0.08} fill="#0a0d13" stroke={edge} strokeWidth="0.001" vectorEffect="non-scaling-stroke" />
      <circle cx={cx} cy={y + h * 0.22} r={w * 0.12} fill="#5a1414" />
      <circle cx={cx} cy={y + h * 0.5} r={w * 0.12} fill="#5a4a14" />
      <circle cx={cx} cy={y + h * 0.78} r={w * 0.12} fill="#14401f" />
    </g>);
  }
  return null;
}

function QRGlyph({ x, y, w, h, seed }) {
  const cells = useMemo(() => {
    let s = 0; for (let i = 0; i < seed.length; i++) s = (s * 31 + seed.charCodeAt(i)) >>> 0;
    const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
    const n = 7, out = [];
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const finder = (r < 3 && c < 3) || (r < 3 && c >= n - 3) || (r >= n - 3 && c < 3);
      out.push({ r, c, on: finder ? (r === 0 || r === 2 || c === 0 || c === 2 || (r === 1 && c === 1)) : rng() > 0.5 });
    }
    return { n, out };
  }, [seed]);
  const cw = w / cells.n, ch = h / cells.n;
  return (<g>
    <rect x={x} y={y} width={w} height={h} fill="#e9edf2" rx={w * 0.04} />
    {cells.out.filter((d) => d.on).map((d, i) => (
      <rect key={i} x={x + d.c * cw} y={y + d.r * ch} width={cw} height={ch} fill="#0a0c11" />
    ))}
  </g>);
}

function Scene({ image, className, style }) {
  const pal = PALETTES[image.palette] || PALETTES.dusk;
  const sid = "sg" + image.id.slice(0, 6);
  // Real backend: render the actual image bytes. Boxes are normalized over the
  // same box, so preserveAspectRatio="none" keeps the overlay aligned.
  if (image.src) {
    return (
      <img src={image.src} className={className} alt=""
        style={{ objectFit: "fill", display: "block", background: "#0a0c11", ...(style || {}) }} />
    );
  }
  return (
    <svg className={className} style={style} viewBox="0 0 1 0.625" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      <defs>
        <linearGradient id={sid + "sky"} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={pal.sky[0]} />
          <stop offset="55%" stopColor={pal.sky[1]} />
          <stop offset="100%" stopColor={pal.sky[2]} />
        </linearGradient>
        <radialGradient id={sid + "sun"} cx="0.7" cy="0.32" r="0.5">
          <stop offset="0%" stopColor={pal.sun} stopOpacity="0.5" />
          <stop offset="100%" stopColor={pal.sun} stopOpacity="0" />
        </radialGradient>
        <linearGradient id={sid + "gnd"} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={pal.haze} stopOpacity="0.5" />
          <stop offset="100%" stopColor={pal.ground} />
        </linearGradient>
        <filter id={sid + "grain"}>
          <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" stitchTiles="stitch" />
          <feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.5 0" />
        </filter>
      </defs>
      <rect x="0" y="0" width="1" height="0.625" fill={`url(#${sid}sky)`} />
      <rect x="0" y="0" width="1" height="0.625" fill={`url(#${sid}sun)`} />
      <rect x="0" y="0.42" width="1" height="0.205" fill={`url(#${sid}gnd)`} />
      <line x1="0" y1="0.42" x2="1" y2="0.42" stroke={pal.haze} strokeWidth="0.002" opacity="0.5" />
      {/* perspective ground lines for road-like scenes */}
      {image.palette !== "indoor" && (
        <g stroke={pal.haze} strokeWidth="0.0015" opacity="0.25">
          {[0.2, 0.4, 0.6, 0.8].map((p, i) => (
            <line key={i} x1={p} y1="0.42" x2={(p - 0.5) * 2.4 + 0.5} y2="0.625" />
          ))}
        </g>
      )}
      <g>{image.objects.map((o, i) => silhouette(o, "s" + i))}</g>
      <rect x="0" y="0" width="1" height="0.625" filter={`url(#${sid}grain)`} opacity="0.5" />
      <rect x="0" y="0" width="1" height="0.625" fill="black" opacity="0.12" />
      {/* vignette */}
      <radialGradient id={sid + "vig"} cx="0.5" cy="0.5" r="0.7">
        <stop offset="60%" stopColor="black" stopOpacity="0" />
        <stop offset="100%" stopColor="black" stopOpacity="0.45" />
      </radialGradient>
      <rect x="0" y="0" width="1" height="0.625" fill={`url(#${sid}vig)`} />
    </svg>
  );
}

Object.assign(window, { Icon, Scene, vhelp });
