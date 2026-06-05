/* ============================================================================
   Veridian Studio — mock data layer
   Deterministic, seeded generation so scenes + detections are stable across
   reloads. Mirrors the sensVeridian payload shapes: bbox [x1,y1,x2,y2] in
   normalized 0..1, confidence, class, plus QR decode / face identity / mask.
   Exposed on window.VD.
   ========================================================================== */
(function () {
  "use strict";

  // ---- deterministic RNG --------------------------------------------------
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function strSeed(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }
  function hex(rng, n) {
    const c = "0123456789abcdef"; let s = "";
    for (let i = 0; i < n; i++) s += c[Math.floor(rng() * 16)];
    return s;
  }
  function pick(rng, arr) { return arr[Math.floor(rng() * arr.length)]; }
  function rnd(rng, a, b) { return a + rng() * (b - a); }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

  // ---- IoU ----------------------------------------------------------------
  function iou(a, b) {
    if (!a || !b) return 0;
    const ax2 = a[0] + a[2], ay2 = a[1] + a[3], bx2 = b[0] + b[2], by2 = b[1] + b[3];
    const ix = Math.max(0, Math.min(ax2, bx2) - Math.max(a[0], b[0]));
    const iy = Math.max(0, Math.min(ay2, by2) - Math.max(a[1], b[1]));
    const inter = ix * iy;
    const uni = a[2] * a[3] + b[2] * b[3] - inter;
    return uni <= 0 ? 0 : inter / uni;
  }

  // ---- class metadata -----------------------------------------------------
  const CLASSES = {
    car:           { label: "car",           glyph: "car",    model: "amod" },
    truck:         { label: "truck",         glyph: "truck",  model: "amod" },
    person:        { label: "person",        glyph: "person", model: "amod" },
    bicycle:       { label: "bicycle",       glyph: "bike",   model: "amod" },
    traffic_light: { label: "traffic light", glyph: "tlight", model: "amod" },
    stop_sign:     { label: "stop sign",     glyph: "sign",   model: "amod" },
    face:          { label: "face",          glyph: "face",   model: "fd" },
    qr:            { label: "QR code",       glyph: "qr",     model: "qrcode" },
  };

  const PEOPLE = [
    "A. Okafor", "R. Mehta", "L. Tanaka", "S. Novak", "J. Park",
    "M. Costa", "D. Ahmed", "E. Fischer", "C. Romano", "T. Nguyen",
    "K. Singh", "V. Petrova",
  ];
  const QR_PAYLOADS = [
    "PKG-7741-AX", "SKU://4820-1192", "https://lat.tc/asset/0x4f", "BIN-C12-R08",
    "LOT#2026-0419", "WH3-AISLE-22", "PALLET-009273", "RMA-55821",
  ];

  // ---- object generation --------------------------------------------------
  // state: match | miss (gt only, model missed) | fp (pred only, false pos) |
  //        mismatch (overlap but wrong class/identity) | low_conf
  function makeObject(rng, cls, idx) {
    const w = clamp(rnd(rng, 0.06, 0.26), 0.04, 0.32);
    const h = cls === "person" ? w * rnd(rng, 1.9, 2.6)
      : cls === "qr" ? w
      : cls === "face" ? w * rnd(rng, 1.1, 1.35)
      : w * rnd(rng, 0.55, 0.85);
    const x = clamp(rnd(rng, 0.04, 0.92 - w), 0.02, 0.95 - w);
    const y = clamp(rnd(rng, 0.18, 0.9 - h), 0.05, 0.95 - h);
    const gt = [x, y, w, Math.min(h, 0.92 - y)];

    const roll = rng();
    let state, pred, conf;
    if (roll < 0.60) {            // clean match
      state = "match";
      const j = 0.012;
      pred = [x + rnd(rng, -j, j), y + rnd(rng, -j, j), w * rnd(rng, 0.95, 1.05), gt[3] * rnd(rng, 0.95, 1.05)];
      conf = rnd(rng, 0.82, 0.99);
    } else if (roll < 0.74) {     // loose match (geometry drift)
      state = "match";
      const j = 0.05;
      pred = [x + rnd(rng, -j, j), y + rnd(rng, -j, j), w * rnd(rng, 0.78, 1.22), gt[3] * rnd(rng, 0.8, 1.2)];
      conf = rnd(rng, 0.55, 0.82);
    } else if (roll < 0.84) {     // miss — no prediction
      state = "miss"; pred = null; conf = 0;
    } else if (roll < 0.93) {     // false positive — pred, no gt
      state = "fp";
      pred = [clamp(x + rnd(rng, -0.1, 0.1), 0.02, 0.9), clamp(y + rnd(rng, -0.1, 0.1), 0.05, 0.9), w * 0.9, gt[3] * 0.9];
      conf = rnd(rng, 0.31, 0.6);
    } else {                       // mismatch — class/identity disagreement
      state = "mismatch";
      const j = 0.03;
      pred = [x + rnd(rng, -j, j), y + rnd(rng, -j, j), w * rnd(rng, 0.9, 1.1), gt[3] * rnd(rng, 0.9, 1.1)];
      conf = rnd(rng, 0.45, 0.78);
    }

    const o = {
      id: "det_" + idx + "_" + hex(rng, 4),
      cls,
      model: CLASSES[cls].model,
      gt: state === "fp" ? null : gt,
      pred: pred ? pred.map((v) => +v.toFixed(4)) : null,
      conf: +conf.toFixed(3),
      state,
      reviewed: false,
      verdict: null, // 'accepted' | 'rejected' | 'edited'
    };
    o.iou = +iou(o.gt, o.pred).toFixed(3);

    // segmentation polygon for a subset of amod objects (SAM mask path)
    if ((cls === "car" || cls === "truck" || cls === "person") && rng() < 0.4 && o.pred) {
      o.mask = polyFromBox(rng, o.pred);
    }

    // identity for faces
    if (cls === "face") {
      const gtPerson = pick(rng, PEOPLE);
      let predPerson = gtPerson;
      if (state === "mismatch") predPerson = pick(rng, PEOPLE.filter((p) => p !== gtPerson));
      o.identity = {
        gt: gtPerson,
        pred: state === "miss" ? null : (state === "fp" ? "unknown" : predPerson),
        sim: +rnd(rng, state === "mismatch" ? 0.41 : 0.62, 0.97).toFixed(3),
        person_id: "P-" + String(strSeed(gtPerson) % 9000 + 1000),
      };
    }
    // decoded text for QR
    if (cls === "qr") {
      const gtText = pick(rng, QR_PAYLOADS);
      let predText = gtText;
      if (state === "mismatch") predText = gtText.slice(0, Math.max(2, gtText.length - 3)) + "\u2588\u2588?";
      o.decoded = { gt: gtText, pred: state === "miss" ? null : (state === "fp" ? "(garbled)" : predText) };
    }
    return o;
  }

  function polyFromBox(rng, b) {
    const [x, y, w, h] = b;
    const cx = x + w / 2, cy = y + h / 2;
    const pts = [];
    const n = 9;
    for (let i = 0; i < n; i++) {
      const ang = (i / n) * Math.PI * 2;
      const rx = (w / 2) * rnd(rng, 0.72, 1.0);
      const ry = (h / 2) * rnd(rng, 0.72, 1.0);
      pts.push([+(cx + Math.cos(ang) * rx).toFixed(4), +(cy + Math.sin(ang) * ry).toFixed(4)]);
    }
    return pts;
  }

  // ---- image generation ---------------------------------------------------
  function makeImage(datasetId, classPool, idx, palette) {
    const seedStr = datasetId + ":" + idx;
    const rng = mulberry32(strSeed(seedStr));
    const sha = hex(rng, 12) + hex(rng, 12);
    const nObj = 1 + Math.floor(rng() * 5);
    const objects = [];
    for (let i = 0; i < nObj; i++) objects.push(makeObject(rng, pick(rng, classPool), i));

    // agreement: fraction of gt detections matched well
    const gtCount = objects.filter((o) => o.gt).length;
    const matched = objects.filter((o) => o.state === "match").length;
    const conflicts = objects.filter((o) => o.state === "fp" || o.state === "miss" || o.state === "mismatch").length;
    const agreement = gtCount + objects.filter((o) => o.state === "fp").length === 0
      ? 1 : matched / Math.max(1, matched + conflicts);

    let status = "unreviewed";
    const sr = rng();
    if (sr < 0.32) status = "verified";
    else if (sr < 0.46) status = "flagged";

    return {
      id: sha,
      datasetId,
      idx,
      w: 1280,
      h: 800,
      seed: strSeed(seedStr),
      palette,
      d0_ft: +rnd(rng, 4, 9).toFixed(1),
      augmented: rng() < 0.18,
      objects,
      agreement: +agreement.toFixed(3),
      conflicts,
      status,
      captured: "2026-0" + (1 + (idx % 5)) + "-" + String(10 + (idx % 18)).padStart(2, "0"),
    };
  }

  // ---- datasets -----------------------------------------------------------
  const DATASET_DEFS = [
    {
      id: "street_scenes",
      name: "Street Scenes — Automotive",
      desc: "Urban + highway captures for AMOD multi-object detection.",
      models: ["amod"],
      classPool: ["car", "car", "truck", "person", "person", "bicycle", "traffic_light", "stop_sign"],
      n: 28,
      palette: "dusk",
      runId: "baseline",
    },
    {
      id: "access_faces",
      name: "Access Control — Faces",
      desc: "Lobby + door-camera frames for face detection + recognition.",
      models: ["fd", "fr"],
      classPool: ["face", "face", "face", "person"],
      n: 22,
      palette: "indoor",
      runId: "baseline",
    },
    {
      id: "warehouse_qr",
      name: "Warehouse Labels — QR",
      desc: "Pallet + bin QR captures across lighting + range.",
      models: ["qrcode"],
      classPool: ["qr", "qr", "qr"],
      n: 18,
      palette: "warehouse",
      runId: "baseline",
    },
    {
      id: "distance_eval",
      name: "Distance Sweep — Eval",
      desc: "Synthetic distance-swept augmentations for range robustness.",
      models: ["amod", "fd", "qrcode"],
      classPool: ["car", "person", "face", "qr", "stop_sign"],
      n: 16,
      palette: "dusk",
      runId: "augmented",
    },
  ];

  const datasets = DATASET_DEFS.map((d) => {
    const images = [];
    for (let i = 0; i < d.n; i++) images.push(makeImage(d.id, d.classPool, i, d.palette));
    const totalAgree = images.reduce((s, im) => s + im.agreement, 0) / images.length;
    const totalConf = images.reduce((s, im) => s + im.conflicts, 0);
    const reviewed = images.filter((im) => im.status === "verified").length;
    return {
      id: d.id, name: d.name, desc: d.desc, models: d.models, runId: d.runId,
      palette: d.palette, images, kind: "vision",
      count: images.length,
      agreement: +totalAgree.toFixed(3),
      conflicts: totalConf,
      reviewed,
    };
  });

  // ---- models + version history ------------------------------------------
  function versionHistory(modelId, name, base, weightsBase) {
    const rng = mulberry32(strSeed(modelId + "v"));
    const versions = [];
    const v = base.split(".").map(Number);
    let p = 0.71, r = 0.66, ag = 0.70;
    const dates = ["2025-09-14", "2025-11-02", "2026-01-20", "2026-03-08", "2026-04-19"];
    for (let i = 0; i < dates.length; i++) {
      p = clamp(p + rnd(rng, 0.005, 0.05), 0, 0.985);
      r = clamp(r + rnd(rng, 0.005, 0.055), 0, 0.985);
      ag = clamp(ag + rnd(rng, 0.01, 0.05), 0, 0.985);
      const map = clamp((p + r) / 2 + rnd(rng, -0.03, 0.03), 0, 0.99);
      versions.push({
        version: v[0] + "." + v[1] + "." + i,
        weights_sha: hex(rng, 40),
        date: dates[i],
        metrics: {
          precision: +p.toFixed(3), recall: +r.toFixed(3),
          mAP: +map.toFixed(3), agreement: +ag.toFixed(3),
          f1: +((2 * p * r) / (p + r)).toFixed(3),
        },
        notes: i === dates.length - 1 ? "current" : (i === 0 ? "initial export" : "retrain"),
        current: i === dates.length - 1,
      });
    }
    return versions.reverse();
  }

  const models = [
    {
      id: "amod", display_name: "AutomotiveMultiObjectDetection", short: "AMOD",
      input: "320×320×3", weights_path: "all-models/AutomotiveMultiObjectDetection/amod-cpnx-8.2.0.h5",
      classes: 6, color: "amod", versions: versionHistory("amod", "AMOD", "8.2.0"),
    },
    {
      id: "qrcode", display_name: "QRCodeDetection", short: "QR",
      input: "320×320×1", weights_path: "all-models/QRCode/qr-code-detection-final.h5",
      classes: 1, color: "qr", versions: versionHistory("qrcode", "QR", "1.4.0"),
    },
    {
      id: "fd", display_name: "FaceDetection", short: "FD",
      input: "320×320×3", weights_path: "all-models/FaceDetection/fd_lnd_hp-fpga-8.1.0.h5",
      classes: 1, color: "fd", versions: versionHistory("fd", "FD", "8.1.0"),
    },
    {
      id: "fr", display_name: "FaceRecognition", short: "FR",
      input: "112×112×3", weights_path: "all-models/FaceRecognition/fr-fpga-8.1.1.h5",
      classes: 12, color: "fr", versions: versionHistory("fr", "FR", "8.1.1"), depends_on: "fd",
    },
    {
      id: "aed", display_name: "AcousticEventDetection", short: "AED",
      input: "16kHz mono", weights_path: "all-models/Audio/aed-mel-2.0.0.h5",
      classes: 7, color: "aed", versions: versionHistory("aed", "AED", "2.0.0"),
    },
  ];

  // ---- import sources (label files) --------------------------------------
  const importFormats = [
    { id: "yolo", label: "YOLO txt", ext: ".txt", note: "class cx cy w h (normalized) per line" },
    { id: "coco", label: "COCO JSON", ext: ".json", note: "images / annotations / categories arrays" },
    { id: "csv", label: "CSV / parquet", ext: ".csv", note: "one row per box, xyxy or xywh columns" },
  ];

  // ---- helpers exposed ----------------------------------------------------
  function getDataset(id) { return datasets.find((d) => d.id === id); }
  function getImage(datasetId, imageId) {
    const d = getDataset(datasetId);
    return d ? d.images.find((im) => im.id === imageId) : null;
  }

  // model -> class pool, used when spinning up a dataset for a new use-case tag
  const MODEL_CLASS_POOLS = {
    amod: ["car", "car", "truck", "person", "person", "bicycle", "traffic_light", "stop_sign"],
    fd: ["face", "face", "face", "person"],
    fr: ["face", "face", "person"],
    qrcode: ["qr", "qr", "qr"],
  };
  let _customSeq = 0;
  function createDataset({ name, desc, models, palette, n }) {
    const base = (name || "tag").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").slice(0, 24) || "dataset";
    let id = base;
    while (datasets.some((d) => d.id === id)) id = base + "_" + (++_customSeq);
    let pool = [];
    (models && models.length ? models : ["amod"]).forEach((m) => { pool = pool.concat(MODEL_CLASS_POOLS[m] || []); });
    if (!pool.length) pool = MODEL_CLASS_POOLS.amod;
    const count = Math.max(6, n || 16);
    const images = [];
    for (let i = 0; i < count; i++) images.push(makeImage(id, pool, i, palette || "dusk"));
    const totalAgree = images.reduce((s, im) => s + im.agreement, 0) / images.length;
    const totalConf = images.reduce((s, im) => s + im.conflicts, 0);
    const reviewed = images.filter((im) => im.status === "verified").length;
    const ds = {
      id, name: name || "Untitled", desc: desc || "", models: models || ["amod"], runId: "ingest",
      palette: palette || "dusk", images, count: images.length, kind: "vision",
      agreement: +totalAgree.toFixed(3), conflicts: totalConf, reviewed, custom: true,
    };
    datasets.push(ds);
    return ds;
  }

  // ---- audio datasets (waveform + label-track) ---------------------------
  const AUDIO_LABELS = ["speech", "music", "siren", "alarm", "keyword", "noise", "silence"];
  const KEYWORDS = ['"hey lattice"', '"wake"', '"stop"', '"open door"'];
  function makeClip(dsId, idx) {
    const rng = mulberry32(strSeed(dsId + ":clip:" + idx));
    const dur = +rnd(rng, 18, 64).toFixed(1);
    const N = 220;
    const wave = [];
    let env = 0.2;
    for (let i = 0; i < N; i++) {
      env = clamp(env + rnd(rng, -0.18, 0.18), 0.05, 1);
      wave.push(+(env * (0.4 + 0.6 * Math.abs(Math.sin(i * 0.3 + rng() * 2)))).toFixed(3));
    }
    const nSeg = 2 + Math.floor(rng() * 4);
    const segs = [];
    let t = rnd(rng, 0.0, 0.06);
    for (let i = 0; i < nSeg && t < 0.95; i++) {
      const w = rnd(rng, 0.08, 0.26);
      const end = Math.min(0.98, t + w);
      const lbl = pick(rng, AUDIO_LABELS);
      const roll = rng();
      let state = "match", predLbl = lbl, conf = rnd(rng, 0.8, 0.98);
      if (roll > 0.62 && roll < 0.74) { state = "mismatch"; predLbl = pick(rng, AUDIO_LABELS.filter((x) => x !== lbl)); conf = rnd(rng, 0.4, 0.7); }
      else if (roll >= 0.74 && roll < 0.84) { state = "miss"; predLbl = null; conf = 0; }
      else if (roll >= 0.84 && roll < 0.92) { state = "fp"; conf = rnd(rng, 0.3, 0.55); }
      segs.push({
        id: "seg_" + idx + "_" + i, start: +t.toFixed(4), end: +end.toFixed(4),
        gt: state === "fp" ? null : lbl, pred: predLbl, conf: +conf.toFixed(3), state,
        keyword: lbl === "keyword" ? pick(rng, KEYWORDS) : null,
        reviewed: false,
      });
      t = end + rnd(rng, 0.01, 0.08);
    }
    const conflicts = segs.filter((s) => s.state !== "match").length;
    return { id: hex(rng, 16), name: "clip_" + String(idx).padStart(3, "0") + "_" + pick(rng, ["lobby", "dock", "floor", "gate", "yard"]) + ".wav", dur, wave, segments: segs, conflicts, status: rng() < 0.3 ? "verified" : rng() < 0.45 ? "flagged" : "unreviewed", idx };
  }
  function createAudioDataset({ name, desc, n }) {
    const base = (name || "audio").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").slice(0, 24) || "audio";
    let id = base;
    while (datasets.some((d) => d.id === id)) id = base + "_" + (++_customSeq);
    const count = Math.max(6, n || 14);
    const clips = [];
    for (let i = 0; i < count; i++) clips.push(makeClip(id, i));
    const conflicts = clips.reduce((s, c) => s + c.conflicts, 0);
    const segN = clips.reduce((s, c) => s + c.segments.length, 0);
    const matched = clips.reduce((s, c) => s + c.segments.filter((x) => x.state === "match").length, 0);
    const ds = {
      id, name: name || "Acoustic Events", desc: desc || "", models: ["aed"], runId: "ingest",
      kind: "audio", clips, count: clips.length, images: [],
      agreement: +(matched / Math.max(1, segN)).toFixed(3), conflicts,
      reviewed: clips.filter((c) => c.status === "verified").length, custom: true,
    };
    datasets.push(ds);
    return ds;
  }

  // seed one audio dataset so the modality is reachable out of the box
  createAudioDataset({ name: "Acoustic Events — Edge", desc: "Edge-mic captures for acoustic event + keyword detection.", n: 14 });

  window.VD = {
    datasets, models, CLASSES, PEOPLE, importFormats,
    iou, getDataset, getImage, createDataset, createAudioDataset,
    storage: { engine: "PostgreSQL", host: "veridian-db.lattice.internal", db: "sensveridian", migratedFrom: "DuckDB" },
  };
})();
