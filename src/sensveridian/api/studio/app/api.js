/* ============================================================================
   Veridian Studio — data-access adapter (the backend seam).

   The whole UI talks to ONE async interface: window.VeridianAPI. Today it is
   backed by MockAdapter (the in-memory window.VD store). Flip
   window.VERIDIAN_CONFIG.backend to "rest" to use RestAdapter, which maps every
   method to a documented HTTP/WS call (see API_CONTRACT.md). No screen code
   changes between the two — only this file.

   Architecture: window.VD is the client-side cache/store (datasets, models,
   class metadata). VeridianAPI is the I/O layer that hydrates the cache on
   boot, reads through it, and persists mutations to the server.
   ========================================================================== */
(function () {
  "use strict";

  // Defaults; a window.VERIDIAN_CONFIG set BEFORE this script (e.g. the inline
  // config in index.html) wins, so the bundled studio boots straight into REST.
  window.VERIDIAN_CONFIG = Object.assign({
    backend: "mock",                 // "mock" | "rest"
    baseUrl: "/api/v1",              // REST root  (e.g. https://veridian.lattice.internal/api/v1)
    wsUrl: "/api/v1/ws",             // WebSocket root for ingest job progress
    latencyMs: 0,                    // simulate network latency in mock mode
    token: null,                     // bearer token for RestAdapter
  }, window.VERIDIAN_CONFIG || {});

  const cfg = () => window.VERIDIAN_CONFIG;
  const sleep = (ms) => (ms ? new Promise((r) => setTimeout(r, ms)) : Promise.resolve());

  /* --------------------------------------------------------------------- *
   * MockAdapter — resolves from the in-memory window.VD store.            *
   * Read methods return slices of the cache; writes mutate it in place.   *
   * --------------------------------------------------------------------- */
  const reviewStore = {};   // targetId -> { verdict, ts }
  const jobStore = {};      // jobId -> job

  const MockAdapter = {
    name: "mock",

    /** Boot: in mock mode the cache is already populated by data.js. */
    async hydrate() { await sleep(cfg().latencyMs); return { datasets: window.VD.datasets.length, models: window.VD.models.length }; },

    /* ---- reads ---- */
    async listDatasets() {
      await sleep(cfg().latencyMs);
      return window.VD.datasets.map((d) => ({
        id: d.id, name: d.name, desc: d.desc, kind: d.kind, models: d.models,
        count: d.count, agreement: d.agreement, conflicts: d.conflicts, reviewed: d.reviewed, runId: d.runId,
      }));
    },
    async getDataset(id) { await sleep(cfg().latencyMs); return window.VD.getDataset(id) || null; },
    async getImage(datasetId, imageId) { await sleep(cfg().latencyMs); return window.VD.getImage(datasetId, imageId) || null; },
    async getClip(datasetId, clipId) {
      await sleep(cfg().latencyMs);
      const d = window.VD.getDataset(datasetId);
      return d && d.clips ? d.clips.find((c) => c.id === clipId) || null : null;
    },
    async listModels() { await sleep(cfg().latencyMs); return window.VD.models; },
    /** predictions for one image, optionally narrowed to a run/model (layer). */
    async getPredictions(datasetId, imageId /*, runId, modelId */) {
      await sleep(cfg().latencyMs);
      const im = window.VD.getImage(datasetId, imageId);
      return im ? im.objects : [];
    },
    async getRegressions(modelId, baseVersion, candVersion) {
      await sleep(cfg().latencyMs);
      return window.__buildFlips ? window.__buildFlips(modelId, baseVersion, candVersion) : [];
    },
    async getLineage() { await sleep(cfg().latencyMs); return { nodes: window.__LIN_NODES || [], edges: window.__LIN_EDGES || [] }; },

    /* ---- writes ---- */
    /** Persist a human verdict for a detection / segment / image / flip. */
    async saveReview(targetId, patch) {
      await sleep(cfg().latencyMs);
      reviewStore[targetId] = { ...(reviewStore[targetId] || {}), ...patch, ts: Date.now() };
      return { ok: true, targetId, serverTs: Date.now() };
    },
    async bulkReview(targetIds, patch) {
      await sleep(cfg().latencyMs);
      targetIds.forEach((id) => { reviewStore[id] = { ...patch, ts: Date.now() }; });
      return { ok: true, n: targetIds.length };
    },

    /** Submit an ingest job (video auto-label). Returns a job handle; progress
        is delivered via subscribeJob() (WS in rest mode). */
    async createIngestJob(spec) {
      await sleep(cfg().latencyMs);
      const jobId = "job_" + Math.random().toString(16).slice(2, 10);
      jobStore[jobId] = { jobId, status: "queued", spec, progress: 0 };
      // materialize datasets for any new tags (mock convenience; the real
      // backend does this server-side and the UI just polls the job).
      (spec.groups || []).forEach((g) => {
        if (g.createDataset) window.VD.createDataset({ name: g.name, desc: g.desc, models: g.models, palette: g.palette, n: g.n });
      });
      return { jobId, status: "queued" };
    },
    async createDatasetFromImport(spec) {
      await sleep(cfg().latencyMs);
      const ds = spec.kind === "audio"
        ? window.VD.createAudioDataset({ name: spec.name, desc: spec.desc })
        : window.VD.createDataset({ name: spec.name, desc: spec.desc, models: spec.models, palette: spec.palette || "dusk", n: spec.n });
      return { ok: true, datasetId: ds.id };
    },
    /** Optional fake progress stream for the mock UI. */
    subscribeJob(jobId, onTick) {
      let p = 0; const iv = setInterval(() => { p = Math.min(100, p + 4); onTick({ jobId, progress: p, status: p >= 100 ? "done" : "running" }); if (p >= 100) clearInterval(iv); }, 120);
      return () => clearInterval(iv);
    },
    /** Browse local dataset folders (mock: synthesize from existing datasets). */
    async browseDatasets() {
      await sleep(cfg().latencyMs);
      return { root: "(mock)", entries: window.VD.datasets.filter((d) => d.kind !== "audio").map((d) => ({ name: d.name, path: d.id, images: d.count, videos: 0, has_labels: false, kind: "image" })) };
    },
    async getJob(jobId) { await sleep(cfg().latencyMs); return jobStore[jobId] || { jobId, status: "done" }; },
  };

  /* --------------------------------------------------------------------- *
   * RestAdapter — same interface, mapped to HTTP/WS. Not active unless     *
   * VERIDIAN_CONFIG.backend === "rest". See API_CONTRACT.md for shapes.    *
   * --------------------------------------------------------------------- */
  const RestAdapter = {
    name: "rest",
    _h() { const h = { "Content-Type": "application/json" }; if (cfg().token) h.Authorization = "Bearer " + cfg().token; return h; },
    async _get(path) { const r = await fetch(cfg().baseUrl + path, { headers: this._h() }); if (!r.ok) throw new Error(`GET ${path} → ${r.status}`); return r.json(); },
    async _send(method, path, body) { const r = await fetch(cfg().baseUrl + path, { method, headers: this._h(), body: body ? JSON.stringify(body) : undefined }); if (!r.ok) throw new Error(`${method} ${path} → ${r.status}`); return r.json(); },

    async hydrate() { return this._get("/health"); },
    async listDatasets() { return this._get("/datasets"); },
    async getDataset(id) { return this._get(`/datasets/${id}`); },
    async getImage(datasetId, imageId) { return this._get(`/datasets/${datasetId}/images/${imageId}`); },
    async getClip(datasetId, clipId) { return this._get(`/datasets/${datasetId}/clips/${clipId}`); },
    async listModels() { return this._get("/models"); },
    async getPredictions(datasetId, imageId, runId, modelId) {
      const q = new URLSearchParams({ ...(runId ? { run_id: runId } : {}), ...(modelId ? { model_id: modelId } : {}) });
      return this._get(`/datasets/${datasetId}/images/${imageId}/predictions?${q}`);
    },
    async getRegressions(modelId, baseVersion, candVersion) { return this._get(`/models/${modelId}/regressions?base=${baseVersion}&candidate=${candVersion}`); },
    async getLineage() { return this._get("/lineage"); },

    async saveReview(targetId, patch) { return this._send("PUT", `/reviews/${encodeURIComponent(targetId)}`, patch); },
    async bulkReview(targetIds, patch) { return this._send("POST", "/reviews:bulk", { target_ids: targetIds, patch }); },
    async createIngestJob(spec) { return this._send("POST", "/ingest/jobs", spec); },
    async createDatasetFromImport(spec) { return this._send("POST", "/datasets:import", spec); },
    /** Live job progress over WebSocket. */
    subscribeJob(jobId, onTick) {
      const ws = new WebSocket(cfg().wsUrl + `/jobs/${jobId}`);
      ws.onmessage = (e) => { try { onTick(JSON.parse(e.data)); } catch (_) {} };
      return () => ws.close();
    },
    /** List local dataset folders under the server's datasets root. */
    async browseDatasets() { return this._get("/fs/datasets"); },
    async getJob(jobId) { return this._get(`/ingest/jobs/${jobId}`); },
  };

  window.VeridianAPI = cfg().backend === "rest" ? RestAdapter : MockAdapter;
  window.VeridianAdapters = { MockAdapter, RestAdapter };
  /** Swap backends at runtime (e.g. from a settings toggle). */
  window.setVeridianBackend = (mode) => { cfg().backend = mode; window.VeridianAPI = mode === "rest" ? RestAdapter : MockAdapter; };
})();
