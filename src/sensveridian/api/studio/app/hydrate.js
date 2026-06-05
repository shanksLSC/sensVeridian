/* ============================================================================
   Veridian Studio — REST hydration layer.

   The screens read a SYNCHRONOUS window.VD cache (datasets/models with nested
   images+objects). In mock mode data.js fills it. In REST mode this module
   fills it from the live backend: a boot hydrate (models + dataset summaries)
   plus lazy ensureDataset()/ensureImage() that the shell awaits on navigation.

   Loaded after data.js + api.js so it can reuse window.VeridianAPI (RestAdapter)
   and overlay the mock cache with server data. No-op when backend !== "rest".
   ========================================================================== */
(function () {
  "use strict";
  const cfg = () => window.VERIDIAN_CONFIG || {};
  const isRest = () => cfg().backend === "rest";

  /** Boot: replace mock datasets/models with server data (keeps VD helpers). */
  async function hydrateBoot() {
    if (!isRest()) return { mock: true };
    try {
      const models = await window.VeridianAPI.listModels();
      if (Array.isArray(models) && models.length) {
        models.forEach((m) => {
          const old = (window.VD.models || []).find((x) => x.id === m.id);
          if (old && old.color) m.color = old.color;          // keep palette hints
          if (!m.versions) m.versions = [];
        });
        window.VD.models = models;
      }
    } catch (e) { console.warn("[veridian] hydrate models failed", e); }

    try {
      const datasets = await window.VeridianAPI.listDatasets();
      // Mutate the array IN PLACE: window.VD.getDataset/getImage (defined in
      // data.js) close over this exact array reference, so reassigning it would
      // leave them searching the stale mock data.
      const shaped = (datasets || []).map((d) =>
        Object.assign({ images: [], clips: [], kind: d.kind || "vision" }, d));
      window.VD.datasets.length = 0;
      shaped.forEach((d) => window.VD.datasets.push(d));
    } catch (e) { console.warn("[veridian] hydrate datasets failed", e); }

    window.VD.storage = Object.assign({}, window.VD.storage, {
      engine: "PostgreSQL", db: "sensveridian", host: "localhost:5432", live: true,
    });
    return { datasets: window.VD.datasets.length, models: window.VD.models.length };
  }

  /** Ensure a dataset's images (vision) or clips (audio) are loaded. */
  async function ensureDataset(id) {
    if (!isRest()) return window.VD.getDataset(id);
    const full = await window.VeridianAPI.getDataset(id);
    if (!full) return null;
    let ds = (window.VD.datasets || []).find((x) => x.id === id);
    if (!ds) { ds = { id, images: [], clips: [] }; window.VD.datasets.push(ds); }
    Object.assign(ds, full);
    ds.images = full.images || [];
    ds.clips = full.clips || [];
    (ds.images || []).forEach((im) => {
      im.datasetId = id;
      if (!im.objects) im.objects = [];
      if (im.objects.length) im._loaded = true;
    });
    ds._loaded = true;
    return ds;
  }

  /** Ensure a single image's full detections are loaded (canvas detail). */
  async function ensureImage(datasetId, imageId) {
    if (!isRest()) return window.VD.getImage(datasetId, imageId);
    let ds = window.VD.getDataset(datasetId);
    if (!ds || !ds._loaded) ds = await ensureDataset(datasetId);
    const full = await window.VeridianAPI.getImage(datasetId, imageId);
    let im = ds && ds.images ? ds.images.find((x) => x.id === imageId) : null;
    if (!full) return im || null;
    if (!im) { im = { id: imageId }; if (ds) ds.images.push(im); }
    Object.assign(im, full);
    im.datasetId = datasetId;
    if (!im.objects) im.objects = [];
    im._loaded = true;
    return im;
  }

  if (window.VD) {
    window.VD.ensureDataset = ensureDataset;
    window.VD.ensureImage = ensureImage;
  }
  // Wire the boot hydrate into the RestAdapter so shell's hydrate() populates VD.
  if (window.VeridianAdapters && window.VeridianAdapters.RestAdapter) {
    window.VeridianAdapters.RestAdapter.hydrate = hydrateBoot;
  }
  window.VeridianHydrate = hydrateBoot;
})();
