/* ============================================================================
   Veridian Studio — Tweaks. Defines TWEAK_DEFAULTS + VeridianTweaks.
   useTweaks is owned by App (shell.jsx); this only renders the panel UI.
   ========================================================================== */
window.TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "confThreshold": 0.30,
  "showPred": true,
  "showGT": true,
  "showMasks": true,
  "showLabels": true,
  "models": { "amod": true, "qrcode": true, "fd": true, "fr": true },
  "autoAccept": false,
  "trustThreshold": 0.85,
  "accent": "#5b7cfa",
  "density": "regular"
}/*EDITMODE-END*/;

const ACCENTS = ["#5b7cfa", "#22d3ee", "#34d39a", "#f5a524", "#c084fc"];

function VeridianTweaks({ t, setTweak }) {
  React.useEffect(() => {
    const r = document.documentElement.style;
    r.setProperty("--accent", t.accent);
    // light/dark companion + dim wash derived from the accent
    r.setProperty("--accent-2", t.accent);
    r.setProperty("--accent-dim", t.accent + "26");
  }, [t.accent]);

  React.useEffect(() => {
    const sz = { compact: 12.5, regular: 13.5, comfy: 14.5 }[t.density] || 13.5;
    document.body.style.fontSize = sz + "px";
  }, [t.density]);

  const MODELS = window.VD.models;
  const toggleModel = (id) => setTweak("models", { ...t.models, [id]: !t.models[id] });

  return (
    <TweaksPanel title="Tweaks">
      <TweakSection label="Overlay" />
      <TweakSlider label="Confidence threshold" value={t.confThreshold} min={0} max={0.95} step={0.01}
        onChange={(v) => setTweak("confThreshold", v)} />
      <TweakToggle label="Prediction boxes" value={t.showPred} onChange={(v) => setTweak("showPred", v)} />
      <TweakToggle label="Ground-truth boxes" value={t.showGT} onChange={(v) => setTweak("showGT", v)} />
      <TweakToggle label="Segmentation masks" value={t.showMasks} onChange={(v) => setTweak("showMasks", v)} />
      <TweakToggle label="Labels" value={t.showLabels} onChange={(v) => setTweak("showLabels", v)} />

      <TweakSection label="Visible models" />
      {MODELS.map((m) => (
        <TweakToggle key={m.id} label={m.short + " — " + m.display_name} value={t.models[m.id] !== false}
          onChange={() => toggleModel(m.id)} />
      ))}

      <TweakSection label="Pre-labeling" />
      <TweakToggle label="Confidence-gated auto-accept" value={t.autoAccept} onChange={(v) => setTweak("autoAccept", v)} />
      <TweakSlider label="Trust threshold" value={t.trustThreshold} min={0.5} max={0.98} step={0.01} onChange={(v) => setTweak("trustThreshold", v)} />

      <TweakSection label="Appearance" />
      <TweakColor label="Accent" value={t.accent} options={ACCENTS} onChange={(v) => setTweak("accent", v)} />
      <TweakRadio label="Density" value={t.density} options={["compact", "regular", "comfy"]}
        onChange={(v) => setTweak("density", v)} />
    </TweaksPanel>
  );
}

window.VeridianTweaks = VeridianTweaks;
