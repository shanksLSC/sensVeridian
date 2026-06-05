-- ============================================================================
-- Veridian Studio / sensVeridian — PostgreSQL schema.
--
-- This is the canonical DDL for the PostgreSQL backend. It mirrors the embedded
-- DuckDB schema (store/schema.sql) 1:1 and adds the tables the Veridian Studio
-- UI needs. JSON columns become jsonb; image_id / clip_id stay sha-256 hex
-- (content-addressed, dedup-friendly).
--
-- Apply with:  psql "$DATABASE_URL_SYNC" -f schema_pg.sql
-- (or via sensveridian.store.pg.PgStore.migrate(), which executes this file).
-- ============================================================================

-- ---- existing tables (ported from DuckDB store/schema.sql) ------------------
CREATE TABLE IF NOT EXISTS images (
  image_id    TEXT PRIMARY KEY,                 -- sha-256 of decoded pixels
  path        TEXT,
  width       INTEGER,
  height      INTEGER,
  dataset_id  TEXT,                             -- NEW: which dataset owns this frame
  ingested_at TIMESTAMPTZ DEFAULT now(),
  metadata    JSONB
);
CREATE INDEX IF NOT EXISTS idx_images_dataset ON images(dataset_id);

CREATE TABLE IF NOT EXISTS models (
  model_id     TEXT PRIMARY KEY,
  display_name TEXT,
  version      TEXT,
  weights_path TEXT,
  weights_sha  TEXT,
  input_spec   TEXT,                            -- e.g. '320x320x3'
  n_classes    INTEGER,
  depends_on   TEXT                             -- e.g. fr depends_on 'fd'
);

CREATE TABLE IF NOT EXISTS runs (
  run_id     TEXT PRIMARY KEY,
  started_at TIMESTAMPTZ DEFAULT now(),
  code_hash  TEXT,
  notes      TEXT
);

CREATE TABLE IF NOT EXISTS predictions_summary (
  image_id TEXT,
  run_id   TEXT,
  model_id TEXT,
  present  BOOLEAN,
  count    INTEGER,
  extras   JSONB,
  PRIMARY KEY (image_id, run_id, model_id)
);

CREATE TABLE IF NOT EXISTS predictions_raw (
  image_id TEXT,
  run_id   TEXT,
  model_id TEXT,
  payload  JSONB,
  PRIMARY KEY (image_id, run_id, model_id)
);
-- fast drill-down into detection payloads
CREATE INDEX IF NOT EXISTS idx_pred_raw_payload ON predictions_raw USING gin (payload);

CREATE TABLE IF NOT EXISTS augmentations (
  augmented_image_id TEXT PRIMARY KEY,
  parent_image_id    TEXT NOT NULL REFERENCES images(image_id),
  method             TEXT NOT NULL,
  step_index         INTEGER,
  delta_ft           DOUBLE PRECISION,
  params             JSONB,
  created_at         TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_aug_parent ON augmentations(parent_image_id);

CREATE TABLE IF NOT EXISTS image_depth_stats (
  image_id      TEXT,
  model_id      TEXT,
  detection_idx INTEGER,
  bbox_xyxy     JSONB,
  d_initial_ft  DOUBLE PRECISION,
  source        TEXT DEFAULT 'zoe',             -- 'zoe' | 'manual'
  PRIMARY KEY (image_id, model_id, detection_idx)
);

CREATE TABLE IF NOT EXISTS image_bg_plates (
  image_id   TEXT PRIMARY KEY,
  plate_path TEXT,
  mask_sha   TEXT,
  inpainter  TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- ---- new tables (Veridian Studio) ------------------------------------------

-- datasets get first-class rows (DuckDB inferred them; here they are explicit)
CREATE TABLE IF NOT EXISTS datasets (
  dataset_id TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  descr      TEXT,
  kind       TEXT NOT NULL DEFAULT 'vision',    -- 'vision' | 'audio'
  models     TEXT[] DEFAULT '{}',
  palette    TEXT,
  run_id     TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- version history per model (the model manager + regression review)
CREATE TABLE IF NOT EXISTS model_versions (
  model_id    TEXT NOT NULL REFERENCES models(model_id),
  version     TEXT NOT NULL,
  weights_sha TEXT,
  released_on DATE,
  metrics     JSONB,                            -- {precision,recall,mAP,f1,agreement}
  notes       TEXT,
  is_current  BOOLEAN DEFAULT false,
  PRIMARY KEY (model_id, version)
);

-- a layer = a named annotation source on a dataset
CREATE TABLE IF NOT EXISTS layers (
  layer_id   TEXT PRIMARY KEY,                  -- e.g. 'baseline:amod' | 'gt:verified'
  dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
  type       TEXT NOT NULL,                     -- 'prediction' | 'ground-truth' | 'consensus'
  model_id   TEXT,
  version    TEXT,
  run_id     TEXT,
  source     TEXT,                              -- 'human' | 'import:coco' | 'model' ...
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_layers_dataset ON layers(dataset_id);

-- ground-truth boxes parsed from an imported label set (per image, per layer)
CREATE TABLE IF NOT EXISTS gt_boxes (
  gt_id      BIGSERIAL PRIMARY KEY,
  layer_id   TEXT NOT NULL REFERENCES layers(layer_id),
  dataset_id TEXT NOT NULL,
  image_id   TEXT,                              -- sha-256 once matched to a frame
  image_ref  TEXT,                              -- original filename/key from the label set
  cls        TEXT,
  box        JSONB,                             -- [x,y,w,h] normalized 0..1
  meta       JSONB
);
CREATE INDEX IF NOT EXISTS idx_gt_boxes_image ON gt_boxes(image_id);
CREATE INDEX IF NOT EXISTS idx_gt_boxes_layer ON gt_boxes(layer_id);

-- human verdicts (verify / accept / reject / edit / flag)
CREATE TABLE IF NOT EXISTS reviews (
  target_id   TEXT PRIMARY KEY,                 -- det id | seg id | 'img:<imageId>' | 'flip:<detId>'
  target_kind TEXT,                             -- 'detection' | 'segment' | 'image' | 'flip'
  dataset_id  TEXT,
  verdict     TEXT NOT NULL,                    -- 'accepted'|'rejected'|'edited'|'flagged'
  box         JSONB,                            -- present when edited: [x,y,w,h] normalized
  reviewer    TEXT,
  ts          TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_reviews_dataset ON reviews(dataset_id);

-- audio clips + segments (audio modality)
CREATE TABLE IF NOT EXISTS clips (
  clip_id    TEXT PRIMARY KEY,                  -- sha-256
  dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
  name       TEXT,
  duration_s DOUBLE PRECISION,
  waveform   JSONB,                             -- downsampled peaks
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS segments (
  segment_id TEXT PRIMARY KEY,
  clip_id    TEXT NOT NULL REFERENCES clips(clip_id),
  start_frac DOUBLE PRECISION,                  -- 0..1 of duration
  end_frac   DOUBLE PRECISION,
  gt_label   TEXT,
  pred_label TEXT,
  conf       DOUBLE PRECISION,
  state      TEXT,                              -- match|miss|fp|mismatch
  keyword    TEXT
);
CREATE INDEX IF NOT EXISTS idx_segments_clip ON segments(clip_id);

-- ingest jobs (video auto-label / import / connection runs)
CREATE TABLE IF NOT EXISTS ingest_jobs (
  job_id       TEXT PRIMARY KEY,
  source       TEXT,                            -- 'video' | 'import' | 'connection'
  status       TEXT DEFAULT 'queued',           -- queued|running|done|error
  spec         JSONB,
  progress     INTEGER DEFAULT 0,
  stage        TEXT,
  frames_done  INTEGER DEFAULT 0,
  frames_total INTEGER,
  error        TEXT,
  created_at   TIMESTAMPTZ DEFAULT now(),
  finished_at  TIMESTAMPTZ
);

-- watched external sources (S3 bucket / folder)
CREATE TABLE IF NOT EXISTS connections (
  connection_id TEXT PRIMARY KEY,
  uri        TEXT NOT NULL,
  schedule   TEXT,                              -- 'hourly'|'15min'|'on-object'
  pipeline   TEXT,                              -- 'auto-label:amod'|'import'|'stage'
  enabled    BOOLEAN DEFAULT true,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- exported sets (lineage downstream)
CREATE TABLE IF NOT EXISTS exports (
  export_id  TEXT PRIMARY KEY,
  name       TEXT,
  sql        TEXT,
  uri        TEXT,
  dataset_id TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- ---- summary view (port of v_image_summary_wide) ---------------------------
-- A materialized view so the dataset grid stays fast; REFRESH after each ingest
-- (PgStore.refresh_summary_view()). The unique index lets us REFRESH CONCURRENTLY.
CREATE MATERIALIZED VIEW IF NOT EXISTS v_image_summary_wide AS
SELECT
  i.image_id,
  i.path,
  i.dataset_id,
  BOOL_OR(CASE WHEN s.model_id = 'amod'   THEN s.present END) AS amod_present,
  MAX(CASE WHEN s.model_id = 'amod'   THEN s.count   END) AS n_amod,
  BOOL_OR(CASE WHEN s.model_id = 'qrcode' THEN s.present END) AS qrc_present,
  MAX(CASE WHEN s.model_id = 'qrcode' THEN s.count   END) AS n_qrc,
  BOOL_OR(CASE WHEN s.model_id = 'fd'     THEN s.present END) AS fd_present,
  MAX(CASE WHEN s.model_id = 'fd'     THEN s.count   END) AS n_fd,
  BOOL_OR(CASE WHEN s.model_id = 'fr'     THEN s.present END) AS fid_present,
  MAX(CASE WHEN s.model_id = 'fr'     THEN s.count   END) AS n_fid
FROM images i
LEFT JOIN predictions_summary s USING (image_id)
GROUP BY i.image_id, i.path, i.dataset_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_v_image_summary_wide_pk
  ON v_image_summary_wide (image_id);
-- REFRESH MATERIALIZED VIEW CONCURRENTLY v_image_summary_wide;  -- after each ingest
