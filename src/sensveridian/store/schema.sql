CREATE TABLE IF NOT EXISTS images (
  image_id VARCHAR PRIMARY KEY,
  path VARCHAR,
  width INTEGER,
  height INTEGER,
  ingested_at TIMESTAMP DEFAULT now()
);
ALTER TABLE images ADD COLUMN IF NOT EXISTS metadata JSON;

CREATE TABLE IF NOT EXISTS models (
  model_id VARCHAR PRIMARY KEY,
  display_name VARCHAR,
  version VARCHAR,
  weights_path VARCHAR,
  weights_sha VARCHAR
);

CREATE TABLE IF NOT EXISTS runs (
  run_id VARCHAR PRIMARY KEY,
  started_at TIMESTAMP DEFAULT now(),
  code_hash VARCHAR,
  notes VARCHAR
);

CREATE TABLE IF NOT EXISTS predictions_summary (
  image_id VARCHAR,
  run_id VARCHAR,
  model_id VARCHAR,
  present BOOLEAN,
  count INTEGER,
  extras JSON,
  PRIMARY KEY (image_id, run_id, model_id)
);

CREATE TABLE IF NOT EXISTS predictions_raw (
  image_id VARCHAR,
  run_id VARCHAR,
  model_id VARCHAR,
  payload JSON,
  PRIMARY KEY (image_id, run_id, model_id)
);

CREATE TABLE IF NOT EXISTS augmentations (
  augmented_image_id VARCHAR PRIMARY KEY,
  parent_image_id VARCHAR NOT NULL,
  method VARCHAR NOT NULL,
  step_index INTEGER,
  delta_ft DOUBLE,
  params JSON,
  created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_aug_parent ON augmentations(parent_image_id);

CREATE TABLE IF NOT EXISTS image_depth_stats (
  image_id VARCHAR,
  model_id VARCHAR,
  detection_idx INTEGER,
  bbox_xyxy JSON,
  d_initial_ft DOUBLE,
  source VARCHAR DEFAULT 'zoe',  -- 'zoe' (ZoeDepth fallback) or 'manual' (operator-supplied)
  PRIMARY KEY (image_id, model_id, detection_idx)
);

ALTER TABLE image_depth_stats ADD COLUMN IF NOT EXISTS source VARCHAR DEFAULT 'zoe';

CREATE TABLE IF NOT EXISTS image_bg_plates (
  image_id VARCHAR PRIMARY KEY,
  plate_path VARCHAR,
  mask_sha VARCHAR,
  inpainter VARCHAR,
  created_at TIMESTAMP DEFAULT now()
);

CREATE OR REPLACE VIEW v_image_summary_wide AS
SELECT
  i.image_id,
  i.path,
  MAX(CASE WHEN s.model_id = 'amod' THEN s.present END) AS amod_present,
  MAX(CASE WHEN s.model_id = 'amod' THEN s.count END) AS n_amod,
  MAX(CASE WHEN s.model_id = 'qrcode' THEN s.present END) AS qrc_present,
  MAX(CASE WHEN s.model_id = 'qrcode' THEN s.count END) AS n_qrc,
  MAX(CASE WHEN s.model_id = 'fd' THEN s.present END) AS fd_present,
  MAX(CASE WHEN s.model_id = 'fd' THEN s.count END) AS n_fd,
  MAX(CASE WHEN s.model_id = 'fr' THEN s.present END) AS fid_present,
  MAX(CASE WHEN s.model_id = 'fr' THEN s.count END) AS n_fid
FROM images i
LEFT JOIN predictions_summary s USING (image_id)
GROUP BY i.image_id, i.path;

