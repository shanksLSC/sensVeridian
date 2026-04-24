# Data Model

## Tables

### images

Unique images processed by the system (keyed by SHA256).

| Column | Type | Notes |
|--------|------|-------|
| `image_id` | VARCHAR | SHA256 of pixel content |
| `path` | VARCHAR | Filesystem path at ingest time |
| `width` | INTEGER | Image width in pixels |
| `height` | INTEGER | Image height in pixels |
| `file_size_bytes` | BIGINT | File size |
| `ingest_time` | TIMESTAMP | When inserted |

### models

Registered oracle models in this session.

| Column | Type | Notes |
|--------|------|-------|
| `model_id` | VARCHAR | Name: "amod", "qrcode", "fd", "fr" |
| `version` | VARCHAR | Model version string |

### runs

Ingest or augmentation runs (for organizing results).

| Column | Type | Notes |
|--------|------|-------|
| `run_id` | VARCHAR | User-provided identifier ("baseline", "distance_eval", etc.) |
| `run_type` | VARCHAR | "ingest" or "augment" |
| `timestamp` | TIMESTAMP | When created |

### predictions_summary

High-level result: was the model present? How many detections?

| Column | Type | Notes |
|--------|------|-------|
| `image_id` | VARCHAR | Foreign key: images |
| `model_id` | VARCHAR | Foreign key: models |
| `run_id` | VARCHAR | Foreign key: runs |
| `present` | BOOLEAN | Did model fire? (≥1 detection) |
| `count` | INTEGER | Number of detections |
| `extras` | JSON | Model-specific extras (e.g., FR matching results) |

### predictions_raw

Full model output as JSON (bboxes, embeddings, decoded text, etc.).

| Column | Type | Notes |
|--------|------|-------|
| `image_id` | VARCHAR | Foreign key: images |
| `model_id` | VARCHAR | Foreign key: models |
| `run_id` | VARCHAR | Foreign key: runs |
| `payload` | JSON | Full nested output |
| `ingest_time` | TIMESTAMP | When stored |

### augmentations

Metadata for generated distance-swept images.

| Column | Type | Notes |
|--------|------|-------|
| `augmented_image_id` | VARCHAR | SHA256 of augmented image |
| `parent_image_id` | VARCHAR | Foreign key: original image |
| `distance_ft` | FLOAT | Simulated distance |
| `params` | JSON | Augmentation parameters (seed, effects, depth source, etc.) |

### image_depth_stats

Per-image depth information from the augmentation pipeline.

| Column | Type | Notes |
|--------|------|-------|
| `image_id` | VARCHAR | Foreign key: images |
| `median_depth_ft` | FLOAT | Overall median metric depth (feet) |
| `min_depth_ft` | FLOAT | Minimum depth in detections |
| `max_depth_ft` | FLOAT | Maximum depth in detections |
| `n_manual_distance` | INTEGER | How many objects used manual override |
| `n_zoe_distance` | INTEGER | How many objects used ZoeDepth |
| `source` | VARCHAR | Depth provenance: "zoe", "manual", or mixed |

### image_bg_plates

Cached inpainted background plates (one per inpainter).

| Column | Type | Notes |
|--------|------|-------|
| `bg_plate_id` | VARCHAR | Identifier |
| `parent_image_id` | VARCHAR | Foreign key: original image |
| `inpainter` | VARCHAR | Inpainter backend ("lama", "sdxl", etc.) |
| `path` | VARCHAR | Filesystem path to saved plate |

## Views

### v_image_summary_wide

Pivoted summary view: one row per image, columns per model.

| Column | Type | Notes |
|--------|------|-------|
| `image_id` | VARCHAR | Image ID |
| `path` | VARCHAR | Image path |
| `width`, `height` | INTEGER | Dimensions |
| `{model}_present` | BOOLEAN | Was model present? (one per model) |
| `n_{model}` | INTEGER | Detection count (one per model) |

Example output:

```
image_id                       path    width height amod_present n_amod ...
abc123...                      img.jpg   640   480  True         2     ...
```

## Sample Data & Examples

### images Table (Sample)

```
image_id                                  | path                              | width | height | file_size_bytes | ingest_time
--------------------------------------|---------------------------------------|-------|--------|-----------------|---------------------------
abc123def456789...                   | /data/Images/qr_code/gard_00000.png   | 640   | 480    | 150000          | 2026-04-24 10:15:32
xyz789uvw012345...                   | /data/Images/face_pics/john_01.jpg    | 1920  | 1080   | 450000          | 2026-04-24 10:16:01
def456ghi789jkl...                   | /data/Images/multi/cars_scene_01.png  | 1024  | 768    | 200000          | 2026-04-24 10:16:45
pqr123stu456vwx...                   | /data/Images/qr_code/gard_00001.jpg   | 640   | 480    | 155000          | 2026-04-24 10:17:12
```

### models Table (Sample)

```
model_id   | version
-----------|----------
amod       | 8.2.0
qrcode     | final
fd         | 8.1.0
fr         | 8.1.1
```

### runs Table (Sample)

```
run_id       | run_type | timestamp
--------------|----------|---------------------------
baseline     | ingest   | 2026-04-24 10:15:00
distance_eval| augment  | 2026-04-24 14:30:15
exp_gpu_v2   | ingest   | 2026-04-24 15:45:22
```

### predictions_summary Table (Sample)

```
image_id                  | model_id | run_id    | present | count | extras
--------------------------|----------|-----------|---------|-------|--------
abc123def456789...       | amod     | baseline  | true    | 2     | {"car": 2}
abc123def456789...       | qrcode   | baseline  | true    | 1     | {"decoded": "abc123xyz"}
abc123def456789...       | fd       | baseline  | true    | 0     | {}
abc123def456789...       | fr       | baseline  | false   | 0     | {}
xyz789uvw012345...       | amod     | baseline  | false   | 0     | {}
xyz789uvw012345...       | fd       | baseline  | true    | 3     | {}
xyz789uvw012345...       | fr       | baseline  | true    | 2     | {"matched": ["john_doe_001", "jane_doe_002"]}
def456ghi789jkl...       | amod     | baseline  | true    | 5     | {"car": 3, "person": 2}
```

### predictions_raw Table (Sample, payload field abbreviated)

```
image_id                  | model_id | run_id   | payload (JSON) | ingest_time
--------------------------|----------|---------|------|---------------------------
abc123def456789...       | amod     | baseline | {"detections": [{"bbox": [10, 20, 100, 150], "class_id": 1, "confidence": 0.95}, {"bbox": [150, 50, 300, 200], "class_id": 1, "confidence": 0.87}]} | 2026-04-24 10:15:45
abc123def456789...       | qrcode   | baseline | {"detections": [{"bbox": [200, 100, 400, 300], "points": [...]}], "decoded_strings": ["abc123xyz"]} | 2026-04-24 10:16:10
xyz789uvw012345...       | fd       | baseline | {"detections": [{"bbox": [50, 30, 180, 280], "crop_data": "..."}, {...}]} | 2026-04-24 10:16:35
xyz789uvw012345...       | fr       | baseline | {"detections": [{"embedding": [0.12, -0.45, 0.67, ...], "matched_ids": ["john_doe_001"]}]} | 2026-04-24 10:17:00
```

### augmentations Table (Sample)

```
augmented_image_id            | parent_image_id            | distance_ft | params (JSON)
------|--------------------------|--------------|--------
abc123_5ft_seed42            | abc123def456789...        | 5.0         | {"inpainter": "lama", "dof_blur": true, "haze": 0.3, "seed": 42}
abc123_6ft_seed42            | abc123def456789...        | 6.0         | {"inpainter": "lama", "dof_blur": true, "haze": 0.35, "seed": 42}
abc123_7ft_seed42            | abc123def456789...        | 7.0         | {"inpainter": "lama", "dof_blur": true, "haze": 0.4, "seed": 42}
def456_8ft_seed123           | def456ghi789jkl...        | 8.0         | {"inpainter": "lama", "dof_blur": false, "haze": 0.0, "seed": 123}
def456_9ft_seed123           | def456ghi789jkl...        | 9.0         | {"inpainter": "lama", "dof_blur": false, "haze": 0.0, "seed": 123}
```

### image_depth_stats Table (Sample)

```
image_id                  | median_depth_ft | min_depth_ft | max_depth_ft | n_manual_distance | n_zoe_distance | source
--------------------------|-----------------|--------------|--------------|-------------------|----------------|--------
abc123def456789...       | 5.2             | 4.1          | 6.8          | 0                 | 2              | zoe
xyz789uvw012345...       | 5.0             | 5.0          | 5.0          | 3                 | 0              | manual
def456ghi789jkl...       | 7.5             | 5.2          | 9.1          | 1                 | 4              | mixed
pqr123stu456vwx...       | 5.2             | 4.1          | 6.8          | 0                 | 2              | zoe
```

### image_bg_plates Table (Sample)

```
bg_plate_id              | parent_image_id            | inpainter | path
-------------------------|----|------|------
abc123_lama             | abc123def456789...        | lama      | ./cache/bg_plates/abc123def456789_lama.png
xyz789_lama             | xyz789uvw012345...        | lama      | ./cache/bg_plates/xyz789uvw012345_lama.png
def456_lama             | def456ghi789jkl...        | lama      | ./cache/bg_plates/def456ghi789jkl_lama.png
```

### v_image_summary_wide View (Sample)

```
image_id               | path                             | width | height | amod_present | n_amod | qrc_present | n_qrc | fd_present | n_fd | fr_present | n_fr
-----------------------|----------------------------------|-------|--------|--------------|--------|-------------|-------|------------|------|------------|------
abc123def456789...    | /data/Images/qr_code/gard.png    | 640   | 480    | true         | 2      | true        | 1     | false      | 0    | false      | 0
xyz789uvw012345...    | /data/Images/face_pics/john.jpg  | 1920  | 1080   | false        | 0      | false       | 0     | true       | 3    | true       | 2
def456ghi789jkl...    | /data/Images/multi/cars_01.png   | 1024  | 768    | true         | 5      | false       | 0     | false      | 0    | false      | 0
pqr123stu456vwx...    | /data/Images/qr_code/gard_01.jpg | 640   | 480    | true         | 2      | false       | 0     | false      | 0    | false      | 0
```

## Sample Queries & Results

### Query 1: Get all images with QR codes

```sql
SELECT image_id, path, n_qrc FROM v_image_summary_wide WHERE qrc_present ORDER BY n_qrc DESC;
```

**Output:**
```
image_id               | path                          | n_qrc
-----------------------|--------------------------------|-------
abc123def456789...    | /data/Images/qr_code/gard.png | 1
pqr123stu456vwx...    | /data/Images/qr_code/gard_01  | 0
```

### Query 2: Count detections per model (summary)

```sql
SELECT model_id, COUNT(*) as num_images, SUM(count) as total_detections
FROM predictions_summary
GROUP BY model_id
ORDER BY total_detections DESC;
```

**Output:**
```
model_id | num_images | total_detections
----------|----------|------------------
amod     | 3         | 9
fd       | 2         | 3
qrcode   | 4         | 1
fr       | 1         | 2
```

### Query 3: Images with multiple detection types

```sql
SELECT 
  image_id, 
  path,
  (amod_present::int + fd_present::int + qrc_present::int + fr_present::int) as model_hits,
  n_amod + n_fd + n_qrc + n_fr as total_detections
FROM v_image_summary_wide
WHERE (amod_present OR fd_present OR qrc_present OR fr_present)
ORDER BY total_detections DESC;
```

**Output:**
```
image_id               | path                         | model_hits | total_detections
-----------------------|------------------------------|------------|------------------
def456ghi789jkl...    | /data/Images/multi/cars.png  | 1          | 5
xyz789uvw012345...    | /data/Images/face_pics/john  | 2          | 5
abc123def456789...    | /data/Images/qr_code/gard.png| 2          | 3
pqr123stu456vwx...    | /data/Images/qr_code/gard_01 | 1          | 2
```

### Query 4: Face matches (FR results)

```sql
SELECT 
  image_id, 
  path,
  n_fd as faces_detected,
  n_fr as faces_recognized
FROM v_image_summary_wide
WHERE fr_present
ORDER BY n_fr DESC;
```

**Output:**
```
image_id               | path                         | faces_detected | faces_recognized
-----------------------|------------------------------|----------------|------------------
xyz789uvw012345...    | /data/Images/face_pics/john  | 3              | 2
```

### Query 5: Depth statistics from augmentation pipeline

```sql
SELECT
  image_id,
  source,
  n_manual_distance,
  n_zoe_distance,
  ROUND(median_depth_ft, 2) as median_depth_ft,
  ROUND(max_depth_ft - min_depth_ft, 2) as depth_range_ft
FROM image_depth_stats
ORDER BY depth_range_ft DESC;
```

**Output:**
```
image_id               | source | n_manual_distance | n_zoe_distance | median_depth_ft | depth_range_ft
-----------------------|--------|-------------------|----------------|-----------------|----------------
def456ghi789jkl...    | mixed  | 1                 | 4              | 7.50            | 3.90
abc123def456789...    | zoe    | 0                 | 2              | 5.20            | 2.70
xyz789uvw012345...    | manual | 3                 | 0              | 5.00            | 0.00
pqr123stu456vwx...    | zoe    | 0                 | 2              | 5.20            | 2.70
```

### Query 6: Augmentation count per image

```sql
SELECT 
  parent_image_id,
  COUNT(*) as num_augmentations,
  ROUND(MIN(distance_ft), 2) as min_distance,
  ROUND(MAX(distance_ft), 2) as max_distance,
  ROUND(AVG(distance_ft), 2) as avg_distance
FROM augmentations
GROUP BY parent_image_id
ORDER BY num_augmentations DESC;
```

**Output:**
```
parent_image_id        | num_augmentations | min_distance | max_distance | avg_distance
-----------------------|-------------------|--------------|--------------|---------------
abc123def456789...    | 3                 | 5.00         | 7.00         | 6.00
def456ghi789jkl...    | 2                 | 8.00         | 9.00         | 8.50
```

### Query 7: Images by detection quality (high-confidence detections)

```sql
SELECT 
  ps.image_id,
  ps.model_id,
  ps.count,
  ROUND(AVG(CAST(pr.payload->>'confidence' AS FLOAT)), 3) as avg_confidence
FROM predictions_summary ps
JOIN predictions_raw pr ON ps.image_id = pr.image_id AND ps.model_id = pr.model_id
WHERE ps.present
GROUP BY ps.image_id, ps.model_id, ps.count
ORDER BY avg_confidence DESC;
```

**Output:**
```
image_id               | model_id | count | avg_confidence
-----------------------|----------|-------|----------------
abc123def456789...    | amod     | 2     | 0.910
xyz789uvw012345...    | fd       | 3     | 0.850
def456ghi789jkl...    | amod     | 5     | 0.820
```

### Query 8: Decoded QR codes with image info

```sql
SELECT 
  img.image_id,
  img.path,
  img.width,
  img.height,
  pr.payload->>'decoded_strings' as qr_text
FROM predictions_raw pr
JOIN images img ON pr.image_id = img.image_id
WHERE pr.model_id = 'qrcode'
  AND pr.payload->>'decoded_strings' IS NOT NULL;
```

**Output:**
```
image_id               | path                          | width | height | qr_text
-----------------------|-------------------------------|-------|--------|-------------------
abc123def456789...    | /data/Images/qr_code/gard.png | 640   | 480    | abc123xyz
```

### Query 9: Models missing from images (false negatives)

```sql
SELECT 
  image_id,
  path,
  (CASE WHEN NOT amod_present THEN 'amod, ' ELSE '' END ||
   CASE WHEN NOT fd_present THEN 'fd, ' ELSE '' END ||
   CASE WHEN NOT qrc_present THEN 'qrcode, ' ELSE '' END ||
   CASE WHEN NOT fr_present THEN 'fr' ELSE '' END) as missing_models
FROM v_image_summary_wide
WHERE NOT (amod_present AND fd_present AND qrc_present AND fr_present)
ORDER BY image_id;
```

**Output:**
```
image_id               | path                             | missing_models
-----------------------|----------------------------------|-------------------
abc123def456789...    | /data/Images/qr_code/gard.png    | fd, fr
def456ghi789jkl...    | /data/Images/multi/cars_01.png   | fd, qrcode, fr
pqr123stu456vwx...    | /data/Images/qr_code/gard_01.jpg | amod, fd, qrcode, fr
xyz789uvw012345...    | /data/Images/face_pics/john.jpg  | amod, qrcode
```

### Query 10: Summary statistics across all runs

```sql
SELECT 
  COUNT(DISTINCT image_id) as total_images,
  COUNT(DISTINCT run_id) as total_runs,
  SUM(CASE WHEN present THEN 1 ELSE 0 END) as detections_present,
  COUNT(*) as total_predictions
FROM predictions_summary;
```

**Output:**
```
total_images | total_runs | detections_present | total_predictions
--------------|----------|-------------------|-------------------
4            | 2         | 7                 | 16
```
