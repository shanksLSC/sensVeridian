# Sample Queries

All queries run against DuckDB via `sv query "<SQL>"` or programmatically.

## Basic Statistics

### Images & Run Counts

```sql
SELECT COUNT(*) as total_images FROM images;
SELECT COUNT(*) as total_runs FROM runs;
```

### Predictions per Model

```sql
SELECT model_id, COUNT(*) as num_images, SUM(count) as total_detections
FROM predictions_summary
GROUP BY model_id
ORDER BY total_detections DESC;
```

### Detection Summary (Pivoted)

```sql
SELECT 
  image_id,
  path,
  width,
  height,
  amod_present, n_amod,
  qrc_present, n_qrc,
  fd_present, n_fd,
  fr_present, n_fr
FROM v_image_summary_wide
LIMIT 10;
```

## Model-Specific Queries

### AMOD: All Multi-Object Detections

```sql
SELECT image_id, COUNT(*) as object_count, 
       array_agg(bbox) as bboxes,
       array_agg(class_id) as classes
FROM (
  SELECT image_id, 
         (payload->'detections'->0->>'bbox')::VARCHAR as bbox,
         (payload->'detections'->0->>'class_id')::VARCHAR as class_id
  FROM predictions_raw WHERE model_id='amod'
)
GROUP BY image_id;
```

### QRCode: Decoded Text Extraction

```sql
SELECT image_id, 
       (payload->'decoded_strings')::VARCHAR as qr_text,
       COUNT(*) as num_qr_codes
FROM predictions_raw 
WHERE model_id='qrcode' AND (payload->'decoded_strings')::VARCHAR IS NOT NULL
GROUP BY image_id;
```

### Face Detection: Crop Count

```sql
SELECT image_id, 
       array_length((payload->'crops')::VARCHAR[], 1) as num_crops
FROM predictions_raw 
WHERE model_id='fd';
```

### Face Recognition: Matched Identities

```sql
SELECT image_id,
       (payload->'matched_person_ids')::VARCHAR as matched_ids,
       (payload->'match_distances')::VARCHAR as distances
FROM predictions_raw
WHERE model_id='fr';
```

## Augmentation & Depth Queries

### Images with Distance Data

```sql
SELECT image_id, 
       source,
       n_manual_distance,
       n_zoe_distance,
       median_depth_ft,
       min_depth_ft,
       max_depth_ft
FROM image_depth_stats
ORDER BY median_depth_ft DESC;
```

### Sources of Depth Estimation

```sql
SELECT source, COUNT(*) as num_images
FROM image_depth_stats
GROUP BY source;
```

### Augmented Images from Specific Parent

```sql
SELECT parent_image_id, 
       COUNT(*) as num_augmentations,
       MIN(distance_ft) as min_distance,
       MAX(distance_ft) as max_distance
FROM augmentations
GROUP BY parent_image_id;
```

### Augmentation Parameters (JSON)

```sql
SELECT augmented_image_id, 
       params->>'inpainter' as inpainter_used,
       params->>'dof_blur' as dof_enabled,
       params->>'haze_strength' as atmospheric_effect
FROM augmentations;
```

## Export to Parquet for Analytics

### Full Wide Summary

```bash
sv export --to /tmp/ground_truth_wide.parquet
```

### Filtered: QR-Positive Images Only

```bash
sv export --to /tmp/qr_positive.parquet \
  --sql "SELECT * FROM v_image_summary_wide WHERE qrc_present ORDER BY image_id"
```

### Raw Detections for AMOD

```bash
sv export --to /tmp/amod_raw.parquet \
  --sql "SELECT image_id, payload FROM predictions_raw WHERE model_id='amod'"
```

### Depth Statistics

```bash
sv export --to /tmp/depth_stats.parquet \
  --sql "SELECT * FROM image_depth_stats ORDER BY median_depth_ft"
```

## Python API

Programmatic access via DuckStore:

```python
from sensveridian.store.duck import DuckStore

store = DuckStore()

# Query as DataFrame
df = store.query_df("SELECT * FROM v_image_summary_wide")
print(df)

# Export to Parquet
store.export_parquet("/tmp/results.parquet", sql="SELECT * FROM v_image_summary_wide")
```
