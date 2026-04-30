# CLI Reference

Entrypoint: `sv` (installed via `pyproject.toml`) or `python -m sensveridian.cli`.

## Top-level commands

| Command | What it does |
|---------|-------------|
| `sv ingest` | Run oracles on images and write to cache |
| `sv query` | Execute arbitrary SQL against DuckDB |
| `sv export` | Export a SQL result to parquet |
| `sv stats` | Show row counts for all tables |
| `sv refresh-metadata` | Recompute the `images.metadata` JSON column for one or many images |
| `sv faces ...` | Face registry subcommands |
| `sv augment ...` | Augmentation subcommands |

> All long-running commands now print a tqdm progress bar over the per-image loop,
> showing live counters for `ingested`, `writes`, and (for augment) `generated`.
> Inner ingest bars triggered by `--auto-run-oracle` collapse on completion so
> the outer augmentation bar stays clean.

## ingest

Run the oracle pipeline on a folder (recursive) or a single file.

```
sv ingest <image_root>
  [--run-id STR]           Run identifier (default: "baseline")
  [--models STR]           Comma-separated model IDs (default: "amod,qrcode,fd,fr")
  [--skip-existing BOOL]   Skip (image_id, run_id) already processed (default: True)
  [--conf FLOAT]           Generic detection confidence threshold override (applied
                           to any runner that exposes a `conf_threshold` attribute,
                           e.g. AMOD).
```

After every per-image batch of writes, `sv ingest` recomputes the `images.metadata`
JSON column for that image so it always reflects the most recent model runs,
augmentation state, and per-runner confidence thresholds.

### Examples

```bash
# Baseline ingest, all four models
sv ingest /data3/ssharma8/all-models/Images --run-id baseline

# Only AMOD + QR on a folder, force reprocess
sv ingest /data3/ssharma8/all-models/Images --models amod,qrcode --skip-existing False

# Custom run tag for an experiment
sv ingest /data3/ssharma8/all-models/Images --run-id exp_2026_04_24_cpu

# AMOD-only with a stricter detection threshold (0.5 instead of the runner default)
sv ingest /data3/ssharma8/datasets/mod2_class/images/test \
  --models amod --conf 0.5 --run-id mod2_test_amod_c05
```

## query

Run any SQL against the live DuckDB file.

```
sv query "<SQL>"
```

### Examples

```bash
sv query "select * from v_image_summary_wide limit 5"
sv query "select model_id, count(*) from predictions_summary group by model_id"
sv query "select image_id, present, count from predictions_summary where model_id='qrcode'"
```

## export

Export a SQL result to parquet for pandas/analytics pipelines.

```
sv export --to <path.parquet> [--sql "<SQL>"]
```

### Examples

```bash
# Full summary view
sv export --to /tmp/ground_truth_wide.parquet

# Only QR-positive images
sv export --to /tmp/qr_positive.parquet \
  --sql "SELECT image_id, path, n_qrc FROM v_image_summary_wide WHERE qrc_present"

# Raw detections for a specific run
sv export --to /tmp/raw_baseline.parquet \
  --sql "SELECT * FROM predictions_raw WHERE run_id='baseline'"
```

## stats

Quick count of rows across all tables.

```bash
sv stats
# Output:
# images: 4
# runs: 1
# predictions_summary: 16
# predictions_raw: 16
# augmentations: 5
```

## refresh-metadata

Recompute the `images.metadata` JSON column from the current contents of
`images`, `augmentations`, `predictions_summary`, `predictions_raw`, and
`models`. Useful after backfilling data, changing runner settings, or after
running augmentation pipelines that did not propagate metadata for some reason.

```
sv refresh-metadata
  [--run-id STR]      Only refresh images that have predictions for this run
  [--image-id STR]    Refresh exactly one image_id (overrides --run-id)
```

### Examples

```bash
# Refresh metadata for every image touched by run "mod2_test_amod_c05"
sv refresh-metadata --run-id mod2_test_amod_c05

# Refresh just one image
sv refresh-metadata --image-id 329acdb6e9850f346b06d04fd73aec48f28d793e5c8...

# Refresh metadata across the entire DB
sv refresh-metadata
```

A tqdm bar tracks per-image progress. See
[Data Model → images.metadata schema](data-model.md#imagesmetadata-json-schema)
for the JSON layout.

## faces

Manage the registered-faces lookup used by FR.

```
sv faces seed [--n INT] [--embedding-dim INT] [--clear-first]
sv faces list
sv faces clear
```

### Examples

```bash
# Seed 8 dummy identities (clears first)
sv faces seed --n 8 --clear-first

# Inspect registered identities
sv faces list

# Seed with a custom embedding dim (e.g., 512)
sv faces seed --n 16 --embedding-dim 512 --clear-first

# Clear all
sv faces clear
```

## augment distance

Generate distance-swept versions of each detected object.

```
sv augment distance <image_or_folder>
  --sam-checkpoint PATH       SAM .pth checkpoint path (required)
  [--d-max-ft FLOAT]          Upper distance threshold in feet
  [--step-ft FLOAT]           Distance step in feet
  [--step-m FLOAT]            Distance step in meters (alternative to --step-ft;
                              mutually exclusive)
  [--n-steps INT]             Run exactly N augmentation steps (overrides
                              the d-max-ft cap when set)
  [--source-models STR]       Detection sources (default: "amod,fd,qrcode")
  [--models STR]              Models to re-run on generated images when
                              --auto-run-oracle is set (default: "amod,qrcode,fd,fr")
  [--conf FLOAT]              Generic detection confidence threshold override
                              applied during the auto-run-oracle pass.
  [--run-id STR]              Run tag for auto oracle rerun (default: "augmented")
  [--auto-run-oracle]         Re-run oracles on generated images
  [--d0-ft FLOAT]             Manual initial distance (feet)
  [--d0-map PATH]             JSON file with per-image/per-detection overrides
  [--camera TEXT]             Camera profile for calibration fallback (e.g. imx219)
  [--camera-native WxH]       Override camera native mode (e.g. 1640x1232)
```

You must provide at least one of `--d-max-ft` or `--n-steps`. Step size is
required and may be supplied as either `--step-ft` or `--step-m` (only one).

### Examples

```bash
# One image, 5ft -> 10ft at 1ft steps
sv augment distance /data3/ssharma8/all-models/Images/qr_code/gard_00000.png \
  --d-max-ft 10 --step-ft 1 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth

# Whole folder, full ground truth populated on outputs too
sv augment distance /data3/ssharma8/all-models/Images \
  --d-max-ft 15 --step-ft 2 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth \
  --auto-run-oracle --run-id distance_eval_2026_04

# Known capture distance: bench tests shot at exactly 5 ft; skip ZoeDepth entirely
sv augment distance /data3/ssharma8/all-models/Images/qr_code \
  --d-max-ft 10 --step-ft 1 --d0-ft 5.0 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth

# Mixed: precise per-image distances from a JSON file
sv augment distance /data3/ssharma8/all-models/Images \
  --d-max-ft 15 --step-ft 1 \
  --d0-map /path/to/d0_overrides.json \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth

# Camera-calibrated fallback (replaces ZoeDepth fallback)
sv augment distance /data3/ssharma8/all-models/Images \
  --d-max-ft 15 --step-ft 1 \
  --camera imx219 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth

# Camera-calibrated fallback with native mode override
sv augment distance /data3/ssharma8/all-models/Images \
  --d-max-ft 15 --step-ft 1 \
  --camera imx219 --camera-native 1640x1232 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth

# Fixed N=5 steps using meters with an AMOD-only oracle rerun at conf=0.5
sv augment distance /data3/ssharma8/datasets/mod2_class/images/test \
  --step-m 10 --n-steps 5 \
  --source-models amod --models amod --conf 0.5 \
  --auto-run-oracle --run-id mod2_test_amod_c05 \
  --sam-checkpoint /data3/ssharma8/model-cache/sam/sam_vit_b_01ec64.pth
```

## augment list

List augmentations for a given parent image.

```
sv augment list <parent_image_id>
```

### Example

```bash
sv augment list 329acdb6e9850f346b06d04fd73aec48f28d793e5c8...
```

## augment miniaturize

Generate distance augmentation by shrinking the full frame and padding back to original size.

```
sv augment miniaturize <image_or_folder>
  [--d-max-ft FLOAT]          Upper distance threshold in feet
  [--step-ft FLOAT]           Distance step in feet
  [--step-m FLOAT]            Distance step in meters (alternative to --step-ft;
                              mutually exclusive)
  [--n-steps INT]             Run exactly N augmentation steps (overrides
                              the d-max-ft cap when set)
  [--source-models STR]       Detection sources for baseline distance (default: "amod,fd,qrcode")
  [--models STR]              Models to re-run on generated images when
                              --auto-run-oracle is set (default: "amod,qrcode,fd,fr")
  [--conf FLOAT]              Generic detection confidence threshold override
                              applied during the auto-run-oracle pass.
  [--run-id STR]              Run tag for auto oracle rerun (default: "miniaturized")
  [--auto-run-oracle]         Re-run oracles on generated images
  [--pad-mode STR]            black | replicate | reflect (default: "black")
  [--d0-ft FLOAT]             Manual initial distance (feet)
  [--d0-map PATH]             JSON file with per-image/per-detection overrides
  [--camera TEXT]             Camera profile for calibration fallback (e.g. imx219)
  [--camera-native WxH]       Override camera native mode (e.g. 1640x1232)
```

You must provide at least one of `--d-max-ft` or `--n-steps`. Step size is
required and may be supplied as either `--step-ft` or `--step-m` (only one).

### Examples

```bash
# Fast deterministic distance simulation
sv augment miniaturize /data3/ssharma8/all-models/Images \
  --d-max-ft 15 --step-ft 1 \
  --pad-mode black

# Calibration-backed distance baseline
sv augment miniaturize /data3/ssharma8/all-models/Images \
  --d-max-ft 15 --step-ft 1 \
  --camera imx219 --camera-native 1640x1232 \
  --pad-mode reflect

# Manual known baseline with oracle rerun
sv augment miniaturize /data3/ssharma8/all-models/Images \
  --d-max-ft 10 --step-ft 1 \
  --d0-ft 5.0 --auto-run-oracle --run-id mini_eval

# Generate exactly 5 miniaturized variants every 10 m, AMOD-only at conf=0.5
sv augment miniaturize /data3/ssharma8/datasets/mod2_class/images/test \
  --step-m 10 --n-steps 5 \
  --source-models amod --models amod --conf 0.5 \
  --pad-mode black --auto-run-oracle --run-id mod2_test_amod_c05
```
