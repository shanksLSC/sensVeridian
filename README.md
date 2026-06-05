# sensVeridian

sensVeridian builds a queryable ground-truth cache for image evaluation pipelines and can synthesize distance-sweep augmentations for stress-testing model behavior at range.

## Current scope

- Oracle models (float `.h5`):
  - `AMOD`: `/data3/ssharma8/all-models/AutomotiveMultiObjectDetection/amod-cpnx-8.2.0.h5`
  - `QRCode`: `/data3/ssharma8/all-models/QRCode/qr-code-detection-final.h5`
  - `FaceDetection`: `/data3/ssharma8/all-models/FaceDetection/fd_lnd_hp-fpga-8.1.0.h5`
  - `FaceRecognition`: `/data3/ssharma8/all-models/FaceRecognition/fr-fpga-8.1.1.h5`
- Storage:
  - DuckDB for analytics and model outputs.
  - Redis for registered faces.
- Augmentation:
  - ZoeDepth (metric depth), SAM masks, SDXL inpainting, distance sweep generation.

## Install (py310)

```bash
/data3/ssharma8/py310/bin/python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quickstart

1. Start Redis (local default `redis://localhost:6379/0`).
2. Seed a dummy face registry:
   ```bash
   sv faces seed --n 8
   ```
3. Ingest images:
   ```bash
   sv ingest /path/to/images --run-id baseline
   ```
4. Query cache:
   ```bash
   sv stats
   sv query "select * from v_image_summary_wide limit 10"
   ```
5. Create distance sweep augmentations:
   ```bash
   sv augment distance /path/to/images --d-max-ft 10 --step-ft 1 --auto-run-oracle
   ```

## Veridian Studio (web UI + PostgreSQL backend)

A FastAPI backend + bundled web UI for ingesting local folders, running the
oracle models, and reviewing predictions on the real images, backed by
PostgreSQL.

```bash
pip install -e ".[api]"
cp .env.example .env            # DATABASE_URL, VERIDIAN_DATASETS_ROOT, ...
psql "$DATABASE_URL_SYNC" -f src/sensveridian/store/schema_pg.sql
uvicorn sensveridian.api.main:app --port 8000
# open the UI:
xdg-open http://localhost:8000/studio
```

See `docs/docs/veridian-studio/` — backend integration, running the Studio UI,
and the per-model output interpreters.

## Notes

- Detection decoding is handled by per-model interpreters in
  `src/sensveridian/postprocessors/` (ported from the MLHILS reference
  post-processors). Each detection model registers an interpreter; FR uses a
  cosine embedding. See `docs/docs/veridian-studio/postprocessors.md`.
- Add new models by implementing a runner under `src/sensveridian/runners/`,
  registering it in `orchestrator.py`, and adding its interpreter to
  `sensveridian.postprocessors.DETECTION_INTERPRETERS`.

