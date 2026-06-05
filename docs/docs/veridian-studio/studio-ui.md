# Running the Studio UI

The Veridian Studio front-end is served by the backend at `/studio`. From there
you pick a folder of images (or videos) on disk, run the oracle models, watch
live progress, and review the predictions on the real images.

## Launch

```bash
pip install -e ".[api]"
cp .env.example .env            # set DATABASE_URL, VERIDIAN_DATASETS_ROOT, ...
psql "$DATABASE_URL_SYNC" -f src/sensveridian/store/schema_pg.sql
uvicorn sensveridian.api.main:app --port 8000
```

Open **http://localhost:8000/studio**. The storage badge (bottom-left) shows the
live PostgreSQL connection and `api: rest`.

## Ingest a folder

1. **New source → Local dataset folder**. The picker lists folders under
   `VERIDIAN_DATASETS_ROOT` (default `/data3/ssharma8/datasets`) with image/video
   counts.
2. Select one or more folders. For each, choose the models to run (AMOD / QR /
   FD / FR) and a max-images cap (default 200, adjustable). Optionally enable
   **confidence-gated pre-labelling** (auto-accept confident detections into the
   ground-truth layer).
3. **Run models.** A live progress view streams the stages
   (Decoding → Sampling → Auto-labelling → Hash+dedup → Writing to Postgres)
   over a WebSocket. Images run through the oracle pipeline; videos are
   frame-split first (OpenCV target-fps sampler).
4. On completion, open the new dataset.

## Review

- **Grid** shows every ingested image with its agreement, conflicts, and status.
- **Canvas** shows the real image with predicted boxes (decoded by the model's
  interpreter — see [Model interpreters](postprocessors.md)), the detections
  panel, distance sweep, and confidence filter. Accept / reject / edit / flag
  detections, or verify a whole frame; verdicts persist to the `reviews` table
  and form the dataset's ground-truth layer.

## What runs where

- **API** (FastAPI, async) serves the UI and the read endpoints.
- **Ingest worker** runs the synchronous `Orchestrator` + runners off the event
  loop (in-process in dev; arq/Redis in production).
- **PostgreSQL** stores images (content-addressed by sha-256, with the on-disk
  `path`), predictions, reviews, and datasets. Raw image bytes are served from
  the filesystem via `/datasets/{id}/images/{imageId}/raw`.

See [Backend integration](backend.md) for architecture and the remaining infra
seams.
