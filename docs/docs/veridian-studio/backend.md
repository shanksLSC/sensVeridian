# Veridian Studio backend

The Veridian Studio backend is a thin FastAPI layer over the existing
sensVeridian pipeline. The studio front-end is **served by the backend** at
`/studio` and talks to it over REST + WebSocket. You pick a folder of images (or
videos) on disk, run the oracle models, watch progress, and review the
predictions on the real images — all backed by PostgreSQL.

## Architecture

```
 Veridian Studio (browser, served at /studio)        FastAPI app (sensveridian.api)
   window.VeridianAPI (REST)  ──HTTP──▶  routers ──▶ AsyncPgStore (asyncpg)  ─┐
   hydrate.js fills window.VD            /studio static mount                  │
                            ◀──WS /ws/jobs── progress                          ▼
                                   ingest worker (sensveridian.ingest)   PostgreSQL
                                     OpenCV frame sampler (videos)
                                     Orchestrator + runners  ──▶ PgStore (sync, psycopg)
                                     reads images from VERIDIAN_DATASETS_ROOT
```

### The sync / async split

The `Orchestrator` is **synchronous** (it calls `store.query_df(sql)` and
consumes a pandas `DataFrame`), so it cannot use an async store unchanged. The
integration splits by layer:

| Layer | Store | Driver |
|---|---|---|
| API request handlers (on the event loop) | `sensveridian.store.pg_async.AsyncPgStore` | `asyncpg` |
| Ingest worker / `Orchestrator` (blocking CPU/GPU) | `sensveridian.store.pg.PgStore` | `psycopg` (sync) |

`PgStore` mirrors every `DuckStore` method, so `Orchestrator(store=PgStore(...))`
runs with no Orchestrator changes.

## Raw data store: the local filesystem (no S3)

Media lives on disk under `VERIDIAN_DATASETS_ROOT` (default
`/data3/ssharma8/datasets`). `images.image_id` stays the sha-256 of the decoded
pixels (content-addressed dedup); the filesystem `path` is stored and indexed,
and `GET /datasets/{id}/images/{imageId}/raw` serves the bytes from it
(validated to live under the datasets/frames roots).

## Layout

```
src/sensveridian/
├── store/{pg.py (sync), pg_async.py (async), schema_pg.sql}
├── api/
│   ├── main.py            # app + CORS + routers + WS + /studio static mount + model-seed lifespan
│   ├── config.py          # settings (datasets_root, studio_dir, max_ingest_images, ...)
│   ├── fusion.py / classmaps.py / lint.py / importers.py   # pure logic
│   ├── routers/{datasets,models,reviews,ingest,lineage,connections,fs}.py
│   └── studio/            # the bundled no-build front-end (served at /studio)
├── ingest/
│   ├── kinds.py           # image/video extensions + discovery (stdlib only)
│   ├── frames.py          # OpenCV target-fps stride sampler (dataset-generator convention)
│   └── worker.py          # local-source ingest: images + videos -> models -> Postgres + WS
└── runners/aed.py
```

## Local-folder ingest (images + videos)

`POST /ingest/jobs` with `source:"local"` and groups carrying a `path` (relative
to the datasets root), `models`, `maxImages`, and `kind`:

- **Images** — the first `maxImages` files are symlink-staged into a per-run dir
  (so the model weights hash once) and `Orchestrator.ingest` runs over them.
- **Videos** — sampled to frames with the OpenCV target-fps stride sampler, then
  `Orchestrator.ingest` runs over the frames.
- Each frame is dataset-tagged; with a `trustThreshold`, confident detections
  are auto-accepted into the ground-truth layer; the summary matview is
  refreshed. Progress streams over `WS /ws/jobs/{jobId}` (in-process in dev;
  Redis pub/sub under arq in prod).

## Model catalogue

A FastAPI lifespan hook seeds the `models` + `model_versions` tables from the
static model cards + `sensveridian.config.MODELS` (no TensorFlow import), so
`/models` and the ingest model picker work before any ingest.

## Running

```bash
pip install -e ".[api]"          # FastAPI, asyncpg + psycopg drivers, arq
cp .env.example .env             # DATABASE_URL, VERIDIAN_DATASETS_ROOT, ...
psql "$DATABASE_URL_SYNC" -f src/sensveridian/store/schema_pg.sql
uvicorn sensveridian.api.main:app --port 8000
# open the UI:
xdg-open http://localhost:8000/studio
# ingest worker (prod, Redis-backed). Dev runs in-process — no worker needed.
VERIDIAN_USE_ARQ=1 arq sensveridian.ingest.worker.WorkerSettings
```

## Endpoints

| Method | Path | Backed by |
|---|---|---|
| GET | `/fs/datasets` · `/fs/browse` | folders under the datasets root (picker) |
| POST | `/ingest/jobs` · GET `/ingest/jobs/{id}` | local ingest (images/videos) + worker |
| WS | `/ws/jobs/{jobId}` | worker progress |
| GET | `/datasets` · `/datasets/{id}` | datasets + batched grid (images + fused detections + `src`) |
| GET | `/datasets/{id}/images/{imageId}` · `/raw` | fused detections · raw image bytes |
| GET | `/datasets/{id}/images/{imageId}/predictions` | fusion narrowed by run/model |
| GET | `/datasets/{id}/clips/{clipId}` · `/layers` | audio clips · layers |
| POST | `/datasets:import` | parse labels → GT layer + lint |
| GET | `/models` · `/models/{id}/regressions` · POST `:promote` | model catalogue |
| PUT | `/reviews/{targetId}` · POST `/reviews:bulk` | human verdicts |
| GET | `/lineage` · POST `/connections` | provenance DAG · watched local folder |

## Detection decoding — per-model interpreters

Each detection model has a **Python interpreter** that turns its raw head
tensors into decoded, NMS'd detections — ported from the reference C
post-processors at `/data3/ssharma8/projects/lattice-internal/postptocessors_MLHILS/src`.
They live in `sensveridian.postprocessors` and are selected by a registry:

| model | interpreter | head |
|---|---|---|
| `amod` | `multiobject.interpret` | FCOS, 6 tensors (3 scales), 8 classes |
| `qrcode` | `qrcode.interpret` | 4-anchor single-class |
| `fd` | `detection.interpret_face` | 2-anchor + 5 landmarks |
| `fr` | `embedding` | L2-normalized cosine (matched via FaceRegistry) |

The runners call `postprocessors.interpret(model_id, outputs, img_w, img_h,
conf)` instead of dumping raw candidates, so the stored boxes are real (decoded
+ NMS). **Convention: every new detection model added for ingestion must
register an interpreter** in `postprocessors.DETECTION_INTERPRETERS`; an
unregistered model raises a clear error. See `docs/veridian-studio/postprocessors.md`.

Note: a few header constants (e.g. `MOD_MAX_FPGA_OUTPUT`, the AMOD class-name
order) were not shipped with the weights and are calibrated against real outputs
in `postprocessors/constants.py`; confirm them against the MLHILS headers when
available.

## Remaining seams (require live infrastructure)

- **arq/Redis progress in production** — dev streams over an in-process bus; the
  arq worker publishes to Redis pub/sub (`worker.redis_progress_stream`).
- **Connection watchers** — `POST /connections` records a watched local folder;
  a cron/inotify watcher that auto-creates ingest jobs is a seam.
- **AED model weights** — `runners/aed.py` uses an energy-based placeholder;
  `load()` is the seam for a real acoustic classifier.
- **Alembic migrations** — `schema_pg.sql` is the canonical DDL; an Alembic env
  can wrap it for versioned migrations.
- **Imported label ↔ frame matching**, **FR identity names** (`name_map`), and
  **regression version↔run mapping** are documented hooks in `pg_async.py`.
- **Dataset-aggregate memoization** — exact agreement/conflicts fuse per image;
  precompute into a stats table for very large datasets.
