# Veridian Studio backend

The Veridian Studio backend is a thin FastAPI layer over the existing
sensVeridian pipeline. The studio front-end talks to one async interface
(`window.VeridianAPI`); flipping `window.VERIDIAN_CONFIG.backend` to `"rest"`
points it at this API. No UI code changes — only the adapter switches.

This page documents how the design handoff was integrated into this repository.

## Architecture

```
 Veridian Studio (browser)            FastAPI app (sensveridian.api)
   window.VeridianAPI  ──HTTP/WS──▶     routers ──▶ AsyncPgStore  ─┐
                                        WS /ws/jobs ◀── progress    │
                                                                    ▼
                                   ingest worker (sensveridian.ingest)
                                     OpenCV frame sampler            PostgreSQL
                                     Orchestrator + runners  ──▶ PgStore (sync)
```

### The sync / async split (the key integration decision)

The handoff scaffold made a single `PgStore` fully `async`. The existing
`Orchestrator` is **synchronous** — it calls `store.query_df(sql)` and consumes
a pandas `DataFrame`, and writes via synchronous `upsert_*` calls — so it cannot
use an async store unchanged. Rather than rewrite the Orchestrator, the
integration splits responsibilities by layer:

| Layer | Store | Driver | Why |
|---|---|---|---|
| API request handlers (I/O bound, on the event loop) | `sensveridian.store.pg_async.AsyncPgStore` | `asyncpg` | non-blocking reads/writes for the UI |
| Ingest worker / `Orchestrator` (blocking CPU/GPU) | `sensveridian.store.pg.PgStore` | `psycopg` (sync) | keeps the Orchestrator unchanged, off the event loop |

Both target the same schema (`store/schema_pg.sql`). `PgStore` mirrors every
`DuckStore` method name and signature (`ensure_run`, `upsert_image`,
`upsert_image_metadata`, `upsert_model`, `upsert_summary`, `upsert_raw`,
`insert_augmentation`, `upsert_depth_stat`, `query_df`, `migrate`, `close`, …),
so `Orchestrator(store=PgStore(...), registry=...)` works with no Orchestrator
changes.

## Layout

```
src/sensveridian/
├── store/
│   ├── duck.py             # existing DuckDB store (unchanged)
│   ├── pg.py               # NEW sync PgStore (DuckStore-compatible) — worker/Orchestrator
│   ├── pg_async.py         # NEW async read/build store — API
│   └── schema_pg.sql       # NEW canonical PostgreSQL DDL
├── api/                    # NEW FastAPI app
│   ├── main.py             # app + CORS + routers + WS
│   ├── config.py           # settings (reuses sensveridian.config)
│   ├── db.py               # async engine/session
│   ├── deps.py             # store + bearer-auth dependencies
│   ├── schemas.py          # Pydantic request/response models
│   ├── fusion.py           # detection fusion (pred ↔ ground truth) — pure
│   ├── classmaps.py        # model/class metadata — pure
│   ├── lint.py             # label-lint health check — pure
│   ├── importers.py        # COCO/YOLO/CSV/VOC parsers — pure
│   └── routers/            # datasets, models, reviews, ingest, lineage, connections
├── ingest/
│   ├── frames.py           # OpenCV target-fps stride sampler (dataset-generator convention)
│   └── worker.py           # auto-label worker: Orchestrator + sync PgStore + WS progress
└── runners/
    └── aed.py              # NEW AcousticEventDetection runner (audio modality)
```

## Detection fusion

`GET /datasets/{id}/images/{imageId}` returns the merged pred↔ground-truth view
the canvas renders. The logic lives in `api/fusion.py` (pure, unit-tested) and
handles the **actual** runner payload shapes:

- **AMOD / QR / FD** detections come from `payload["detections"]`; **FR** identities
  come from `payload["recognized"]` (`matched_person_id` / `score`).
- **FD/FR** boxes are absolute pixels; **AMOD/QR** boxes are auto-detected
  (normalized vs pixels). All are normalized to `[x, y, w, h]` in `0..1`.
- State (`match` / `miss` / `fp` / `mismatch`) and `agreement` / `conflicts`
  match the front-end reference (`design_reference/app/data.js`).
- With no ground-truth layer yet, predictions are treated as provisional ground
  truth (the auto-label semantic), so a freshly ingested set is not shown as
  "all false positives".

## Running

```bash
pip install -e ".[api]"          # FastAPI, async + sync Postgres drivers, arq
cp .env.example .env             # set DATABASE_URL, SV_REDIS_URL, media/frames roots

# apply the schema (canonical path)
psql "postgresql://veridian:veridian@localhost:5432/sensveridian" -f src/sensveridian/store/schema_pg.sql

uvicorn sensveridian.api.main:app --reload --port 8000

# ingest worker (prod, Redis-backed). Dev runs in-process — no worker needed.
VERIDIAN_USE_ARQ=1 arq sensveridian.ingest.worker.WorkerSettings
```

Point the studio at `http://localhost:8000/api/v1`
(`window.VERIDIAN_CONFIG.backend = "rest"`).

## Endpoints

| Method | Path | Backed by |
|---|---|---|
| GET | `/datasets` | `datasets` + per-image rollups |
| GET | `/datasets/{id}` | dataset + images/clips (paginated) |
| GET | `/datasets/{id}/images/{imageId}` | `images` + `predictions_raw` + GT → `Detection[]` |
| GET | `/datasets/{id}/images/{imageId}/predictions` | fusion narrowed by run/model |
| GET | `/datasets/{id}/clips/{clipId}` | `clips` + `segments` |
| GET | `/datasets/{id}/layers` | `layers` |
| POST | `/datasets:import` | parse labels → GT layer + lint |
| GET | `/models` · `/models/{id}/regressions` | `models` + `model_versions` |
| POST | `/models/{id}/versions/{v}:promote` | flip `is_current` |
| PUT | `/reviews/{targetId}` · POST `/reviews:bulk` | `reviews` |
| POST | `/ingest/jobs` · GET `/ingest/jobs/{id}` · POST `/ingest/uploads` | `ingest_jobs` + worker |
| WS | `/ws/jobs/{jobId}` | worker progress |
| POST | `/connections` | `connections` |
| GET | `/lineage` | provenance DAG |

## Remaining seams (require live infrastructure)

These are wired with clear interfaces but need external services or assets to
exercise fully:

- **arq/Redis progress in production.** Dev streams progress over an in-process
  bus; when running the arq worker out-of-process, the API's WS handler should
  use `worker.redis_progress_stream` (subscribes to Redis pub/sub).
- **Presigned uploads.** `POST /ingest/uploads` returns local staging paths;
  wire to S3 (`boto3.generate_presigned_url`) for production.
- **Connection watchers.** `POST /connections` records the source; an arq cron /
  S3-event watcher that auto-creates ingest jobs is a seam.
- **AED model weights.** `runners/aed.py` uses a deterministic energy-based
  segmenter; `load()` is the seam for a real acoustic classifier.
- **Alembic migrations.** `schema_pg.sql` is the canonical DDL (apply via `psql`
  or `PgStore.migrate()`); an Alembic env can wrap it for versioned migrations.
- **Label ↔ frame matching.** Imported `gt_boxes` keep the original `image_ref`;
  resolving `image_id` (sha-256) for imported labels against ingested frames is
  a matching step.
- **FR identity names.** `AsyncPgStore` exposes a `name_map` hook
  (`person_id → display name`) for FR identities; wire it to `FaceRegistry`.
- **Regression version↔run mapping.** `/regressions` diffs two *runs*; a
  `model_versions.run_id` column would make the version→run mapping explicit.
- **Dataset-aggregate memoization.** Exact `agreement`/`conflicts` fuse per
  image; for very large datasets, precompute into a stats table / matview.
