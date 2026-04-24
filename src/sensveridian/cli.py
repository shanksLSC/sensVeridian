from __future__ import annotations

from pathlib import Path
from typing import Optional
import uuid
import typer
from rich.console import Console
from rich.table import Table

from .config import SETTINGS
from .store.duck import DuckStore
from .store.faces_registry import FaceRegistry
from .seed_faces import seed_dummy_faces
from .orchestrator import Orchestrator
from .augmentation.distance_sweep import DistanceAugmentor
from .augmentation.manual_distance import DistanceOverrides


app = typer.Typer(help="sensVeridian CLI")
faces_app = typer.Typer(help="Face registry operations")
augment_app = typer.Typer(help="Augmentation operations")
app.add_typer(faces_app, name="faces")
app.add_typer(augment_app, name="augment")
console = Console()


def _store() -> DuckStore:
    schema_path = Path(__file__).resolve().parent / "store" / "schema.sql"
    s = DuckStore(db_path=SETTINGS.db_path, schema_path=schema_path)
    s.migrate()
    return s


def _registry() -> FaceRegistry:
    return FaceRegistry(redis_url=SETTINGS.redis_url)


@app.command("ingest")
def ingest(
    image_root: Path = typer.Argument(..., exists=True),
    run_id: str = typer.Option("baseline", help="Run identifier."),
    models: str = typer.Option("amod,qrcode,fd,fr", help="Comma-separated model IDs."),
    skip_existing: bool = typer.Option(True, help="Skip images already present for this run."),
):
    s = _store()
    try:
        reg = _registry()
        orch = Orchestrator(store=s, registry=reg)
        selected = {m.strip() for m in models.split(",") if m.strip()}
        res = orch.ingest(image_root=image_root, run_id=run_id, selected_models=selected, skip_existing=skip_existing)
        console.print(f"Ingest complete: seen={res.images_seen}, ingested={res.images_ingested}, writes={res.predictions_written}")
    finally:
        s.close()


@app.command("query")
def query(sql: str = typer.Argument(..., help="SQL query executed on DuckDB")):
    s = _store()
    try:
        df = s.query_df(sql)
        if df.empty:
            console.print("No rows.")
            return
        table = Table(show_header=True)
        for col in df.columns:
            table.add_column(str(col))
        for _, row in df.iterrows():
            table.add_row(*[str(row[c]) for c in df.columns])
        console.print(table)
    finally:
        s.close()


@app.command("export")
def export(
    to: Path = typer.Option(..., help="Output parquet path."),
    sql: str = typer.Option("SELECT * FROM v_image_summary_wide", help="SQL statement to export."),
):
    s = _store()
    try:
        s.export_parquet(sql=sql, out_path=to)
        console.print(f"Exported to {to}")
    finally:
        s.close()


@app.command("stats")
def stats():
    s = _store()
    try:
        counts = {
            "images": int(s.query_df("select count(*) c from images")["c"].iloc[0]),
            "runs": int(s.query_df("select count(*) c from runs")["c"].iloc[0]),
            "predictions_summary": int(s.query_df("select count(*) c from predictions_summary")["c"].iloc[0]),
            "predictions_raw": int(s.query_df("select count(*) c from predictions_raw")["c"].iloc[0]),
            "augmentations": int(s.query_df("select count(*) c from augmentations")["c"].iloc[0]),
        }
        for k, v in counts.items():
            console.print(f"{k}: {v}")
    finally:
        s.close()


@faces_app.command("seed")
def faces_seed(
    n: int = typer.Option(8, help="Number of dummy entries."),
    embedding_dim: int = typer.Option(128, help="Embedding vector length."),
    clear_first: bool = typer.Option(False, help="Clear existing registry first."),
):
    reg = _registry()
    if clear_first:
        reg.clear()
    seed_dummy_faces(registry=reg, n=n, embedding_dim=embedding_dim)
    console.print(f"Seeded {n} dummy face records.")


@faces_app.command("list")
def faces_list():
    reg = _registry()
    entries = reg.list_entries()
    if not entries:
        console.print("No entries.")
        return
    table = Table(show_header=True)
    table.add_column("person_id")
    table.add_column("name")
    table.add_column("embedding_dim")
    for e in entries:
        table.add_row(e.person_id, e.name, str(len(e.embedding)))
    console.print(table)


@faces_app.command("clear")
def faces_clear():
    reg = _registry()
    reg.clear()
    console.print("Face registry cleared.")


@augment_app.command("distance")
def augment_distance(
    image_or_folder: Path = typer.Argument(..., exists=True),
    d_max_ft: float = typer.Option(..., help="Upper distance threshold in feet."),
    step_ft: float = typer.Option(..., help="Distance step in feet."),
    source_models: str = typer.Option("amod,fd,qrcode", help="Detection sources for object extraction."),
    run_id: str = typer.Option("augmented", help="Run id used if auto oracle run is enabled."),
    auto_run_oracle: bool = typer.Option(False, help="Run oracle models on generated images."),
    sam_checkpoint: Path = typer.Option(..., help="Path to SAM checkpoint .pth"),
    d0_ft: Optional[float] = typer.Option(
        None,
        "--d0-ft",
        help="Manual initial distance (feet) applied to every detection. Lowest-priority override; ZoeDepth is used if unset.",
    ),
    d0_map: Optional[Path] = typer.Option(
        None,
        "--d0-map",
        exists=True,
        help="JSON file with per-image / per-detection manual distance overrides. See docs for schema.",
    ),
):
    s = _store()
    try:
        reg = _registry()
        orch = Orchestrator(store=s, registry=reg)
        aug = DistanceAugmentor(store=s, orchestrator=orch, sam_checkpoint=str(sam_checkpoint), device=SETTINGS.device)
        roots = [image_or_folder] if image_or_folder.is_file() else sorted([p for p in image_or_folder.rglob("*") if p.is_file()])
        out_root = SETTINGS.cache_dir / "augmentations" / f"{run_id}_{uuid.uuid4().hex[:8]}"
        total = 0
        selected = [m.strip() for m in source_models.split(",") if m.strip()]

        if d0_map is not None:
            overrides = DistanceOverrides.from_json(d0_map)
            if d0_ft is not None and overrides.global_ft is None:
                overrides.global_ft = float(d0_ft)
        elif d0_ft is not None:
            overrides = DistanceOverrides(global_ft=float(d0_ft))
        else:
            overrides = DistanceOverrides.empty()

        console.print(
            f"Manual distance: global_ft={overrides.global_ft}, image_entries={len(overrides.images)}"
        )

        for p in roots:
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            total += aug.augment_image(
                image_path=p,
                run_id=run_id,
                d_max_ft=d_max_ft,
                step_ft=step_ft,
                source_models=selected,
                out_dir=out_root,
                auto_run_oracle=auto_run_oracle,
                overrides=overrides,
            )
        console.print(f"Distance augmentation complete. Generated {total} images under {out_root}")
    finally:
        s.close()


@augment_app.command("list")
def augment_list(parent_image_id: str):
    s = _store()
    try:
        df = s.query_df(f"select * from augmentations where parent_image_id='{parent_image_id}' order by step_index")
        if df.empty:
            console.print("No augmentations found.")
            return
        table = Table(show_header=True)
        for c in df.columns:
            table.add_column(str(c))
        for _, row in df.iterrows():
            table.add_row(*[str(row[c]) for c in df.columns])
        console.print(table)
    finally:
        s.close()


if __name__ == "__main__":
    app()

