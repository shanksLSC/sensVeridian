from __future__ import annotations

from pathlib import Path
from typing import Optional
import uuid
import typer
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

from .config import SETTINGS
from .store.duck import DuckStore
from .store.faces_registry import FaceRegistry
from .seed_faces import seed_dummy_faces
from .orchestrator import Orchestrator
from .augmentation.distance_sweep import DistanceAugmentor
from .augmentation.frame_miniaturize import FrameMiniaturizer
from .augmentation.camera import get_camera_profile
from .augmentation.manual_distance import DistanceOverrides


app = typer.Typer(help="sensVeridian CLI")
faces_app = typer.Typer(help="Face registry operations")
augment_app = typer.Typer(help="Augmentation operations")
app.add_typer(faces_app, name="faces")
app.add_typer(augment_app, name="augment")
console = Console()
M_TO_FT = 3.28084


def _store() -> DuckStore:
    schema_path = Path(__file__).resolve().parent / "store" / "schema.sql"
    s = DuckStore(db_path=SETTINGS.db_path, schema_path=schema_path)
    s.migrate()
    return s


def _registry() -> FaceRegistry:
    return FaceRegistry(redis_url=SETTINGS.redis_url)


def _resolve_step_ft(step_ft: Optional[float], step_m: Optional[float]) -> float:
    if step_ft is not None and step_m is not None:
        raise typer.BadParameter("Provide only one of --step-ft or --step-m.")
    if step_ft is None and step_m is None:
        raise typer.BadParameter("Either --step-ft or --step-m is required.")
    if step_m is not None:
        if step_m <= 0:
            raise typer.BadParameter("--step-m must be > 0.")
        return float(step_m) * M_TO_FT
    if step_ft is None or step_ft <= 0:
        raise typer.BadParameter("--step-ft must be > 0.")
    return float(step_ft)


@app.command("ingest")
def ingest(
    image_root: Path = typer.Argument(..., exists=True),
    run_id: str = typer.Option("baseline", help="Run identifier."),
    models: str = typer.Option("amod,qrcode,fd,fr", help="Comma-separated model IDs."),
    skip_existing: bool = typer.Option(True, help="Skip images already present for this run."),
    conf: Optional[float] = typer.Option(None, "--conf", help="Generic detection confidence threshold override."),
):
    s = _store()
    try:
        reg = _registry()
        orch = Orchestrator(store=s, registry=reg, conf_threshold=conf)
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


@app.command("refresh-metadata")
def refresh_metadata(
    run_id: Optional[str] = typer.Option(None, help="Only refresh images that have predictions for this run."),
    image_id: Optional[str] = typer.Option(None, help="Refresh only one image_id."),
):
    s = _store()
    try:
        reg = _registry()
        orch = Orchestrator(store=s, registry=reg)
        n = orch.refresh_metadata(run_id=run_id, image_id=image_id)
        console.print(f"Metadata refresh complete. Updated {n} image rows.")
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
    d_max_ft: Optional[float] = typer.Option(None, help="Upper distance threshold in feet."),
    step_ft: Optional[float] = typer.Option(None, help="Distance step in feet."),
    step_m: Optional[float] = typer.Option(None, help="Distance step in meters."),
    n_steps: Optional[int] = typer.Option(None, help="Run exactly N augmentation steps (overrides d_max_ft cap)."),
    source_models: str = typer.Option("amod,fd,qrcode", help="Detection sources for object extraction."),
    models: str = typer.Option("amod,qrcode,fd,fr", help="Comma-separated model IDs for auto-run-oracle reruns."),
    conf: Optional[float] = typer.Option(None, "--conf", help="Generic detection confidence threshold override."),
    run_id: str = typer.Option("augmented", help="Run id used if auto oracle run is enabled."),
    auto_run_oracle: bool = typer.Option(False, help="Run oracle models on generated images."),
    sam_checkpoint: Path = typer.Option(..., help="Path to SAM checkpoint .pth"),
    camera: Optional[str] = typer.Option(
        None,
        "--camera",
        help="Camera profile for calibration-based distance estimation (for example: imx219). Replaces ZoeDepth fallback when set.",
    ),
    camera_native: Optional[str] = typer.Option(
        None,
        "--camera-native",
        help="Override camera native resolution as WxH (for example: 1640x1232).",
    ),
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
    step_ft_resolved = _resolve_step_ft(step_ft=step_ft, step_m=step_m)
    if d_max_ft is not None and d_max_ft <= 0:
        raise typer.BadParameter("--d-max-ft must be > 0.")
    if n_steps is not None and n_steps <= 0:
        raise typer.BadParameter("--n-steps must be > 0.")
    if d_max_ft is None and n_steps is None:
        raise typer.BadParameter("Provide --d-max-ft or --n-steps.")

    s = _store()
    try:
        reg = _registry()
        orch = Orchestrator(store=s, registry=reg, conf_threshold=conf)
        camera_profile = None
        if camera is not None:
            camera_profile = get_camera_profile(camera)
            if camera_native:
                try:
                    native_w, native_h = [int(x.strip()) for x in camera_native.lower().split("x", 1)]
                except Exception as exc:  # pragma: no cover - argument validation path
                    raise typer.BadParameter("--camera-native must be in WxH format, for example 1640x1232") from exc
                if native_w <= 0 or native_h <= 0:
                    raise typer.BadParameter("--camera-native dimensions must be positive integers")
                camera_profile = camera_profile.with_native_resolution(native_w_px=native_w, native_h_px=native_h)

        aug = DistanceAugmentor(
            store=s,
            orchestrator=orch,
            sam_checkpoint=str(sam_checkpoint),
            device=SETTINGS.device,
            camera_profile=camera_profile,
        )
        roots = [image_or_folder] if image_or_folder.is_file() else sorted([p for p in image_or_folder.rglob("*") if p.is_file()])
        out_root = SETTINGS.cache_dir / "augmentations" / f"{run_id}_{uuid.uuid4().hex[:8]}"
        total = 0
        selected = [m.strip() for m in source_models.split(",") if m.strip()]
        rerun_models = {m.strip() for m in models.split(",") if m.strip()}

        if d0_map is not None:
            overrides = DistanceOverrides.from_json(d0_map)
            if d0_ft is not None and overrides.global_ft is None:
                overrides.global_ft = float(d0_ft)
        elif d0_ft is not None:
            overrides = DistanceOverrides(global_ft=float(d0_ft))
        else:
            overrides = DistanceOverrides.empty()

        console.print(
            f"Manual distance: global_ft={overrides.global_ft}, image_entries={len(overrides.images)}, step_ft={step_ft_resolved}, n_steps={n_steps}"
        )
        if camera_profile is not None:
            console.print(
                f"Camera calibration enabled: {camera_profile.name} ({camera_profile.native_w_px}x{camera_profile.native_h_px})"
            )

        image_roots = [p for p in roots if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        bar = tqdm(image_roots, desc=f"augment-distance[{run_id}]", unit="img")
        for p in bar:
            total += aug.augment_image(
                image_path=p,
                run_id=run_id,
                d_max_ft=d_max_ft,
                step_ft=step_ft_resolved,
                source_models=selected,
                out_dir=out_root,
                auto_run_oracle=auto_run_oracle,
                overrides=overrides,
                n_steps=n_steps,
                rerun_models=rerun_models,
            )
            bar.set_postfix(generated=total, refresh=False)
        console.print(f"Distance augmentation complete. Generated {total} images under {out_root}")
    finally:
        s.close()


@augment_app.command("miniaturize")
def augment_miniaturize(
    image_or_folder: Path = typer.Argument(..., exists=True),
    d_max_ft: Optional[float] = typer.Option(None, help="Upper distance threshold in feet."),
    step_ft: Optional[float] = typer.Option(None, help="Distance step in feet."),
    step_m: Optional[float] = typer.Option(None, help="Distance step in meters."),
    n_steps: Optional[int] = typer.Option(None, help="Run exactly N augmentation steps (overrides d_max_ft cap)."),
    source_models: str = typer.Option("amod,fd,qrcode", help="Detection sources for baseline distance estimation."),
    models: str = typer.Option("amod,qrcode,fd,fr", help="Comma-separated model IDs for auto-run-oracle reruns."),
    conf: Optional[float] = typer.Option(None, "--conf", help="Generic detection confidence threshold override."),
    run_id: str = typer.Option("miniaturized", help="Run id used if auto oracle run is enabled."),
    auto_run_oracle: bool = typer.Option(False, help="Run oracle models on generated images."),
    pad_mode: str = typer.Option(
        "black",
        "--pad-mode",
        help="Padding mode after downscale: black | replicate | reflect",
    ),
    camera: Optional[str] = typer.Option(
        None,
        "--camera",
        help="Camera profile for calibration-based distance fallback (for example: imx219). Replaces ZoeDepth fallback when set.",
    ),
    camera_native: Optional[str] = typer.Option(
        None,
        "--camera-native",
        help="Override camera native resolution as WxH (for example: 1640x1232).",
    ),
    d0_ft: Optional[float] = typer.Option(
        None,
        "--d0-ft",
        help="Manual initial distance (feet) applied to every detection. Lowest-priority override.",
    ),
    d0_map: Optional[Path] = typer.Option(
        None,
        "--d0-map",
        exists=True,
        help="JSON file with per-image / per-detection manual distance overrides.",
    ),
):
    step_ft_resolved = _resolve_step_ft(step_ft=step_ft, step_m=step_m)
    if d_max_ft is not None and d_max_ft <= 0:
        raise typer.BadParameter("--d-max-ft must be > 0.")
    if n_steps is not None and n_steps <= 0:
        raise typer.BadParameter("--n-steps must be > 0.")
    if d_max_ft is None and n_steps is None:
        raise typer.BadParameter("Provide --d-max-ft or --n-steps.")

    s = _store()
    try:
        reg = _registry()
        orch = Orchestrator(store=s, registry=reg, conf_threshold=conf)

        camera_profile = None
        if camera is not None:
            camera_profile = get_camera_profile(camera)
            if camera_native:
                try:
                    native_w, native_h = [int(x.strip()) for x in camera_native.lower().split("x", 1)]
                except Exception as exc:  # pragma: no cover - argument validation path
                    raise typer.BadParameter("--camera-native must be in WxH format, for example 1640x1232") from exc
                if native_w <= 0 or native_h <= 0:
                    raise typer.BadParameter("--camera-native dimensions must be positive integers")
                camera_profile = camera_profile.with_native_resolution(native_w_px=native_w, native_h_px=native_h)

        mini = FrameMiniaturizer(
            store=s,
            orchestrator=orch,
            device=SETTINGS.device,
            camera_profile=camera_profile,
        )
        roots = [image_or_folder] if image_or_folder.is_file() else sorted([p for p in image_or_folder.rglob("*") if p.is_file()])
        out_root = SETTINGS.cache_dir / "augmentations" / f"{run_id}_{uuid.uuid4().hex[:8]}"
        total = 0
        selected = [m.strip() for m in source_models.split(",") if m.strip()]
        rerun_models = {m.strip() for m in models.split(",") if m.strip()}

        if d0_map is not None:
            overrides = DistanceOverrides.from_json(d0_map)
            if d0_ft is not None and overrides.global_ft is None:
                overrides.global_ft = float(d0_ft)
        elif d0_ft is not None:
            overrides = DistanceOverrides(global_ft=float(d0_ft))
        else:
            overrides = DistanceOverrides.empty()

        console.print(
            f"Frame miniaturize: pad_mode={pad_mode}, global_ft={overrides.global_ft}, image_entries={len(overrides.images)}, step_ft={step_ft_resolved}, n_steps={n_steps}"
        )
        if camera_profile is not None:
            console.print(
                f"Camera calibration enabled: {camera_profile.name} ({camera_profile.native_w_px}x{camera_profile.native_h_px})"
            )

        image_roots = [p for p in roots if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        bar = tqdm(image_roots, desc=f"augment-miniaturize[{run_id}]", unit="img")
        for p in bar:
            total += mini.augment_image(
                image_path=p,
                run_id=run_id,
                d_max_ft=d_max_ft,
                step_ft=step_ft_resolved,
                source_models=selected,
                out_dir=out_root,
                auto_run_oracle=auto_run_oracle,
                overrides=overrides,
                pad_mode=pad_mode,
                n_steps=n_steps,
                rerun_models=rerun_models,
            )
            bar.set_postfix(generated=total, refresh=False)
        console.print(f"Frame miniaturization complete. Generated {total} images under {out_root}")
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

