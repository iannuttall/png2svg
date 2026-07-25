"""png2svg CLI: init a project, build the SVG, compare against the reference."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import typer
from PIL import Image

from . import compare as cmp
from .model import Project, load_project, save_project, sha256_file
from .render import RENDERER, render_svg
from .svggen import generate_svg

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _estimate_background(img: Image.Image) -> list[int]:
    """Median colour of the 2px border ring."""
    arr = np.asarray(img.convert("RGBA"))
    ring = np.concatenate(
        [
            arr[:2].reshape(-1, 4),
            arr[-2:].reshape(-1, 4),
            arr[:, :2].reshape(-1, 4),
            arr[:, -2:].reshape(-1, 4),
        ]
    )
    return [int(v) for v in np.median(ring, axis=0)]


@app.command()
def init(
    image: Path = typer.Argument(..., exists=True, dir_okay=False),
    project: Path = typer.Option(..., "--project", "-p"),
) -> None:
    """Create a project directory around a source PNG."""
    project.mkdir(parents=True, exist_ok=True)
    (project / "source").mkdir(exist_ok=True)
    dest = project / "source" / image.name
    shutil.copy2(image, dest)
    img = Image.open(dest)
    proj = Project(
        source_path=f"source/{image.name}",
        width=img.width,
        height=img.height,
        sha256=sha256_file(dest),
        background=_estimate_background(img),
        view_box=[0, 0, img.width, img.height],
        shapes=[],
        notes=[f"renderer: {RENDERER}"],
    )
    save_project(project, proj)
    typer.echo(
        f"initialised {project}/project.json "
        f"({img.width}x{img.height}, background={proj.background})"
    )


@app.command()
def build(project: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    """Generate generated/current.svg from project.json."""
    proj = load_project(project)
    svg = generate_svg(proj)
    out = project / "generated"
    out.mkdir(exist_ok=True)
    (out / "current.svg").write_text(svg)
    typer.echo(f"wrote {out / 'current.svg'} ({len(svg)} bytes)")


@app.command()
def check(
    project: Path = typer.Argument(..., exists=True, file_okay=False),
    label: str = typer.Option("latest", "--label", "-l"),
    supersample: int = typer.Option(
        4, "--supersample", "-s",
        help="render at Nx and box-downsample; cancels resvg's quarter-pixel "
             "edge quantisation so the metrics score the model, not the "
             "rasteriser. Use 1 to see raw renderer output.",
    ),
) -> None:
    """Build, render, and compare against the reference. Prints metrics."""
    proj = load_project(project)
    svg = generate_svg(proj)
    gen = project / "generated"
    gen.mkdir(exist_ok=True)
    (gen / "current.svg").write_text(svg)

    ref = Image.open(project / proj.source_path)
    render = render_svg(svg, proj.width, proj.height, supersample=supersample)
    out_dir = project / "comparisons" / label
    bg = tuple(proj.background[:3])
    metrics = cmp.compare(ref, render, bg, out_dir=out_dir)
    metrics["renderer"] = RENDERER
    metrics["supersample"] = supersample
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def analyse(project: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    """First-pass deterministic analysis: components, segments, paint probes.

    Writes analysis/features.json and analysis/overlay.png. Coordinates are
    mask-derived proposals (±0.5px); refine with png2svg.measure before
    committing geometry to the model.
    """
    from .analyse import analyse_image

    proj = load_project(project)
    img = Image.open(project / proj.source_path)
    features, overlay = analyse_image(img, proj.background)
    out = project / "analysis"
    out.mkdir(exist_ok=True)
    (out / "features.json").write_text(json.dumps(features, indent=2) + "\n")
    overlay.save(out / "overlay.png")
    for c in features["components"]:
        kinds = [s["kind"] for s in c["segments"]]
        typer.echo(
            f"component {c['id']}: bbox={c['bbox']} area={c['area_px']} "
            f"paint={c['paint']['kind']} segments="
            f"{kinds.count('line')}L/{kinds.count('arc')}A/{kinds.count('curve')}C "
            f"corners={c['n_corners']}"
        )
    typer.echo(f"wrote {out / 'features.json'} and overlay.png")


@app.command()
def residuals(
    project: Path = typer.Argument(..., exists=True, file_okay=False),
    label: str = typer.Option("latest", "--label", "-l"),
    threshold: float = typer.Option(5.0, "--threshold", "-t"),
) -> None:
    """Localise remaining errors in a saved comparison (colour + edge clusters)."""
    from .residuals import find_residuals

    proj = load_project(project)
    comp_dir = project / "comparisons" / label
    if not (comp_dir / "render.png").exists():
        raise typer.BadParameter(f"no comparison at {comp_dir}; run `check` first")
    ref = Image.open(project / proj.source_path)
    render = Image.open(comp_dir / "render.png")
    rep = find_residuals(ref, render, tuple(proj.background[:3]), threshold)
    (comp_dir / "residuals.json").write_text(json.dumps(rep, indent=2) + "\n")
    typer.echo(json.dumps(rep, indent=2))


@app.command()
def recolor(
    project: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output", "-o"),
    rotate: float = typer.Option(0.0, "--rotate", help="hue rotation in degrees"),
    mapping: str = typer.Option(
        "", "--map",
        help="anchor remaps '#old=#new,...': each colour gets the Lab delta "
        "of its nearest anchor",
    ),
) -> None:
    """Write a colour-variant copy of a project (geometry untouched)."""
    import colorsys
    import shutil

    from . import compare as cmp_mod

    def hex_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    def to_lab(c):
        arr = np.array([[c]], dtype=float)
        return cmp_mod.linear_to_lab(cmp_mod.srgb_to_linear(arr))[0, 0]

    anchors = []
    if mapping:
        for pair in mapping.split(","):
            old, new = pair.strip().split("=")
            o, nw = hex_rgb(old), hex_rgb(new)
            anchors.append((to_lab(o), np.array(nw, float) - np.array(o, float)))

    def transform(hexc: str) -> str:
        r, g, b = hex_rgb(hexc)
        if anchors:
            lab = to_lab((r, g, b))
            deltas = [(np.linalg.norm(lab - a), d) for a, d in anchors]
            _, delta = min(deltas, key=lambda t: t[0])
            r, g, b = (int(np.clip(v, 0, 255)) for v in (np.array([r, g, b], float) + delta))
        if rotate:
            hh, ll, ss = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
            r, g, b = (
                round(v * 255)
                for v in colorsys.hls_to_rgb((hh + rotate / 360) % 1, ll, ss)
            )
        return f"#{r:02x}{g:02x}{b:02x}"

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "color" and isinstance(v, str):
                    o[k] = transform(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    proj = load_project(project)
    data = proj.to_dict()
    walk(data["model"]["shapes"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "source").mkdir(exist_ok=True)
    shutil.copy2(project / proj.source_path, output / proj.source_path)
    from .model import Project as P

    save_project(output, P.from_dict(data))
    typer.echo(f"wrote colour variant to {output}/project.json")


@app.command()
def validate(project: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    """Standalone-SVG validation: XML, security, determinism, scale, halos."""
    import xml.etree.ElementTree as ET

    proj = load_project(project)
    svg = generate_svg(proj)
    report: dict = {"renderer": RENDERER, "checks": {}}
    checks = report["checks"]

    # XML parses and contains no raster/script/external references
    root = ET.fromstring(svg)
    banned = [el.tag.split("}")[-1] for el in root.iter()
              if el.tag.split("}")[-1] in ("image", "script", "foreignObject", "use")]
    refs_ok = "http://www.w3.org/1999/xlink" not in svg.replace(
        'xmlns="http://www.w3.org/2000/svg"', "")
    checks["xml_parses"] = True
    checks["no_raster_or_script"] = not banned and "data:" not in svg and refs_ok
    checks["has_viewbox"] = root.get("viewBox") is not None

    # deterministic re-generation
    checks["deterministic"] = generate_svg(load_project(project)) == svg

    # sharp at 1x/4x/16x: alpha histogram must stay bimodal (few mid values)
    for scale in (1, 4, 16):
        img = render_svg(svg, proj.width * scale, proj.height * scale)
        a = np.asarray(img)[..., 3]
        mid = float(((a > 16) & (a < 240)).mean())
        checks[f"alpha_mid_fraction_{scale}x"] = round(mid, 5)

    # halo check: composite on hard backgrounds, look for bright/dark fringes
    img = render_svg(svg, proj.width, proj.height)
    arr = np.asarray(img).astype(float)
    alpha = arr[..., 3:] / 255.0
    for name, bgc in [("white", (255, 255, 255)), ("black", (0, 0, 0))]:
        flat = arr[..., :3] * alpha + np.array(bgc) * (1 - alpha)
        edge_band = (alpha[..., 0] > 0.02) & (alpha[..., 0] < 0.98)
        if name == "white":
            fringe = float((flat[edge_band] > 250).all(axis=-1).mean())
        else:
            fringe = float((flat[edge_band] < 5).all(axis=-1).mean())
        checks[f"halo_free_on_{name}"] = fringe < 0.9  # partial pixels must carry colour

    # model complexity
    nodes = sum(len(s["d"]) for s in proj.shapes)
    checks["model_path_nodes"] = nodes
    checks["shapes"] = len(proj.shapes)

    out = project / "validation-metrics.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    typer.echo(json.dumps(report, indent=2))
    failed = [k for k, v in checks.items() if v is False]
    if failed:
        typer.echo(f"FAILED: {failed}")
        raise typer.Exit(1)


@app.command()
def export(
    project: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Write the standalone SVG, verifying it is raster- and script-free."""
    proj = load_project(project)
    svg = generate_svg(proj)
    import xml.etree.ElementTree as ET

    root = ET.fromstring(svg)
    bad = [
        el.tag
        for el in root.iter()
        if el.tag.split("}")[-1] in ("image", "script", "foreignObject")
    ]
    if bad:
        raise typer.Exit(f"export blocked: contains {bad}")
    output.write_text(svg)
    typer.echo(f"wrote {output} ({len(svg)} bytes)")


if __name__ == "__main__":
    app()
