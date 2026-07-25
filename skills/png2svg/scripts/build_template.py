#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.5.1",
#     "pillow>=12.3.0",
#     "resvg-py>=0.3.3",
#     "scipy>=1.18.0",
#     "typer>=0.26.8",
# ]
# ///
"""TEMPLATE: assemble project.json from measurements.

Copy this next to your work, edit the marked sections, and run it with
`uv run --no-project build_<name>.py`. It reads
`<project>/analysis/measurements.json` and rewrites `<project>/project.json`.

Keep it a pure function of the measurements: no hand-typed coordinates. When
a number needs to change you change the measurement or the constraint, rerun,
and the whole model moves consistently.

See references/model.md for the full schema.
"""

import json
import os
import sys
from pathlib import Path


# --- locate the skill's bundled png2svg package -----------------------------
def _find_skill() -> Path:
    """Locate the skill directory, wherever this file was copied to."""
    here = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    candidates = []
    if env := os.environ.get("PNG2SVG_SKILL"):
        candidates.append(Path(env).expanduser())
    candidates += [here.parent.parent, here.parent / "png2svg"]
    for base in [cwd, *cwd.parents, Path.home()]:
        candidates += [base / ".claude" / "skills" / "png2svg",
                       base / ".agents" / "skills" / "png2svg",
                       base / "png2svg"]
    for cand in candidates:
        # verify: a directory named png2svg is not necessarily the package
        if (cand / "scripts" / "png2svg" / "__init__.py").is_file():
            return cand
    raise SystemExit(
        "could not find the png2svg skill directory.\n"
        "Set PNG2SVG_SKILL=/path/to/skills/png2svg and rerun."
    )


sys.path.insert(0, str(_find_skill() / "scripts"))

from png2svg.model import load_project, save_project  # noqa: E402

# ==== EDIT: project ========================================================
PROJECT = Path("work/name")
M = json.loads((PROJECT / "analysis" / "measurements.json").read_text())
V = M["vertices"]


def subpath(names, close=True):
    """Straight-sided subpath through named vertices."""
    segs = [["M" if k == 0 else "L", *V[n]] for k, n in enumerate(names)]
    if close:
        segs.append(["Z"])
    return segs


# ==== EDIT: shapes =========================================================
# Order matters: shapes paint in list order, so later shapes sit on top.
# A counter (hole) is a second subpath in the SAME shape, wound the OPPOSITE
# way round — there is no fill-rule, so that is what cuts the hole.
#
# Fill types: solid / linear / radial / conic. Any fill may add
# "rect": [x, y, w, h] to bound its region. Conic angles are degrees with
# 0 = +x and positive clockwise on screen.

shapes = [
    {
        "id": "main",
        "type": "path",
        "d": subpath(["A", "B", "C"]),
        "fills": [{"type": "solid", "color": "#000000"}],
    },
]

# ==== EDIT: notes ==========================================================
# Record the decisions a future reader cannot recover from the numbers: what
# overlaps what, which relationships you enforced, what you deliberately
# discarded (watermarks, artefacts).
notes = [
    f"polygon deviation max {M.get('polygon_deviation_max')}px",
]

proj = load_project(PROJECT)
proj.shapes = shapes
proj.notes = notes
save_project(PROJECT, proj)
print("model written:", sum(len(s["d"]) for s in proj.shapes), "nodes")
