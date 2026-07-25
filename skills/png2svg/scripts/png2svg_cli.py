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
"""Run the png2svg CLI straight out of the skill, with no install step.

    uv run --no-project scripts/png2svg_cli.py init logo.png --project work/x

uv reads the dependency block above and builds a cached ephemeral
environment on first run; the png2svg package itself is the sibling
directory, so nothing is fetched from a registry for it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from png2svg.cli import app  # noqa: E402

if __name__ == "__main__":
    app()
