from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_cold_import_keeps_viz_and_field_surfaces_lazy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH")
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = (
        src_path if not existing_path else os.pathsep.join([src_path, existing_path])
    )

    code = """
import json
import sys

import torchsky

names = (
    "numpy",
    "torch",
    "torchsky.cpp",
    "fastapi",
    "torchsky.viz",
    "torchsky.map",
    "torchsky.map.field",
)
print(json.dumps({name: name in sys.modules for name in names}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    imported = json.loads(result.stdout)
    assert imported == {
        "fastapi": False,
        "numpy": False,
        "torch": False,
        "torchsky.cpp": False,
        "torchsky.map": False,
        "torchsky.map.field": False,
        "torchsky.viz": False,
    }


def test_map_import_keeps_field_surface_lazy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH")
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = (
        src_path if not existing_path else os.pathsep.join([src_path, existing_path])
    )

    code = """
import json
import sys

import torchsky.map

names = (
    "torchsky.map.field",
    "torchsky.map.healpix_spectral",
)
print(json.dumps({name: name in sys.modules for name in names}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    imported = json.loads(result.stdout)
    assert imported == {
        "torchsky.map.field": False,
        "torchsky.map.healpix_spectral": False,
    }
