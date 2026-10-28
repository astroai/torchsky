import ast
from pathlib import Path


def test_torchsky_src_does_not_import_astropy_io_fits():
    repo = Path(__file__).parents[1]
    offenders = []
    roots = [repo / "src" / "torchsky", repo / "tests", repo / "scripts", repo / "examples"]
    for root in roots:
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    forbidden = any(
                        alias.name == "astropy.io.fits"
                        or alias.name.startswith("astropy.io.fits.")
                        for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom):
                    forbidden = (node.module or "").startswith("astropy.io.fits") or (
                        node.module == "astropy.io"
                        and any(alias.name == "fits" for alias in node.names)
                    )
                else:
                    forbidden = False
                if forbidden:
                    offenders.append(f"{path}:{node.lineno}: Astropy FITS import")
    assert not offenders, "\n".join(offenders)


def test_torchsky_does_not_import_python_fitsio():
    repo = Path(__file__).parents[1]
    roots = [
        repo / "src" / "torchsky",
        repo / "tests",
        repo / "scripts",
        repo / "examples",
    ]
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            for line_number, line in enumerate(path.read_text().splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if stripped.startswith("import fitsio") or stripped.startswith(
                    "from fitsio"
                ):
                    offenders.append(f"{path}:{line_number}: {stripped}")
    assert not offenders, "\n".join(offenders)


def test_torchsky_does_not_import_torchfits_private_modules():
    source_root = Path(__file__).parents[1] / "src" / "torchsky"
    offenders = []
    for path in source_root.rglob("*.py"):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if "torchfits._" in line:
                offenders.append(f"{path}:{line_number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_fits_surfaces_are_not_owned_by_torchsky():
    import importlib.util

    import torchfits

    for name in (
        "io",
        "table",
        "hdu",
        "cache",
        "datasets",
        "dataloader",
        "header_parser",
        "core",
    ):
        assert importlib.util.find_spec(f"torchsky.{name}") is None
    for name in ("read", "write", "open", "read_header", "read_tensor"):
        assert hasattr(torchfits, name)
    assert hasattr(torchfits.table, "read")
    assert hasattr(torchfits.hdu, "HDUList")
