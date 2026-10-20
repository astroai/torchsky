"""
Smoke tests for torchsky.viz server.

Requires torchsky[viz] installed: fastapi, uvicorn, pillow.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchfits

# Skip entire module if viz extras not available
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")  # httpx is used by FastAPI TestClient

from fastapi.testclient import TestClient  # noqa: E402  # type: ignore


# ---------------------------------------------------------------------------  # type: ignore
# Fixtures
# ---------------------------------------------------------------------------  # type: ignore


@pytest.fixture
def tmp_fits_dir(tmp_path: Path):
    """Create a temp dir with a minimal valid FITS file."""
    fits_path = tmp_path / "test_image.fits"
    _write_minimal_fits(fits_path, shape=(128, 128))
    return tmp_path


@pytest.fixture
def client(tmp_fits_dir: Path):
    from torchsky.viz.server import create_app

    app = create_app(root_dir=str(tmp_fits_dir), cache_mb=32)
    return TestClient(app)


def _write_minimal_fits(path: Path, shape: tuple[int, int] = (64, 64)) -> None:
    """Write a minimal FITS file with random float data and basic WCS."""
    import numpy as np

    h, w = shape
    data = (torch.randn(h, w) * 100 + 500).numpy().astype(np.float32)  # type: ignore

    # Build FITS header (2880-byte blocks)  # type: ignore
    def card(key: str, value) -> bytes:
        if isinstance(value, bool):
            v = "T" if value else "F"
            line = f"{key:<8}= {v:>20}"  # type: ignore
        elif isinstance(value, int):
            line = f"{key:<8}= {value:>20}"  # type: ignore
        elif isinstance(value, float):
            line = f"{key:<8}= {value:>20.10E}"  # type: ignore
        elif isinstance(value, str):
            line = f"{key:<8}= '{value:<8}'"  # type: ignore
        else:
            line = f"{key:<8}  {value}"  # type: ignore
        return line.ljust(80).encode("ascii")

    cards = [
        card("SIMPLE", True),
        card("BITPIX", -32),  # type: ignore
        card("NAXIS", 2),
        card("NAXIS1", w),
        card("NAXIS2", h),
        card("EXTEND", True),
        card("CTYPE1", "RA---TAN"),  # type: ignore
        card("CTYPE2", "DEC--TAN"),  # type: ignore
        card("CRPIX1", w // 2),
        card("CRPIX2", h // 2),
        card("CRVAL1", 150.0),
        card("CRVAL2", 2.0),
        card("CD1_1", -5.0e-5),  # type: ignore
        card("CD1_2", 0.0),
        card("CD2_1", 0.0),
        card("CD2_2", 5.0e-5),  # type: ignore
        b"END" + b" " * 77,
    ]
    header_bytes = b"".join(cards)  # type: ignore
    # Pad to 2880-byte block  # type: ignore
    pad = (2880 - len(header_bytes) % 2880) % 2880  # type: ignore
    header_bytes += b" " * pad  # type: ignore

    # Data in big-endian float32  # type: ignore
    data_bytes = data.byteswap().tobytes()
    pad_data = (2880 - len(data_bytes) % 2880) % 2880  # type: ignore
    data_bytes += b"\x00" * pad_data

    path.write_bytes(header_bytes + data_bytes)  # type: ignore


# ---------------------------------------------------------------------------  # type: ignore
# Tests: /api/fits
# ---------------------------------------------------------------------------  # type: ignore


def test_health(client: TestClient):
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert "version" in j
    assert "root" not in j


def test_health_exposes_root_when_configured(tmp_fits_dir: Path):
    from torchsky.viz.server import create_app

    app = create_app(root_dir=str(tmp_fits_dir), expose_root_in_health=True)
    c = TestClient(app)
    r = c.get("/api/health")
    assert r.status_code == 200
    assert "root" in r.json()


def test_create_app_defaults_to_no_cors_origins(tmp_fits_dir: Path):
    from fastapi.middleware.cors import CORSMiddleware
    from torchsky.viz.server import create_app

    app = create_app(root_dir=str(tmp_fits_dir))
    cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)

    assert cors.kwargs["allow_origins"] == []
    assert cors.kwargs["allow_credentials"] is False


def test_static_viz_html_escapes_metadata_and_tracks_aria_state():
    html = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "torchsky"
        / "viz"
        / "static"
        / "index.html"
    ).read_text()

    assert "htmlEscape(fmt(dr.dr_ratio))" in html
    assert "htmlEscape(fmt(s.min))" in html
    assert "htmlEscape(fmt(wcs.crval1))" in html
    assert 'id="stretch-group" role="group"' in html
    assert 'id="cmap-group" role="group"' in html
    assert "setAttribute('aria-pressed', 'false')" in html
    assert "setAttribute('aria-pressed', 'true')" in html


def test_file_tree(client: TestClient):
    r = client.get("/api/fits/tree")
    assert r.status_code == 200
    items = r.json()["items"]
    assert any(i["name"] == "test_image.fits" for i in items)


def test_fits_info(client: TestClient):
    r = client.get("/api/fits/info?path=test_image.fits&hdu=0")
    assert r.status_code == 200
    j = r.json()
    assert j["shape"][-2:] == [128, 128]  # type: ignore
    assert j["bitpix"] == -32  # type: ignore


def test_fits_histogram(client: TestClient):
    r = client.get("/api/fits/histogram?path=test_image.fits&hdu=0")
    assert r.status_code == 200
    j = r.json()
    assert "bins" in j and "counts" in j
    assert len(j["bins"]) == 256
    assert j["dr"]["recommended_stretch"] in ("linear", "log", "asinh")
    assert "vmin_zscale" in j
    assert "vmax_zscale" in j


def test_fits_header(client: TestClient):
    r = client.get("/api/fits/header?path=test_image.fits&hdu=0")  # type: ignore
    assert r.status_code == 200
    j = r.json()
    assert "NAXIS1" in j["header"]  # type: ignore
    assert int(j["header"]["NAXIS1"]) == 128  # type: ignore


# ---------------------------------------------------------------------------  # type: ignore
# Tests: /api/hips
# ---------------------------------------------------------------------------  # type: ignore


def test_hips_register(client: TestClient):
    r = client.get(
        "/api/hips/register?path=test_image.fits&hdu=0&stretch=zscale&cmap=gray"
    )
    assert r.status_code == 200
    j = r.json()
    assert "id" in j
    assert "max_order" in j
    assert j["shape"] == [128, 128]
    assert j["max_order"] >= 1


def test_hips_properties(client: TestClient):
    sid = client.get("/api/hips/register?path=test_image.fits").json()["id"]
    r = client.get(f"/api/hips/{sid}/properties")
    assert r.status_code == 200
    assert "hips_tile_format" in r.text
    assert "png" in r.text


def test_hips_tile_returns_png(client: TestClient):
    sid = client.get("/api/hips/register?path=test_image.fits").json()["id"]
    # Request a low-order tile that covers most of the image  # type: ignore
    r = client.get(f"/api/hips/{sid}/Norder3/Dir0/Npix0.png")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"  # type: ignore
    # Verify PNG magic bytes
    assert r.content[:4] == b"\x89PNG"


def test_hips_tile_wrong_dir_rejected(client: TestClient):
    """HiPS Dir must equal Npix // 10000 (IVOA grouping)."""
    sid = client.get("/api/hips/register?path=test_image.fits").json()["id"]
    r = client.get(f"/api/hips/{sid}/Norder3/Dir9/Npix0.png")
    assert r.status_code == 400
    assert "Dir mismatch" in r.json()["detail"]


def test_hips_preview(client: TestClient):
    sid = client.get("/api/hips/register?path=test_image.fits").json()["id"]
    r = client.get(f"/api/hips/{sid}/preview.png")
    assert r.status_code == 200
    assert r.content[:4] == b"\x89PNG"


def test_hips_list(client: TestClient):
    client.get("/api/hips/register?path=test_image.fits")
    r = client.get("/api/hips/list")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
    assert len(r.json()) >= 1


# ---------------------------------------------------------------------------  # type: ignore
# Tests: path security
# ---------------------------------------------------------------------------  # type: ignore


def test_path_traversal_blocked(client: TestClient):
    r = client.get("/api/fits/info?path=../../../etc/passwd")
    assert r.status_code == 403


def test_path_prefix_sibling_directory_blocked(tmp_path: Path):
    """Regression: string prefix checks wrongly allowed ../<root_name>_extra/... paths."""  # type: ignore
    from torchsky.viz.server import create_app

    vault = tmp_path / "vault"
    vault.mkdir()
    sibling = tmp_path / "vault_extra"
    sibling.mkdir()
    _write_minimal_fits(vault / "inside.fits")
    _write_minimal_fits(sibling / "outside.fits")

    app = create_app(root_dir=str(vault), cache_mb=32)
    c = TestClient(app)
    r = c.get("/api/fits/info?path=../vault_extra/outside.fits&hdu=0")
    assert r.status_code == 403


# ---------------------------------------------------------------------------  # type: ignore
# Tests: /api/validate
# ---------------------------------------------------------------------------  # type: ignore


def test_validate_register_and_panel(client: TestClient, tmp_fits_dir: Path):
    # Write model FITS
    _write_minimal_fits(tmp_fits_dir / "model.fits", shape=(128, 128))

    body = {
        "truth_path": "test_image.fits",
        "model_path": "model.fits",
        "truth_hdu": 0,
        "model_hdu": 0,
    }
    r = client.post(
        "/api/validate/register",
        json=body,
        headers={"Content-Type": "application/json"},
    )  # type: ignore
    assert r.status_code == 200
    pid = r.json()["id"]
    assert len(pid) == 64

    # Panel PNGs
    r2 = client.get(f"/api/validate/{pid}/panel_pngs")
    assert r2.status_code == 200
    pngs = r2.json()
    for k in ("truth", "model", "residual", "pull"):
        assert k in pngs
        assert pngs[k].startswith("data:image/png;base64,")

    # Pull histogram
    r3 = client.get(f"/api/validate/{pid}/pull_histogram")
    assert r3.status_code == 200
    hist = r3.json()
    assert "bins" in hist and "counts" in hist
    assert "fit" in hist


# ---------------------------------------------------------------------------  # type: ignore
# Tests: /api/moc (requires a MOC file — create a minimal one)
# ---------------------------------------------------------------------------  # type: ignore


def test_moc_stats_synthetic(client: TestClient, tmp_fits_dir: Path):
    """Test MOC endpoint with a synthetic MOC created via torchsky."""
    from torchsky.sphere.moc import MOC
    from torchsky.sphere.healpix import query_disc
    from torchsky.sphere.core import lonlat_to_unit_xyz
    import math

    vec = lonlat_to_unit_xyz(150.0, 2.0)
    pix = query_disc(8, vec, math.radians(5.0), nest=True)  # nside=8, 5° radius
    uniq = (4 * 8 * 8 + pix).to(torch.int64)
    moc = MOC(uniq)
    moc_path = tmp_fits_dir / "test.moc"
    moc.write_fits(str(moc_path), overwrite=True)

    r = client.get("/api/moc/stats?path=test.moc")
    assert r.status_code == 200
    j = r.json()
    assert j["max_order"] >= 0
    assert j["area_sq_deg"] > 0


def test_catalog_filter_moc_column_discovery(tmp_path: Path):
    """filter_moc must resolve RAJ2000/DEJ2000 like /load (no KeyError → 500)."""
    import math

    from torchsky.sphere.core import lonlat_to_unit_xyz
    from torchsky.sphere.moc import MOC
    from torchsky.viz.server import create_app
    from torchsky.sphere.healpix import query_disc

    root = tmp_path
    torchfits.write(
        str(root / "cat.fits"),
        {
            "RAJ2000": torch.tensor([150.0, 150.1], dtype=torch.float64),
            "DEJ2000": torch.tensor([2.0, 2.1], dtype=torch.float64),
        },
        header={"EXTNAME": "CAT"},  # type: ignore
        overwrite=True,
    )
    vec = lonlat_to_unit_xyz(150.05, 2.05)
    pix = query_disc(8, vec, math.radians(10.0), nest=True)
    uniq = (4 * 8 * 8 + pix).to(torch.int64)
    MOC(uniq).write_fits(str(root / "foot.moc"), overwrite=True)

    app = create_app(root_dir=str(root), cache_mb=16)
    c = TestClient(app)
    hdu_ok = None
    for hdu in (0, 1):
        r_load = c.get(f"/api/catalog/load?path=cat.fits&hdu={hdu}")  # type: ignore
        if r_load.status_code == 200:  # type: ignore
            hdu_ok = hdu
            break
    assert hdu_ok is not None, r_load.text  # type: ignore
    r_f = c.get(
        f"/api/catalog/filter_moc?catalog_path=cat.fits&moc_path=foot.moc&hdu={hdu_ok}",
    )
    assert r_f.status_code == 200, r_f.text
    body = r_f.json()
    assert body["n_in_moc"] >= 1


def test_catalog_load_caps_rows_exactly(tmp_path: Path):
    """`max_rows` is a hard cap: floor-division decimation used to overshoot it."""
    from torchsky.viz.server import create_app

    n = 1000
    torchfits.write(
        str(tmp_path / "cat.fits"),
        {
            "RA": torch.linspace(0.0, 10.0, n, dtype=torch.float64),
            "DEC": torch.zeros(n, dtype=torch.float64),
        },
        header={"EXTNAME": "CAT"},  # type: ignore
        overwrite=True,
    )
    c = TestClient(create_app(root_dir=str(tmp_path), cache_mb=8))
    hdu = 1 if c.get("/api/catalog/load?path=cat.fits&hdu=1").status_code == 200 else 0
    r = c.get(f"/api/catalog/load?path=cat.fits&hdu={hdu}&max_rows=300")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_total"] == n
    assert body["n_returned"] <= 300
    assert len(body["ra"]) == body["n_returned"]


def _write_header_only_image(path: Path, shape: tuple[int, int]) -> None:
    """Write a FITS header advertising ``shape`` with no pixel data at all."""
    h, w = shape
    cards = [
        f"{'SIMPLE':<8}= {'T':>20}",
        f"{'BITPIX':<8}= {-32:>20}",
        f"{'NAXIS':<8}= {2:>20}",
        f"{'NAXIS1':<8}= {w:>20}",
        f"{'NAXIS2':<8}= {h:>20}",
    ]
    header = b"".join(line.ljust(80).encode() for line in cards)
    header += b"END".ljust(80)
    header += b" " * ((2880 - len(header) % 2880) % 2880)
    path.write_bytes(header)


def test_hips_register_rejects_oversize_image_before_loading(tmp_path: Path):
    """The pixel budget must be enforced from header metadata, not after decode.

    The fixture has no pixel data, so decoding it would fail with 422; a 413
    proves the guard ran before the read.
    """
    from torchsky.viz.server import create_app

    _write_header_only_image(tmp_path / "huge.fits", (100_000, 100_000))
    c = TestClient(create_app(root_dir=str(tmp_path), max_image_pixels=1024))
    r = c.get("/api/hips/register?path=huge.fits&hdu=0")
    assert r.status_code == 413, r.text
    assert "max allowed" in r.json()["detail"]


def test_validate_register_rejects_oversize_image(tmp_path: Path):
    from torchsky.viz.server import create_app

    _write_minimal_fits(tmp_path / "a.fits", shape=(64, 64))
    _write_minimal_fits(tmp_path / "b.fits", shape=(64, 64))
    c = TestClient(create_app(root_dir=str(tmp_path), max_image_pixels=1024))
    r = c.post(
        "/api/validate/register",
        json={"truth_path": "a.fits", "model_path": "b.fits"},
    )
    assert r.status_code == 413


def test_viz_router_modules_import() -> None:
    import importlib

    for name in ("fits", "hips", "moc", "validate", "catalog"):
        importlib.import_module(f"torchsky.viz.routers.{name}")


def test_moc_geojson_and_fits(client: TestClient, tmp_fits_dir: Path):
    """Exercise the MOC GeoJSON/FITS endpoints (not used by the bundled UI)."""
    import math

    from torchsky.sphere.core import lonlat_to_unit_xyz
    from torchsky.sphere.healpix import query_disc
    from torchsky.sphere.moc import MOC

    vec = lonlat_to_unit_xyz(150.0, 2.0)
    pix = query_disc(8, vec, math.radians(5.0), nest=True)
    uniq = (4 * 8 * 8 + pix).to(torch.int64)
    MOC(uniq).write_fits(str(tmp_fits_dir / "overlay.moc"), overwrite=True)

    r = client.get("/api/moc/geojson?path=overlay.moc")
    assert r.status_code == 200, r.text
    assert r.json()["type"] == "FeatureCollection"

    r = client.get("/api/moc/fits?path=overlay.moc")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/fits")


def test_validate_chi2_requires_sigma(client: TestClient, tmp_fits_dir: Path):
    """The chi2 endpoints must reject a pair registered without a sigma image."""
    _write_minimal_fits(tmp_fits_dir / "chi2_model.fits", shape=(128, 128))
    body = {
        "truth_path": "test_image.fits",
        "model_path": "chi2_model.fits",
        "truth_hdu": 0,
        "model_hdu": 0,
    }
    r = client.post("/api/validate/register", json=body)
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    assert client.get(f"/api/validate/{pid}/chi2_stats").status_code == 422
    assert client.get(f"/api/validate/{pid}/chi2_map").status_code == 422
