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

    assert cors.options["allow_origins"] == []
    assert cors.options["allow_credentials"] is False


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
