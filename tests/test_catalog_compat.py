import numpy as np
import pytest
import torch
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from pathlib import Path
import tempfile

pytest.importorskip("scipy")

import torchsky
from torchsky.catalogs import crossmatch_sky


def _has_native_table_io() -> bool:
    try:
        import torchsky.cpp  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def _fits_table_column_to_float64_tensor(table, name: str) -> torch.Tensor:
    """``torchsky.table.read`` returns PyArrow; convert a numeric column to torch."""
    col = table.column(name)
    if hasattr(col, "combine_chunks"):
        col = col.combine_chunks()
    return torch.tensor(
        np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64).copy(),
        dtype=torch.float64,
    )


def _astropy_crossmatch(
    left_ra: torch.Tensor,
    left_dec: torch.Tensor,
    right_ra: torch.Tensor,
    right_dec: torch.Tensor,
    *,
    radius_arcsec: float,
) -> tuple[list[int], list[int], np.ndarray]:
    if right_ra.numel() == 0 or left_ra.numel() == 0:
        return [], [], np.empty(0, dtype=np.float64)

    left = SkyCoord(
        ra=left_ra.cpu().numpy() * u.deg, dec=left_dec.cpu().numpy() * u.deg
    )
    right = SkyCoord(
        ra=right_ra.cpu().numpy() * u.deg,
        dec=right_dec.cpu().numpy() * u.deg,
    )
    right_idx, sep2d, _ = left.match_to_catalog_sky(right)
    keep = sep2d.arcsec <= radius_arcsec
    left_idx = np.nonzero(keep)[0].tolist()
    matched_right = right_idx[keep].tolist()
    sep_arcsec = np.asarray(sep2d.arcsec[keep], dtype=np.float64)
    return left_idx, matched_right, sep_arcsec


def _catalog_fixture_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "upstream_astropy_catalogs"


def _load_real_catalog(
    filename: str,
    *,
    ra_col: str,
    dec_col: str,
) -> tuple[Table, torch.Tensor, torch.Tensor]:
    table = Table.read(_catalog_fixture_dir() / filename, format="votable")
    ra = np.asarray(table[ra_col], dtype=np.float64)
    dec = np.asarray(table[dec_col], dtype=np.float64)
    return table, torch.from_numpy(ra), torch.from_numpy(dec)


def _real_catalogs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, left_ra, left_dec = _load_real_catalog(
        "irsa-nph-m31.xml",
        ra_col="ra",
        dec_col="dec",
    )
    _, vizier_ra, vizier_dec = _load_real_catalog(
        "vizier_b2_votable.xml",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
    )
    _, gaia_ra, gaia_dec = _load_real_catalog(
        "binary2_masked_strings.xml",
        ra_col="ra",
        dec_col="dec",
    )

    perm = torch.tensor([6, 1, 11, 3, 14, 9, 0, 7, 12, 5, 15, 8, 2, 16, 4, 10, 13, 17])
    right_ra = torch.cat((left_ra[perm], vizier_ra, gaia_ra))
    right_dec = torch.cat((left_dec[perm], vizier_dec, gaia_dec))
    return left_ra, left_dec, right_ra, right_dec


def _real_catalog_tables() -> tuple[Table, Table]:
    irsa_table, irsa_ra, irsa_dec = _load_real_catalog(
        "irsa-nph-m31.xml",
        ra_col="ra",
        dec_col="dec",
    )
    vizier_table, vizier_ra, vizier_dec = _load_real_catalog(
        "vizier_b2_votable.xml",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
    )
    gaia_table, gaia_ra, gaia_dec = _load_real_catalog(
        "binary2_masked_strings.xml",
        ra_col="ra",
        dec_col="dec",
    )

    perm = np.array([6, 1, 11, 3, 14, 9, 0, 7, 12, 5, 15, 8, 2, 16, 4, 10, 13, 17])
    left_table = Table(
        {
            "RA": irsa_ra.numpy(),
            "DEC": irsa_dec.numpy(),
            "SOURCE": np.full(len(irsa_table), "irsa_2mass_psc", dtype=object),
            "DESIGNATION": np.asarray(irsa_table["designation"]).astype(str),
            "J_MAG": np.asarray(irsa_table["j_m"], dtype=np.float64),
        }
    )
    right_table = Table(
        {
            "RA": np.concatenate(
                (
                    irsa_ra.numpy()[perm],
                    vizier_ra.numpy(),
                    gaia_ra.numpy(),
                )
            ),
            "DEC": np.concatenate(
                (
                    irsa_dec.numpy()[perm],
                    vizier_dec.numpy(),
                    gaia_dec.numpy(),
                )
            ),
            "SOURCE": np.concatenate(
                (
                    np.full(len(perm), "irsa_2mass_psc", dtype=object),
                    np.full(len(vizier_table), "vizier_mash", dtype=object),
                    np.full(len(gaia_table), "gaia_dr2", dtype=object),
                )
            ),
            "LABEL": np.concatenate(
                (
                    np.asarray(irsa_table["designation"]).astype(str)[perm],
                    np.asarray(vizier_table["PNG"]).astype(str),
                    np.asarray(gaia_table["source_id"]).astype(str),
                )
            ),
        }
    )
    return left_table, right_table


def test_crossmatch_sky_matches_astropy_skycoord_on_random_catalogs() -> None:
    rng = torch.Generator().manual_seed(321)
    n_left = 64
    n_right = 96

    left_ra = torch.rand(n_left, generator=rng, dtype=torch.float64) * 360.0
    left_dec = torch.rad2deg(
        torch.asin((torch.rand(n_left, generator=rng, dtype=torch.float64) * 2.0) - 1.0)
    )
    right_ra = torch.rand(n_right, generator=rng, dtype=torch.float64) * 360.0
    right_dec = torch.rad2deg(
        torch.asin(
            (torch.rand(n_right, generator=rng, dtype=torch.float64) * 2.0) - 1.0
        )
    )

    right_ra[:16] = (
        left_ra[:16]
        + (torch.rand(16, generator=rng, dtype=torch.float64) - 0.5) * 0.00015
    )
    right_dec[:16] = (
        left_dec[:16]
        + (torch.rand(16, generator=rng, dtype=torch.float64) - 0.5) * 0.00015
    )

    result = crossmatch_sky(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=2.0,
        left_chunk_size=13,
        right_chunk_size=17,
    )
    exp_left, exp_right, exp_sep = _astropy_crossmatch(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=2.0,
    )

    assert result.left_index.tolist() == exp_left
    assert result.right_index.tolist() == exp_right
    np.testing.assert_allclose(
        result.separation_arcsec.cpu().numpy(),
        exp_sep,
        atol=1e-6,
        rtol=0.0,
    )


def test_crossmatch_sky_matches_astropy_on_wrap_and_pole_cases() -> None:
    left_ra = torch.tensor([359.9998, 5.0, 220.0], dtype=torch.float64)
    left_dec = torch.tensor([0.0, 89.9990, 89.9985], dtype=torch.float64)
    right_ra = torch.tensor([0.0001, 185.0, 40.0, 10.0], dtype=torch.float64)
    right_dec = torch.tensor([0.0, 89.9991, 88.5, 89.9987], dtype=torch.float64)

    result = crossmatch_sky(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=12.0,
        left_chunk_size=2,
        right_chunk_size=2,
    )
    exp_left, exp_right, exp_sep = _astropy_crossmatch(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=12.0,
    )

    assert result.left_index.tolist() == exp_left
    assert result.right_index.tolist() == exp_right
    np.testing.assert_allclose(
        result.separation_arcsec.cpu().numpy(),
        exp_sep,
        atol=1e-6,
        rtol=0.0,
    )


def test_crossmatch_sky_matches_astropy_on_real_votable_catalogs() -> None:
    left_ra, left_dec, right_ra, right_dec = _real_catalogs()

    result = crossmatch_sky(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=0.1,
        left_chunk_size=7,
        right_chunk_size=11,
    )
    exp_left, exp_right, exp_sep = _astropy_crossmatch(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=0.1,
    )

    assert result.left_index.tolist() == exp_left
    assert result.right_index.tolist() == exp_right
    np.testing.assert_allclose(
        result.separation_arcsec.cpu().numpy(),
        exp_sep,
        atol=1e-8,
        rtol=0.0,
    )


@pytest.mark.skipif(
    not _has_native_table_io(),
    reason="torchsky native FITS table extension unavailable",
)
def test_crossmatch_sky_end_to_end_matches_astropy_after_fits_read_on_real_catalogs() -> (
    None
):
    left_table, right_table = _real_catalog_tables()
    # FITS table write only needs float RA/DEC columns; extra string/object columns
    # can trip Astropy FITS verification on newer releases (e.g. varlen descriptors).
    left_roundtrip = left_table[["RA", "DEC"]]
    right_roundtrip = right_table[["RA", "DEC"]]

    with tempfile.TemporaryDirectory() as tmpdir:
        left_path = Path(tmpdir) / "left_catalog.fits"
        right_path = Path(tmpdir) / "right_catalog.fits"
        left_roundtrip.write(left_path, format="fits", overwrite=True)
        right_roundtrip.write(right_path, format="fits", overwrite=True)

        left_loaded = torchsky.table.read(
            str(left_path),
            columns=["RA", "DEC"],
        )
        right_loaded = torchsky.table.read(
            str(right_path),
            columns=["RA", "DEC"],
        )

        left_ra = _fits_table_column_to_float64_tensor(left_loaded, "RA")
        left_dec = _fits_table_column_to_float64_tensor(left_loaded, "DEC")
        right_ra = _fits_table_column_to_float64_tensor(right_loaded, "RA")
        right_dec = _fits_table_column_to_float64_tensor(right_loaded, "DEC")

        result = crossmatch_sky(
            left_ra,
            left_dec,
            right_ra,
            right_dec,
            radius_arcsec=2.0,
            left_chunk_size=32,
            right_chunk_size=64,
        )
        exp_left, exp_right, exp_sep = _astropy_crossmatch(
            left_ra,
            left_dec,
            right_ra,
            right_dec,
            radius_arcsec=2.0,
        )

        assert result.left_index.tolist() == exp_left
        assert result.right_index.tolist() == exp_right
        np.testing.assert_allclose(
            result.separation_arcsec.cpu().numpy(),
            exp_sep,
            atol=1e-6,
            rtol=0.0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_crossmatch_sky_matches_cpu_results_on_cuda() -> None:
    left_ra, left_dec, right_ra, right_dec = _real_catalogs()

    cpu_result = crossmatch_sky(
        left_ra,
        left_dec,
        right_ra,
        right_dec,
        radius_arcsec=2.0,
        left_chunk_size=24,
        right_chunk_size=48,
    )
    cuda_result = crossmatch_sky(
        left_ra.cuda(),
        left_dec.cuda(),
        right_ra.cuda(),
        right_dec.cuda(),
        radius_arcsec=2.0,
        left_chunk_size=24,
        right_chunk_size=48,
    )

    assert cuda_result.left_index.device.type == "cuda"
    assert cuda_result.right_index.device.type == "cuda"
    assert cuda_result.separation_arcsec.device.type == "cuda"
    assert cuda_result.left_index.cpu().tolist() == cpu_result.left_index.tolist()
    assert cuda_result.right_index.cpu().tolist() == cpu_result.right_index.tolist()
    torch.testing.assert_close(
        cuda_result.separation_arcsec.cpu(),
        cpu_result.separation_arcsec.cpu(),
        atol=1e-6,
        rtol=0.0,
    )


def test_crossmatch_sky_per_row_radius() -> None:
    left_ra = torch.tensor([10.0, 10.0], dtype=torch.float64)
    left_dec = torch.tensor([0.0, 0.0], dtype=torch.float64)
    right_ra = torch.tensor([10.0 + 0.5 / 3600.0, 10.0 + 2.0 / 3600.0], dtype=torch.float64)
    right_dec = torch.tensor([0.0, 0.0], dtype=torch.float64)
    radii = torch.tensor([1.0, 3.0], dtype=torch.float64)
    all_hits = crossmatch_sky(
        left_ra, left_dec, right_ra, right_dec, radius_arcsec=radii, find="all"
    )
    pairs = list(zip(all_hits.left_index.tolist(), all_hits.right_index.tolist()))
    assert (0, 0) in pairs
    assert (0, 1) not in pairs
    assert (1, 0) in pairs
    assert (1, 1) in pairs
