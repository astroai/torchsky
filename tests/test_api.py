"""Public torchsky API checks; FITS behavior is tested in torchfits."""

import pytest
import torch

import torchsky
from torchsky.cli import build_parser


def test_top_level_namespaces_expose_science_modules() -> None:
    assert set(torchsky.__all__) == {
        "read",
        "write",
        "open",
        "get_header",
        "get_wcs",
        "WCS",
        "Header",
        "Card",
        "HDUList",
        "TensorHDU",
        "TableHDU",
        "io",
        "table",
        "wcs",
        "sphere",
        "spectral",
        "spectrophotometry",
        "map",
        "scene",
        "field",
        "transforms",
        "datasets",
        "cache",
        "interop",
        "encoders",
        "catalogs",
        "quality",
    }
    assert callable(torchsky.io.create_dataloader)
    assert hasattr(torchsky.transforms, "Normalize")
    assert callable(torchsky.cache.clear)
    assert isinstance(torchsky.cache.stats(), dict)
    assert torchsky.wcs.WCS is torchsky.WCS
    assert hasattr(torchsky.sphere, "great_circle_distance")
    with pytest.raises(AttributeError):
        torchsky.coords  # noqa: B018


def test_map_root_keeps_workflow_api_small() -> None:
    assert set(torchsky.map.__all__) == {
        "coadd_image_plane",
        "coadd_image_plane_cube",
        "coadd_image_plane_multiband",
        "stack_map_product",
        "MapProductSpec",
        "TargetRenderSpec",
        "MapBandpassSpec",
        "MapProductResult",
        "MapResult",
        "MultiBandMapResult",
        "SpectralCubeMapResult",
    }
    assert "fit_field_model" not in torchsky.map.__all__
    assert torchsky.field.fit is torchsky.map.field.fit_field_model
    assert torchsky.field.model is torchsky.map.field.make_field_model


def test_field_namespace_uses_short_composable_verbs() -> None:
    field = torchsky.field
    domain = field.Domain(
        center_ra_deg=0.5,
        center_dec_deg=0.5,
        spatial_scale_deg=1.0,
        wavelength_center=450.0,
        wavelength_scale=100.0,
    )
    model = field.model("sine_implicit", domain, hidden_dim=8, hidden_layers=1)
    assert isinstance(model, field.Sine)
    assert field.preset("fast", domain).__class__ is field.SkyEncoderMLP
    obs = field.Observation(
        image=torch.zeros((3, 4)),
        header={
            "NAXIS": 2,
            "NAXIS1": 4,
            "NAXIS2": 3,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": 2.0,
            "CRPIX2": 2.0,
            "CRVAL1": 0.5,
            "CRVAL2": 0.5,
            "CDELT1": -0.01,
            "CDELT2": 0.01,
        },
        spectral_min=400.0,
        spectral_max=500.0,
    )
    rendered = field.render(model, obs, num_spectral_samples=1)
    assert rendered.shape == obs.image.shape

    ra = torch.full((2,), 0.5)
    dec = torch.full((2,), 0.5)
    assert field.sample(
        model, obs, ra_deg=ra, dec_deg=dec, num_spectral_samples=1
    ).shape == (2,)
    assert field.metrics(rendered, obs.image)["weighted_rmse"] >= 0.0
    assert callable(field.quality)
    assert callable(field.residual)


def test_map_cli_exposes_resampling_kernel_option() -> None:
    args = build_parser().parse_args(
        [
            "map",
            "out.fits",
            "a.fits",
            "b.fits",
            "--kernel",
            "mitchell_netravali",
            "--value-semantics",
            "flux",
        ]
    )
    assert args.kernel == "mitchell_netravali"
    assert args.value_semantics == "flux"


def test_fits_compatibility_aliases_delegate_to_torchfits() -> None:
    import torchfits
    import torchsky.cache as sky_cache

    assert torchsky.io.fits.write is torchfits.write
    assert torchsky.io.fits.open is torchfits.open
    assert torchsky.io.fits.get_header is torchfits.get_header
    assert torchsky.io.read is torchfits.read
    assert sky_cache.clear is torchfits.cache.clear
