"""Public torchsky API checks; FITS behavior is tested in torchfits."""

import torch

import torchsky
from torchsky.cli import build_parser


def test_top_level_namespaces_expose_science_modules() -> None:
    assert set(torchsky.__all__) == {
        "WCS",
        "PrecisionWarning",
        "wcs",
        "sphere",
        "spectral",
        "spectrophotometry",
        "redshift",
        "log_lambda",
        "extinction",
        "cosmology",
        "map",
        "scene",
        "field",
        "transforms",
        "encoders",
        "catalogs",
        "coords",
        "quality",
    }
    assert hasattr(torchsky.transforms, "Normalize")
    assert torchsky.wcs.WCS is torchsky.WCS
    assert hasattr(torchsky.sphere, "great_circle_distance")
    assert callable(torchsky.coords.transform)
    assert torchsky.coords.CatalogState is torchsky.wcs.CatalogState
    assert set(torchsky.coords.__all__) == {
        "CatalogState",
        "EarthOrientation",
        "Site",
        "apply_dcr",
        "apply_ephemeris_offset",
        "apply_refraction",
        "apply_stellar_aberration",
        "convert_celestial",
        "dcr_offset",
        "heliocentric_rv_to_lsr",
        "icrs_to_ecliptic_of_date",
        "icrs_to_galactocentric",
        "observed_altaz",
        "propagate_proper_motion",
        "propagate_space_motion",
        "transform",
    }


def test_map_root_keeps_workflow_api_small() -> None:
    assert set(torchsky.map.__all__) == {
        "Observation",
        "IFUObservation",
        "ImageObservation",
        "Target",
        "CubeTarget",
        "ResampleConfig",
        "SkyConfig",
        "OutlierConfig",
        "DQPolicy",
        "DQRejectionLog",
        "QualityProduct",
        "StackConfig",
        "MapResult",
        "CubeResult",
        "MultibandResult",
        "HealpixResult",
        "ApertureUncertainty",
        "ProductSpec",
        "BandpassSpec",
        "ProductResult",
        "UncertaintyProduct",
        "TargetRenderSpec",
        "AdaptiveResampleConfig",
        "plan",
        "plan_cube",
        "plan_ifu_cube_response",
        "plan_product",
        "resample",
        "resample_healpix",
        "blot",
        "detect_outliers",
        "match_sky",
        "coadd",
        "coadd_multiband",
        "coadd_image_plane",
        "coadd_cube",
        "build_ifu_cube",
        "coadd_healpix",
        "stack_product",
        "aperture_uncertainty",
        "adaptive_reprojection_response",
    }
    assert "fit_field_model" not in torchsky.map.__all__
    assert torchsky.field.fit is torchsky.map.field.fit_field_model
    assert torchsky.field.model is torchsky.map.field.make_field_model
    for removed in (
        "Polygon",
        "MultiPolygon",
        "Circle",
        "Cone",
        "Cap",
        "Box",
        "Ellipse",
        "NativeExactSphericalRegion",
        "HealSparseMap",
        "SparseMap",
        "query_cone",
    ):
        assert not hasattr(torchsky.sphere, removed)
    for removed in (
        "max_pixrad",
        "ring_to_nested",
        "nested_to_ring",
        "neighbours",
        "query_cone",
    ):
        assert not hasattr(torchsky.sphere.healpix, removed)
    for removed in (
        "get_values_pix",
        "lookup_With_valid",
        "generate_healpix_map",
        "write",
    ):
        assert not hasattr(torchsky.sphere.SparseHealpixMap, removed)
    assert callable(torchsky.sphere.query_polygon)
    assert callable(torchsky.sphere.sample_map)
    assert callable(torchsky.catalogs.crossmatch)
    assert callable(torchsky.wcs.from_tabular)
    assert callable(torchsky.wcs.sample_gwcs)
    assert not hasattr(torchsky.scene, "coadd")
    assert not hasattr(torchsky.scene, "plan")
    assert callable(torchsky.map.coadd)
    assert callable(torchsky.scene.render)


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
    from torchsky.map.field.core import SineImplicitField, SkyEncoderMLPField

    assert isinstance(model, SineImplicitField)
    assert isinstance(field.preset("fast", domain), SkyEncoderMLPField)
    obs = field.FieldObservation(
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


def test_fits_io_is_owned_by_torchfits() -> None:
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
    for name in (
        "read",
        "write",
        "open",
        "get_header",
        "get_wcs",
        "Header",
        "Card",
        "HDUList",
        "TensorHDU",
        "TableHDU",
    ):
        assert not hasattr(torchsky, name)
    assert callable(torchfits.read)
    assert callable(torchfits.table.read)
