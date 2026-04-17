from __future__ import annotations

from torchsky.map import MapBandpassSpec, MapProductSpec
from torchsky.map.target import TargetRenderSpec


def _simple_header() -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": 6,
        "NAXIS2": 5,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 3.0,
        "CRPIX2": 2.5,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }


def test_target_render_spec_from_header_tracks_shape() -> None:
    spec = TargetRenderSpec.from_header(_simple_header(), value_semantics="flux")
    assert spec.shape == (5, 6)
    assert spec.value_semantics == "flux"


def test_target_render_spec_from_map_product_pulls_band_metadata() -> None:
    product = MapProductSpec(
        target_header=_simple_header(),
        bands=(
            MapBandpassSpec(
                label="f444w",
                spectral_min=42000.0,
                spectral_max=47000.0,
                transmission_center=44400.0,
                transmission_sigma=800.0,
            ),
        ),
        value_semantics="surface_brightness",
    )
    spec = TargetRenderSpec.from_map_product(product)
    assert spec.band_label == "f444w"
    assert spec.transmission_center == 44400.0
    assert spec.shape == (5, 6)


def test_map_product_target_render_specs_preserve_band_order_and_psf() -> None:
    product = MapProductSpec(
        target_header=_simple_header(),
        bands=(
            MapBandpassSpec(label="blue", transmission_center=4500.0),
            MapBandpassSpec(label="red", transmission_center=6500.0),
        ),
        value_semantics="flux",
    )
    specs = product.target_render_specs(psf_specs=("psf-blue", "psf-red"))
    assert tuple(spec.band_label for spec in specs) == ("blue", "red")
    assert tuple(spec.psf_spec for spec in specs) == ("psf-blue", "psf-red")
    assert all(spec.value_semantics == "flux" for spec in specs)
