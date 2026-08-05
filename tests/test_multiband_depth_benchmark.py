from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import pytest
import torch
import torchsky
from astropy.io import fits

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from bench_multiband_depth import (  # noqa: E402
    _classical_smoke_artifact,
    _foam_grid_side_for_size,
    _gate0_artifact,
    _legendre_basis,
    _target_response_gate,
    _updec_classical_detection_artifact,
    _updec_classical_validation_artifact,
    _updec_control_union_artifact,
    _updec_effective_psf_artifact,
    calibrate_matched_filter_whitening,
    calibrate_purity_threshold,
    coadd_synthetic_exposures,
    common_support_mask,
    combine_whitened_chi_responses,
    detection_statistic,
    exclude_catalogue_neighborhoods,
    find_detections,
    flux_color_metrics,
    forced_linear_flux,
    gaussian_nll,
    gaussian_psf_contract,
    generate_truth_catalog,
    inject_gaussian_sources_mjysr,
    independent_union_statistic,
    inject_common_world_exposures,
    make_seed_manifest,
    make_updec_common_world_manifest,
    main as multiband_main,
    maximum_cardinality_angular_match,
    maximum_cardinality_match,
    measure_gaussian_source_flux_jy,
    render_truth,
    score_detections,
    simple_tan_header,
    simulate_dithered_exposures,
    split_identifiers,
    stack_effective_psf,
    updec_patt_split,
    weighted_chi_detection_statistic,
    weighted_matched_filter_snr,
)


def test_seed_catalog_render_and_artifact_are_deterministic() -> None:
    first = _gate0_artifact(seed=101, size=32, bands=6, sources=10)
    second = _gate0_artifact(seed=101, size=32, bands=6, sources=10)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert len(set(first["seeds"].values())) == 4
    split = first["split"]
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])
    assert first["spectral_samples"] == 129
    assert all(
        value == 0.0
        for values in first["noiseless_forced_photometry"].values()
        for value in values
    )


def test_truth_catalog_has_required_strata_and_conserves_flux() -> None:
    catalog = generate_truth_catalog(
        height=48, width=48, bands=6, source_count=10, seed=81
    )
    assert {
        "continuum_point",
        "compact_galaxy",
        "emission_line",
        "blue_dropout",
        "red_dropout",
    }.issubset(catalog.kind)
    cube = render_truth(catalog, height=48, width=48, spatial_oversample=4)
    torch.testing.assert_close(
        cube.sum(dim=(-2, -1)),
        catalog.band_flux.sum(dim=0),
        rtol=2e-12,
        atol=2e-12,
    )
    recomputed_flux = torch.trapezoid(
        catalog.spectral_flux_density[:, None] * catalog.band_response[None],
        catalog.spectral_coordinate,
        dim=-1,
    )
    torch.testing.assert_close(catalog.band_flux, recomputed_flux)
    assert catalog.spectral_coordinate.numel() == 129
    assert 2.0 < torch.linalg.vector_norm(catalog.xy[-1] - catalog.xy[-2]) < 4.0


def test_detector_mjysr_injection_conserves_jy_and_exposes_psf_mismatch() -> None:
    header = simple_tan_header(32, 32, pixel_scale_arcsec=0.04)
    header["BUNIT"] = "MJy/sr"
    wcs = torchsky.WCS(header)
    ra, dec = wcs.pixel_to_world(
        torch.tensor([15.35], dtype=torch.float64),
        torch.tensor([16.2], dtype=torch.float64),
    )
    world = torch.stack((ra, dec), dim=-1)
    flux = torch.tensor([3.0e-8], dtype=torch.float64)
    contract = gaussian_psf_contract("F200W")
    result = inject_gaussian_sources_mjysr(
        torch.zeros((32, 32), dtype=torch.float64),
        header=header,
        world_radec_deg=world,
        flux_jy=flux,
        psf_fwhm_arcsec=contract.truth_fwhm_arcsec,
        spatial_oversample=contract.spatial_oversample,
    )
    assert result.accepted.tolist() == [True]
    torch.testing.assert_close(result.rendered_flux_jy, flux, rtol=2e-12, atol=0.0)
    exact = measure_gaussian_source_flux_jy(
        result.delta,
        header=header,
        world_radec_deg=world[0],
        psf_fwhm_arcsec=contract.truth_fwhm_arcsec,
        spatial_oversample=contract.spatial_oversample,
    )
    perturbed = measure_gaussian_source_flux_jy(
        result.delta,
        header=header,
        world_radec_deg=world[0],
        psf_fwhm_arcsec=contract.fitted_fwhm_arcsec,
        spatial_oversample=contract.spatial_oversample,
    )
    assert exact == pytest.approx(float(flux[0]), rel=2e-12)
    assert abs(perturbed / float(flux[0]) - 1.0) > abs(
        exact / float(flux[0]) - 1.0
    )


def test_common_world_injection_conserves_flux_and_blank_is_unchanged(
    tmp_path: Path,
) -> None:
    paths = []
    headers = [
        simple_tan_header(32, 32, pixel_scale_arcsec=0.04),
        simple_tan_header(
            32,
            32,
            pixel_scale_arcsec=0.04,
            dither_xy_px=(0.35, -0.2),
        ),
    ]
    for index, header in enumerate(headers):
        path = tmp_path / f"exposure-{index}_cal.fits"
        science = fits.ImageHDU(
            data=torch.zeros((32, 32), dtype=torch.float32).numpy(),
            name="SCI",
        )
        for key, value in header.items():
            if key not in {"NAXIS", "NAXIS1", "NAXIS2"}:
                science.header[key] = value
        science.header["BUNIT"] = "MJy/sr"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                science,
                fits.ImageHDU(
                    data=torch.ones((32, 32), dtype=torch.float32).numpy(),
                    name="ERR",
                ),
                fits.ImageHDU(
                    data=torch.zeros((32, 32), dtype=torch.int16).numpy(),
                    name="DQ",
                ),
            ]
        ).writeto(path)
        paths.append(path)

    ra, dec = torchsky.WCS(headers[0]).pixel_to_world(
        torch.tensor([15.25], dtype=torch.float64),
        torch.tensor([16.1], dtype=torch.float64),
    )
    world = torch.stack((ra, dec), dim=-1)
    flux = torch.tensor([3.0e-8], dtype=torch.float64)
    overrides, audit = inject_common_world_exposures(
        paths,
        band="F200W",
        world_radec_deg=world,
        flux_jy=flux,
    )
    assert set(overrides) == set(paths)
    assert all(row["accepted"] == [True] for row in audit)
    assert all(row["rendered_flux_jy"][0] == pytest.approx(float(flux[0])) for row in audit)
    assert audit[0]["source_xy"] != audit[1]["source_xy"]

    blank, blank_audit = inject_common_world_exposures(
        paths,
        band="F200W",
        world_radec_deg=torch.empty((0, 2), dtype=torch.float64),
        flux_jy=torch.empty((0,), dtype=torch.float64),
    )
    assert all(torch.count_nonzero(torch.from_numpy(image)) == 0 for image in blank.values())
    assert all(row["accepted"] == [] and row["input_unchanged"] for row in blank_audit)


def test_common_world_manifest_is_deterministic_and_quadrant_balanced() -> None:
    header = simple_tan_header(128, 128, pixel_scale_arcsec=0.04)
    first = make_updec_common_world_manifest(
        target_header=header,
        source_count=28,
        seed=44,
        margin_px=8.0,
    )
    second = make_updec_common_world_manifest(
        target_header=header,
        source_count=28,
        seed=44,
        margin_px=8.0,
    )
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert [first["quadrant"].count(value) for value in range(4)] == [7, 7, 7, 7]
    assert len(set(first["reference_ab_magnitude"])) == 7
    assert set(first["band_flux_jy"]) == {
        "F090W",
        "F150W",
        "F200W",
        "F277W",
        "F356W",
        "F444W",
    }
    assert all(
        len(values) == 28 for values in first["band_flux_jy"].values()
    )


def test_common_support_mask_intersects_erodes_and_gates_manifest() -> None:
    first_weight = torch.ones((128, 128), dtype=torch.float32)
    second_weight = torch.ones((128, 128), dtype=torch.float32)
    second_weight[:20] = 0.0
    second_weight[-20:] = 0.0
    second_weight[:, :20] = 0.0
    second_weight[:, -20:] = 0.0
    second_weight[64, 64] = float("nan")
    support = common_support_mask(
        (first_weight, second_weight),
        erode_radius_px=2,
    )

    assert support.dtype is torch.bool
    assert not bool(support[21, 64])
    assert bool(support[22, 64])
    assert not bool(support[62:67, 62:67].any())
    manifest = make_updec_common_world_manifest(
        target_header=simple_tan_header(128, 128, pixel_scale_arcsec=0.04),
        source_count=28,
        seed=44,
        margin_px=8.0,
        valid_mask=support,
    )
    for x, y in manifest["target_xy"]:
        assert bool(support[round(y), round(x)])
    assert manifest["placement"]["valid_mask_pixel_count"] == int(support.sum())
    assert len(manifest["placement"]["valid_mask_sha256"]) == 64


def test_common_support_can_tolerate_sparse_footprint_holes() -> None:
    weight = torch.ones((21, 21), dtype=torch.float32)
    weight[2::4, 2::4] = 0.0

    strict = common_support_mask((weight,), erode_radius_px=2)
    tolerant = common_support_mask(
        (weight,),
        erode_radius_px=2,
        min_footprint_fraction=0.8,
    )

    assert int(tolerant.sum()) > int(strict.sum())
    assert not bool(tolerant[weight == 0].any())
    with pytest.raises(ValueError, match="matching 2-D"):
        common_support_mask(
            (weight,),
            erode_radius_px=2,
            min_footprint_fraction=0.0,
        )


def test_catalogue_neighborhood_exclusion_is_circular_and_bounded() -> None:
    support = torch.ones((21, 21), dtype=torch.bool)
    result = exclude_catalogue_neighborhoods(
        support,
        torch.tensor([[10.0, 10.0], [-100.0, -100.0]], dtype=torch.float64),
        radius_px=2.5,
    )

    assert not bool(result[10, 10])
    assert not bool(result[10, 12])
    assert bool(result[10, 13])
    assert bool(result[0, 0])


def test_manifest_samples_sparse_support_pixels_without_rejection_failure() -> None:
    support = torch.zeros((128, 128), dtype=torch.bool)
    support[18:34, 18:34] = True
    support[18:34, 94:110] = True
    support[94:110, 18:34] = True
    support[94:110, 94:110] = True

    manifest = make_updec_common_world_manifest(
        target_header=simple_tan_header(128, 128, pixel_scale_arcsec=0.04),
        source_count=12,
        seed=9,
        margin_px=8.0,
        min_separation_px=5.0,
        valid_mask=support,
    )

    assert manifest["placement"]["sampling"] == "shuffled_valid_pixels"
    assert [manifest["quadrant"].count(value) for value in range(4)] == [3, 3, 3, 3]
    for x, y in manifest["target_xy"]:
        assert bool(support[round(y), round(x)])


def test_manifest_cli_reads_final_weight_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    header = simple_tan_header(64, 64, pixel_scale_arcsec=0.04)
    reference = tmp_path / "reference.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=torch.zeros((64, 64), dtype=torch.float32).numpy(),
                header=fits.Header(header),
                name="SCI",
            ),
        ]
    ).writeto(reference)
    weight_path = tmp_path / "weight.fits"
    weight = torch.ones((64, 64), dtype=torch.float32)
    weight[:12] = 0.0
    torchsky.write(str(weight_path), weight, header=header)
    output = tmp_path / "manifest.json"
    support_mask_path = tmp_path / "support_mask.fits"
    avoidance_path = tmp_path / "avoidance.json"
    avoidance_world = torchsky.WCS(header).pixel_to_world(
        torch.tensor([[32.0, 32.0]], dtype=torch.float64)
    )
    avoidance_path.write_text(
        json.dumps({"world_radec_deg": avoidance_world.tolist()})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_multiband_depth.py",
            "--profile",
            "updec-adass-manifest",
            "--reference-path",
            str(reference),
            "--cutout-size",
            "64",
            "--crop-x0",
            "0",
            "--crop-y0",
            "0",
            "--batch-sources",
            "8",
            "--margin-px",
            "10",
            "--min-separation-px",
            "6",
            "--support-weight-paths",
            str(weight_path),
            "--support-erode-radius-px",
            "2",
            "--support-footprint-fraction-grid",
            "0.8,1.0",
            "--support-mask-out",
            str(support_mask_path),
            "--avoidance-catalogue",
            str(avoidance_path),
            "--avoidance-radius-px",
            "3",
            "--json-out",
            str(output),
        ],
    )

    assert multiband_main() == 0
    artifact = json.loads(output.read_text())
    assert artifact["common_support"]["weight_paths"] == [str(weight_path)]
    assert artifact["common_support"]["erode_radius_px"] == 2
    assert artifact["common_support"]["raw_intersection_pixel_count"] == int(
        (weight > 0).sum()
    )
    assert artifact["common_support"]["individual"][0]["pixel_count"] == int(
        (weight > 0).sum()
    )
    assert len(artifact["common_support"]["quadrant_pixel_count"]) == 4
    sweep = artifact["common_support"]["footprint_fraction_sweep"]
    assert [row["min_footprint_fraction"] for row in sweep] == [0.8, 1.0]
    assert sweep[0]["pixel_count"] >= sweep[1]["pixel_count"]
    assert artifact["placement"]["margin_px"] == 10.0
    assert artifact["placement"]["min_separation_px"] == 6.0
    assert all(y >= 14.0 for _, y in artifact["target_xy"])
    assert artifact["common_support"]["mask_path"] == str(support_mask_path)
    assert len(artifact["common_support"]["mask_sha256"]) == 64
    assert artifact["common_support"]["avoidance_source_count"] == 1
    expected_support = exclude_catalogue_neighborhoods(
        common_support_mask((weight,), erode_radius_px=2),
        torchsky.WCS(header).world_to_pixel(avoidance_world),
        radius_px=3,
    )
    torch.testing.assert_close(
        torchsky.read(str(support_mask_path)),
        expected_support.to(torch.uint8),
    )


def test_forced_flux_color_and_heldout_likelihood() -> None:
    catalog = generate_truth_catalog(
        height=32,
        width=32,
        bands=4,
        source_count=5,
        seed=38,
        include_close_pair=False,
    )
    cube = render_truth(catalog, height=32, width=32)
    measured = forced_linear_flux(cube, catalog)
    torch.testing.assert_close(measured, catalog.band_flux, rtol=2e-11, atol=2e-11)
    metrics = flux_color_metrics(measured, catalog.band_flux)
    assert max(abs(value) for value in metrics["median_fractional_flux_bias"]) < 2e-11
    assert max(abs(value) for value in metrics["median_adjacent_color_bias_mag"]) < 2e-11
    assert gaussian_nll(cube, cube, 0.2) == pytest.approx(torch.log(torch.tensor(0.2)).item())


def test_blank_truth_scene() -> None:
    catalog = generate_truth_catalog(
        height=32, width=32, bands=3, source_count=0, seed=4
    )
    cube = render_truth(catalog, height=32, width=32)
    assert cube.shape == (3, 32, 32)
    assert torch.count_nonzero(cube) == 0


def test_split_is_disjoint_complete_and_deterministic() -> None:
    identifiers = [f"scene-{index}" for index in range(20)]
    split = split_identifiers(identifiers, seed=19)
    assert split == split_identifiers(identifiers, seed=19)
    assert set(split.train).isdisjoint(split.validation)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.validation).isdisjoint(split.test)
    assert set(split.train + split.validation + split.test) == set(identifiers)
    assert [updec_patt_split(value) for value in range(1, 10)] == [
        "train",
        "train",
        "validation",
        "train",
        "train",
        "test",
        "train",
        "validation",
        "test",
    ]
    with pytest.raises(ValueError, match="PATT_NUM"):
        updec_patt_split(10)


def test_detector_recovers_noiseless_bright_sources_and_null_is_empty() -> None:
    catalog = generate_truth_catalog(
        height=48,
        width=48,
        bands=6,
        source_count=1,
        seed=5,
        include_close_pair=False,
    )
    cube = render_truth(catalog, height=48, width=48)
    statistic = detection_statistic(cube, 0.01)
    detections, scores = find_detections(
        statistic, threshold=float(statistic.max()) * 0.8, nms_radius_px=2
    )
    match = maximum_cardinality_match(catalog.xy, detections, radius_px=1.0)
    assert len(match.pairs) == 1
    assert match.distances_px[0] < 0.75
    assert scores.shape == (1,)

    null_statistic = detection_statistic(torch.zeros_like(cube), 0.01)
    null_detections, _ = find_detections(null_statistic, threshold=1.0)
    assert null_detections.shape == (0, 2)


def test_weighted_matched_filter_uses_spatial_inverse_variance() -> None:
    template = torch.tensor(
        [[0.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 0.0]],
        dtype=torch.float64,
    )
    image = torch.zeros((9, 9), dtype=torch.float64)
    amplitude = 3.0
    image[3:6, 3:6] = amplitude * template
    inverse_variance = torch.ones_like(image)
    inverse_variance[3, 4] = 4.0

    snr = weighted_matched_filter_snr(image, inverse_variance, template)
    expected = amplitude * torch.sqrt(
        torch.sum(inverse_variance[3:6, 3:6] * template.square())
    )

    torch.testing.assert_close(snr[4, 4], expected)
    assert int(torch.argmax(snr)) == 4 * image.shape[1] + 4


def test_weighted_chi_detection_combines_positive_band_significance() -> None:
    template = torch.tensor(
        [[0.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 0.0]],
        dtype=torch.float64,
    )
    images = torch.zeros((3, 9, 9), dtype=torch.float64)
    images[0, 3:6, 3:6] = 2.0 * template
    images[1, 3:6, 3:6] = 3.0 * template
    images[2, 3:6, 3:6] = -100.0 * template
    inverse_variances = torch.ones_like(images)

    statistic = weighted_chi_detection_statistic(
        images,
        inverse_variances,
        (template, template, template),
    )
    template_norm = torch.linalg.vector_norm(template)

    torch.testing.assert_close(
        statistic[4, 4],
        torch.sqrt(torch.tensor(13.0, dtype=torch.float64)) * template_norm,
    )


def test_blank_responses_robustly_calibrate_whitened_chi() -> None:
    generator = torch.Generator().manual_seed(72)
    responses = torch.randn(
        (3, 128, 128),
        generator=generator,
        dtype=torch.float64,
    )
    responses[:, :4, :4] = 100.0
    support = torch.ones((128, 128), dtype=torch.bool)

    centers, scales = calibrate_matched_filter_whitening(responses, support)
    statistic = combine_whitened_chi_responses(responses, centers, scales)

    torch.testing.assert_close(
        centers, torch.zeros(3, dtype=responses.dtype), atol=0.03, rtol=0.0
    )
    torch.testing.assert_close(
        scales, torch.ones(3, dtype=responses.dtype), atol=0.04, rtol=0.0
    )
    assert statistic.shape == support.shape
    assert torch.isfinite(statistic).all()


def test_effective_psf_stack_registers_fractional_source_phases() -> None:
    dtype = torch.float64
    yy, xx = torch.meshgrid(
        torch.arange(64, dtype=dtype),
        torch.arange(64, dtype=dtype),
        indexing="ij",
    )
    source_xy = torch.tensor([[17.2, 18.7], [44.6, 42.3]], dtype=dtype)
    difference = torch.zeros((64, 64), dtype=dtype)
    for position in source_xy:
        profile = torch.exp(
            -0.5
            * (
                (xx - position[0]).square()
                + (yy - position[1]).square()
            )
            / 1.3**2
        )
        difference += profile / profile.sum()

    effective = stack_effective_psf(difference, source_xy, radius_px=6)

    torch.testing.assert_close(effective.sum(), torch.tensor(1.0, dtype=dtype))
    peak_yx = torch.stack(
        torch.unravel_index(torch.argmax(effective), effective.shape)
    ).tolist()
    assert peak_yx == [6, 6]
    torch.testing.assert_close(effective, effective.flip((0, 1)), atol=2e-2, rtol=0.0)


def test_effective_psf_artifact_reads_torchsky_products(tmp_path: Path) -> None:
    header = simple_tan_header(32, 32, pixel_scale_arcsec=0.04)
    header["BUNIT"] = "MJy/sr"
    source_xy = torch.tensor([[10.2, 11.3], [21.4, 20.1]], dtype=torch.float64)
    world = torchsky.WCS(header).pixel_to_world(source_xy)
    yy, xx = torch.meshgrid(
        torch.arange(32, dtype=torch.float64),
        torch.arange(32, dtype=torch.float64),
        indexing="ij",
    )
    positive = torch.zeros((32, 32), dtype=torch.float64)
    for position in source_xy:
        profile = torch.exp(
            -0.5
            * (
                (xx - position[0]).square()
                + (yy - position[1]).square()
            )
            / 1.1**2
        )
        positive += profile / profile.sum()
    blank_root = tmp_path / "blank"
    positive_root = tmp_path / "positive"
    relative = Path("F200W/products/torchsky_bilinear_sci.fits")
    (blank_root / relative.parent).mkdir(parents=True)
    (positive_root / relative.parent).mkdir(parents=True)
    torchsky.write(str(blank_root / relative), torch.zeros_like(positive), header=header)
    torchsky.write(str(positive_root / relative), positive, header=header)
    manifest_path = tmp_path / "manifest.json"
    flux_jy = (0.04 / 3600.0 * math.pi / 180.0) ** 2 * 1.0e6
    manifest_path.write_text(
        json.dumps(
            {
                "world_radec_deg": world.tolist(),
                "band_flux_jy": {"F200W": [flux_jy, flux_jy]},
            }
        )
    )

    artifact = _updec_effective_psf_artifact(
        positive_product_root=positive_root,
        blank_product_root=blank_root,
        injection_manifest_path=manifest_path,
        output_dir=tmp_path / "psfs",
        bands=("F200W",),
        methods=("torchsky_bilinear",),
        radius_px=4,
    )

    assert artifact["source_count"] == 2
    row = artifact["rows"][0]
    assert row["band"] == "F200W"
    assert row["method"] == "torchsky_bilinear"
    assert row["negative_lobe_fraction"] < 0.002
    assert abs(row["median_fractional_flux_bias"]) < 0.02
    assert len(row["aperture_flux_jy"]) == 2
    psf = torchsky.read(row["effective_psf_path"])
    assert psf.shape == (9, 9)
    torch.testing.assert_close(psf.sum(), torch.tensor(1.0), atol=1e-6, rtol=0.0)


def test_classical_detection_artifact_uses_mesh_psf_and_blank_whitening(
    tmp_path: Path,
) -> None:
    size = 64
    band = "F200W"
    method = "torchsky_bilinear"
    header = simple_tan_header(size, size, pixel_scale_arcsec=0.04)
    coordinates = torch.arange(-4, 5, dtype=torch.float32)
    psf_y, psf_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    psf = torch.exp(-0.5 * (psf_x.square() + psf_y.square()) / 1.2**2)
    psf /= psf.sum()
    generator = torch.Generator().manual_seed(84)
    image = 0.01 * torch.randn((size, size), generator=generator)
    image += 0.2
    image[28:37, 28:37] += 0.5 * psf
    weight = torch.full_like(image, 1.0e4)
    product_root = tmp_path / "products"
    product_dir = product_root / band / "products"
    product_dir.mkdir(parents=True)
    torchsky.write(
        str(product_dir / f"{method}_sci.fits"),
        image,
        header=header,
    )
    torchsky.write(
        str(product_dir / f"{method}_wht.fits"),
        weight,
        header=header,
    )
    psf_root = tmp_path / "psfs"
    psf_root.mkdir()
    torchsky.write(
        str(psf_root / f"{band}_{method}_effective_psf.fits"),
        psf,
    )
    support_path = tmp_path / "support.fits"
    torchsky.write(
        str(support_path),
        torch.ones((size, size), dtype=torch.uint8),
        header=header,
    )

    artifact = _updec_classical_detection_artifact(
        product_root=product_root,
        effective_psf_root=psf_root,
        support_mask_path=support_path,
        bands=(band,),
        method=method,
        detection_threshold=5.0,
        background_block_size=16,
        statistic_out=tmp_path / "statistic.fits",
    )

    detections = torch.tensor(artifact["detection_xy"])
    assert artifact["detection_count"] >= 1
    assert float(torch.linalg.vector_norm(detections - 32.0, dim=1).min()) < 1.0
    assert len(artifact["whitening"]["centers"]) == 1
    assert Path(artifact["statistic_path"]).is_file()


def test_control_union_deduplicates_and_covers_all_method_peaks(
    tmp_path: Path,
) -> None:
    paths = []
    for method, xy in (
        ("torchsky_bilinear", [[10.0, 10.0], [40.0, 40.0]]),
        ("drizzle_square_pf0p800", [[12.0, 10.0], [46.0, 40.0]]),
    ):
        path = tmp_path / f"{method}.json"
        path.write_text(
            json.dumps(
                {
                    "method": method,
                    "support_mask_path": "/shared/support.fits",
                    "detection_count": len(xy),
                    "detection_xy": xy,
                    "world_radec_deg": [
                        [150.0 + 1.0e-5 * x, 2.0 + 1.0e-5 * y] for x, y in xy
                    ],
                    "scores": [7.0] * len(xy),
                }
            )
        )
        paths.append(path)

    artifact = _updec_control_union_artifact(
        control_json_paths=paths,
        dedup_radius_px=6.0,
    )

    assert artifact["input_detection_count"] == 4
    assert artifact["union_count"] == 2
    assert artifact["max_input_to_union_distance_px"] == pytest.approx(6.0)
    assert artifact["world_radec_deg"] == [
        [150.0001, 2.0001],
        [150.0004, 2.0004],
    ]


def test_classical_validation_excludes_control_sources_and_calibrates_purity(
    tmp_path: Path,
) -> None:
    method = "torchsky_bilinear"
    control_path = tmp_path / "control.json"
    control_path.write_text(
        json.dumps(
            {
                "method": method,
                "detection_xy": [[2.0, 2.0]],
            }
        )
    )
    detection_path = tmp_path / "detections.json"
    detection_path.write_text(
        json.dumps(
            {
                "method": method,
                "detection_xy": [
                    [2.2, 2.1],
                    [10.2, 10.1],
                    [20.1, 20.2],
                    [40.0, 40.0],
                ],
                "scores": [12.0, 9.0, 8.0, 7.0],
            }
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "batch_id": "validation-000",
                "target_xy": [[10.0, 10.0], [20.0, 20.0]],
                "reference_ab_magnitude": [27.0, 28.0],
            }
        )
    )

    artifact = _updec_classical_validation_artifact(
        detection_json_paths=(detection_path,),
        manifest_paths=(manifest_path,),
        control_json_path=control_path,
        control_radius_px=2.0,
        match_radius_px=1.0,
        target_purity=0.95,
    )

    assert artifact["selected_threshold"] == pytest.approx(8.0)
    assert artifact["pooled"]["completeness"] == 1.0
    assert artifact["pooled"]["purity"] == 1.0
    assert artifact["input_rows"][0]["control_neighborhood_count"] == 1
    assert artifact["completeness_by_ab_magnitude"]["28"]["matched_count"] == 1


def test_dithered_exposures_and_production_classical_coadds_are_deterministic() -> None:
    catalog = generate_truth_catalog(
        height=20,
        width=20,
        bands=3,
        source_count=2,
        seed=17,
        include_close_pair=False,
    )
    truth = render_truth(catalog, height=20, width=20)
    first = simulate_dithered_exposures(
        truth, seed=91, exposure_count=2, noise_sigma=0.01
    )
    second = simulate_dithered_exposures(
        truth, seed=91, exposure_count=2, noise_sigma=0.01
    )
    torch.testing.assert_close(first.images, second.images, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        first.dither_xy_px, second.dither_xy_px, rtol=0.0, atol=0.0
    )
    assert float(first.dither_xy_px.abs().max()) <= 0.45
    assert first.dither_xy_px[0].tolist() == [0.0, 0.0]

    bilinear, bilinear_weight = coadd_synthetic_exposures(
        first, resampling_kernel="bilinear"
    )
    drizzle, drizzle_weight = coadd_synthetic_exposures(
        first, resampling_kernel="drizzle_square"
    )
    assert bilinear.shape == drizzle.shape == truth.shape
    assert bool(torch.all(bilinear_weight >= 0))
    assert bool(torch.all(drizzle_weight >= 0))
    assert torch.isfinite(bilinear).all()
    assert torch.isfinite(drizzle).all()
    # Both production paths should recover this small subpixel-dithered scene,
    # but their interpolation kernels are intentionally not identical.
    assert float(torch.sqrt(torch.mean((bilinear.double() - truth).square()))) < 0.02
    assert float(torch.sqrt(torch.mean((drizzle.double() - truth).square()))) < 0.02
    assert not torch.equal(bilinear, drizzle)


def test_classical_smoke_reports_three_honest_rows() -> None:
    artifact = _classical_smoke_artifact(
        seed=101,
        size=20,
        bands=3,
        sources=3,
        device="cpu",
        exposure_count=2,
    )
    assert artifact["depth_claim"] is False
    rows = artifact["methods"]
    assert [row["method"] for row in rows] == [
        "weighted_bilinear",
        "drizzle_square",
        "chi2_coadd",
    ]
    assert rows[2]["reuses_product"] == "weighted_bilinear"
    assert all(row["null_detections"] == 0 for row in rows)
    assert all(row["runtime_seconds"] > 0 for row in rows)
    assert all(row["detection"]["purity"] >= 0.0 for row in rows)


def test_classical_smoke_includes_phase0_foam_row_when_requested() -> None:
    artifact = _classical_smoke_artifact(
        seed=101,
        size=16,
        bands=3,
        sources=3,
        device="cpu",
        exposure_count=2,
        include_foam=True,
    )
    # resolution-matched Foam grid (cells kept at data scale) must preserve sources
    assert artifact["foam_grid_side"] == 5
    rows = artifact["methods"]
    names = [row["method"] for row in rows]
    assert names[:3] == ["weighted_bilinear", "drizzle_square", "chi2_coadd"]
    assert names[3:] == ["independent_foam", "joint_foam"]
    foam, joint = rows[3], rows[4]
    assert foam["representation"] == "foam_per_band_surface_brightness"
    assert joint["representation"] == "foam_joint_spectral_surface_brightness"
    assert joint["basis_count"] < 3
    assert joint["basis_count"] >= 1
    gate_summary = artifact["target_response_gate"]
    assert gate_summary["baseline_method"] == "chi2_coadd"
    for row in (foam, joint):
        assert row["depth_claim"] is False
        assert row["statistic"] == "independent_union"
        assert row["null_detections"] == 0
        assert row["runtime_seconds"] > 0
        for key in (
            "threshold_from_validation_null",
            "detection",
            "forced_photometry",
            "heldout_gaussian_nll",
            "image_rmse",
            "target_response",
        ):
            assert key in row
        assert row["target_response"]["baseline_method"] == "chi2_coadd"
        assert row["target_response"]["signal_preserved"] is True
        assert (
            row["target_response"]["median_recovery_fraction"]
            >= row["target_response"]["median_baseline_recovery_fraction"] - 0.25
        )
        assert math.isfinite(float(row["image_rmse"]))
        assert math.isfinite(float(row["heldout_gaussian_nll"]))


def test_joint_foam_spectral_basis_preserves_continuum_but_rejects_perturbation() -> None:
    from torchsky.map.field import solve_spectral_cell_coefficients

    device = torch.device("cpu")
    dtype = torch.float64
    bands, cells = 5, 8
    basis = _legendre_basis(bands, 2, device=device, dtype=dtype)
    assert basis.shape == (bands, 2)
    assert torch.allclose(basis[:, 0], torch.ones(bands, dtype=dtype))
    lam = torch.linspace(-1.0, 1.0, steps=bands, device=device, dtype=dtype)
    assert torch.allclose(basis[:, 1], lam)
    generator = torch.Generator().manual_seed(7)
    a0 = torch.randn(cells, dtype=dtype, generator=generator)
    a1 = torch.randn(cells, dtype=dtype, generator=generator)
    continuum = a0[None, :] + lam[:, None] * a1[None, :]
    fitted = solve_spectral_cell_coefficients(
        basis, continuum.clone(), ridge=1.0e-12
    )
    reconstructed = torch.einsum("bk,ck->bc", basis, fitted)
    assert torch.allclose(reconstructed, continuum, atol=1.0e-8)
    perturbation = torch.randn(bands, cells, dtype=dtype, generator=generator)
    noise_amp = 0.1 * float(torch.linalg.vector_norm(continuum))
    noisy = continuum + noise_amp * perturbation / float(
        torch.linalg.vector_norm(perturbation)
    )
    fitted_noisy = solve_spectral_cell_coefficients(
        basis, noisy.clone(), ridge=1.0e-12
    )
    reconstructed_noisy = torch.einsum("bk,ck->bc", basis, fitted_noisy)
    residual = torch.linalg.vector_norm(noisy - reconstructed_noisy) / float(
        torch.linalg.vector_norm(noisy - continuum)
    )
    assert residual < 1.0
    assert residual > 0.0


def test_target_response_gate_flags_attenuation_and_passes_preserved() -> None:
    def _row(method: str, bias: list[float]) -> dict[str, Any]:
        return {
            "method": method,
            "forced_photometry": {"median_fractional_flux_bias": bias},
        }

    rows = [
        _row("chi2_coadd", [0.02, -0.01, 0.03]),
        _row("preserved_method", [-0.02, 0.01, -0.01]),
        _row("attenuating_method", [-0.8, -0.85, -0.75]),
    ]
    summary = _target_response_gate(rows, baseline_method="chi2_coadd")
    assert summary["baseline_method"] == "chi2_coadd"
    assert rows[1]["target_response"]["signal_preserved"] is True
    assert rows[2]["target_response"]["signal_preserved"] is False
    empty = [_row("chi2_coadd", [0.0]), _row("dead_method", [])]
    _target_response_gate(empty, baseline_method="chi2_coadd")
    assert empty[1]["target_response"]["signal_preserved"] is False
    pytest.raises(ValueError, _target_response_gate, [_row("only", [0.0])])


def test_foam_grid_side_is_resolution_matched_and_bounded() -> None:
    assert _foam_grid_side_for_size(16) == 5
    assert _foam_grid_side_for_size(20) == 6
    assert _foam_grid_side_for_size(64, max_side=8) == 8
    assert _foam_grid_side_for_size(64, max_side=0) == 21
    pytest.raises(ValueError, _foam_grid_side_for_size, 12)


def test_independent_union_statistic_does_not_combine_subthreshold_bands() -> None:
    images = torch.zeros((2, 17, 17), dtype=torch.float64)
    images[:, 8, 8] = 1.0
    union = independent_union_statistic(
        images, 1.0, template_sigmas_px=(1.0,)
    )
    chi = detection_statistic(images, 1.0, template_sigmas_px=(1.0,))
    assert float(chi[8, 8]) == pytest.approx(math.sqrt(2) * float(union[8, 8]))


def test_angular_nms_and_matching_use_target_pixel_scale() -> None:
    statistic = torch.zeros((9, 9), dtype=torch.float64)
    statistic[4, 4] = 10.0
    statistic[4, 6] = 9.0
    angular_xy, _ = find_detections(
        statistic,
        threshold=8.0,
        nms_radius_arcsec=0.12,
        pixel_scale_arcsec=0.05,
    )
    assert angular_xy.tolist() == [[4.0, 4.0]]
    match = maximum_cardinality_angular_match(
        torch.tensor([[4.0, 4.0]]),
        torch.tensor([[5.0, 4.0]]),
        radius_arcsec=0.06,
        pixel_scale_arcsec=0.05,
    )
    assert match.pairs == ((0, 0),)


def test_matching_maximizes_cardinality_for_close_pair_and_counts_duplicate() -> None:
    truth = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    detections = torch.tensor(
        [[0.45, 0.0], [0.1, 0.0], [0.0, 0.2]], dtype=torch.float64
    )
    match = maximum_cardinality_match(truth, detections, radius_px=0.65)
    assert match.pairs == ((0, 1), (1, 0))
    assert match.unmatched_truth == ()
    assert match.unmatched_detections == (2,)
    score = score_detections(
        truth, detections, radius_px=0.65, area_square_arcmin=2.0
    )
    assert score["completeness"] == 1.0
    assert score["purity"] == pytest.approx(2 / 3)
    assert score["false_detections_per_square_arcmin"] == 0.5


def test_matching_is_order_invariant_in_cardinality() -> None:
    truth = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    detections = torch.tensor([[0.6, 0.0], [1.5, 0.0], [2.2, 0.0]])
    first = maximum_cardinality_match(truth, detections, radius_px=0.65)
    second = maximum_cardinality_match(
        truth.flip(0), detections.flip(0), radius_px=0.65
    )
    assert len(first.pairs) == len(second.pairs) == 3


def test_validation_purity_threshold_uses_lowest_accepted_score() -> None:
    scores = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0])
    labels = torch.tensor([True, True, False, True, False])
    assert calibrate_purity_threshold(scores, labels, target_purity=0.75) == 6.0
    assert calibrate_purity_threshold(scores, labels, target_purity=0.9) == 8.0
    assert calibrate_purity_threshold(
        scores, torch.zeros_like(labels), target_purity=0.95
    ) == float("inf")
    tied_scores = torch.tensor([9.0, 8.0, 8.0])
    tied_labels = torch.tensor([True, True, False])
    assert (
        calibrate_purity_threshold(tied_scores, tied_labels, target_purity=0.75)
        == 9.0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_detector_and_matching_cpu_cuda_parity() -> None:
    generator = torch.Generator().manual_seed(92)
    cube = torch.randn((4, 32, 32), generator=generator, dtype=torch.float64)
    cpu_statistic = detection_statistic(cube, torch.tensor([1.0, 1.2, 0.8, 1.5]))
    cuda_statistic = detection_statistic(
        cube.cuda(), torch.tensor([1.0, 1.2, 0.8, 1.5], device="cuda")
    )
    torch.testing.assert_close(cpu_statistic, cuda_statistic.cpu(), rtol=1e-12, atol=1e-12)
    cpu_xy, cpu_scores = find_detections(cpu_statistic, threshold=5.0)
    cuda_xy, cuda_scores = find_detections(cuda_statistic, threshold=5.0)
    torch.testing.assert_close(cpu_xy, cuda_xy.cpu())
    torch.testing.assert_close(cpu_scores, cuda_scores.cpu(), rtol=1e-12, atol=1e-12)
    assert maximum_cardinality_match(cpu_xy, cpu_xy, radius_px=0.0).pairs == tuple(
        (index, index) for index in range(cpu_xy.shape[0])
    )


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="root_seed"):
        make_seed_manifest(-1)
    with pytest.raises(ValueError, match="unique"):
        split_identifiers(["x", "x"], seed=1)
    with pytest.raises(ValueError, match="below one"):
        split_identifiers(["a", "b", "c"], seed=1, validation_fraction=0.5, test_fraction=0.5)
    with pytest.raises(ValueError, match="at least three"):
        generate_truth_catalog(height=32, width=32, bands=2, source_count=1, seed=1)
