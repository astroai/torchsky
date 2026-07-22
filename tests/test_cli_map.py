import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import torchsky

from torchsky.cli import build_parser
from torchsky.cli.map_command import _dispatch_map


def _simple_header(shape: tuple[int, int] = (8, 8)) -> dict:
    h, w = shape
    return {
        "SIMPLE": True,
        "BITPIX": -32,
        "NAXIS": 2,
        "NAXIS1": w,
        "NAXIS2": h,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": w / 2 + 0.5,
        "CRPIX2": h / 2 + 0.5,
        "CRVAL1": 10.0,
        "CRVAL2": 20.0,
        "CD1_1": -0.0002777777778,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.0002777777778,
    }


def test_map_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["map", "out.fits", "in1.fits", "in2.fits"])
    assert args.output == "out.fits"
    assert args.inputs == ["in1.fits", "in2.fits"]
    assert args.dry_run is False
    assert args.method == "coadd"
    assert args.hdu == 0
    assert args.weights is None
    assert args.photometric_scales is None
    assert args.background_offsets is None
    assert args.stack == "weighted_mean"
    assert args.kernel == "bilinear"
    assert args.weight_maps is None
    assert args.mask_maps is None
    assert args.primary_beam_maps is None
    assert args.inr_spectral_source == "svo"


def test_map_parser_help_is_renderable(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["map", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--union-footprint" in help_text
    assert "1.05 = 5%" in help_text


@mock.patch("torchsky.cli.map_command._expand_inputs")
def test_dispatch_map_weights_mismatch(mock_expand):
    mock_expand.return_value = ["in1.fits", "in2.fits"]
    args = argparse.Namespace(
        inputs=["in*.fits"],
        weights=[1.0],  # only 1 weight for 2 inputs
        method="coadd",
    )
    with pytest.raises(SystemExit, match="--weights count must match expanded"):
        _dispatch_map(args)


@mock.patch("torchsky.cli.map_command._expand_inputs")
def test_dispatch_map_weight_map_count_mismatch(mock_expand):
    mock_expand.return_value = ["in1.fits", "in2.fits"]
    args = argparse.Namespace(
        inputs=["in*.fits"],
        weights=None,
        weight_maps=["in1_wht.fits"],
        mask_maps=None,
        method="coadd",
    )
    with pytest.raises(SystemExit, match="--weight-maps count must match expanded"):
        _dispatch_map(args)


@mock.patch("torchsky.cli.map_command._expand_inputs")
def test_dispatch_map_rejects_invalid_inr_tile(mock_expand):
    mock_expand.return_value = ["in1.fits"]
    args = argparse.Namespace(
        inputs=["in1.fits"],
        weights=None,
        weight_maps=None,
        mask_maps=None,
        primary_beam_maps=None,
        method="inr",
        inr_render_tile=0,
    )
    with pytest.raises(SystemExit, match="--inr-render-tile must be positive"):
        _dispatch_map(args)


@mock.patch("torchsky.cli.map_command._expand_inputs")
def test_dispatch_map_rejects_cropping_footprint_padding(mock_expand):
    mock_expand.return_value = ["in1.fits"]
    args = argparse.Namespace(
        inputs=["in1.fits"],
        weights=None,
        method="coadd",
        footprint_padding=0.9,
    )
    with pytest.raises(SystemExit, match="--footprint-padding"):
        _dispatch_map(args)


@mock.patch("torchsky.cli.map_command._expand_inputs")
@mock.patch("torchsky.cli.map_command._run_coadd")
def test_dispatch_map_coadd(mock_run_coadd, mock_expand):
    mock_expand.return_value = ["in1.fits", "in2.fits"]
    mock_run_coadd.return_value = 0

    args = argparse.Namespace(inputs=["in*.fits"], weights=None, method="coadd")

    rc = _dispatch_map(args)
    assert rc == 0
    assert args.inputs == ["in1.fits", "in2.fits"]
    assert args.weights == [1.0, 1.0]
    mock_run_coadd.assert_called_once_with(args)


@mock.patch("torchsky.cli.map_command._expand_inputs")
@mock.patch("torchsky.cli.map_command._run_inr")
def test_dispatch_map_inr(mock_run_inr, mock_expand):
    mock_expand.return_value = ["in1.fits"]
    mock_run_inr.return_value = 0

    args = argparse.Namespace(inputs=["in1.fits"], weights=None, method="inr")

    rc = _dispatch_map(args)
    assert rc == 0
    assert args.inputs == ["in1.fits"]
    assert args.weights == [1.0]
    mock_run_inr.assert_called_once_with(args)


def test_safe_name():
    from torchsky.cli.map_command import _safe_name

    assert _safe_name("F277W") == "F277W"
    assert _safe_name("F277W/CLEAR") == "F277W_CLEAR"
    assert _safe_name("") == "band"


def test_run_coadd_plan_reference_out_of_range():
    from torchsky.cli.map_command import _run_coadd

    args = argparse.Namespace(
        inputs=["a.fits", "b.fits"],
        weights=[1.0, 1.0],
        hdu=0,
        reference=None,
        reference_hdu=0,
        plan_reference=5,
        auto_tile=False,
        tile_pixels=None,
        tile_above_megapixels=32.0,
        kernel="bilinear",
        stack="weighted_mean",
        drizzle_pixfrac=1.0,
        value_semantics="surface_brightness",
        output="out.fits",
        device=None,
        photometric_scales=[1.0, 1.0],
        background_offsets=[0.0, 0.0],
        weight_maps=[None, None],
        weight_map_hdu=0,
        mask_maps=[None, None],
        mask_map_hdu=0,
        primary_beam_maps=[None, None],
        primary_beam_hdu=None,
    )
    with pytest.raises(SystemExit, match="--plan-reference 5 out of range"):
        _run_coadd(args)


def test_run_coadd_accepts_cli_weight_and_mask_maps(tmp_path) -> None:
    from torchsky.cli.map_command import _run_coadd

    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    weight_a = tmp_path / "a_wht.fits"
    mask_b = tmp_path / "b_mask.fits"
    output = tmp_path / "out.fits"
    header = _simple_header()
    torchsky.write(str(path_a), torch.full((8, 8), 2.0), header=header, overwrite=True)
    torchsky.write(str(path_b), torch.full((8, 8), 10.0), header=header, overwrite=True)
    wmap = torch.ones((8, 8), dtype=torch.float32)
    wmap[:, :4] = 3.0
    mask = torch.ones((8, 8), dtype=torch.float32)
    mask[:, 4:] = 0.0
    torchsky.write(str(weight_a), wmap, header=header, overwrite=True)
    torchsky.write(str(mask_b), mask, header=header, overwrite=True)

    args = argparse.Namespace(
        inputs=[str(path_a), str(path_b)],
        weights=[1.0, 1.0],
        photometric_scales=[1.0, 1.0],
        background_offsets=[0.0, 0.0],
        weight_maps=[str(weight_a), None],
        weight_map_hdu=0,
        mask_maps=[None, str(mask_b)],
        mask_map_hdu=0,
        primary_beam_maps=[None, None],
        primary_beam_hdu=None,
        hdu=0,
        reference=None,
        reference_hdu=0,
        plan_reference=0,
        auto_tile=False,
        tile_pixels=None,
        tile_above_megapixels=32.0,
        kernel="bilinear",
        stack="weighted_mean",
        drizzle_pixfrac=1.0,
        value_semantics="surface_brightness",
        output=str(output),
        device="cpu",
    )

    assert _run_coadd(args) == 0
    data = torchsky.read(str(output), device="cpu")
    expected_left = (2.0 * 3.0 + 10.0) / 4.0
    # Interior rows (1-6) should match; edge pixels may be 0 from bilinear resampling
    torch.testing.assert_close(
        data[1:7, 1:4],
        torch.full((6, 3), expected_left, dtype=torch.float32),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        data[1:7, 5:7],
        torch.full((6, 2), 2.0, dtype=torch.float32),
        atol=1e-5,
        rtol=0.0,
    )


def test_run_coadd_accepts_cli_photometric_scale_and_background(tmp_path) -> None:
    from torchsky.cli.map_command import _run_coadd

    path_ref = tmp_path / "ref.fits"
    path_raw = tmp_path / "raw.fits"
    output = tmp_path / "out.fits"
    header = _simple_header()
    torchsky.write(str(path_ref), torch.full((8, 8), 2.0), header=header, overwrite=True)
    torchsky.write(str(path_raw), torch.full((8, 8), 6.0), header=header, overwrite=True)

    args = argparse.Namespace(
        inputs=[str(path_ref), str(path_raw)],
        weights=[1.0, 1.0],
        photometric_scales=[1.0, 2.0],
        background_offsets=[0.0, 5.0],
        weight_maps=[None, None],
        weight_map_hdu=0,
        mask_maps=[None, None],
        mask_map_hdu=0,
        primary_beam_maps=[None, None],
        primary_beam_hdu=None,
        hdu=0,
        reference=None,
        reference_hdu=0,
        plan_reference=0,
        auto_tile=False,
        tile_pixels=None,
        tile_above_megapixels=32.0,
        kernel="bilinear",
        stack="weighted_mean",
        drizzle_pixfrac=1.0,
        value_semantics="surface_brightness",
        output=str(output),
        device="cpu",
    )

    assert _run_coadd(args) == 0
    data = torchsky.read(str(output), device="cpu")
    torch.testing.assert_close(
        data[1:7, 1:7],
        torch.full((6, 6), 2.0, dtype=torch.float32),
        atol=1e-5,
        rtol=0.0,
    )


def test_run_coadd_dry_run_does_not_execute_mapmaker(monkeypatch, tmp_path) -> None:
    from torchsky.cli import map_command

    path = tmp_path / "input.fits"
    header = _simple_header()
    torchsky.write(str(path), torch.ones((8, 8)), header=header, overwrite=True)

    def fail_run(*args, **kwargs):
        raise AssertionError("ImagePlaneMapMaker should not run during dry-run")

    monkeypatch.setattr(map_command, "stack_map_product", fail_run)

    args = argparse.Namespace(
        inputs=[str(path)],
        weights=[1.0],
        photometric_scales=[1.0],
        background_offsets=[0.0],
        weight_maps=[None],
        weight_map_hdu=0,
        mask_maps=[None],
        mask_map_hdu=0,
        primary_beam_maps=[None],
        primary_beam_hdu=None,
        hdu=0,
        reference=None,
        reference_hdu=0,
        plan_reference=0,
        auto_tile=False,
        tile_pixels=None,
        tile_above_megapixels=32.0,
        kernel="bilinear",
        stack="weighted_mean",
        drizzle_pixfrac=1.0,
        value_semantics="surface_brightness",
        output=str(tmp_path / "out.fits"),
        device="cpu",
        dry_run=True,
    )

    assert map_command._run_coadd(args) == 0


def test_run_inr_single_uses_reference_header(monkeypatch, tmp_path) -> None:
    import torch.nn as nn
    from torchsky.cli import map_command

    input_path = tmp_path / "input.fits"
    reference_path = tmp_path / "reference.fits"
    output_path = tmp_path / "out.fits"
    input_header = _simple_header((8, 8))
    reference_header = _simple_header((5, 7))

    def fake_read(path, *, hdu=0, return_header=False, device=None):
        header = reference_header if str(path) == str(reference_path) else input_header
        image = torch.zeros(
            (int(header["NAXIS2"]), int(header["NAXIS1"])), dtype=torch.float32
        )
        if return_header:
            return image, dict(header)
        return image

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(()))

    written: dict[str, object] = {}

    monkeypatch.setattr(map_command, "read", fake_read)
    monkeypatch.setattr("torchsky.map.image_plane.read", fake_read)
    monkeypatch.setattr(
        map_command,
        "_paths_to_field_observations",
        lambda *args, **kwargs: [mock.Mock(image=torch.zeros((8, 8)))],
    )
    monkeypatch.setattr(
        map_command,
        "_inr_spectral_field_kwargs",
        lambda args, band: {
            "spectral_min": 1.0,
            "spectral_max": 2.0,
            "transmission_center": 1.5,
        },
    )
    monkeypatch.setattr(
        map_command,
        "estimate_field_domain",
        lambda *args, **kwargs: SimpleNamespace(
            center_ra_deg=0.0,
            center_dec_deg=0.0,
            spatial_scale_deg=1.0,
            wavelength_center=1.5,
            wavelength_scale=0.5,
        ),
    )
    monkeypatch.setattr(map_command, "make_field_model", lambda *args, **kwargs: TinyModel())
    monkeypatch.setattr(map_command, "fit_field_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        map_command,
        "render_field_product",
        lambda model, product, **kwargs: SimpleNamespace(
            data=torch.ones((1, *product.shape)), header=product.target_header
        ),
    )
    monkeypatch.setattr(
        map_command,
        "write",
        lambda path, data, *, header, overwrite: written.update(
            path=path, data=data, header=header
        ),
    )

    args = argparse.Namespace(
        inputs=[str(input_path)],
        weights=[1.0],
        background_offsets=[0.25],
        weight_maps=[None],
        weight_map_hdu=0,
        mask_maps=[None],
        mask_map_hdu=0,
        hdu=0,
        reference=reference_path,
        reference_hdu=0,
        plan_reference=0,
        inr_band="F277W",
        inr_spectral_source="builtin",
        inr_model="sine_implicit",
        inr_patch_size=0,
        inr_steps=0,
        inr_lr=1e-3,
        inr_patches_per_obs=1,
        inr_obs_per_step=1,
        inr_spectral_samples=1,
        inr_render_tile=512,
        inr_max_pixels=100,
        output=str(output_path),
        device="cpu",
        dry_run=False,
    )

    assert map_command._run_inr_single(args) == 0
    assert written["path"] == str(output_path)
    assert tuple(written["data"].shape) == (5, 7)
    assert written["header"]["NAXIS1"] == 7
    assert written["header"]["NAXIS2"] == 5


def test_run_inr_single_dry_run_skips_fit(monkeypatch, tmp_path) -> None:
    from torchsky.cli import map_command

    reference_header = _simple_header((6, 7))

    def fake_read(path, *, hdu=0, return_header=False, device=None):
        image = torch.zeros((6, 7), dtype=torch.float32)
        if return_header:
            return image, dict(reference_header)
        return image

    monkeypatch.setattr(map_command, "read", fake_read)
    monkeypatch.setattr("torchsky.map.image_plane.read", fake_read)
    monkeypatch.setattr(
        map_command,
        "_paths_to_field_observations",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("field observations should not be read during dry-run")
        ),
    )
    monkeypatch.setattr(
        map_command,
        "_inr_spectral_field_kwargs",
        lambda args, band: {
            "spectral_min": 1.0,
            "spectral_max": 2.0,
            "transmission_center": 1.5,
        },
    )

    args = argparse.Namespace(
        inputs=[str(tmp_path / "input.fits")],
        weights=[1.0],
        background_offsets=[0.0],
        weight_maps=[None],
        weight_map_hdu=0,
        mask_maps=[None],
        mask_map_hdu=0,
        hdu=0,
        reference=tmp_path / "reference.fits",
        reference_hdu=0,
        plan_reference=0,
        inr_band="F277W",
        inr_spectral_source="builtin",
        inr_model="sine_implicit",
        inr_render_tile=512,
        inr_max_pixels=100,
        output=str(tmp_path / "out.fits"),
        device="cpu",
        dry_run=True,
    )

    assert map_command._run_inr_single(args) == 0


def test_paths_to_field_observations_keeps_background_offsets(monkeypatch) -> None:
    from torchsky.cli import map_command

    def fake_read(path, *, hdu=0, return_header=False, device=None):
        image = torch.zeros((4, 4), dtype=torch.float32)
        header = _simple_header((4, 4))
        if return_header:
            return image, header
        return image

    monkeypatch.setattr(map_command, "read", fake_read)

    observations = map_command._paths_to_field_observations(
        ["a.fits", "b.fits"],
        hdu=0,
        spectral_kw={
            "spectral_min": 1.0,
            "spectral_max": 2.0,
            "transmission_center": 1.5,
        },
        read_device="cpu",
        weights=[1.0, 2.0],
        background_offsets=[0.0, 0.75],
    )

    assert [obs.background_offset for obs in observations] == [0.0, 0.75]
    torch.testing.assert_close(
        observations[1].weight,
        torch.full((4, 4), 2.0, dtype=torch.float32),
    )


def test_run_inr_per_filter_keeps_group_weights_and_maps(monkeypatch, tmp_path) -> None:
    from torchsky.cli import map_command

    paths = [str(tmp_path / name) for name in ("a.fits", "b.fits", "c.fits")]
    filters = {"a.fits": "F277W", "b.fits": "F356W", "c.fits": "F277W"}
    captured: list[argparse.Namespace] = []

    def fake_read(path, *, hdu=0, return_header=False, device=None):
        header = _simple_header()
        header["FILTER"] = filters[Path(path).name]
        image = torch.zeros((8, 8), dtype=torch.float32)
        if return_header:
            return image, header
        return image

    def fake_run_single(ns):
        captured.append(ns)
        return 0

    monkeypatch.setattr(map_command, "read", fake_read)
    monkeypatch.setattr(map_command, "_run_inr_single", fake_run_single)

    args = argparse.Namespace(
        inputs=paths,
        weights=[1.0, 2.0, 3.0],
        background_offsets=[0.1, 0.2, 0.3],
        weight_maps=["a_wht.fits", "b_wht.fits", "c_wht.fits"],
        mask_maps=["a_mask.fits", "b_mask.fits", "c_mask.fits"],
        primary_beam_maps=[None, None, None],
        hdu=0,
        output=str(tmp_path / "out"),
        per_filter=True,
        inr_band=None,
        device="cpu",
    )

    assert map_command._run_inr(args) == 0
    by_band = {ns.inr_band: ns for ns in captured}
    assert by_band["F277W"].inputs == [paths[0], paths[2]]
    assert by_band["F277W"].weights == [1.0, 3.0]
    assert by_band["F277W"].background_offsets == [0.1, 0.3]
    assert by_band["F277W"].weight_maps == ["a_wht.fits", "c_wht.fits"]
    assert by_band["F277W"].mask_maps == ["a_mask.fits", "c_mask.fits"]
    assert by_band["F356W"].inputs == [paths[1]]
    assert by_band["F356W"].weights == [2.0]
