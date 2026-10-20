import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import torchsky

from torchsky.cli import build_parser
from torchsky.cli.map_command import _dispatch_map
import torchfits


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
    assert args.plan_json is None
    assert args.method == "coadd"
    assert args.workflow == "native"
    assert args.hdu == 0
    assert args.weights is None
    assert args.photometric_scales is None
    assert args.background_offsets is None
    assert args.stack is None
    assert args.clip_sigma == 4.0
    assert args.clip_iterations == 3
    assert args.kernel is None
    assert args.background_mode is None
    assert args.weight_maps is None
    assert args.mask_maps is None
    assert args.primary_beam_maps is None
    assert args.inr_spectral_source == "svo"
    assert args.inr_spatial_sampling == "center"
    assert args.field_amp == "none"


@pytest.mark.parametrize(
    ("workflow", "kernel", "background_mode"),
    [
        ("native", "bilinear", "none"),
        ("drizzle", "drizzle_square", "none"),
        ("swarp", "lanczos3", "per-exposure"),
        ("montage", "conservative_polygon", "overlap"),
    ],
)
def test_map_workflow_presets_expand_to_explicit_controls(
    workflow: str,
    kernel: str,
    background_mode: str,
) -> None:
    from torchsky.cli.map_command import _resolve_map_workflow

    args = SimpleNamespace(
        workflow=workflow,
        kernel=None,
        stack=None,
        background_mode=None,
        sky=False,
    )

    _resolve_map_workflow(args)

    assert args.kernel == kernel
    assert args.stack == "weighted_mean"
    assert args.background_mode == background_mode


def test_map_workflow_explicit_controls_override_preset() -> None:
    from torchsky.cli.map_command import _resolve_map_workflow

    args = SimpleNamespace(
        workflow="swarp",
        kernel="bilinear",
        stack="median",
        background_mode="none",
        sky=False,
    )

    _resolve_map_workflow(args)

    assert args.kernel == "bilinear"
    assert args.stack == "median"
    assert args.background_mode == "none"


def test_map_parser_accepts_sigma_clipped_stack() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "map",
            "out.fits",
            "in.fits",
            "--stack",
            "sigma_clipped",
            "--clip-sigma",
            "3.5",
            "--clip-iterations",
            "5",
        ]
    )

    assert args.stack == "sigma_clipped"
    assert args.clip_sigma == 3.5
    assert args.clip_iterations == 5


def test_map_sky_alias_rejects_conflicting_background_mode() -> None:
    from torchsky.cli.map_command import _resolve_map_workflow

    args = SimpleNamespace(
        workflow="native",
        kernel=None,
        stack=None,
        background_mode="overlap",
        sky=True,
    )

    with pytest.raises(SystemExit, match="--sky conflicts with --background-mode"):
        _resolve_map_workflow(args)


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
        method="field",
        inr_render_tile=0,
    )
    with pytest.raises(SystemExit, match="--inr-render-tile must be positive"):
        _dispatch_map(args)


@mock.patch("torchsky.cli.map_command._expand_inputs")
def test_dispatch_map_rejects_field_amp_without_cuda(mock_expand):
    mock_expand.return_value = ["in1.fits"]
    args = argparse.Namespace(
        inputs=["in1.fits"],
        weights=None,
        weight_maps=None,
        mask_maps=None,
        primary_beam_maps=None,
        method="field",
        field_amp="float16",
        device="cpu",
    )

    with pytest.raises(SystemExit, match="requires CUDA"):
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
    torchfits.write(str(path_a), torch.full((8, 8), 2.0), header=header, overwrite=True)
    torchfits.write(str(path_b), torch.full((8, 8), 10.0), header=header, overwrite=True)
    wmap = torch.ones((8, 8), dtype=torch.float32)
    wmap[:, :4] = 3.0
    mask = torch.ones((8, 8), dtype=torch.float32)
    mask[:, 4:] = 0.0
    torchfits.write(str(weight_a), wmap, header=header, overwrite=True)
    torchfits.write(str(mask_b), mask, header=header, overwrite=True)

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
    data = torchfits.read(str(output), device="cpu")
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
    torchfits.write(
        str(path_ref), torch.full((8, 8), 2.0), header=header, overwrite=True
    )
    torchfits.write(
        str(path_raw), torch.full((8, 8), 6.0), header=header, overwrite=True
    )

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
    data = torchfits.read(str(output), device="cpu")
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
    torchfits.write(str(path), torch.ones((8, 8)), header=header, overwrite=True)

    def fail_run(*args, **kwargs):
        raise AssertionError("ImagePlaneMapMaker should not run during dry-run")

    monkeypatch.setattr(map_command, "stack_product", fail_run)

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


def test_cli_map_dry_run_writes_resolved_plan_json(monkeypatch, tmp_path) -> None:
    from torchsky.cli import main

    input_path = tmp_path / "input.fits"
    output_path = tmp_path / "output.fits"
    plan_path = tmp_path / "plans" / "output.plan.json"
    torchfits.write(
        str(input_path),
        torch.ones((8, 8)),
        header=_simple_header(),
        overwrite=True,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "torchsky",
            "map",
            str(output_path),
            str(input_path),
            "--dry-run",
            "--plan-json",
            str(plan_path),
            "--device",
            "cpu",
        ],
    )

    assert main() == 0
    assert plan_path.is_file()
    assert not output_path.exists()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["schema_version"] == 2
    assert plan["method"] == "coadd"
    assert plan["inputs"][0]["path"] == str(input_path.resolve())
    assert plan["inputs"][0]["hdu"] == 0
    assert plan["inputs"][0]["weight_is_inverse_variance"] is False
    assert plan["inputs"][0]["identity"]["path"] == str(input_path.resolve())
    assert "sha256" in plan["inputs"][0]["identity"]
    assert plan["target"]["shape"] == [9, 9]
    assert plan["execution"]["workflow"] == "native"
    assert plan["execution"]["kernel"] == "bilinear"
    assert plan["execution"]["stack"] == "weighted_mean"
    assert plan["execution"]["clip_sigma"] == 4.0
    assert plan["execution"]["clip_iterations"] == 3
    assert plan["execution"]["scratch_required_bytes"] == 0
    assert plan["execution"]["resolved_device"] == "cpu"
    assert plan["requested_products"] == ["SCI", "WHT", "IVAR", "CTX", "MASK", "NEXP"]
    assert plan["software"]["torchsky"] == torchsky.__version__
    assert plan["runtime"]["dtype"] == "float32"
    assert not list(plan_path.parent.glob("*.tmp"))


def test_cli_explicit_unit_weight_round_trips_as_inverse_variance(
    monkeypatch, tmp_path
) -> None:
    from torchsky.cli import main

    input_path = tmp_path / "unit-weight-input.fits"
    first_output = tmp_path / "unused.fits"
    replay_output = tmp_path / "unit-weight-replay.fits"
    plan_path = tmp_path / "unit-weight.plan.json"
    torchfits.write(
        str(input_path),
        torch.ones((8, 8)),
        header=_simple_header(),
        overwrite=True,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "torchsky",
            "map",
            str(first_output),
            str(input_path),
            "--weights",
            "1.0",
            "--no-union-footprint",
            "--dry-run",
            "--plan-json",
            str(plan_path),
            "--device",
            "cpu",
        ],
    )

    assert main() == 0
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["inputs"][0]["weight_is_inverse_variance"] is True

    monkeypatch.setattr(
        "sys.argv",
        [
            "torchsky",
            "map",
            str(replay_output),
            "--from-plan",
            str(plan_path),
        ],
    )
    assert main() == 0
    with torchfits.hdu.HDUList.fromfile(str(replay_output)) as hdul:
        by_name = {str(hdu.header.get("EXTNAME", "")): hdu for hdu in hdul}
        torch.testing.assert_close(
            by_name["IVAR"].to_tensor("cpu"),
            torch.ones((8, 8)),
        )


def test_cli_sigma_clipped_plan_records_exact_scratch_requirement(
    monkeypatch, tmp_path
) -> None:
    from torchsky.cli import main

    input_path = tmp_path / "clip-input.fits"
    plan_path = tmp_path / "clip.plan.json"
    torchfits.write(
        str(input_path),
        torch.ones((8, 8)),
        header=_simple_header(),
        overwrite=True,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "torchsky",
            "map",
            str(tmp_path / "clip-output.fits"),
            str(input_path),
            "--stack",
            "sigma_clipped",
            "--clip-sigma",
            "3.5",
            "--clip-iterations",
            "5",
            "--dry-run",
            "--plan-json",
            str(plan_path),
            "--device",
            "cpu",
        ],
    )

    assert main() == 0
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    height, width = plan["target"]["shape"]
    assert plan["execution"]["stack"] == "sigma_clipped"
    assert plan["execution"]["clip_sigma"] == 3.5
    assert plan["execution"]["clip_iterations"] == 5
    assert plan["execution"]["scratch_required_bytes"] == height * width * 12


def test_cli_montage_plan_records_global_overlap_scratch_requirement(
    monkeypatch,
    tmp_path,
) -> None:
    from torchsky.cli import main
    from torchsky.map.operators.execution import overlap_background_scratch_bytes

    input_a = tmp_path / "montage-plan-a.fits"
    input_b = tmp_path / "montage-plan-b.fits"
    plan_path = tmp_path / "montage.plan.json"
    for path in (input_a, input_b):
        torchfits.write(
            str(path),
            torch.ones((8, 8)),
            header=_simple_header(),
            overwrite=True,
        )
    monkeypatch.setattr(
        "sys.argv",
        [
            "torchsky",
            "map",
            str(tmp_path / "montage-output.fits"),
            str(input_a),
            str(input_b),
            "--workflow",
            "montage",
            "--dry-run",
            "--plan-json",
            str(plan_path),
            "--device",
            "cpu",
        ],
    )

    assert main() == 0
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    shape = tuple(plan["target"]["shape"])
    assert plan["execution"]["scratch_required_bytes"] == (
        overlap_background_scratch_bytes(
            2,
            shape,
            stacking_method="weighted_mean",
        )
    )


def test_cli_map_from_plan_verifies_identity_and_allows_output_override(
    monkeypatch,
    tmp_path,
) -> None:
    from torchsky.cli import main

    input_path = tmp_path / "input.fits"
    first_output = tmp_path / "first.fits"
    replay_output = tmp_path / "replay.fits"
    plan_path = tmp_path / "map.plan.json"
    torchfits.write(
        str(input_path), torch.ones((8, 8)), header=_simple_header(), overwrite=True
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "torchsky",
            "map",
            str(first_output),
            str(input_path),
            "--dry-run",
            "--plan-json",
            str(plan_path),
            "--device",
            "cpu",
        ],
    )
    assert main() == 0

    monkeypatch.setattr(
        "sys.argv",
        [
            "torchsky",
            "map",
            str(replay_output),
            "--from-plan",
            str(plan_path),
            "--dry-run",
        ],
    )
    assert main() == 0

    torchfits.write(
        str(input_path), torch.zeros((8, 8)), header=_simple_header(), overwrite=True
    )
    with pytest.raises(SystemExit, match="content changed|size changed|checksum"):
        main()


def test_cli_map_writes_standard_mef_products(monkeypatch, tmp_path) -> None:
    from torchsky.cli import main

    input_path = tmp_path / "input.fits"
    output_path = tmp_path / "map.fits"
    torchfits.write(
        str(input_path), torch.ones((8, 8)), header=_simple_header(), overwrite=True
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "torchsky",
            "map",
            str(output_path),
            str(input_path),
            "--weights",
            "4.0",
            "--no-union-footprint",
            "--device",
            "cpu",
        ],
    )

    assert main() == 0

    with torchfits.hdu.HDUList.fromfile(str(output_path)) as hdul:
        names = [str(hdu.header.get("EXTNAME", "")) for hdu in hdul]
        assert names == [
            "SCI",
            "WHT",
            "IVAR",
            "CTX",
            "MASK",
            "NEXP",
            "VAR_DETECTOR",
        ]
        assert str(hdul[3].header["CTX0000"]).startswith("0:0:")
        torch.testing.assert_close(hdul[0].to_tensor("cpu"), torch.ones((8, 8)))


def test_cli_map_writes_requested_noise_correlation_plane(
    monkeypatch, tmp_path
) -> None:
    from torchsky.cli import main

    input_path = tmp_path / "correlation-input.fits"
    output_path = tmp_path / "correlation-map.fits"
    torchfits.write(
        str(input_path), torch.ones((8, 8)), header=_simple_header(), overwrite=True
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "torchsky",
            "map",
            str(output_path),
            str(input_path),
            "--weights",
            "4.0",
            "--no-union-footprint",
            "--output-planes",
            "SCI",
            "WHT",
            "IVAR",
            "NCORR",
            "--device",
            "cpu",
        ],
    )

    assert main() == 0

    with torchfits.hdu.HDUList.fromfile(str(output_path)) as hdul:
        by_name = {str(hdu.header.get("EXTNAME", "")): hdu for hdu in hdul}
        assert list(by_name) == ["SCI", "WHT", "IVAR", "NCORR", "VAR_DETECTOR"]
        assert by_name["NCORR"].header["NCORTYPE"] == "LOCAL_ENBW"
        torch.testing.assert_close(
            by_name["NCORR"].to_tensor("cpu"), torch.ones((8, 8))
        )


def test_cli_map_inr_dry_run_writes_plan_json(monkeypatch, tmp_path) -> None:
    from torchsky.cli import main

    input_path = tmp_path / "input.fits"
    output_path = tmp_path / "output.fits"
    plan_path = tmp_path / "output.inr.plan.json"
    torchfits.write(
        str(input_path),
        torch.ones((8, 8)),
        header=_simple_header(),
        overwrite=True,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "torchsky",
            "map",
            str(output_path),
            str(input_path),
            "--method",
            "field",
            "--inr-band",
            "F277W",
            "--inr-spectral-source",
            "builtin",
            "--inr-spatial-sampling",
            "gauss_legendre_2x2",
            "--dry-run",
            "--plan-json",
            str(plan_path),
            "--device",
            "cpu",
        ],
    )

    assert main() == 0
    assert not output_path.exists()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["schema_version"] == 2
    assert plan["method"] == "field"
    assert plan["spectral"]["band"] == "F277W"
    assert plan["spectral"]["source"] == "builtin"
    assert plan["execution"]["field_preset"] == "balanced"
    assert plan["execution"]["model"] == "sky_lambda_feature_field"
    assert plan["execution"]["spatial_sampling"] == "gauss_legendre_2x2"


def test_cli_map_field_defaults_to_joint_header_derived_bands(
    monkeypatch,
    tmp_path,
) -> None:
    from torchsky.cli import main

    input_paths = [tmp_path / "f277w.fits", tmp_path / "f356w.fits"]
    for path, band in zip(input_paths, ("F277W", "F356W")):
        header = _simple_header()
        header["FILTER"] = band
        torchfits.write(str(path), torch.ones((8, 8)), header=header, overwrite=True)
    plan_path = tmp_path / "joint.plan.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "torchsky",
            "map",
            str(tmp_path / "joint.fits"),
            *(str(path) for path in input_paths),
            "--method",
            "field",
            "--inr-spectral-source",
            "builtin",
            "--dry-run",
            "--plan-json",
            str(plan_path),
            "--no-union-footprint",
            "--device",
            "cpu",
        ],
    )

    assert main() == 0
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["spectral"]["observation_bands"] == ["F277W", "F356W"]
    assert [band["label"] for band in plan["spectral"]["bands"]] == [
        "F277W",
        "F356W",
    ]
    assert plan["target"]["band_count"] == 2


def test_cli_map_field_artifact_resumes_training(monkeypatch, tmp_path) -> None:
    from torchsky.cli import main

    input_path = tmp_path / "input.fits"
    output_path = tmp_path / "field.fits"
    resumed_output_path = tmp_path / "field-resumed.fits"
    replay_output_path = tmp_path / "field-replayed.fits"
    artifact_path = tmp_path / "field.artifact"
    plan_path = tmp_path / "field.plan.json"
    torchfits.write(
        str(input_path),
        torch.ones((8, 8)),
        header=_simple_header(),
        overwrite=True,
    )
    common = [
        str(input_path),
        "--method",
        "field",
        "--field-model",
        "sine_implicit",
        "--inr-band",
        "F277W",
        "--inr-spectral-source",
        "builtin",
        "--inr-steps",
        "1",
        "--inr-patch-size",
        "4",
        "--inr-patches-per-obs",
        "1",
        "--inr-obs-per-step",
        "1",
        "--inr-spectral-samples",
        "3",
        "--no-union-footprint",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(
        "sys.argv",
        [
            "torchsky",
            "map",
            str(output_path),
            *common,
            "--artifact",
            str(artifact_path),
            "--plan-json",
            str(plan_path),
        ],
    )

    assert main() == 0
    manifest = json.loads((artifact_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["kind"] == "sine_implicit"
    assert manifest["fit_config"]["steps"] == 1
    assert output_path.is_file()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["execution"]["artifact"] == str(artifact_path.resolve())
    assert len(plan["execution"]["artifact_identities"]) == 3

    monkeypatch.setattr(
        "sys.argv",
        [
            "torchsky",
            "map",
            str(replay_output_path),
            "--from-plan",
            str(plan_path),
        ],
    )
    assert main() == 0
    assert replay_output_path.is_file()
    replayed_manifest = json.loads(
        (artifact_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert replayed_manifest["fit_config"]["steps"] == 1

    monkeypatch.setattr(
        "sys.argv",
        [
            "torchsky",
            "map",
            str(resumed_output_path),
            *common,
            "--resume",
            str(artifact_path),
        ],
    )
    assert main() == 0
    resumed_manifest = json.loads(
        (artifact_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert resumed_manifest["fit_config"]["steps"] == 2
    assert resumed_output_path.is_file()

    monkeypatch.setattr(
        "sys.argv",
        [
            "torchsky",
            "map",
            str(tmp_path / "stale-replay.fits"),
            "--from-plan",
            str(plan_path),
        ],
    )
    with pytest.raises(SystemExit, match="artifact|content changed|size changed"):
        main()


def test_run_field_single_uses_reference_header(monkeypatch, tmp_path) -> None:
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
    monkeypatch.setattr("torchsky.map.image_plane_common.read", fake_read)
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
    monkeypatch.setattr(
        map_command, "make_field_model", lambda *args, **kwargs: TinyModel()
    )
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
        "write_map_product",
        lambda path, result, *, requested_planes, overwrite: (
            written.update(
                path=path,
                data=result.data.squeeze(0),
                header=result.header,
            )
            or ("SCI",)
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

    assert map_command._run_field_single(args) == 0
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
    monkeypatch.setattr("torchsky.map.image_plane_common.read", fake_read)
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

    assert map_command._run_field_single(args) == 0


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


def test_paths_to_field_observations_keep_each_passband(monkeypatch) -> None:
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
        spectral_kw=[
            {
                "band_name": "F277W",
                "spectral_min": 2.4,
                "spectral_max": 3.1,
                "transmission_center": 2.77,
            },
            {
                "band_name": "F356W",
                "spectral_min": 3.1,
                "spectral_max": 4.0,
                "transmission_center": 3.56,
            },
        ],
        read_device="cpu",
    )

    assert [observation.band_name for observation in observations] == [
        "F277W",
        "F356W",
    ]
    assert [observation.transmission_center for observation in observations] == [
        2.77,
        3.56,
    ]


def test_run_field_per_filter_keeps_group_weights_and_maps(monkeypatch, tmp_path) -> None:
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
    monkeypatch.setattr(map_command, "_run_field_single", fake_run_single)

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

    assert map_command._run_field(args) == 0
    by_band = {ns.inr_band: ns for ns in captured}
    assert set(by_band) == {"F277W", "F356W"}
    assert by_band["F277W"].inputs == [paths[0], paths[2]]
    assert by_band["F277W"].weights == [1.0, 3.0]
    assert by_band["F277W"].background_offsets == [0.1, 0.3]
    assert by_band["F277W"].weight_maps == ["a_wht.fits", "c_wht.fits"]
    assert by_band["F277W"].mask_maps == ["a_mask.fits", "c_mask.fits"]
    assert by_band["F356W"].inputs == [paths[1]]
    assert by_band["F356W"].weights == [2.0]


def test_run_field_per_filter_keeps_running_after_a_group_fails(
    monkeypatch, tmp_path
) -> None:
    from torchsky.cli import map_command

    paths = [str(tmp_path / name) for name in ("a.fits", "b.fits")]
    filters = {"a.fits": "F277W", "b.fits": "F356W"}
    calls: list[str] = []

    def fake_read(path, *, hdu=0, return_header=False, device=None):
        header = _simple_header()
        header["FILTER"] = filters[Path(path).name]
        image = torch.zeros((8, 8), dtype=torch.float32)
        if return_header:
            return image, header
        return image

    def fake_run_single(ns):
        calls.append(ns.inr_band)
        return 1 if ns.inr_band == "F277W" else 0

    monkeypatch.setattr(map_command, "read", fake_read)
    monkeypatch.setattr(map_command, "_run_field_single", fake_run_single)

    args = argparse.Namespace(
        inputs=paths,
        weights=[1.0, 1.0],
        background_offsets=[0.0, 0.0],
        weight_maps=[None, None],
        mask_maps=[None, None],
        primary_beam_maps=[None, None],
        hdu=0,
        output=str(tmp_path / "out"),
        per_filter=True,
        inr_band=None,
        device="cpu",
    )

    # A failing band must not short-circuit the remaining bands, and its status
    # is what the command reports.
    assert map_command._run_field(args) == 1
    assert calls == ["F277W", "F356W"]
