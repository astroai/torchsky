import argparse
from unittest import mock
import pytest

from torchsky.cli import build_parser
from torchsky.cli.map_command import _dispatch_map

def test_map_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["map", "out.fits", "in1.fits", "in2.fits"])
    assert args.output == "out.fits"
    assert args.inputs == ["in1.fits", "in2.fits"]
    assert args.method == "coadd"
    assert args.hdu == 0
    assert args.weights is None
    assert args.stack == "weighted_mean"
    assert args.kernel == "bilinear"
    assert args.inr_spectral_source == "svo"

@mock.patch("torchsky.cli.map_command._expand_inputs")
def test_dispatch_map_weights_mismatch(mock_expand):
    mock_expand.return_value = ["in1.fits", "in2.fits"]
    args = argparse.Namespace(
        inputs=["in*.fits"],
        weights=[1.0],  # only 1 weight for 2 inputs
        method="coadd"
    )
    with pytest.raises(SystemExit, match="--weights count must match expanded"):
        _dispatch_map(args)

@mock.patch("torchsky.cli.map_command._expand_inputs")
@mock.patch("torchsky.cli.map_command._run_coadd")
def test_dispatch_map_coadd(mock_run_coadd, mock_expand):
    mock_expand.return_value = ["in1.fits", "in2.fits"]
    mock_run_coadd.return_value = 0

    args = argparse.Namespace(
        inputs=["in*.fits"],
        weights=None,
        method="coadd"
    )

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

    args = argparse.Namespace(
        inputs=["in1.fits"],
        weights=None,
        method="inr"
    )

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
    )
    with pytest.raises(SystemExit, match="--plan-reference 5 out of range"):
        _run_coadd(args)
