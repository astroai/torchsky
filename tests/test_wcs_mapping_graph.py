from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from torchsky.wcs import (
    AffineMapping,
    AmbiguousPathError,
    AxisSpec,
    CallableMapping,
    DomainSpec,
    Epoch,
    FrameSpec,
    GeneralizedWCS,
    IdentityMapping,
    Mapping,
    ParallelMapping,
    PortRoute,
    PortRoutingMapping,
    TimeSpec,
    TransformEdge,
    TransformGraph,
    UnitConversionMapping,
    UnitSpec,
    UnsupportedInverseError,
    UnsupportedPathError,
    WCS,
)


def _axes(count: int, *, unit: str = "") -> tuple[AxisSpec, ...]:
    return tuple(AxisSpec(f"x{index}", unit) for index in range(count))


def _affine(
    matrix: torch.Tensor,
    offset: torch.Tensor,
    input_axes: tuple[AxisSpec, ...],
    output_axes: tuple[AxisSpec, ...] | None = None,
) -> AffineMapping:
    return AffineMapping(
        matrix,
        offset,
        input_axes,
        input_axes if output_axes is None else output_axes,
    )


def test_unit_axis_frame_domain_semantics_fail_closed() -> None:
    degrees = UnitSpec("degrees")
    radians = UnitSpec("rad")
    assert degrees.symbol == "deg"
    assert degrees.dimension == "angle"
    assert degrees.conversion_factor_to(radians) == pytest.approx(math.pi / 180.0)
    with pytest.raises(ValueError, match="cannot convert"):
        degrees.conversion_factor_to(UnitSpec("m"))
    with pytest.raises(ValueError, match="unsupported unit"):
        UnitSpec("furlong")

    domain = DomainSpec(((-180.0, 180.0), (-90.0, 90.0)))
    values = torch.tensor([[0.0, 0.0], [181.0, 0.0], [0.0, -91.0]])
    torch.testing.assert_close(
        domain.contains(values), torch.tensor([True, False, False])
    )
    torch.testing.assert_close(
        domain.clamp(values),
        torch.tensor([[0.0, 0.0], [180.0, 0.0], [0.0, -90.0]]),
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        AxisSpec("ra", "deg", "pos.eq.ra", domain=domain)
    with pytest.raises(ValueError, match="unique"):
        FrameSpec("sky", [AxisSpec("ra", "deg"), AxisSpec("ra", "deg")])


def test_epoch_format_scale_conversion_and_tensor_state() -> None:
    epoch = Epoch(
        torch.tensor([51544.5, 57754.0], dtype=torch.float64),
        TimeSpec(format="mjd", scale="utc", reference_position="geocenter"),
    )
    tt = epoch.to(TimeSpec(format="jd", scale="tt"))
    assert tt.spec == TimeSpec(format="jd", scale="tt")
    assert torch.all(tt.value > epoch.value + 2_400_000.0)
    roundtrip = tt.to(epoch.spec)
    torch.testing.assert_close(roundtrip.value, epoch.value, atol=2e-10, rtol=0.0)
    metadata, tensors = epoch.to_state()
    json.dumps(metadata)
    restored = Epoch.from_state(metadata, tensors)
    assert restored.spec == epoch.spec
    torch.testing.assert_close(restored.value, epoch.value)
    with pytest.raises(ValueError, match="unsupported time-scale"):
        Epoch(2000.0, TimeSpec(format="jyear", scale="tcb")).to(
            TimeSpec(format="jyear", scale="tt")
        )


def test_serial_parallel_and_exact_inverse_mixed_dimensions() -> None:
    xy = _axes(2)
    uvw = _axes(3)
    affine_2_to_3 = _affine(
        torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.0, 3.0]], dtype=torch.float64),
        torch.tensor([0.5, -2.0, 4.0], dtype=torch.float64),
        xy,
        uvw,
    )
    one = _axes(1)
    scale = _affine(
        torch.tensor([[2.0]], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        one,
    )
    parallel = ParallelMapping((affine_2_to_3, scale))
    inputs = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    expected = torch.cat(
        (affine_2_to_3(inputs[..., :2]), scale(inputs[..., 2:])), dim=-1
    )
    torch.testing.assert_close(parallel(inputs), expected)
    with pytest.raises(UnsupportedInverseError, match="exact inverse"):
        _ = parallel.inverse

    square = _affine(
        torch.tensor([[2.0, 0.5], [-0.25, 1.5]], dtype=torch.float64),
        torch.tensor([1.0, -2.0], dtype=torch.float64),
        xy,
    )
    unit_shift = _affine(
        torch.eye(2, dtype=torch.float64),
        torch.tensor([4.0, 5.0], dtype=torch.float64),
        xy,
    )
    serial = square | unit_shift
    points = torch.tensor([[1.0, 2.0], [-3.0, 4.0]], dtype=torch.float64)
    torch.testing.assert_close(serial.inverse(serial(points)), points)


def test_port_routing_supports_permutation_duplication_and_axis_assembly() -> None:
    inputs = _axes(3)
    outputs = tuple(AxisSpec(f"y{index}") for index in range(4))
    identity_one = IdentityMapping((AxisSpec("local"),))
    double = _affine(
        torch.tensor([[2.0]], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        (AxisSpec("local"),),
        (AxisSpec("out"),),
    )
    routing = PortRoutingMapping(
        inputs,
        outputs,
        (
            PortRoute(identity_one, (2,), (0,)),
            PortRoute(double, (0,), (1,)),
            PortRoute(identity_one, (1,), (2,)),
            PortRoute(identity_one, (0,), (3,)),
        ),
    )
    values = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    torch.testing.assert_close(
        routing(values), torch.tensor([[3.0, 2.0, 2.0, 1.0]], dtype=torch.float64)
    )
    assert not routing.has_exact_inverse

    permutation = PortRoutingMapping(
        inputs,
        tuple(AxisSpec(f"z{index}") for index in range(3)),
        (
            PortRoute(identity_one, (2,), (0,)),
            PortRoute(identity_one, (0,), (1,)),
            PortRoute(identity_one, (1,), (2,)),
        ),
    )
    torch.testing.assert_close(permutation.inverse(permutation(values)), values)


def test_shape_and_unit_validation_reject_invalid_composition() -> None:
    one_degree = (AxisSpec("angle", "deg", "angle"),)
    one_radian = (AxisSpec("angle", "rad", "angle"),)
    degree_identity = IdentityMapping(one_degree)
    radian_identity = IdentityMapping(one_radian)
    with pytest.raises(ValueError, match="serial axis mismatch"):
        _ = degree_identity | radian_identity
    converted = degree_identity | UnitConversionMapping(one_degree, one_radian)
    result = converted(torch.tensor([[180.0]], dtype=torch.float64))
    torch.testing.assert_close(result, torch.tensor([[math.pi]], dtype=torch.float64))
    with pytest.raises(ValueError, match="trailing dimension"):
        degree_identity(torch.ones(2, 2))


def test_transform_graph_validation_selection_and_ambiguity() -> None:
    axes = _axes(2)
    frames = [FrameSpec(name, axes) for name in ("a", "b", "c", "d")]
    identity = IdentityMapping(axes)
    graph = TransformGraph(frames)
    graph.add_transform("a", "b", identity)
    graph.add_transform("b", "d", identity)
    graph.add_transform("a", "c", identity)
    graph.add_transform("c", "d", identity)
    with pytest.raises(AmbiguousPathError, match="multiple shortest"):
        graph.find_path("a", "d")
    selected = graph.get_mapping("a", "d", path=("a", "b", "d"))
    values = torch.tensor([[1.0, 2.0]])
    torch.testing.assert_close(selected(values), values)
    with pytest.raises(ValueError, match="acyclic"):
        graph.add_transform("d", "a", identity)

    disconnected = TransformGraph((frames[0], frames[1], frames[2]))
    disconnected.add_transform("a", "b", identity)
    with pytest.raises(UnsupportedPathError, match="no transform path"):
        disconnected.get_mapping("a", "c")


def test_mapping_graph_json_tensor_serialization_roundtrip() -> None:
    xy = _axes(2)
    uv = tuple(AxisSpec(f"u{index}") for index in range(2))
    affine = _affine(
        torch.tensor([[2.0, 0.5], [-1.0, 3.0]], dtype=torch.float64),
        torch.tensor([4.0, -2.0], dtype=torch.float64),
        xy,
        uv,
    )
    graph = TransformGraph(
        (
            FrameSpec("detector", xy),
            FrameSpec(
                "world",
                uv,
                obstime=Epoch(2025.0, TimeSpec(format="jyear", scale="tt")),
            ),
        ),
        (TransformEdge("detector", "world", affine),),
    )
    generalized = GeneralizedWCS(graph, "detector", "world")
    metadata, tensors = generalized.to_state()
    json.dumps(metadata)
    restored = GeneralizedWCS.from_state(metadata, tensors)
    points = torch.tensor([[1.0, 2.0], [-4.0, 3.0]], dtype=torch.float64)
    torch.testing.assert_close(restored.pixel_to_world(points), generalized.pixel_to_world(points))
    torch.testing.assert_close(restored.world_to_pixel(restored.pixel_to_world(points)), points)
    restored_epoch = restored.graph.frames["world"].obstime
    assert restored_epoch is not None
    assert restored_epoch.spec.scale == "tt"


def test_numerical_inverse_converges_with_bounds_and_fails_closed() -> None:
    axis = (AxisSpec("x"),)
    nonlinear = CallableMapping(
        lambda value: value + 0.1 * value.square(),
        axis,
        axis,
        domain=DomainSpec(((-4.0, 4.0),)),
    )
    inverse = nonlinear.numerical_inverse(
        initial=torch.tensor([0.0], dtype=torch.float64),
        bounds=DomainSpec(((-4.0, 4.0),)),
        max_iterations=20,
        residual_tolerance=1.0e-12,
        relative_tolerance=1.0e-12,
    )
    points = torch.tensor([[-2.0], [0.5], [3.0]], dtype=torch.float64)
    result = inverse.solve(nonlinear(points))
    assert result.converged.all()
    torch.testing.assert_close(result.values, points, atol=1e-10, rtol=1e-10)
    with pytest.raises(RuntimeError, match="failed to converge"):
        inverse(torch.tensor([[100.0]], dtype=torch.float64))

    nonsquare = CallableMapping(lambda value: value[..., :1], _axes(2), _axes(1))
    with pytest.raises(UnsupportedInverseError, match="square"):
        nonsquare.numerical_inverse(
            initial=torch.zeros(2), bounds=DomainSpec(((-1.0, 1.0),) * 2)
        )


def test_mapping_gradcheck_gradgradcheck_and_compile() -> None:
    axes = _axes(2)
    first = _affine(
        torch.tensor([[1.5, 0.2], [-0.1, 0.75]], dtype=torch.float64),
        torch.tensor([0.3, -0.4], dtype=torch.float64),
        axes,
    )
    nonlinear = CallableMapping(
        lambda value: torch.stack(
            (torch.sin(value[..., 0]), value[..., 1].square()), dim=-1
        ),
        axes,
        axes,
    )
    mapping = first | nonlinear
    values = torch.tensor([[0.2, -0.4], [0.7, 0.3]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(mapping, (values,))
    assert torch.autograd.gradgradcheck(mapping, (values,))
    compiled = torch.compile(mapping, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(values), mapping(values))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
                reason="MPS unavailable",
            ),
        ),
    ],
)
def test_mapping_device_forward_inverse_and_jacobian(device: str) -> None:
    dtype = torch.float32 if device == "mps" else torch.float64
    axes = _axes(2)
    mapping = _affine(
        torch.tensor([[2.0, 0.5], [-0.25, 1.5]], dtype=torch.float64),
        torch.tensor([1.0, -2.0], dtype=torch.float64),
        axes,
    ).to(device=device, dtype=dtype)
    values = torch.tensor([[1.0, 2.0], [-3.0, 4.0]], dtype=dtype, device=device)
    output = mapping(values)
    recovered = mapping.inverse.to(device=device, dtype=dtype)(output)
    jacobian = mapping.jacobian(values)
    assert output.device.type == device
    assert jacobian.device.type == device
    torch.testing.assert_close(recovered, values, atol=2e-5, rtol=2e-5)


def test_wcs_as_mapping_and_generalized_wcs_match_astropy() -> None:
    astropy_wcs = pytest.importorskip("astropy.wcs")
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRPIX1": 32.0,
        "CRPIX2": 24.0,
        "CRVAL1": 150.0,
        "CRVAL2": -20.0,
        "CD1_1": -0.02,
        "CD1_2": 0.003,
        "CD2_1": 0.002,
        "CD2_2": 0.018,
    }
    wcs = WCS(header)
    mapping = wcs.as_mapping()
    graph = TransformGraph(
        (
            FrameSpec("detector", mapping.input_axes),
            FrameSpec("icrs", mapping.output_axes, representation="lonlat"),
        ),
        (TransformEdge("detector", "icrs", mapping),),
    )
    generalized = GeneralizedWCS(graph, "detector", "icrs")
    pixels = torch.tensor([[31.0, 23.0], [35.5, 19.25]], dtype=torch.float64)
    expected = astropy_wcs.WCS(header).all_pix2world(pixels.numpy(), 0)
    np.testing.assert_allclose(
        generalized.pixel_to_world(pixels).detach().numpy(), expected, atol=2e-10
    )
    torch.testing.assert_close(
        generalized.world_to_pixel(generalized.pixel_to_world(pixels)),
        pixels,
        atol=2e-8,
        rtol=0.0,
    )
    ra, dec = generalized.pixel_to_world_values(pixels[:, 0], pixels[:, 1])
    torch.testing.assert_close(torch.stack((ra, dec), dim=-1), generalized.pixel_to_world(pixels))
    metadata, tensors = mapping.to_state()
    json.dumps(metadata)
    restored = Mapping.from_state(metadata, tensors)
    torch.testing.assert_close(restored(pixels), mapping(pixels))


def test_compound_mapping_matches_astropy_modeling() -> None:
    models = pytest.importorskip("astropy.modeling.models")
    xy = _axes(2)
    shifted_scaled = ParallelMapping(
        (
            _affine(torch.tensor([[1.0]]), torch.tensor([1.0]), (xy[0],)),
            _affine(torch.tensor([[2.0]]), torch.tensor([0.0]), (xy[1],)),
        )
    )
    angle = math.radians(30.0)
    rotation = _affine(
        torch.tensor(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        ),
        torch.zeros(2),
        xy,
    )
    native = shifted_scaled | rotation
    reference = (models.Shift(1.0) & models.Scale(2.0)) | models.Rotation2D(30.0)
    values = torch.tensor([[0.0, 1.0], [2.0, -3.0]], dtype=torch.float64)
    expected_x, expected_y = reference(values[:, 0].numpy(), values[:, 1].numpy())
    expected = torch.from_numpy(np.stack((expected_x, expected_y), axis=-1))
    torch.testing.assert_close(native(values), expected)


def test_linear_pipeline_matches_gwcs_when_available() -> None:
    pytest.importorskip("gwcs")
    units = pytest.importorskip("astropy.units")
    from astropy.modeling.models import Scale, Shift
    from gwcs.coordinate_frames import Frame2D
    from gwcs.wcs import WCS as GWCS

    detector = Frame2D(name="detector", axes_order=(0, 1), unit=(units.pix, units.pix))
    world = Frame2D(name="world", axes_order=(0, 1), unit=(units.deg, units.deg))
    reference = GWCS([(detector, Shift(1.0) & Scale(2.0)), (world, None)])
    xy = _axes(2)
    native = ParallelMapping(
        (
            _affine(torch.tensor([[1.0]]), torch.tensor([1.0]), (xy[0],)),
            _affine(torch.tensor([[2.0]]), torch.tensor([0.0]), (xy[1],)),
        )
    )
    values = torch.tensor([[0.0, 1.0], [2.0, -3.0]], dtype=torch.float64)
    expected = torch.from_numpy(np.stack(reference(values[:, 0], values[:, 1]), axis=-1))
    torch.testing.assert_close(native(values), expected)
