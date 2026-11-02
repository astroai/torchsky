from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchfits

import torchsky
from torchsky.domain import DomainSpec
from torchsky.map import (
    DQPolicy,
    Observation,
    OutlierConfig,
    ResampleConfig,
    StackConfig,
)
from torchsky.map.io_utils import write_map_product
from torchsky.map.operators import IndexedResponsePlan, apply_response_products
from torchsky.quality import QualityBit


def _header(shape: tuple[int, int] = (16, 16)) -> dict[str, object]:
    height, width = shape
    return {
        "NAXIS": 2,
        "NAXIS1": width,
        "NAXIS2": height,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": width / 2 + 0.5,
        "CRPIX2": height / 2 + 0.5,
        "CRVAL1": 42.0,
        "CRVAL2": -7.0,
        "CDELT1": -0.001,
        "CDELT2": 0.001,
    }


def test_dq_policy_bit_matrix_is_explicit_and_preserves_original_flags() -> None:
    dq = torch.tensor([[0, 1, 2, 4, 8, 10]], dtype=torch.int32)
    policy = DQPolicy(
        reject_bits=1 | 4,
        propagate_bits=1 | 2 | 4,
        cosmic_ray_bit=4,
        unknown_bit_policy="ignore",
    )

    assert policy.usability_mask(dq).tolist() == [
        [True, False, True, False, True, True]
    ]
    assert policy.propagated(dq).tolist() == [[0, 1, 2, 4, 0, 2]]
    updated = policy.with_cosmic_rays(
        dq,
        torch.tensor([[False, False, True, False, False, False]]),
    )
    assert updated.tolist() == [[0, 1, 6, 4, 8, 10]]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_one_response_plan_matches_dense_science_variance_dq_context_and_gradient(
    device: str,
) -> None:
    dtype = torch.float32 if device == "mps" else torch.float64
    source_indices = torch.arange(33, dtype=torch.int64, device=device)
    target_indices = torch.zeros(33, dtype=torch.int64, device=device)
    coefficients = torch.linspace(0.25, 1.0, 33, dtype=dtype, device=device)
    plan = IndexedResponsePlan(
        DomainSpec((33,), dtype),
        DomainSpec((1,), dtype),
        source_indices,
        target_indices,
        coefficients,
        contributor_ids=source_indices.clone(),
        default_chunk_size=7,
    )
    science = torch.linspace(-1.0, 2.0, 33, dtype=dtype, device=device).requires_grad_()
    variance = torch.linspace(0.5, 1.5, 33, dtype=dtype, device=device)
    dq = torch.tensor([1 << (index % 6) for index in range(33)], device=device)

    products = apply_response_products(
        plan,
        science,
        variance_components={"detector": variance},
        dq=dq,
        dq_policy=DQPolicy(reject_bits=0, unknown_bit_policy="preserve"),
        chunk_size=5,
    )

    torch.testing.assert_close(products.science, (coefficients @ science).reshape(1))
    torch.testing.assert_close(
        products.variance_components["detector"],
        (coefficients.square() @ variance).reshape(1),
    )
    assert int(products.dq.item()) == 63
    assert products.context.shape == (2, 1)
    assert int(products.context[0, 0]) == 2**32 - 1
    assert int(products.context[1, 0]) == 1
    assert int(products.nexp.item()) == 33
    products.science.sum().backward()
    torch.testing.assert_close(science.grad, coefficients)
    target = torch.tensor([1.25], dtype=dtype, device=device)
    torch.testing.assert_close(
        products.science @ target,
        science.detach() @ plan.adjoint(target, chunk_size=3),
    )


def test_map_dq_is_separate_from_boolean_mask_and_uses_actual_contributors() -> None:
    header = _header((8, 8))
    image = torch.ones((8, 8), dtype=torch.float32)
    policy = DQPolicy(
        reject_bits=1,
        propagate_bits=1 | 2 | 4,
        cosmic_ray_bit=4,
        unknown_bit_policy="ignore",
    )
    first_dq = torch.zeros((8, 8), dtype=torch.int32)
    second_dq = torch.zeros((8, 8), dtype=torch.int32)
    first_dq[3, 3] = 2
    second_dq[3, 3] = 1
    result = torchsky.map.coadd(
        [
            Observation(data=image, wcs=header, dq=first_dq, dq_policy=policy),
            Observation(data=3 * image, wcs=header, dq=second_dq, dq_policy=policy),
        ],
        header,
        device="cpu",
    )

    assert result.dq is not None
    assert result.usability_mask is not None
    assert int(result.dq[3, 3]) == 2
    assert bool(result.usability_mask[3, 3])
    assert float(result.data[3, 3]) == 1.0
    assert int(result.overlap_count[3, 3]) == 1
    assert int(result.context[0, 3, 3]) == 1


def test_invalid_declared_variance_is_unusable_and_never_invents_weight() -> None:
    header = _header((8, 8))
    variance = torch.ones((8, 8), dtype=torch.float32)
    variance[2, 4] = float("nan")
    observation = Observation(
        data=torch.ones((8, 8), dtype=torch.float32),
        wcs=header,
        var_poisson=variance,
    )

    result = torchsky.map.coadd([observation], header, device="cpu")

    assert result.usability_mask is not None
    assert not bool(result.usability_mask[2, 4])
    assert float(result.weight[2, 4]) == 0.0
    assert result.inverse_variance is not None
    assert float(result.inverse_variance[2, 4]) == 0.0
    assert result.dq is not None
    assert int(result.dq[2, 4]) & int(QualityBit.INVALID_VARIANCE)


def test_certified_outlier_mode_preserves_dq_and_reports_rejections() -> None:
    header = _header()
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 16),
        torch.linspace(-1.0, 1.0, 16),
        indexing="ij",
    )
    truth = 1.0 + 3.0 * torch.exp(-7.0 * (x.square() + y.square()))
    images = [truth.clone() for _ in range(4)]
    cosmic_rays = ((3, 4), (6, 11), (12, 8))
    for exposure, (row, column) in enumerate(cosmic_rays):
        images[exposure][row, column] += 100.0
    original_dq = torch.zeros_like(truth, dtype=torch.int64)
    original_dq[1, 1] = 2
    policy = DQPolicy(
        reject_bits=int(QualityBit.COSMIC_RAY | QualityBit.INVALID_VARIANCE),
        propagate_bits=None,
        cosmic_ray_bit=int(QualityBit.COSMIC_RAY),
        unknown_bit_policy="preserve",
    )
    observations = [
        Observation(
            data=image,
            wcs=header,
            dq=original_dq.clone(),
            dq_policy=policy,
            var_poisson=torch.full_like(truth, 0.2),
            var_rnoise=torch.full_like(truth, 0.05),
        )
        for image in images
    ]
    config = OutlierConfig(
        parity_mode=True,
        snr=(5.0, 3.0),
        scale=(1.2, 0.7),
        grow=1,
        dq_policy=policy,
    )

    masks, updated, logs = torchsky.map.detect_outliers(
        observations,
        config=config,
        return_logs=True,
        device="cpu",
    )

    for exposure, position in enumerate(cosmic_rays):
        assert bool(masks[exposure][position])
        assert int(updated[exposure].dq[1, 1]) == 2
        assert int(updated[exposure].dq[position]) & policy.cosmic_ray_bit
    assert not bool(masks[3].any())
    assert sum(entry["primary_count"] for entry in logs) == len(cosmic_rays)
    clean_pixels = sum(mask.numel() for mask in masks) - len(cosmic_rays)
    false_positives = sum(int(mask.sum()) for mask in masks) - len(cosmic_rays)
    assert false_positives / clean_pixels <= 1.0e-3

    result = torchsky.map.coadd(
        observations,
        header,
        resample=ResampleConfig(kernel="drizzle_square", pixfrac=1.0),
        outliers=config,
        stack=StackConfig(method="weighted_mean"),
        device="cpu",
    )

    torch.testing.assert_close(result.data, truth, atol=1.0e-4, rtol=1.0e-4)
    assert result.rejection_log
    assert result.usability_mask is not None and bool(result.usability_mask.all())
    yy, xx = torch.meshgrid(
        torch.arange(16, dtype=result.data.dtype),
        torch.arange(16, dtype=result.data.dtype),
        indexing="ij",
    )
    expected_flux = truth.sum()
    measured_flux = result.data.sum()
    torch.testing.assert_close(measured_flux, expected_flux, rtol=2.0e-3, atol=0.0)
    expected_centroid = torch.stack(
        ((truth * xx).sum() / expected_flux, (truth * yy).sum() / expected_flux)
    )
    measured_centroid = torch.stack(
        (
            (result.data * xx).sum() / measured_flux,
            (result.data * yy).sum() / measured_flux,
        )
    )
    torch.testing.assert_close(
        measured_centroid, expected_centroid, atol=1.0e-2, rtol=0.0
    )


def test_certified_outlier_mode_rejects_undeclared_noise() -> None:
    header = _header((8, 8))
    observations = [Observation(data=torch.ones((8, 8)), wcs=header) for _ in range(3)]
    with pytest.raises(ValueError, match="declared detector variance"):
        torchsky.map.detect_outliers(
            observations,
            config=OutlierConfig(parity_mode=True),
            device="cpu",
        )


def test_product_dq_roundtrip_uses_torchfits_hdu_facade(tmp_path: Path) -> None:
    header = _header((8, 8))
    dq = torch.zeros((8, 8), dtype=torch.int32)
    dq[2, 5] = 2
    policy = DQPolicy(reject_bits=1, cosmic_ray_bit=4, unknown_bit_policy="preserve")
    observation = Observation(
        data=torch.ones((8, 8)),
        wcs=header,
        dq=dq,
        dq_policy=policy,
    )
    product = torchsky.map.plan_product([observation])
    result = torchsky.map.stack_product([observation], product)
    path = tmp_path / "product.fits"

    written = write_map_product(
        path,
        result,
        requested_planes=("SCI", "WHT", "DQ", "CTX", "MASK", "NEXP"),
        overwrite=True,
    )

    assert written == ("SCI", "WHT", "DQ", "CTX", "MASK", "NEXP")
    roundtrip = torchfits.read(str(path), hdu="DQ")
    torch.testing.assert_close(roundtrip.to(torch.int64), dq.to(torch.int64))
