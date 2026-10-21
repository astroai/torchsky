from __future__ import annotations

import math

import pytest
import torch

import torchsky.sphere.pseudocl as pseudocl_module
from torchsky.sphere import alm_index, alm_size
from torchsky.sphere.pseudocl import (
    ApodizedMask,
    BandpowerBinning,
    ModeCouplingError,
    ModeCouplingWorkspace,
    PseudoClField,
    _packed_cross_spectrum,
    coupled_spectra,
    pure_alms,
    template_deprojection_bias,
)
from torchsky.sphere.sht import CARGeometry, analyze, synthesize


def _geometry(lmax: int, *, device: str = "cpu") -> CARGeometry:
    del device
    nlat = 2 * lmax + 1
    nlon = 4 * lmax + 1
    return CARGeometry.from_header(
        {
            "NAXIS1": nlon,
            "NAXIS2": nlat,
            "CTYPE1": "RA---CAR",
            "CTYPE2": "DEC--CAR",
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CRPIX1": 1.0,
            "CRPIX2": (nlat + 1.0) / 2.0,
            "CDELT1": 360.0 / nlon,
            "CDELT2": -180.0 / (nlat - 1),
        },
        spin_convention="iau",
    )


def _random_alm(
    lmax: int,
    *,
    leading: tuple[int, ...] = (),
    dtype: torch.dtype = torch.complex128,
    device: torch.device | str = "cpu",
    seed: int = 12,
) -> torch.Tensor:
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn(
        (*leading, alm_size(lmax)), generator=generator, dtype=real_dtype
    )
    imag = torch.randn(
        (*leading, alm_size(lmax)), generator=generator, dtype=real_dtype
    )
    for ell in range(lmax + 1):
        imag[..., alm_index(ell, 0, lmax)] = 0.0
    return torch.complex(real, imag).to(device=device, dtype=dtype)


def _field_from_alm(
    alm: torch.Tensor,
    geometry: CARGeometry,
    lmax: int,
    mask: torch.Tensor,
    *,
    spin: int = 0,
    beam: torch.Tensor | None = None,
    pixel_window: torch.Tensor | None = None,
) -> PseudoClField:
    values = synthesize(alm, geometry, lmax=lmax, spin=spin).values
    return PseudoClField(
        values,
        mask,
        geometry,
        lmax,
        spin=spin,
        beam=beam,
        pixel_window=pixel_window,
    )


def _wigner_3j_000(first: int, second: int, third: int) -> float:
    total = first + second + third
    if total % 2 or third < abs(first - second) or third > first + second:
        return 0.0
    half = total // 2
    ratio = math.factorial(half) / (
        math.factorial(half - first)
        * math.factorial(half - second)
        * math.factorial(half - third)
    )
    root = math.sqrt(
        math.factorial(total - 2 * first)
        * math.factorial(total - 2 * second)
        * math.factorial(total - 2 * third)
        / math.factorial(total + 1)
    )
    return (-1.0 if half % 2 else 1.0) * ratio * root


def _wigner_3j(
    first: int, second: int, third: int, m_first: int, m_second: int, m_third: int
) -> float:
    if m_first + m_second + m_third != 0:
        return 0.0
    if third < abs(first - second) or third > first + second:
        return 0.0
    if abs(m_first) > first or abs(m_second) > second or abs(m_third) > third:
        return 0.0
    triangle = (
        math.factorial(first + second - third)
        * math.factorial(first - second + third)
        * math.factorial(-first + second + third)
        / math.factorial(first + second + third + 1)
    )
    magnetic = math.prod(
        math.factorial(value)
        for value in (
            first + m_first,
            first - m_first,
            second + m_second,
            second - m_second,
            third + m_third,
            third - m_third,
        )
    )
    lower = max(
        0,
        second - third - m_first,
        first - third + m_second,
    )
    upper = min(
        first + second - third,
        first - m_first,
        second + m_second,
    )
    total = 0.0
    for index in range(lower, upper + 1):
        denominator = math.prod(
            math.factorial(value)
            for value in (
                index,
                first + second - third - index,
                first - m_first - index,
                second + m_second - index,
                third - second + m_first + index,
                third - first - m_second + index,
            )
        )
        total += (-1.0 if index % 2 else 1.0) / denominator
    sign = -1.0 if (first - second - m_third) % 2 else 1.0
    return sign * math.sqrt(triangle * magnetic) * total


def _direct_scalar_coupling(mask_cl: torch.Tensor, lmax: int) -> torch.Tensor:
    result = torch.zeros((lmax + 1, lmax + 1), dtype=mask_cl.dtype)
    for ell in range(lmax + 1):
        for ell_prime in range(lmax + 1):
            value = 0.0
            for mask_ell in range(2 * lmax + 1):
                wigner = _wigner_3j_000(ell, ell_prime, mask_ell)
                value += (2 * mask_ell + 1) * float(mask_cl[mask_ell]) * wigner * wigner
            result[ell, ell_prime] = (2 * ell_prime + 1) * value / (4.0 * math.pi)
    return result


def _direct_spin_channels(
    mask_cl: torch.Tensor, lmax: int, spins: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.zeros((lmax + 1, lmax + 1), dtype=mask_cl.dtype)
    second = torch.zeros_like(first)
    for ell in range(max(spins), lmax + 1):
        for ell_prime in range(max(spins), lmax + 1):
            for mask_ell in range(2 * lmax + 1):
                prefactor = (
                    (2 * ell_prime + 1)
                    * (2 * mask_ell + 1)
                    * float(mask_cl[mask_ell])
                    / (4.0 * math.pi)
                )
                if 0 in spins:
                    scalar = _wigner_3j(ell, ell_prime, mask_ell, 0, 0, 0)
                    spin = _wigner_3j(ell, ell_prime, mask_ell, 2, -2, 0)
                    first[ell, ell_prime] += prefactor * scalar * spin
                    second[ell, ell_prime] += prefactor * scalar * spin
                else:
                    spin = _wigner_3j(ell, ell_prime, mask_ell, 2, -2, 0)
                    first[ell, ell_prime] += prefactor * spin * spin
                    second[ell, ell_prime] += (
                        prefactor
                        * spin
                        * spin
                        * (-1.0 if (ell + ell_prime + mask_ell) % 2 else 1.0)
                    )
    return first, second


def test_bandpower_binning_roundtrip_and_validation() -> None:
    binning = BandpowerBinning.from_edges([0, 2, 5], 4, weighting="2ell+1")
    bandpowers = torch.tensor([1.25, -0.5], dtype=torch.float64)
    torch.testing.assert_close(binning.bin(binning.unbin(bandpowers)), bandpowers)
    with pytest.raises(ValueError, match="strictly increasing"):
        BandpowerBinning.from_edges([0, 2, 2], 4)
    with pytest.raises(ValueError, match="must be identity"):
        BandpowerBinning(torch.ones(2, 5), torch.ones(5, 2))


def test_full_sky_identity_auto_cross_decoupling_and_windows() -> None:
    lmax = 4
    geometry = _geometry(lmax)
    mask = torch.ones(geometry.shape, dtype=torch.float64)
    first_alm = _random_alm(lmax, seed=3)
    second_alm = _random_alm(lmax, seed=8)
    first = _field_from_alm(first_alm, geometry, lmax, mask)
    second = _field_from_alm(second_alm, geometry, lmax, mask)
    expected_auto = coupled_spectra(first)
    expected_cross = coupled_spectra(first, second)
    direct_auto = analyze(first.values, geometry, lmax=lmax).values
    direct_cross = analyze(second.values, geometry, lmax=lmax).values
    direct_auto_field = PseudoClField(first.values, mask, geometry, lmax)
    direct_cross_field = PseudoClField(second.values, mask, geometry, lmax)
    torch.testing.assert_close(
        expected_auto, coupled_spectra(direct_auto_field), atol=2e-12, rtol=2e-12
    )
    torch.testing.assert_close(
        expected_cross,
        coupled_spectra(direct_auto_field, direct_cross_field),
        atol=2e-12,
        rtol=2e-12,
    )
    torch.testing.assert_close(direct_auto, first_alm, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(direct_cross, second_alm, atol=2e-11, rtol=2e-11)
    binning = BandpowerBinning.from_edges(list(range(lmax + 2)), lmax)
    workspace = ModeCouplingWorkspace.from_fields(first, second, binning)
    theory = torch.linspace(0.2, 1.0, lmax + 1, dtype=torch.float64)
    torch.testing.assert_close(
        workspace.couple_theory(theory), theory, atol=3e-11, rtol=3e-11
    )
    torch.testing.assert_close(
        workspace.decouple(expected_cross), expected_cross, atol=4e-11, rtol=4e-11
    )
    torch.testing.assert_close(
        workspace.bandpower_windows(),
        torch.eye(lmax + 1, dtype=torch.float64),
        atol=5e-11,
        rtol=5e-11,
    )


def test_constant_mask_beam_pixel_window_and_noise_contracts() -> None:
    lmax = 4
    geometry = _geometry(lmax)
    constant = 0.37
    mask = torch.full(geometry.shape, constant, dtype=torch.float64)
    beam = torch.linspace(1.0, 0.8, lmax + 1, dtype=torch.float64)
    pixel = torch.linspace(0.95, 0.75, lmax + 1, dtype=torch.float64)
    first = _field_from_alm(
        _random_alm(lmax), geometry, lmax, mask, beam=beam, pixel_window=pixel
    )
    second = _field_from_alm(
        _random_alm(lmax, seed=15), geometry, lmax, mask, beam=beam, pixel_window=pixel
    )
    binning = BandpowerBinning.from_edges(list(range(lmax + 2)), lmax)
    workspace = ModeCouplingWorkspace.from_fields(first, second, binning)
    theory = torch.linspace(0.4, 1.2, lmax + 1, dtype=torch.float64)
    transfer = (beam * pixel).square()
    expected = constant * constant * transfer * theory
    torch.testing.assert_close(
        workspace.couple_theory(theory), expected, atol=4e-11, rtol=4e-11
    )
    noise = torch.linspace(0.01, 0.03, lmax + 1, dtype=torch.float64)
    torch.testing.assert_close(
        workspace.decouple(expected + noise, noise_bias=noise),
        theory,
        atol=5e-11,
        rtol=5e-11,
    )


def test_single_harmonic_mode_has_analytic_coupled_auto_cross_power() -> None:
    lmax = 4
    geometry = _geometry(lmax)
    first_mask_value = 0.4
    second_mask_value = 0.7
    first_alm = torch.zeros(alm_size(lmax), dtype=torch.complex128)
    second_alm = torch.zeros_like(first_alm)
    first_coefficient = torch.tensor(0.6 - 0.25j, dtype=torch.complex128)
    second_coefficient = torch.tensor(-0.2 + 0.4j, dtype=torch.complex128)
    mode = alm_index(3, 2, lmax)
    first_alm[mode] = first_coefficient
    second_alm[mode] = second_coefficient
    first = _field_from_alm(
        first_alm,
        geometry,
        lmax,
        torch.full(geometry.shape, first_mask_value, dtype=torch.float64),
    )
    second = _field_from_alm(
        second_alm,
        geometry,
        lmax,
        torch.full(geometry.shape, second_mask_value, dtype=torch.float64),
    )
    expected_auto = torch.zeros(lmax + 1, dtype=torch.float64)
    expected_cross = torch.zeros_like(expected_auto)
    expected_auto[3] = first_mask_value**2 * 2.0 * abs(first_coefficient) ** 2 / 7.0
    expected_cross[3] = (
        first_mask_value
        * second_mask_value
        * 2.0
        * torch.real(first_coefficient * torch.conj(second_coefficient))
        / 7.0
    )
    torch.testing.assert_close(
        coupled_spectra(first), expected_auto, atol=3e-12, rtol=3e-12
    )
    torch.testing.assert_close(
        coupled_spectra(first, second), expected_cross, atol=3e-12, rtol=3e-12
    )


def test_low_l_correlation_action_matches_direct_wigner_3j_oracle() -> None:
    lmax = 4
    geometry = _geometry(lmax)
    mask_lmax = 2 * lmax
    mask_alm = torch.zeros(alm_size(mask_lmax), dtype=torch.complex128)
    mask_alm[alm_index(0, 0, mask_lmax)] = math.sqrt(4.0 * math.pi) * 0.65
    mask_alm[alm_index(1, 0, mask_lmax)] = 0.17
    mask_alm[alm_index(2, 1, mask_lmax)] = 0.08 - 0.04j
    mask = synthesize(mask_alm, geometry, lmax=mask_lmax).values
    field = PseudoClField(
        torch.zeros(geometry.shape, dtype=torch.float64), mask, geometry, lmax
    )
    binning = BandpowerBinning.from_edges([0, 2, 5], lmax)
    workspace = ModeCouplingWorkspace.from_fields(field, None, binning)
    basis = torch.eye(lmax + 1, dtype=torch.float64)
    correlation_matrix = workspace.couple_theory(basis).transpose(0, 1)
    oracle = _direct_scalar_coupling(workspace.mask_spectrum.cpu(), lmax)
    torch.testing.assert_close(correlation_matrix, oracle, atol=4e-11, rtol=4e-11)


def test_spin_coupled_spectra_have_eb_block_ordering() -> None:
    lmax = 3
    geometry = _geometry(lmax)
    mask = torch.ones(geometry.shape, dtype=torch.float64)
    scalar_alm = _random_alm(lmax, seed=5)
    spin_alm = _random_alm(lmax, leading=(2,), seed=6)
    for component in range(2):
        for ell in range(2):
            for m in range(ell + 1):
                spin_alm[component, alm_index(ell, m, lmax)] = 0.0
    scalar = _field_from_alm(scalar_alm, geometry, lmax, mask)
    spin = _field_from_alm(spin_alm, geometry, lmax, mask, spin=2)
    scalar_spin = coupled_spectra(scalar, spin)
    spin_spin = coupled_spectra(spin)
    assert scalar_spin.shape == (2, lmax + 1)
    assert spin_spin.shape == (4, lmax + 1)
    recovered_spin = analyze(spin.values, geometry, lmax=lmax, spin=2).values
    expected_scalar_spin = torch.stack(
        [
            coupled_spectra(
                scalar,
                _field_from_alm(recovered_spin[index], geometry, lmax, mask),
            )
            for index in range(2)
        ]
    )
    torch.testing.assert_close(
        scalar_spin, expected_scalar_spin, atol=3e-11, rtol=3e-11
    )
    torch.testing.assert_close(spin_spin[1], spin_spin[2], atol=3e-11, rtol=3e-11)
    spin_binning = BandpowerBinning.from_edges([2, 4], lmax)
    scalar_spin_workspace = ModeCouplingWorkspace.from_fields(
        scalar, spin, spin_binning
    )
    spin_workspace = ModeCouplingWorkspace.from_fields(spin, None, spin_binning)
    mixed_theory = torch.randn(2, lmax + 1, dtype=torch.float64)
    spin_theory = torch.randn(4, lmax + 1, dtype=torch.float64)
    mixed_theory[:, :2] = 0.0
    spin_theory[:, :2] = 0.0
    torch.testing.assert_close(
        scalar_spin_workspace.couple_theory(mixed_theory),
        mixed_theory,
        atol=5e-11,
        rtol=5e-11,
    )
    torch.testing.assert_close(
        spin_workspace.couple_theory(spin_theory), spin_theory, atol=5e-11, rtol=5e-11
    )


def test_low_l_spin_blocks_match_direct_wigner_oracles() -> None:
    lmax = 4
    geometry = _geometry(lmax)
    mask_lmax = 2 * lmax
    mask_alm = torch.zeros(alm_size(mask_lmax), dtype=torch.complex128)
    mask_alm[alm_index(0, 0, mask_lmax)] = math.sqrt(4.0 * math.pi) * 0.7
    mask_alm[alm_index(2, 0, mask_lmax)] = 0.12
    mask = synthesize(mask_alm, geometry, lmax=mask_lmax).values
    scalar = PseudoClField(torch.zeros_like(mask), mask, geometry, lmax)
    spin_values = torch.zeros((2, *geometry.shape), dtype=torch.float64)
    spin = PseudoClField(spin_values, mask, geometry, lmax, spin=2)
    binning = BandpowerBinning.from_edges([2, 3, 4, 5], lmax)
    mixed_workspace = ModeCouplingWorkspace.from_fields(scalar, spin, binning)
    spin_workspace = ModeCouplingWorkspace.from_fields(spin, None, binning)
    mixed_first, _ = _direct_spin_channels(
        mixed_workspace.mask_spectrum.cpu(), lmax, (0, 2)
    )
    spin_plus, spin_minus = _direct_spin_channels(
        spin_workspace.mask_spectrum.cpu(), lmax, (2, 2)
    )
    basis_mixed = torch.eye(2 * (lmax + 1), dtype=torch.float64).reshape(
        2 * (lmax + 1), 2, lmax + 1
    )
    mixed_action = (
        mixed_workspace.couple_theory(basis_mixed)
        .reshape(2 * (lmax + 1), 2 * (lmax + 1))
        .transpose(0, 1)
    )
    mixed_zero = torch.zeros_like(mixed_first)
    mixed_oracle = torch.cat(
        (
            torch.cat((mixed_first, mixed_zero), dim=1),
            torch.cat((mixed_zero, mixed_first), dim=1),
        ),
        dim=0,
    )
    torch.testing.assert_close(mixed_action, mixed_oracle, atol=8e-11, rtol=8e-11)
    block_basis = torch.eye(4 * (lmax + 1), dtype=torch.float64).reshape(
        4 * (lmax + 1), 4, lmax + 1
    )
    block_action = (
        spin_workspace.couple_theory(block_basis)
        .reshape(4 * (lmax + 1), 4 * (lmax + 1))
        .transpose(0, 1)
    )
    m_plus = 0.5 * (spin_plus + spin_minus)
    m_minus = 0.5 * (spin_plus - spin_minus)
    zero = torch.zeros_like(m_plus)
    oracle = torch.cat(
        (
            torch.cat((m_plus, zero, zero, m_minus), dim=1),
            torch.cat((zero, m_plus, -m_minus, zero), dim=1),
            torch.cat((zero, -m_minus, m_plus, zero), dim=1),
            torch.cat((m_minus, zero, zero, m_plus), dim=1),
        ),
        dim=0,
    )
    torch.testing.assert_close(block_action, oracle, atol=8e-11, rtol=8e-11)


def test_spin_transfer_noise_and_deprojection_bias_all_blocks() -> None:
    lmax = 3
    geometry = _geometry(lmax)
    mask = torch.ones(geometry.shape, dtype=torch.float64)
    beam_a = torch.linspace(1.0, 0.8, lmax + 1, dtype=torch.float64)
    beam_b = torch.linspace(0.9, 0.7, lmax + 1, dtype=torch.float64)
    pixel_a = torch.linspace(0.95, 0.85, lmax + 1, dtype=torch.float64)
    pixel_b = torch.linspace(0.92, 0.82, lmax + 1, dtype=torch.float64)
    first_alm = _random_alm(lmax, leading=(2,), seed=31)
    second_alm = _random_alm(lmax, leading=(2,), seed=32)
    first_alm[..., :2] = 0.0
    second_alm[..., :2] = 0.0
    first = _field_from_alm(
        first_alm,
        geometry,
        lmax,
        mask,
        spin=2,
        beam=beam_a,
        pixel_window=pixel_a,
    )
    second = _field_from_alm(
        second_alm,
        geometry,
        lmax,
        mask,
        spin=2,
        beam=beam_b,
        pixel_window=pixel_b,
    )
    binning = BandpowerBinning.from_edges([2, 3, 4], lmax)
    workspace = ModeCouplingWorkspace.from_fields(first, second, binning)
    theory = torch.randn(4, lmax + 1, dtype=torch.float64)
    theory[:, :2] = 0.0
    transfer = beam_a * pixel_a * beam_b * pixel_b
    torch.testing.assert_close(
        workspace.couple_theory(theory), theory * transfer, atol=6e-11, rtol=6e-11
    )
    noise = torch.rand(4, lmax + 1, dtype=torch.float64)
    noise[:, :2] = 0.0
    coupled_noise = workspace.couple_noise(noise)
    deprojection = torch.full_like(coupled_noise, 0.013)
    observed = workspace.couple_theory(theory) + coupled_noise + deprojection
    torch.testing.assert_close(
        workspace.decouple(
            observed,
            noise_bias=coupled_noise,
            deprojection_bias=deprojection,
        ),
        binning.bin(theory),
        atol=8e-11,
        rtol=8e-11,
    )

    template = first.values.detach()
    contaminated = 2.5 * template
    template_field = PseudoClField(
        contaminated,
        mask,
        geometry,
        lmax,
        spin=2,
        templates=template.unsqueeze(0),
    )
    torch.testing.assert_close(
        template_field.deprojected_values(),
        torch.zeros_like(contaminated),
        atol=2e-12,
        rtol=2e-12,
    )
    simulations = synthesize(
        _random_alm(lmax, leading=(24, 2), seed=33), geometry, lmax=lmax, spin=2
    ).values
    raw_field = PseudoClField(simulations, mask, geometry, lmax, spin=2)
    bias = template_deprojection_bias(template_field, simulations)
    cleaned_field = PseudoClField(
        simulations,
        mask,
        geometry,
        lmax,
        spin=2,
        templates=template.unsqueeze(0),
    )
    torch.testing.assert_close(
        (coupled_spectra(cleaned_field) - bias).mean(0),
        coupled_spectra(raw_field).mean(0),
        atol=3e-11,
        rtol=3e-11,
    )


def test_pure_eb_derivative_mask_and_b_leakage() -> None:
    lmax = 5
    geometry = _geometry(lmax)
    latitudes = geometry.latitudes(dtype=torch.float64, device="cpu")
    mask = torch.cos(latitudes).square().unsqueeze(1).expand(geometry.shape).clone()
    apodized = ApodizedMask.from_mask(mask, geometry, lmax)
    e_alm = _random_alm(lmax, seed=41)
    b_alm = torch.zeros_like(e_alm)
    for ell in range(2):
        for m in range(ell + 1):
            e_alm[alm_index(ell, m, lmax)] = 0.0
    values = synthesize(torch.stack((e_alm, b_alm)), geometry, lmax=lmax, spin=2).values
    standard = PseudoClField(values, mask, geometry, lmax, spin=2)
    purified = PseudoClField(
        values,
        mask,
        geometry,
        lmax,
        spin=2,
        apodized_mask=apodized,
        purify_e=True,
        purify_b=True,
    )
    standard_b = coupled_spectra(standard)[3, 2:].sum()
    pure_coefficients = pure_alms(purified)
    pure_b = _packed_cross_spectrum(pure_coefficients[1], pure_coefficients[1], lmax)[
        2:
    ].sum()
    assert float(pure_b) < 0.05 * float(standard_b)
    torch.testing.assert_close(
        coupled_spectra(purified)[3, 2:].sum(), pure_b, atol=2e-12, rtol=2e-12
    )
    invalid = mask.clone()
    invalid[0] = 0.1
    with pytest.raises(ValueError, match="band-limited|boundary"):
        ApodizedMask.from_mask(invalid, geometry, lmax)


def test_monte_carlo_decoupling_recovers_scalar_theory() -> None:
    lmax = 3
    sample_count = 384
    geometry = _geometry(lmax)
    latitudes = geometry.latitudes(dtype=torch.float64, device="cpu")
    mask = (0.72 + 0.18 * torch.sin(latitudes)).unsqueeze(1).expand(geometry.shape)
    theory = torch.tensor([0.8, 0.65, 0.45, 0.3], dtype=torch.float64)
    alms = _random_alm(lmax, leading=(sample_count,), seed=19)
    for m in range(lmax + 1):
        for ell in range(m, lmax + 1):
            index = alm_index(ell, m, lmax)
            scale = (
                math.sqrt(float(theory[ell]))
                if m == 0
                else math.sqrt(float(theory[ell]) / 2.0)
            )
            alms[:, index] = alms[:, index] * scale
    maps = synthesize(alms, geometry, lmax=lmax).values
    field = PseudoClField(maps, mask, geometry, lmax)
    binning = BandpowerBinning.from_edges(list(range(lmax + 2)), lmax)
    workspace = ModeCouplingWorkspace.from_fields(field, None, binning)
    estimate = workspace.decouple(coupled_spectra(field)).mean(0)
    torch.testing.assert_close(estimate, theory, atol=0.16, rtol=0.16)


def test_monte_carlo_spin_recovery_and_eb_leakage_decoupling() -> None:
    lmax = 3
    sample_count = 320
    geometry = _geometry(lmax)
    latitudes = geometry.latitudes(dtype=torch.float64, device="cpu")
    mask = (0.68 + 0.2 * torch.cos(2.0 * latitudes)).unsqueeze(1).expand(geometry.shape)
    theory = torch.zeros(4, lmax + 1, dtype=torch.float64)
    theory[0, 2:] = torch.tensor([0.7, 0.5], dtype=torch.float64)
    theory[3, 2:] = torch.tensor([0.25, 0.18], dtype=torch.float64)
    alms = _random_alm(lmax, leading=(sample_count, 2), seed=51)
    for component, block in ((0, 0), (1, 3)):
        for m in range(lmax + 1):
            for ell in range(m, lmax + 1):
                index = alm_index(ell, m, lmax)
                scale = 0.0
                if ell >= 2:
                    scale = (
                        math.sqrt(float(theory[block, ell]))
                        if m == 0
                        else math.sqrt(float(theory[block, ell]) / 2.0)
                    )
                alms[:, component, index] = alms[:, component, index] * scale
    maps = synthesize(alms, geometry, lmax=lmax, spin=2).values
    field = PseudoClField(maps, mask, geometry, lmax, spin=2)
    binning = BandpowerBinning.from_edges([2, 3, 4], lmax)
    workspace = ModeCouplingWorkspace.from_fields(field, None, binning)
    estimate = workspace.decouple(coupled_spectra(field)).mean(0)
    torch.testing.assert_close(
        estimate[[0, 3]], theory[[0, 3], 2:], atol=0.14, rtol=0.2
    )
    torch.testing.assert_close(
        estimate[[1, 2]], torch.zeros_like(estimate[[1, 2]]), atol=0.09, rtol=0.0
    )


def test_coupling_adjoint_gradcheck_compile_and_memory_structure() -> None:
    lmax = 4
    geometry = _geometry(lmax)
    mask = torch.ones(geometry.shape, dtype=torch.float64)
    field = _field_from_alm(_random_alm(lmax), geometry, lmax, mask)
    binning = BandpowerBinning.from_edges([0, 2, 5], lmax)
    workspace = ModeCouplingWorkspace.from_fields(
        field, None, binning, node_block_size=2
    )
    theory = torch.linspace(0.3, 1.1, lmax + 1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(workspace.couple_theory, (theory,))
    probe = torch.linspace(-0.4, 0.6, lmax + 1, dtype=torch.float64)
    coupled = workspace.couple_theory(theory)
    adjoint = torch.autograd.grad(coupled, theory, probe, create_graph=True)[0]
    torch.testing.assert_close(torch.dot(coupled, probe), torch.dot(theory, adjoint))
    differentiable_values = field.values.detach().requires_grad_(True)
    differentiable_mask = torch.ones(
        geometry.shape, dtype=torch.float64, requires_grad=True
    )
    differentiable_field = PseudoClField(
        differentiable_values, differentiable_mask, geometry, lmax
    )
    (map_gradient,) = torch.autograd.grad(
        coupled_spectra(differentiable_field).sum(), differentiable_values
    )
    assert torch.all(torch.isfinite(map_gradient))
    differentiable_workspace = ModeCouplingWorkspace.from_fields(
        differentiable_field, None, binning
    )
    (mask_gradient,) = torch.autograd.grad(
        differentiable_workspace.couple_theory(theory).sum(), differentiable_mask
    )
    assert torch.all(torch.isfinite(mask_gradient))
    compiled = torch.compile(workspace.couple_theory, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(theory), workspace.couple_theory(theory))
    windows = workspace.bandpower_windows()
    torch.testing.assert_close(
        windows @ binning.unbinning_matrix,
        torch.eye(binning.nband, dtype=torch.float64),
        atol=6e-11,
        rtol=6e-11,
    )
    retained = [
        value
        for value in (
            workspace.mask_spectrum,
            workspace.transfer,
            workspace.quadrature_nodes,
            workspace.quadrature_weights,
            workspace.mask_correlation,
            workspace.coupling_matrix,
        )
    ]
    assert max(value.numel() for value in retained) <= (2 * lmax + 1)


def test_spin_block_adjoint_gradcheck_compile_windows_and_memory() -> None:
    lmax = 3
    geometry = _geometry(lmax)
    mask = torch.ones(geometry.shape, dtype=torch.float64)
    alms = _random_alm(lmax, leading=(2,), seed=61)
    alms[..., :2] = 0.0
    field = _field_from_alm(alms, geometry, lmax, mask, spin=2)
    binning = BandpowerBinning.from_edges([2, 3, 4], lmax)
    workspace = ModeCouplingWorkspace.from_fields(
        field, None, binning, node_block_size=2
    )
    theory = torch.randn(4, lmax + 1, dtype=torch.float64)
    theory[:, :2] = 0.0
    theory.requires_grad_(True)
    assert torch.autograd.gradcheck(workspace.couple_theory, (theory,))
    probe = torch.randn_like(theory)
    probe[:, :2] = 0.0
    coupled = workspace.couple_theory(theory)
    (adjoint,) = torch.autograd.grad(coupled, theory, probe, create_graph=True)
    torch.testing.assert_close(torch.sum(coupled * probe), torch.sum(theory * adjoint))
    compiled = torch.compile(workspace.couple_theory, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(theory), coupled)
    windows = workspace.bandpower_windows()
    unbinning = torch.zeros(4, lmax + 1, 4, binning.nband, dtype=torch.float64)
    for component in range(4):
        unbinning[component, :, component, :] = binning.unbinning_matrix
    identity = torch.einsum("obic,icjd->objd", windows, unbinning)
    torch.testing.assert_close(
        identity.reshape(4 * binning.nband, 4 * binning.nband),
        torch.eye(4 * binning.nband, dtype=torch.float64),
        atol=8e-11,
        rtol=8e-11,
    )
    retained = (
        workspace.mask_spectrum,
        workspace.transfer,
        workspace.quadrature_nodes,
        workspace.quadrature_weights,
        workspace.mask_correlation,
        workspace.coupling_matrix,
    )
    assert max(value.numel() for value in retained) <= (4 * binning.nband) ** 2


def test_high_l_spin_coupling_uses_node_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    lmax = 64
    node_block_size = 5
    nodes, weights = pseudocl_module._gauss_legendre(
        2 * lmax + 1, dtype=torch.float64, device=torch.device("cpu")
    )
    correlation = 0.7 + 0.1 * nodes.square()
    transfer = torch.linspace(1.0, 0.75, lmax + 1, dtype=torch.float64)
    theory = torch.zeros(4, lmax + 1, dtype=torch.float64)
    theory[:, 2:] = 1.0
    maximum_basis_size = 0
    original = pseudocl_module._basis_values

    def recorded(
        lmax_value: int, block_nodes: torch.Tensor, channel: str
    ) -> torch.Tensor:
        nonlocal maximum_basis_size
        result = original(lmax_value, block_nodes, channel)
        maximum_basis_size = max(maximum_basis_size, result.numel())
        return result

    monkeypatch.setattr(pseudocl_module, "_basis_values", recorded)
    result = pseudocl_module._coupling_action(
        theory,
        nodes,
        weights,
        correlation,
        transfer,
        node_block_size,
        (2, 2),
    )
    assert result.shape == theory.shape
    assert torch.all(torch.isfinite(result))
    assert maximum_basis_size <= (lmax + 1) * node_block_size


def test_rank_and_condition_diagnostics_fail_closed() -> None:
    lmax = 3
    geometry = _geometry(lmax)
    mask = torch.ones(geometry.shape, dtype=torch.float64)
    beam = torch.ones(lmax + 1, dtype=torch.float64)
    beam[2:] = 0.0
    field = _field_from_alm(_random_alm(lmax), geometry, lmax, mask, beam=beam)
    binning = BandpowerBinning.from_edges(list(range(lmax + 2)), lmax)
    with pytest.raises(ModeCouplingError, match="rank deficient"):
        ModeCouplingWorkspace.from_fields(field, None, binning)
    healthy_beam = torch.linspace(1.0, 0.7, lmax + 1, dtype=torch.float64)
    healthy = _field_from_alm(
        _random_alm(lmax), geometry, lmax, mask, beam=healthy_beam
    )
    with pytest.raises(ModeCouplingError, match="condition number"):
        ModeCouplingWorkspace.from_fields(
            healthy, None, binning, maximum_condition_number=1.0
        )


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_pseudocl_device_dtype_contract(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    lmax = 2
    geometry = _geometry(lmax)
    mask = torch.ones(geometry.shape, dtype=torch.float32, device=device)
    alm = _random_alm(lmax, dtype=torch.complex64, device=device)
    field = _field_from_alm(alm, geometry, lmax, mask)
    binning = BandpowerBinning.from_edges(
        list(range(lmax + 2)), lmax, dtype=torch.float32, device=device
    )
    workspace = ModeCouplingWorkspace.from_fields(field, None, binning)
    coupled = coupled_spectra(field)
    decoupled = workspace.decouple(coupled)
    assert coupled.device.type == device
    assert coupled.dtype == torch.float32
    assert decoupled.device.type == device
    assert decoupled.dtype == torch.float32
    spin_alm = _random_alm(lmax, leading=(2,), dtype=torch.complex64, device=device)
    spin_alm[..., :2] = 0.0
    spin_field = _field_from_alm(spin_alm, geometry, lmax, mask, spin=2)
    spin_binning = BandpowerBinning.from_edges(
        [2, 3], lmax, dtype=torch.float32, device=device
    )
    spin_workspace = ModeCouplingWorkspace.from_fields(spin_field, None, spin_binning)
    spin_coupled = coupled_spectra(spin_field)
    spin_decoupled = spin_workspace.decouple(spin_coupled)
    assert spin_coupled.device.type == device
    assert spin_coupled.dtype == torch.float32
    assert spin_decoupled.device.type == device
    assert spin_decoupled.dtype == torch.float32
