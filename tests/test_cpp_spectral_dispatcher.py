from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake

from torchsky import _ops
from torchsky._native import cpp as _cpp
from torchsky.sphere import spectral


MIGRATED = (
    "spectral_scalar_alm2map_direct",
    "spectral_ring_fourier_modes",
    "spectral_ring_fourier_modes_spin_conj",
    "spectral_ring_fourier_synthesis",
    "spectral_spin_ring_finalize",
    "spectral_spin_map2alm_ring_concat",
    "spectral_spin_interpolate_concat",
    "spectral_spin_integrate_concat",
    "spectral_spin_integrate_recurrence",
    "spectral_spin_interpolate_recurrence",
    "spectral_spin_map2alm_ring_recurrence",
    "spectral_spin_alm2map_ring_recurrence",
    "spectral_scalar_integrate_recurrence",
    "spectral_scalar_interpolate_recurrence",
)


def _inputs() -> dict[str, torch.Tensor | int | float]:
    torch.manual_seed(4)
    nside = 1
    lmax = 2
    mmax = 2
    spin = 1
    nalm = spectral.alm_size(lmax, mmax)
    starts, lengths, theta, _ = spectral._ring_layout_for_nside(nside)
    npix = 12
    phase_neg = spectral._ring_phi0_phase_for_nside(nside, mmax, sign=-1)
    phase_pos = spectral._ring_phi0_phase_for_nside(nside, mmax, sign=1)
    y_plus, y_minus = spectral._ring_spin_basis_concat(nside, lmax, mmax, spin)
    return {
        "lmax": lmax,
        "mmax": mmax,
        "spin": spin,
        "nalm": nalm,
        "npix": npix,
        "starts": starts,
        "lengths": lengths,
        "theta": theta,
        "weight": torch.rand(theta.shape, dtype=torch.float64),
        "phase_neg": phase_neg,
        "phase_pos": phase_pos,
        "y_plus": y_plus,
        "y_minus": y_minus,
        "rows": torch.randn((2, npix), dtype=torch.complex128),
        "p_plus": torch.randn(npix, dtype=torch.complex128),
        "modes": torch.randn((mmax + 1, theta.numel()), dtype=torch.complex128),
        "coeff_plus": torch.randn(nalm, dtype=torch.complex128),
        "coeff_minus": torch.randn(nalm, dtype=torch.complex128),
        "maps": torch.randn((2, npix), dtype=torch.float64),
    }


def _basis_analysis_reference(
    q: torch.Tensor,
    u: torch.Tensor,
    ycp_t: torch.Tensor,
    ycm_t: torch.Tensor,
    pixel_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    polarization = torch.complex(q, u)
    return (
        polarization @ ycp_t * pixel_weight,
        polarization.conj() @ ycm_t * pixel_weight,
    )


def _basis_synthesis_reference(
    coeff_plus: torch.Tensor,
    coeff_minus: torch.Tensor,
    y_plus: torch.Tensor,
    y_minus: torch.Tensor,
    y_plus_m0: torch.Tensor,
    y_minus_m0: torch.Tensor,
) -> torch.Tensor:
    p_plus_pos = coeff_plus @ y_plus
    p_minus_pos = coeff_minus @ y_minus
    if y_plus_m0.shape[0] == coeff_plus.shape[0]:
        p_plus = p_plus_pos
        p_minus = p_minus_pos
    else:
        p_plus_m0 = coeff_plus[: y_plus_m0.shape[0]] @ y_plus_m0
        p_minus_m0 = coeff_minus[: y_minus_m0.shape[0]] @ y_minus_m0
        p_plus = p_plus_pos + (p_minus_pos - p_minus_m0).conj()
        p_minus = p_minus_pos + (p_plus_pos - p_plus_m0).conj()
    return torch.stack(
        (-0.5 * (p_plus + p_minus).real, -0.5 * (p_plus - p_minus).imag)
    )


def test_spectral_dispatcher_schemas_are_clean_and_basis_schemas_are_private() -> None:
    assert _ops.has_spectral_dispatcher()
    for name in MIGRATED:
        schema = str(getattr(torch.ops.torchsky, name).default._schema)
        assert schema.startswith(f"torchsky::{name}(")
        assert "_cpu" not in schema
    for name in (
        "_spectral_spin_map2alm_from_basis",
        "_spectral_spin_alm2map_from_basis",
    ):
        schema = str(getattr(torch.ops.torchsky, name).default._schema)
        assert schema.startswith(f"torchsky::{name}(")
        assert torch._C._dispatch_has_kernel_for_dispatch_key(f"torchsky::{name}", "CPU")
    assert hasattr(_cpp, "_healpix_spin_map2alm_from_basis_cpu")
    assert hasattr(_cpp, "_healpix_spin_alm2map_from_basis_cpu")


def test_private_basis_dispatcher_parity_and_compatibility() -> None:
    torch.manual_seed(12)
    npix, nalm, m0 = 7, 5, 2
    q = torch.randn(npix, dtype=torch.float64)
    u = torch.randn(npix, dtype=torch.float64)
    ycp_t = torch.randn((npix, nalm), dtype=torch.complex128)
    ycm_t = torch.randn((npix, nalm), dtype=torch.complex128)
    coeff_plus = torch.randn(nalm, dtype=torch.complex128)
    coeff_minus = torch.randn(nalm, dtype=torch.complex128)
    y_plus = torch.randn((nalm, npix), dtype=torch.complex128)
    y_minus = torch.randn((nalm, npix), dtype=torch.complex128)
    y_plus_m0 = torch.randn((m0, npix), dtype=torch.complex128)
    y_minus_m0 = torch.randn((m0, npix), dtype=torch.complex128)

    analysis = torch.ops.torchsky._spectral_spin_map2alm_from_basis(
        q, u, ycp_t, ycm_t, 0.3
    )
    torch.testing.assert_close(
        analysis, _basis_analysis_reference(q, u, ycp_t, ycm_t, 0.3)
    )
    synthesis = torch.ops.torchsky._spectral_spin_alm2map_from_basis(
        coeff_plus, coeff_minus, y_plus, y_minus, y_plus_m0, y_minus_m0
    )
    torch.testing.assert_close(
        synthesis,
        _basis_synthesis_reference(
            coeff_plus, coeff_minus, y_plus, y_minus, y_plus_m0, y_minus_m0
        ),
    )
    with pytest.warns(DeprecationWarning, match="torch.ops.torchsky"):
        compatibility = _cpp._healpix_spin_alm2map_from_basis_cpu(
            coeff_plus, coeff_minus, y_plus, y_minus, y_plus_m0, y_minus_m0
        )
    torch.testing.assert_close(compatibility, synthesis)


def test_private_basis_dispatcher_autograd_matches_tensor_adjoint() -> None:
    torch.manual_seed(13)
    npix, nalm, m0 = 6, 4, 2
    q = torch.randn(npix, dtype=torch.float64, requires_grad=True)
    u = torch.randn(npix, dtype=torch.float64, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_()
    u_ref = u.detach().clone().requires_grad_()
    ycp_t = torch.randn((npix, nalm), dtype=torch.complex128)
    ycm_t = torch.randn((npix, nalm), dtype=torch.complex128)
    probe_plus = torch.randn(nalm, dtype=torch.complex128)
    probe_minus = torch.randn(nalm, dtype=torch.complex128)
    actual = torch.ops.torchsky._spectral_spin_map2alm_from_basis(
        q, u, ycp_t, ycm_t, 0.4
    )
    expected = _basis_analysis_reference(q_ref, u_ref, ycp_t, ycm_t, 0.4)
    torch.real(torch.vdot(probe_plus, actual[0]) + torch.vdot(probe_minus, actual[1])).backward()
    torch.real(
        torch.vdot(probe_plus, expected[0]) + torch.vdot(probe_minus, expected[1])
    ).backward()
    torch.testing.assert_close(q.grad, q_ref.grad)
    torch.testing.assert_close(u.grad, u_ref.grad)

    coeff_plus = torch.randn(nalm, dtype=torch.complex128, requires_grad=True)
    coeff_minus = torch.randn(nalm, dtype=torch.complex128, requires_grad=True)
    coeff_plus_ref = coeff_plus.detach().clone().requires_grad_()
    coeff_minus_ref = coeff_minus.detach().clone().requires_grad_()
    y_plus = torch.randn((nalm, npix), dtype=torch.complex128)
    y_minus = torch.randn((nalm, npix), dtype=torch.complex128)
    y_plus_m0 = torch.randn((m0, npix), dtype=torch.complex128)
    y_minus_m0 = torch.randn((m0, npix), dtype=torch.complex128)
    map_probe = torch.randn((2, npix), dtype=torch.float64)
    actual_map = torch.ops.torchsky._spectral_spin_alm2map_from_basis(
        coeff_plus, coeff_minus, y_plus, y_minus, y_plus_m0, y_minus_m0
    )
    expected_map = _basis_synthesis_reference(
        coeff_plus_ref,
        coeff_minus_ref,
        y_plus,
        y_minus,
        y_plus_m0,
        y_minus_m0,
    )
    (actual_map * map_probe).sum().backward()
    (expected_map * map_probe).sum().backward()
    torch.testing.assert_close(coeff_plus.grad, coeff_plus_ref.grad)
    torch.testing.assert_close(coeff_minus.grad, coeff_minus_ref.grad)


def test_spectral_dispatcher_parity_with_legacy_adapters() -> None:
    x = _inputs()
    lmax, mmax, spin = x["lmax"], x["mmax"], x["spin"]
    starts, lengths, theta = x["starts"], x["lengths"], x["theta"]
    rows, modes = x["rows"], x["modes"]
    cp, cm = x["coeff_plus"], x["coeff_minus"]
    yp, ym = x["y_plus"], x["y_minus"]
    weight = x["weight"]
    assert isinstance(lmax, int) and isinstance(mmax, int) and isinstance(spin, int)
    assert isinstance(starts, torch.Tensor) and isinstance(lengths, torch.Tensor)
    assert isinstance(theta, torch.Tensor) and isinstance(rows, torch.Tensor)
    assert isinstance(modes, torch.Tensor) and isinstance(cp, torch.Tensor)
    assert isinstance(cm, torch.Tensor) and isinstance(yp, torch.Tensor)
    assert isinstance(ym, torch.Tensor) and isinstance(weight, torch.Tensor)

    cos_theta = torch.cos(theta)
    phi = torch.zeros_like(theta)
    direct = torch.ops.torchsky.spectral_scalar_alm2map_direct(
        cp.unsqueeze(0), cos_theta, phi, lmax, mmax
    )
    torch.testing.assert_close(
        direct,
        _cpp._healpix_scalar_alm2map_direct_cpu(cp.unsqueeze(0), cos_theta, phi, lmax, mmax),
    )

    fourier = torch.ops.torchsky.spectral_ring_fourier_modes(rows, starts, lengths, mmax)
    torch.testing.assert_close(
        fourier, _cpp._healpix_ring_fourier_modes_cpu(rows, starts, lengths, mmax)
    )
    p_plus = x["p_plus"]
    assert isinstance(p_plus, torch.Tensor)
    spin_modes = torch.ops.torchsky.spectral_ring_fourier_modes_spin_conj(
        p_plus, starts, lengths, mmax
    )
    legacy_spin_modes = _cpp._healpix_ring_fourier_modes_spin_conj_cpu(
        p_plus, starts, lengths, mmax
    )
    torch.testing.assert_close(torch.stack(spin_modes), legacy_spin_modes)
    torch.testing.assert_close(
        torch.ops.torchsky.spectral_ring_fourier_synthesis(
            fourier, starts, lengths, x["npix"]
        ),
        _cpp._healpix_ring_fourier_synthesis_cpu(fourier, starts, lengths),
    )

    interp = torch.ops.torchsky.spectral_spin_interpolate_concat(
        cp, cm, yp, ym, lmax, mmax
    )
    legacy_interp = _cpp._healpix_spin_interpolate_concat_cpu(cp, cm, yp, ym, lmax, mmax)
    torch.testing.assert_close(interp, legacy_interp)
    integrated = torch.ops.torchsky.spectral_spin_integrate_concat(
        interp[0], interp[1], yp, ym, lmax, mmax, 0.25
    )
    legacy_integrated = _cpp._healpix_spin_integrate_concat_cpu(
        interp[0], interp[1], yp, ym, lmax, mmax, 0.25
    )
    torch.testing.assert_close(integrated, legacy_integrated)

    phase_neg, phase_pos = x["phase_neg"], x["phase_pos"]
    assert isinstance(phase_neg, torch.Tensor) and isinstance(phase_pos, torch.Tensor)
    ring_concat = torch.ops.torchsky.spectral_spin_map2alm_ring_concat(
        p_plus, starts, lengths, phase_neg, yp, ym, lmax, mmax, 0.25
    )
    legacy_ring_concat = _cpp._healpix_spin_map2alm_ring_concat_cpu(
        p_plus, starts, lengths, phase_neg, yp, ym, lmax, mmax, 0.25
    )
    torch.testing.assert_close(ring_concat, legacy_ring_concat)

    modes_minus = modes.conj().resolve_conj()
    spin_integrated = torch.ops.torchsky.spectral_spin_integrate_recurrence(
        modes, modes_minus, theta, weight, lmax, mmax, spin
    )
    torch.testing.assert_close(
        spin_integrated,
        _cpp._healpix_spin_integrate_recurrence_cpu(
            modes, modes_minus, theta, weight, lmax, mmax, spin
        ),
    )
    torch.testing.assert_close(
        torch.ops.torchsky.spectral_spin_interpolate_recurrence(
            spin_integrated, theta, lmax, mmax, spin
        ),
        _cpp._healpix_spin_interpolate_recurrence_cpu(
            spin_integrated, theta, lmax, mmax, spin
        ),
    )
    scalar_integrated = torch.ops.torchsky.spectral_scalar_integrate_recurrence(
        modes, theta, weight, lmax, mmax
    )
    torch.testing.assert_close(
        scalar_integrated,
        _cpp._healpix_scalar_integrate_recurrence_cpu(modes, theta, weight, lmax, mmax),
    )
    torch.testing.assert_close(
        torch.ops.torchsky.spectral_scalar_interpolate_recurrence(
            scalar_integrated, theta, lmax, mmax
        ),
        _cpp._healpix_scalar_interpolate_recurrence_cpu(
            scalar_integrated, theta, lmax, mmax
        ),
    )

    maps = x["maps"]
    assert isinstance(maps, torch.Tensor)
    fused_analysis = torch.ops.torchsky.spectral_spin_map2alm_ring_recurrence(
        maps, starts, lengths, theta, phase_neg, lmax, mmax, spin, 0.25
    )
    torch.testing.assert_close(
        fused_analysis,
        _cpp._healpix_spin_map2alm_ring_recurrence_cpu(
            maps, starts, lengths, theta, phase_neg, lmax, mmax, spin, 0.25
        ),
    )
    torch.testing.assert_close(
        torch.ops.torchsky.spectral_spin_alm2map_ring_recurrence(
            cp, starts, lengths, theta, phase_pos, lmax, mmax, spin, cm, x["npix"]
        ),
        _cpp._healpix_spin_alm2map_ring_recurrence_cpu(
            cp, starts, lengths, theta, phase_pos, lmax, mmax, spin, cm
        ),
    )
    torch.testing.assert_close(
        torch.ops.torchsky.spectral_spin_ring_finalize(
            interp[0], interp[1], starts, lengths, x["npix"]
        ),
        _cpp._healpix_spin_ring_finalize_cpu(interp[0], interp[1], starts, lengths),
    )


def test_spectral_dispatcher_fake_metadata_all_fixed_shapes() -> None:
    mode = FakeTensorMode()
    with mode:
        starts = torch.empty(3, dtype=torch.int64)
        lengths = torch.empty(3, dtype=torch.int64)
        theta = torch.empty(3, dtype=torch.float64)
        weight = torch.empty(3, dtype=torch.float64)
        phase = torch.empty((3, 3), dtype=torch.complex128)
        basis = torch.empty((6, 3), dtype=torch.complex128)
        coeff = torch.empty(6, dtype=torch.complex128)
        modes = torch.empty((3, 3), dtype=torch.complex128)
        rows = torch.empty((2, 12), dtype=torch.complex128)
        outputs = [
            torch.ops.torchsky.spectral_scalar_alm2map_direct(
                coeff.unsqueeze(0), theta, theta, 2, 2
            ),
            torch.ops.torchsky.spectral_ring_fourier_modes(rows, starts, lengths, 2),
            *torch.ops.torchsky.spectral_ring_fourier_modes_spin_conj(
                rows[0], starts, lengths, 2
            ),
            torch.ops.torchsky.spectral_ring_fourier_synthesis(
                torch.empty((2, 3, 3), dtype=torch.complex128), starts, lengths, 12
            ),
            torch.ops.torchsky.spectral_spin_ring_finalize(
                modes, modes, starts, lengths, 12
            ),
            *torch.ops.torchsky.spectral_spin_map2alm_ring_concat(
                rows[0], starts, lengths, phase, basis, basis, 2, 2, 1.0
            ),
            *torch.ops.torchsky.spectral_spin_interpolate_concat(
                coeff, coeff, basis, basis, 2, 2
            ),
            *torch.ops.torchsky.spectral_spin_integrate_concat(
                modes, modes, basis, basis, 2, 2, 1.0
            ),
            torch.ops.torchsky.spectral_spin_integrate_recurrence(
                modes, modes, theta, weight, 2, 2, 1
            ),
            torch.ops.torchsky.spectral_spin_interpolate_recurrence(
                torch.empty((2, 6), dtype=torch.complex128), theta, 2, 2, 1
            ),
            torch.ops.torchsky.spectral_spin_map2alm_ring_recurrence(
                torch.empty((2, 12), dtype=torch.float64),
                starts,
                lengths,
                theta,
                phase,
                2,
                2,
                1,
                1.0,
            ),
            torch.ops.torchsky.spectral_spin_alm2map_ring_recurrence(
                coeff, starts, lengths, theta, phase, 2, 2, 1, coeff, 12
            ),
            torch.ops.torchsky.spectral_scalar_integrate_recurrence(
                modes, theta, weight, 2, 2
            ),
            torch.ops.torchsky.spectral_scalar_interpolate_recurrence(
                coeff, theta, 2, 2
            ),
            *torch.ops.torchsky._spectral_spin_map2alm_from_basis(
                theta, theta, basis.T.contiguous(), basis.T.contiguous(), 1.0
            ),
            torch.ops.torchsky._spectral_spin_alm2map_from_basis(
                coeff, coeff, basis, basis, basis[:3], basis[:3]
            ),
        ]
    assert all(is_fake(output) for output in outputs)
    assert outputs[0].shape == (1, 3)
    assert outputs[1].shape == (2, 3, 3)
    assert outputs[-4].shape == (3, 3)
    assert outputs[-3].shape == (6,)
    assert outputs[-2].shape == (6,)
    assert outputs[-1].shape == (2, 3)


def test_private_basis_dispatcher_gradcheck_opcheck_and_compile() -> None:
    torch.manual_seed(14)
    npix, nalm, m0 = 4, 3, 1
    q = torch.randn(npix, dtype=torch.float64, requires_grad=True)
    u = torch.randn(npix, dtype=torch.float64, requires_grad=True)
    ycp_t = torch.randn((npix, nalm), dtype=torch.complex128)
    ycm_t = torch.randn((npix, nalm), dtype=torch.complex128)
    coeff_plus = torch.randn(nalm, dtype=torch.complex128, requires_grad=True)
    coeff_minus = torch.randn(nalm, dtype=torch.complex128, requires_grad=True)
    y_plus = torch.randn((nalm, npix), dtype=torch.complex128)
    y_minus = torch.randn((nalm, npix), dtype=torch.complex128)
    y_plus_m0 = torch.randn((m0, npix), dtype=torch.complex128)
    y_minus_m0 = torch.randn((m0, npix), dtype=torch.complex128)

    def analysis(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.torchsky._spectral_spin_map2alm_from_basis(
            first, second, ycp_t, ycm_t, 0.5
        )

    def synthesis(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return torch.ops.torchsky._spectral_spin_alm2map_from_basis(
            first, second, y_plus, y_minus, y_plus_m0, y_minus_m0
        )

    assert torch.autograd.gradcheck(analysis, (q, u), fast_mode=True)
    assert torch.autograd.gradcheck(
        synthesis, (coeff_plus, coeff_minus), fast_mode=True
    )
    torch.library.opcheck(
        torch.ops.torchsky._spectral_spin_map2alm_from_basis.default,
        (q.detach(), u.detach(), ycp_t, ycm_t, 0.5),
    )
    torch.library.opcheck(
        torch.ops.torchsky._spectral_spin_alm2map_from_basis.default,
        (
            coeff_plus.detach(),
            coeff_minus.detach(),
            y_plus,
            y_minus,
            y_plus_m0,
            y_minus_m0,
        ),
    )

    def loss(
        q_value: torch.Tensor,
        u_value: torch.Tensor,
        plus_value: torch.Tensor,
        minus_value: torch.Tensor,
    ) -> torch.Tensor:
        analyzed = analysis(q_value, u_value)
        synthesized = synthesis(plus_value, minus_value)
        return (
            analyzed[0].abs().square().sum()
            + analyzed[1].abs().square().sum()
            + synthesized.square().sum()
        )

    eager_inputs = tuple(
        value.detach().clone().requires_grad_()
        for value in (q, u, coeff_plus, coeff_minus)
    )
    compiled_inputs = tuple(
        value.detach().clone().requires_grad_()
        for value in (q, u, coeff_plus, coeff_minus)
    )
    eager_loss = loss(*eager_inputs)
    eager_loss.backward()
    compiled = torch.compile(loss, fullgraph=True, backend="aot_eager")
    compiled_loss = compiled(*compiled_inputs)
    compiled_loss.backward()
    torch.testing.assert_close(compiled_loss, eager_loss)
    for actual, expected in zip(compiled_inputs, eager_inputs, strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)


def test_spectral_recurrence_adjoint_and_metadata_nondifferentiable() -> None:
    x = _inputs()
    modes = x["modes"].requires_grad_()
    theta = x["theta"].clone().requires_grad_()
    weight = x["weight"].clone().requires_grad_()
    assert isinstance(modes, torch.Tensor) and isinstance(theta, torch.Tensor)
    assert isinstance(weight, torch.Tensor)
    alm = torch.ops.torchsky.spectral_scalar_integrate_recurrence(
        modes, theta, weight, 2, 2
    )
    probe = torch.randn_like(alm)
    loss = torch.real(torch.vdot(probe, alm))
    loss.backward()
    expected = torch.ops.torchsky.spectral_scalar_interpolate_recurrence(
        probe, theta.detach(), 2, 2
    ) * weight.detach().unsqueeze(0)
    torch.testing.assert_close(modes.grad, expected)
    assert theta.grad is None
    assert weight.grad is None

    spin_plus = x["modes"].detach().clone().requires_grad_()
    spin_minus = x["modes"].detach().clone().requires_grad_()
    assert isinstance(spin_plus, torch.Tensor) and isinstance(spin_minus, torch.Tensor)
    spin_alm = torch.ops.torchsky.spectral_spin_integrate_recurrence(
        spin_plus, spin_minus, theta.detach(), weight.detach(), 2, 2, 1
    )
    spin_probe = torch.randn_like(spin_alm)
    torch.real(torch.vdot(spin_probe.reshape(-1), spin_alm.reshape(-1))).backward()
    spin_expected = torch.ops.torchsky.spectral_spin_interpolate_recurrence(
        spin_probe, theta.detach(), 2, 2, 1
    ) * weight.detach().reshape(1, 1, -1)
    torch.testing.assert_close(spin_plus.grad, spin_expected[0])
    torch.testing.assert_close(spin_minus.grad, spin_expected[1])

    rows = x["rows"].clone().requires_grad_()
    starts, lengths = x["starts"], x["lengths"]
    assert isinstance(rows, torch.Tensor) and isinstance(starts, torch.Tensor)
    assert isinstance(lengths, torch.Tensor)
    fourier = torch.ops.torchsky.spectral_ring_fourier_modes(rows, starts, lengths, 2)
    fourier_probe = torch.randn_like(fourier)
    torch.real(torch.vdot(fourier_probe.reshape(-1), fourier.reshape(-1))).backward()
    fourier_expected = torch.ops.torchsky.spectral_ring_fourier_synthesis(
        fourier_probe, starts, lengths, 12
    )
    torch.testing.assert_close(rows.grad, fourier_expected)


def test_spectral_recurrence_gradcheck_opcheck_and_compile() -> None:
    x = _inputs()
    coeff = x["coeff_plus"].clone().requires_grad_()
    theta = x["theta"]
    assert isinstance(coeff, torch.Tensor) and isinstance(theta, torch.Tensor)

    def interpolate(value: torch.Tensor) -> torch.Tensor:
        return torch.ops.torchsky.spectral_scalar_interpolate_recurrence(
            value, theta, 2, 2
        )

    assert torch.autograd.gradcheck(interpolate, (coeff,), fast_mode=True)
    torch.library.opcheck(
        torch.ops.torchsky.spectral_scalar_interpolate_recurrence.default,
        (coeff.detach(), theta, 2, 2),
    )

    coeff_minus = x["coeff_minus"].detach().clone().requires_grad_()
    basis_plus, basis_minus = x["y_plus"], x["y_minus"]
    assert isinstance(coeff_minus, torch.Tensor)
    assert isinstance(basis_plus, torch.Tensor) and isinstance(basis_minus, torch.Tensor)

    def concat(
        first: torch.Tensor, second: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.torchsky.spectral_spin_interpolate_concat(
            first, second, basis_plus, basis_minus, 2, 2
        )

    assert torch.autograd.gradcheck(concat, (coeff, coeff_minus), fast_mode=True)
    torch.library.opcheck(
        torch.ops.torchsky.spectral_spin_interpolate_concat.default,
        (coeff.detach(), coeff_minus.detach(), basis_plus, basis_minus, 2, 2),
    )

    def loss(value: torch.Tensor) -> torch.Tensor:
        output = torch.ops.torchsky.spectral_scalar_interpolate_recurrence(
            value, theta, 2, 2
        )
        return output.abs().square().sum()

    eager = coeff.detach().clone().requires_grad_()
    compiled_input = coeff.detach().clone().requires_grad_()
    eager_loss = loss(eager)
    eager_loss.backward()
    compiled = torch.compile(loss, fullgraph=True, backend="aot_eager")
    compiled_loss = compiled(compiled_input)
    compiled_loss.backward()
    torch.testing.assert_close(compiled_loss, eager_loss)
    torch.testing.assert_close(compiled_input.grad, eager.grad)


def test_spectral_dispatcher_empty_malformed_device_and_no_alias() -> None:
    empty_i64 = torch.empty(0, dtype=torch.int64)
    empty_f64 = torch.empty(0, dtype=torch.float64)
    empty_c128 = torch.empty((1, 0), dtype=torch.complex128)
    modes = torch.ops.torchsky.spectral_ring_fourier_modes(
        torch.empty((1, 0), dtype=torch.complex128), empty_i64, empty_i64, 0
    )
    assert modes.shape == (1, 1, 0)
    synthesized = torch.ops.torchsky.spectral_ring_fourier_synthesis(
        modes, empty_i64, empty_i64, 0
    )
    assert synthesized.shape == (1, 0)
    scalar = torch.ops.torchsky.spectral_scalar_integrate_recurrence(
        empty_c128, empty_f64, empty_f64, 0, 0
    )
    assert scalar.shape == (1,)

    with pytest.raises(RuntimeError, match="complex128"):
        torch.ops.torchsky.spectral_scalar_interpolate_recurrence(
            torch.zeros(1, dtype=torch.complex64), torch.zeros(1, dtype=torch.float64), 0, 0
        )
    with pytest.raises(RuntimeError, match="ring"):
        torch.ops.torchsky.spectral_ring_fourier_modes(
            torch.zeros((1, 3), dtype=torch.complex128),
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([4], dtype=torch.int64),
            0,
        )
    with pytest.raises(RuntimeError, match="npix"):
        torch.ops.torchsky.spectral_ring_fourier_synthesis(
            torch.zeros((1, 1, 1), dtype=torch.complex128),
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int64),
            2,
        )

    x = _inputs()
    coeff, theta = x["coeff_plus"], x["theta"]
    assert isinstance(coeff, torch.Tensor) and isinstance(theta, torch.Tensor)
    output = torch.ops.torchsky.spectral_scalar_interpolate_recurrence(
        coeff, theta, 2, 2
    )
    assert output.untyped_storage().data_ptr() != coeff.untyped_storage().data_ptr()
    basis_plus, basis_minus = x["y_plus"], x["y_minus"]
    coeff_minus = x["coeff_minus"]
    assert isinstance(basis_plus, torch.Tensor) and isinstance(basis_minus, torch.Tensor)
    assert isinstance(coeff_minus, torch.Tensor)
    pair = torch.ops.torchsky.spectral_spin_interpolate_concat(
        coeff, coeff_minus, basis_plus, basis_minus, 2, 2
    )
    input_pointers = {
        tensor.untyped_storage().data_ptr()
        for tensor in (coeff, coeff_minus, basis_plus, basis_minus)
    }
    assert pair[0].untyped_storage().data_ptr() not in input_pointers
    assert pair[1].untyped_storage().data_ptr() not in input_pointers
    assert pair[0].untyped_storage().data_ptr() != pair[1].untyped_storage().data_ptr()
    if torch.cuda.is_available():
        with pytest.raises(NotImplementedError, match="CUDA"):
            torch.ops.torchsky.spectral_scalar_interpolate_recurrence(
                coeff.cuda(), theta.cuda(), 2, 2
            )
