#include <torch/library.h>

#include "torchsky/healpix.h"

namespace torchsky {

std::tuple<torch::Tensor, torch::Tensor> spectral_ring_fourier_modes_spin_conj_cpu(
    const torch::Tensor& row,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    int64_t mmax
) {
    auto stacked = healpix_ring_fourier_modes_spin_conj_cpu(row, starts, lengths, mmax);
    return std::make_tuple(stacked.select(0, 0).clone(), stacked.select(0, 1).clone());
}

std::tuple<torch::Tensor, torch::Tensor> spectral_spin_map2alm_ring_concat_cpu(
    const torch::Tensor& p_plus,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    const torch::Tensor& phase0_neg,
    const torch::Tensor& y_plus,
    const torch::Tensor& y_minus,
    int64_t lmax,
    int64_t mmax,
    double pixel_weight
) {
    auto out = healpix_spin_map2alm_ring_concat_cpu(
        p_plus, starts, lengths, phase0_neg, y_plus, y_minus, lmax, mmax, pixel_weight
    );
    return std::make_tuple(std::move(out.first), std::move(out.second));
}

std::tuple<torch::Tensor, torch::Tensor> spectral_spin_interpolate_concat_cpu(
    const torch::Tensor& coeff_plus,
    const torch::Tensor& coeff_minus,
    const torch::Tensor& y_plus,
    const torch::Tensor& y_minus,
    int64_t lmax,
    int64_t mmax
) {
    auto out = healpix_spin_interpolate_concat_cpu(
        coeff_plus, coeff_minus, y_plus, y_minus, lmax, mmax
    );
    return std::make_tuple(std::move(out.first), std::move(out.second));
}

std::tuple<torch::Tensor, torch::Tensor> spectral_spin_integrate_concat_cpu(
    const torch::Tensor& s_plus,
    const torch::Tensor& s_minus,
    const torch::Tensor& y_plus,
    const torch::Tensor& y_minus,
    int64_t lmax,
    int64_t mmax,
    double pixel_weight
) {
    auto out = healpix_spin_integrate_concat_cpu(
        s_plus, s_minus, y_plus, y_minus, lmax, mmax, pixel_weight
    );
    return std::make_tuple(std::move(out.first), std::move(out.second));
}

std::tuple<torch::Tensor, torch::Tensor> spectral_spin_map2alm_from_basis_cpu(
    const torch::Tensor& q,
    const torch::Tensor& u,
    const torch::Tensor& ycp_t,
    const torch::Tensor& ycm_t,
    double pixel_weight
) {
    auto out = healpix_spin_map2alm_from_basis_cpu(q, u, ycp_t, ycm_t, pixel_weight);
    return std::make_tuple(std::move(out.first), std::move(out.second));
}

torch::Tensor spectral_spin_interpolate_recurrence_cpu(
    const torch::Tensor& alms,
    const torch::Tensor& theta,
    int64_t lmax,
    int64_t mmax,
    int64_t spin
) {
    return healpix_spin_interpolate_recurrence_cpu(alms, theta, lmax, mmax, spin);
}

torch::Tensor spectral_ring_fourier_synthesis_cpu(
    const torch::Tensor& modes,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    int64_t npix
) {
    TORCH_CHECK(npix >= 0, "npix must be non-negative");
    auto out = healpix_ring_fourier_synthesis_cpu(modes, starts, lengths);
    TORCH_CHECK(out.size(-1) == npix, "npix does not match ring layout");
    return out;
}

torch::Tensor spectral_spin_ring_finalize_cpu(
    const torch::Tensor& s_plus,
    const torch::Tensor& s_minus,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    int64_t npix
) {
    TORCH_CHECK(npix >= 0, "npix must be non-negative");
    auto out = healpix_spin_ring_finalize_cpu(s_plus, s_minus, starts, lengths);
    TORCH_CHECK(out.size(-1) == npix, "npix does not match ring layout");
    return out;
}

torch::Tensor spectral_spin_alm2map_ring_recurrence_cpu(
    const torch::Tensor& alm_plus,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    const torch::Tensor& theta,
    const torch::Tensor& phase0_pos,
    int64_t lmax,
    int64_t mmax,
    int64_t spin,
    const torch::Tensor& alm_minus,
    int64_t npix
) {
    TORCH_CHECK(npix >= 0, "npix must be non-negative");
    auto out = healpix_spin_alm2map_ring_recurrence_cpu(
        alm_plus, starts, lengths, theta, phase0_pos, lmax, mmax, spin, alm_minus
    );
    TORCH_CHECK(out.size(-1) == npix, "npix does not match ring layout");
    return out;
}

}

TORCH_LIBRARY_IMPL(torchsky, CPU, m) {
    m.impl("spectral_scalar_alm2map_direct", TORCH_FN(torchsky::healpix_scalar_alm2map_direct_cpu));
    m.impl("spectral_ring_fourier_modes", TORCH_FN(torchsky::healpix_ring_fourier_modes_cpu));
    m.impl("spectral_ring_fourier_modes_spin_conj", TORCH_FN(torchsky::spectral_ring_fourier_modes_spin_conj_cpu));
    m.impl("spectral_ring_fourier_synthesis", TORCH_FN(torchsky::spectral_ring_fourier_synthesis_cpu));
    m.impl("spectral_spin_ring_finalize", TORCH_FN(torchsky::spectral_spin_ring_finalize_cpu));
    m.impl("spectral_spin_map2alm_ring_concat", TORCH_FN(torchsky::spectral_spin_map2alm_ring_concat_cpu));
    m.impl("spectral_spin_interpolate_concat", TORCH_FN(torchsky::spectral_spin_interpolate_concat_cpu));
    m.impl("spectral_spin_integrate_concat", TORCH_FN(torchsky::spectral_spin_integrate_concat_cpu));
    m.impl("spectral_spin_integrate_recurrence", TORCH_FN(torchsky::healpix_spin_integrate_recurrence_cpu));
    m.impl("spectral_spin_interpolate_recurrence", TORCH_FN(torchsky::spectral_spin_interpolate_recurrence_cpu));
    m.impl("spectral_spin_map2alm_ring_recurrence", TORCH_FN(torchsky::healpix_spin_map2alm_ring_recurrence_cpu));
    m.impl("spectral_spin_alm2map_ring_recurrence", TORCH_FN(torchsky::spectral_spin_alm2map_ring_recurrence_cpu));
    m.impl("spectral_scalar_integrate_recurrence", TORCH_FN(torchsky::healpix_scalar_integrate_recurrence_cpu));
    m.impl("spectral_scalar_interpolate_recurrence", TORCH_FN(torchsky::healpix_scalar_interpolate_recurrence_cpu));
    m.impl("_spectral_spin_map2alm_from_basis", TORCH_FN(torchsky::spectral_spin_map2alm_from_basis_cpu));
    m.impl("_spectral_spin_alm2map_from_basis", TORCH_FN(torchsky::healpix_spin_alm2map_from_basis_cpu));
}

namespace torchsky {

void ensure_spectral_cpu_operators_linked() {}

}
