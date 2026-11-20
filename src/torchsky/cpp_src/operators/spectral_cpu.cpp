#include "torchsky/native_nb.h"

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

void register_spectral(nanobind::module_& m) {
    using native::def;
    def(m, "spectral_scalar_alm2map_direct", &healpix_scalar_alm2map_direct_cpu);
    def(m, "spectral_ring_fourier_modes", &healpix_ring_fourier_modes_cpu);
    def(m, "spectral_ring_fourier_modes_spin_conj", &spectral_ring_fourier_modes_spin_conj_cpu);
    def(m, "spectral_ring_fourier_synthesis", &spectral_ring_fourier_synthesis_cpu);
    def(m, "spectral_spin_ring_finalize", &spectral_spin_ring_finalize_cpu);
    def(m, "spectral_spin_map2alm_ring_concat", &spectral_spin_map2alm_ring_concat_cpu);
    def(m, "spectral_spin_interpolate_concat", &spectral_spin_interpolate_concat_cpu);
    def(m, "spectral_spin_integrate_concat", &spectral_spin_integrate_concat_cpu);
    def(m, "spectral_spin_integrate_recurrence", &healpix_spin_integrate_recurrence_cpu);
    def(m, "spectral_spin_interpolate_recurrence", &spectral_spin_interpolate_recurrence_cpu);
    def(m, "spectral_spin_map2alm_ring_recurrence", &healpix_spin_map2alm_ring_recurrence_cpu);
    def(m, "spectral_spin_alm2map_ring_recurrence", &spectral_spin_alm2map_ring_recurrence_cpu);
    def(m, "spectral_scalar_integrate_recurrence", &healpix_scalar_integrate_recurrence_cpu);
    def(m, "spectral_scalar_interpolate_recurrence", &healpix_scalar_interpolate_recurrence_cpu);
}

}  // namespace torchsky
