#pragma once

#include "torchsky_torch.h"

namespace torchsky {

void ensure_spectral_operator_schemas_linked();
void ensure_spectral_cpu_operators_linked();

torch::Tensor healpix_ang2pix_ring_cpu(
    int64_t nside,
    const torch::Tensor& ra_deg,
    const torch::Tensor& dec_deg
);
torch::Tensor healpix_ang2pix_nested_cpu(
    int64_t nside,
    const torch::Tensor& ra_deg,
    const torch::Tensor& dec_deg
);
std::pair<torch::Tensor, torch::Tensor> healpix_pix2ang_ring_cpu(
    int64_t nside,
    const torch::Tensor& pix_ring
);
std::pair<torch::Tensor, torch::Tensor> healpix_pix2ang_nested_cpu(
    int64_t nside,
    const torch::Tensor& pix_nest
);
torch::Tensor healpix_ring2nest_cpu(int64_t nside, const torch::Tensor& pix_ring);
torch::Tensor healpix_nest2ring_cpu(int64_t nside, const torch::Tensor& pix_nest);
torch::Tensor healpix_neighbors_ring_cpu(int64_t nside, const torch::Tensor& pix_ring);
torch::Tensor healpix_neighbors_nested_cpu(int64_t nside, const torch::Tensor& pix_nest);

torch::Tensor healpix_scalar_alm2map_direct_cpu(
    const torch::Tensor& alm_values,
    const torch::Tensor& cos_theta,
    const torch::Tensor& phi,
    int64_t lmax,
    int64_t mmax
);
torch::Tensor healpix_ring_fourier_modes_cpu(
    const torch::Tensor& rows,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    int64_t mmax
);
torch::Tensor healpix_ring_fourier_modes_spin_conj_cpu(
    const torch::Tensor& row,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    int64_t mmax,
    const c10::complex<double>* phase_ptr = nullptr,
    bool from_qu = false
);
torch::Tensor healpix_ring_fourier_synthesis_cpu(
    const torch::Tensor& s_modes,
    const torch::Tensor& starts,
    const torch::Tensor& lengths
);
torch::Tensor healpix_spin_ring_finalize_cpu(
    const torch::Tensor& s_plus,
    const torch::Tensor& s_minus,
    const torch::Tensor& starts,
    const torch::Tensor& lengths
);
std::pair<torch::Tensor, torch::Tensor> healpix_spin_map2alm_from_basis_cpu(
    const torch::Tensor& q,
    const torch::Tensor& u,
    const torch::Tensor& ycp_t,
    const torch::Tensor& ycm_t,
    double pix_w
);
torch::Tensor healpix_spin_alm2map_from_basis_cpu(
    const torch::Tensor& coeff_plus,
    const torch::Tensor& coeff_minus,
    const torch::Tensor& y_plus,
    const torch::Tensor& y_minus,
    const torch::Tensor& y_plus_m0,
    const torch::Tensor& y_minus_m0
);
std::pair<torch::Tensor, torch::Tensor> healpix_spin_interpolate_concat_cpu(
    const torch::Tensor& coeff_plus,
    const torch::Tensor& coeff_minus,
    const torch::Tensor& y_plus,
    const torch::Tensor& y_minus,
    int64_t lmax,
    int64_t mmax
);
std::pair<torch::Tensor, torch::Tensor> healpix_spin_integrate_concat_cpu(
    const torch::Tensor& s_plus,
    const torch::Tensor& s_minus,
    const torch::Tensor& y_plus,
    const torch::Tensor& y_minus,
    int64_t lmax,
    int64_t mmax,
    double pix_w
);
std::pair<torch::Tensor, torch::Tensor> healpix_spin_map2alm_ring_concat_cpu(
    const torch::Tensor& p_plus,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    const torch::Tensor& phase0_neg,
    const torch::Tensor& y_plus,
    const torch::Tensor& y_minus,
    int64_t lmax,
    int64_t mmax,
    double pix_w
);

std::pair<torch::Tensor, torch::Tensor> healpix_get_interp_weights_ring_cpu(
    int64_t nside,
    const torch::Tensor& lon_deg,
    const torch::Tensor& lat_deg
);
std::pair<torch::Tensor, torch::Tensor> healpix_get_interp_weights_nested_cpu(
    int64_t nside,
    const torch::Tensor& lon_deg,
    const torch::Tensor& lat_deg
);
torch::Tensor healpix_get_interp_val_ring_cpu(
    int64_t nside,
    const torch::Tensor& maps_in,
    const torch::Tensor& lon_deg,
    const torch::Tensor& lat_deg
);
torch::Tensor healpix_get_interp_val_nested_cpu(
    int64_t nside,
    const torch::Tensor& maps_in,
    const torch::Tensor& lon_deg,
    const torch::Tensor& lat_deg
);

torch::Tensor healpix_spin_integrate_recurrence_cpu(
    const torch::Tensor& s_plus,
    const torch::Tensor& s_minus,
    const torch::Tensor& theta,
    const torch::Tensor& weight,
    int64_t lmax,
    int64_t mmax,
    int64_t spin
);
torch::Tensor healpix_spin_interpolate_recurrence_cpu(
    const torch::Tensor& alms,
    const torch::Tensor& theta,
    int64_t lmax,
    int64_t mmax,
    int64_t spin,
    const c10::complex<double>* phase_ptr = nullptr,
    const c10::complex<double>* alm0_ptr_in = nullptr,
    const c10::complex<double>* alm1_ptr_in = nullptr,
    c10::complex<double>* eq_plus = nullptr,
    c10::complex<double>* eq_minus = nullptr,
    int64_t eq_start = -1,
    int64_t n_eq = 0,
    int64_t nph_eq = 0
);
torch::Tensor healpix_scalar_integrate_recurrence_cpu(
    const torch::Tensor& s_modes,
    const torch::Tensor& theta,
    const torch::Tensor& weight,
    int64_t lmax,
    int64_t mmax
);
torch::Tensor healpix_scalar_interpolate_recurrence_cpu(
    const torch::Tensor& alm,
    const torch::Tensor& theta,
    int64_t lmax,
    int64_t mmax
);
torch::Tensor healpix_query_disc_cpu(
    int64_t nside,
    const torch::Tensor& vec_xyz,
    double cos_lim,
    bool nest
);
torch::Tensor healpix_spin_map2alm_ring_recurrence_cpu(
    const torch::Tensor& maps,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    const torch::Tensor& theta,
    const torch::Tensor& phase0_neg,
    int64_t lmax,
    int64_t mmax,
    int64_t spin,
    double pix_w
);
torch::Tensor healpix_spin_alm2map_ring_recurrence_cpu(
    const torch::Tensor& alm_plus,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    const torch::Tensor& theta,
    const torch::Tensor& phase0_pos,
    int64_t lmax,
    int64_t mmax,
    int64_t spin,
    const torch::Tensor& alm_minus = torch::Tensor()
);
torch::Tensor sparse_get_values_blocks_cpu(
    const torch::Tensor& query,
    const torch::Tensor& block_index,
    const torch::Tensor& block_values,
    const torch::Tensor& fill
);

}
