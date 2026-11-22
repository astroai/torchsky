#pragma once

#include "torchsky/native.h"

namespace torchsky {


native::Tensor healpix_ang2pix_ring_cpu(
    int64_t nside,
    const native::Tensor& ra_deg,
    const native::Tensor& dec_deg
);
native::Tensor healpix_ang2pix_nested_cpu(
    int64_t nside,
    const native::Tensor& ra_deg,
    const native::Tensor& dec_deg
);
std::pair<native::Tensor, native::Tensor> healpix_pix2ang_ring_cpu(
    int64_t nside,
    const native::Tensor& pix_ring
);
std::pair<native::Tensor, native::Tensor> healpix_pix2ang_nested_cpu(
    int64_t nside,
    const native::Tensor& pix_nest
);
native::Tensor healpix_ring2nest_cpu(int64_t nside, const native::Tensor& pix_ring);
native::Tensor healpix_nest2ring_cpu(int64_t nside, const native::Tensor& pix_nest);
native::Tensor healpix_neighbors_ring_cpu(int64_t nside, const native::Tensor& pix_ring);
native::Tensor healpix_neighbors_nested_cpu(int64_t nside, const native::Tensor& pix_nest);

native::Tensor healpix_scalar_alm2map_direct_cpu(
    const native::Tensor& alm_values,
    const native::Tensor& cos_theta,
    const native::Tensor& phi,
    int64_t lmax,
    int64_t mmax
);
native::Tensor healpix_ring_fourier_modes_cpu(
    const native::Tensor& rows,
    const native::Tensor& starts,
    const native::Tensor& lengths,
    int64_t mmax
);
native::Tensor healpix_ring_fourier_modes_spin_conj_cpu(
    const native::Tensor& row,
    const native::Tensor& starts,
    const native::Tensor& lengths,
    int64_t mmax,
    const std::complex<double>* phase_ptr = nullptr,
    bool from_qu = false
);
native::Tensor healpix_ring_fourier_synthesis_cpu(
    const native::Tensor& s_modes,
    const native::Tensor& starts,
    const native::Tensor& lengths
);
native::Tensor healpix_spin_ring_finalize_cpu(
    const native::Tensor& s_plus,
    const native::Tensor& s_minus,
    const native::Tensor& starts,
    const native::Tensor& lengths
);
std::pair<native::Tensor, native::Tensor> healpix_spin_map2alm_from_basis_cpu(
    const native::Tensor& q,
    const native::Tensor& u,
    const native::Tensor& ycp_t,
    const native::Tensor& ycm_t,
    double pix_w
);std::pair<native::Tensor, native::Tensor> healpix_spin_interpolate_concat_cpu(
    const native::Tensor& coeff_plus,
    const native::Tensor& coeff_minus,
    const native::Tensor& y_plus,
    const native::Tensor& y_minus,
    int64_t lmax,
    int64_t mmax
);
std::pair<native::Tensor, native::Tensor> healpix_spin_integrate_concat_cpu(
    const native::Tensor& s_plus,
    const native::Tensor& s_minus,
    const native::Tensor& y_plus,
    const native::Tensor& y_minus,
    int64_t lmax,
    int64_t mmax,
    double pix_w
);
std::pair<native::Tensor, native::Tensor> healpix_spin_map2alm_ring_concat_cpu(
    const native::Tensor& p_plus,
    const native::Tensor& starts,
    const native::Tensor& lengths,
    const native::Tensor& phase0_neg,
    const native::Tensor& y_plus,
    const native::Tensor& y_minus,
    int64_t lmax,
    int64_t mmax,
    double pix_w
);

std::pair<native::Tensor, native::Tensor> healpix_get_interp_weights_ring_cpu(
    int64_t nside,
    const native::Tensor& lon_deg,
    const native::Tensor& lat_deg
);
std::pair<native::Tensor, native::Tensor> healpix_get_interp_weights_nested_cpu(
    int64_t nside,
    const native::Tensor& lon_deg,
    const native::Tensor& lat_deg
);
native::Tensor healpix_get_interp_val_ring_cpu(
    int64_t nside,
    const native::Tensor& maps_in,
    const native::Tensor& lon_deg,
    const native::Tensor& lat_deg
);
native::Tensor healpix_get_interp_val_nested_cpu(
    int64_t nside,
    const native::Tensor& maps_in,
    const native::Tensor& lon_deg,
    const native::Tensor& lat_deg
);

native::Tensor healpix_spin_integrate_recurrence_cpu(
    const native::Tensor& s_plus,
    const native::Tensor& s_minus,
    const native::Tensor& theta,
    const native::Tensor& weight,
    int64_t lmax,
    int64_t mmax,
    int64_t spin
);
native::Tensor healpix_spin_interpolate_recurrence_cpu(
    const native::Tensor& alms,
    const native::Tensor& theta,
    int64_t lmax,
    int64_t mmax,
    int64_t spin,
    const std::complex<double>* phase_ptr = nullptr,
    const std::complex<double>* alm0_ptr_in = nullptr,
    const std::complex<double>* alm1_ptr_in = nullptr,
    std::complex<double>* eq_plus = nullptr,
    std::complex<double>* eq_minus = nullptr,
    int64_t eq_start = -1,
    int64_t n_eq = 0,
    int64_t nph_eq = 0
);
native::Tensor healpix_scalar_integrate_recurrence_cpu(
    const native::Tensor& s_modes,
    const native::Tensor& theta,
    const native::Tensor& weight,
    int64_t lmax,
    int64_t mmax
);
native::Tensor healpix_scalar_interpolate_recurrence_cpu(
    const native::Tensor& alm,
    const native::Tensor& theta,
    int64_t lmax,
    int64_t mmax
);
native::Tensor healpix_query_disc_nest_ranges_cpu(
    int64_t nside,
    const native::Tensor& vec_xyz,
    double cos_lim
);
native::Tensor healpix_query_disc_cpu(
    int64_t nside,
    const native::Tensor& vec_xyz,
    double cos_lim,
    bool nest
);
native::Tensor healpix_spin_map2alm_ring_recurrence_cpu(
    const native::Tensor& maps,
    const native::Tensor& starts,
    const native::Tensor& lengths,
    const native::Tensor& theta,
    const native::Tensor& phase0_neg,
    int64_t lmax,
    int64_t mmax,
    int64_t spin,
    double pix_w
);
native::Tensor healpix_spin_alm2map_ring_recurrence_cpu(
    const native::Tensor& alm_plus,
    const native::Tensor& starts,
    const native::Tensor& lengths,
    const native::Tensor& theta,
    const native::Tensor& phase0_pos,
    int64_t lmax,
    int64_t mmax,
    int64_t spin,
    const native::Tensor& alm_minus = native::Tensor()
);
native::Tensor sparse_get_values_blocks_cpu(
    const native::Tensor& query,
    const native::Tensor& block_index,
    const native::Tensor& block_values,
    const native::Tensor& fill
);

}
