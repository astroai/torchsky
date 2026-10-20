#include <torch/library.h>

#include "torchsky/healpix.h"

TORCH_LIBRARY_FRAGMENT(torchsky, m) {
    m.def("spectral_scalar_alm2map_direct(Tensor alm, Tensor cos_theta, Tensor phi, int lmax, int mmax) -> Tensor");
    m.def("spectral_ring_fourier_modes(Tensor rows, Tensor starts, Tensor lengths, int mmax) -> Tensor");
    m.def("spectral_ring_fourier_modes_spin_conj(Tensor row, Tensor starts, Tensor lengths, int mmax) -> (Tensor s_plus, Tensor s_minus)");
    m.def("spectral_ring_fourier_synthesis(Tensor modes, Tensor starts, Tensor lengths, int npix) -> Tensor");
    m.def("spectral_spin_ring_finalize(Tensor s_plus, Tensor s_minus, Tensor starts, Tensor lengths, int npix) -> Tensor");
    m.def("spectral_spin_map2alm_ring_concat(Tensor p_plus, Tensor starts, Tensor lengths, Tensor phase0_neg, Tensor y_plus, Tensor y_minus, int lmax, int mmax, float pixel_weight) -> (Tensor c_plus, Tensor c_minus)");
    m.def("spectral_spin_interpolate_concat(Tensor coeff_plus, Tensor coeff_minus, Tensor y_plus, Tensor y_minus, int lmax, int mmax) -> (Tensor s_plus, Tensor s_minus)");
    m.def("spectral_spin_integrate_concat(Tensor s_plus, Tensor s_minus, Tensor y_plus, Tensor y_minus, int lmax, int mmax, float pixel_weight) -> (Tensor coeff_plus, Tensor coeff_minus)");
    m.def("spectral_spin_integrate_recurrence(Tensor s_plus, Tensor s_minus, Tensor theta, Tensor weight, int lmax, int mmax, int spin) -> Tensor");
    m.def("spectral_spin_interpolate_recurrence(Tensor alms, Tensor theta, int lmax, int mmax, int spin) -> Tensor");
    m.def("spectral_spin_map2alm_ring_recurrence(Tensor maps, Tensor starts, Tensor lengths, Tensor theta, Tensor phase0_neg, int lmax, int mmax, int spin, float pixel_weight) -> Tensor");
    m.def("spectral_spin_alm2map_ring_recurrence(Tensor alm_plus, Tensor starts, Tensor lengths, Tensor theta, Tensor phase0_pos, int lmax, int mmax, int spin, Tensor alm_minus, int npix) -> Tensor");
    m.def("spectral_scalar_integrate_recurrence(Tensor modes, Tensor theta, Tensor weight, int lmax, int mmax) -> Tensor");
    m.def("spectral_scalar_interpolate_recurrence(Tensor alm, Tensor theta, int lmax, int mmax) -> Tensor");
}

namespace torchsky {

void ensure_spectral_operator_schemas_linked() {}

}
