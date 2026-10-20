#include <torch/library.h>

#include "torchsky/wcs_operators.h"

TORCH_LIBRARY(torchsky, m) {
    m.def(
        "wcs_pixel_to_world(Tensor x, Tensor y, float crpix0, float crpix1, "
        "float cd00, float cd01, float cd10, float cd11, str proj_code, "
        "float lambda_param, float hpx_H, float hpx_K, float east_x, float east_y, "
        "float north_x, float north_y, float north_z, float radial_x, float radial_y, "
        "float radial_z, float ra_p_rad, float dec_p_rad, float phi_p_rad, "
        "str native_type, float alpha0, float delta0, bool is_tpv, Tensor tpv_idx1, "
        "Tensor tpv_c1, Tensor tpv_idx2, Tensor tpv_c2) -> (Tensor ra, Tensor dec)"
    );
    m.def(
        "wcs_world_to_pixel(Tensor ra, Tensor dec, float crpix0, float crpix1, "
        "float cdi00, float cdi01, float cdi10, float cdi11, str proj_code, "
        "float lambda_param, float hpx_H, float hpx_K, float east_x, float east_y, "
        "float north_x, float north_y, float north_z, float radial_x, float radial_y, "
        "float radial_z, float ra_p_rad, float dec_p_rad, float phi_p_rad, "
        "str native_type, float alpha0, float delta0, bool is_tpv, Tensor tpv_idx1, "
        "Tensor tpv_c1, Tensor tpv_idx2, Tensor tpv_c2) -> (Tensor x, Tensor y)"
    );
}

namespace torchsky {

void ensure_wcs_operator_schemas_linked() {}

}
