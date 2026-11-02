#include <torch/library.h>

#include "torchsky/wcs.h"
#include "torchsky/wcs_operators.h"

namespace torchsky {

WcsTensorPair wcs_pixel_to_world_cpu(
    const torch::Tensor& x,
    const torch::Tensor& y,
    double crpix0,
    double crpix1,
    double cd00,
    double cd01,
    double cd10,
    double cd11,
    const std::string& proj_code,
    double lambda_param,
    double hpx_H,
    double hpx_K,
    double east_x,
    double east_y,
    double north_x,
    double north_y,
    double north_z,
    double radial_x,
    double radial_y,
    double radial_z,
    double ra_p_rad,
    double dec_p_rad,
    double phi_p_rad,
    const std::string& native_type,
    double alpha0,
    double delta0,
    bool is_tpv,
    const torch::Tensor& tpv_idx1,
    const torch::Tensor& tpv_c1,
    const torch::Tensor& tpv_idx2,
    const torch::Tensor& tpv_c2
) {
    const auto x_contiguous = x.contiguous();
    const auto y_contiguous = y.contiguous();
    auto ra = torch::empty_like(x_contiguous);
    auto dec = torch::empty_like(x_contiguous);
    wcs_pixel_to_world_fused_cpu(
        x_contiguous, y_contiguous, ra, dec,
        crpix0, crpix1, cd00, cd01, cd10, cd11,
        proj_code, lambda_param, hpx_H, hpx_K,
        east_x, east_y, north_x, north_y, north_z,
        radial_x, radial_y, radial_z,
        ra_p_rad, dec_p_rad, phi_p_rad,
        native_type, alpha0, delta0,
        is_tpv, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2
    );
    return {ra, dec};
}

WcsTensorPair wcs_world_to_pixel_cpu(
    const torch::Tensor& ra,
    const torch::Tensor& dec,
    double crpix0,
    double crpix1,
    double cdi00,
    double cdi01,
    double cdi10,
    double cdi11,
    const std::string& proj_code,
    double lambda_param,
    double hpx_H,
    double hpx_K,
    double east_x,
    double east_y,
    double north_x,
    double north_y,
    double north_z,
    double radial_x,
    double radial_y,
    double radial_z,
    double ra_p_rad,
    double dec_p_rad,
    double phi_p_rad,
    const std::string& native_type,
    double alpha0,
    double delta0,
    bool is_tpv,
    const torch::Tensor& tpv_idx1,
    const torch::Tensor& tpv_c1,
    const torch::Tensor& tpv_idx2,
    const torch::Tensor& tpv_c2
) {
    const auto ra_contiguous = ra.contiguous();
    const auto dec_contiguous = dec.contiguous();
    auto x = torch::empty_like(ra_contiguous);
    auto y = torch::empty_like(ra_contiguous);
    wcs_world_to_pixel_fused_cpu(
        ra_contiguous, dec_contiguous, x, y,
        crpix0, crpix1, cdi00, cdi01, cdi10, cdi11,
        proj_code, lambda_param, hpx_H, hpx_K,
        east_x, east_y, north_x, north_y, north_z,
        radial_x, radial_y, radial_z,
        ra_p_rad, dec_p_rad, phi_p_rad,
        native_type, alpha0, delta0,
        is_tpv, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2
    );
    return {x, y};
}

void ensure_wcs_cpu_operators_linked() {}

}

TORCH_LIBRARY_IMPL(torchsky, CPU, m) {
    m.impl("wcs_pixel_to_world", TORCH_FN(torchsky::wcs_pixel_to_world_cpu));
    m.impl("wcs_world_to_pixel", TORCH_FN(torchsky::wcs_world_to_pixel_cpu));
}
