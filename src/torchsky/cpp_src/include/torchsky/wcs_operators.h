#pragma once

#include <string>
#include <tuple>

#include "torchsky_torch.h"

namespace torchsky {

using WcsTensorPair = std::tuple<torch::Tensor, torch::Tensor>;

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
);

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
);

void ensure_wcs_operator_schemas_linked();
void ensure_wcs_cpu_operators_linked();

}
