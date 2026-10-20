#pragma once

#include <string>

#include "torchsky_torch.h"

namespace torchsky {

std::pair<torch::Tensor, torch::Tensor> wcs_inverse_spherical_rotation_pole(
    const torch::Tensor& ra_deg,
    const torch::Tensor& dec_deg,
    double phi_p_rad,
    double east_x,
    double east_y,
    double north_x,
    double north_y,
    double north_z,
    double radial_x,
    double radial_y,
    double radial_z
);
std::pair<torch::Tensor, torch::Tensor> wcs_spherical_rotation_pole(
    const torch::Tensor& phi_deg,
    const torch::Tensor& theta_deg,
    double phi_p_rad,
    double east_x,
    double east_y,
    double north_x,
    double north_y,
    double north_z,
    double radial_x,
    double radial_y,
    double radial_z
);
std::pair<torch::Tensor, torch::Tensor> wcs_tan_intermediate_from_radec(
    const torch::Tensor& ra_deg,
    const torch::Tensor& dec_deg,
    double east_x,
    double east_y,
    double north_x,
    double north_y,
    double north_z,
    double radial_x,
    double radial_y,
    double radial_z
);
std::pair<torch::Tensor, torch::Tensor> wcs_zenithal_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg,
    const std::string& proj_code
);
std::pair<torch::Tensor, torch::Tensor> wcs_zpn_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg,
    const torch::Tensor& coeffs
);
std::pair<torch::Tensor, torch::Tensor> wcs_zenithal_deproject(
    const torch::Tensor& phi_deg,
    const torch::Tensor& theta_deg,
    const std::string& proj_code
);
std::pair<torch::Tensor, torch::Tensor> wcs_cylindrical_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg,
    const std::string& proj_code,
    double lambda_param
);
std::pair<torch::Tensor, torch::Tensor> wcs_cylindrical_deproject(
    const torch::Tensor& phi_deg,
    const torch::Tensor& theta_deg,
    const std::string& proj_code,
    double lambda_param
);
std::pair<torch::Tensor, torch::Tensor> wcs_center_pixel_to_world(
    const torch::Tensor& x_in,
    const torch::Tensor& y_in,
    double origin,
    double crpix0,
    double crpix1,
    double cd00,
    double cd01,
    double cd10,
    double cd11,
    const std::string& proj_code,
    double alpha0,
    double cea_eta_scale,
    double hpx_H,
    double hpx_K
);
std::pair<torch::Tensor, torch::Tensor> wcs_center_world_to_pixel(
    const torch::Tensor& ra_deg,
    const torch::Tensor& dec_deg,
    double crpix0,
    double crpix1,
    double cdi00,
    double cdi01,
    double cdi10,
    double cdi11,
    const std::string& proj_code,
    double alpha0,
    double cea_eta_scale,
    double origin,
    double hpx_H,
    double hpx_K
);
std::pair<torch::Tensor, torch::Tensor> wcs_ait_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg
);
std::pair<torch::Tensor, torch::Tensor> wcs_ait_deproject(
    const torch::Tensor& phi_deg,
    const torch::Tensor& theta_deg
);
std::pair<torch::Tensor, torch::Tensor> wcs_mol_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg
);
std::pair<torch::Tensor, torch::Tensor> wcs_mol_deproject(
    const torch::Tensor& phi_deg,
    const torch::Tensor& theta_deg
);
std::pair<torch::Tensor, torch::Tensor> wcs_hpx_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg,
    double H,
    double K
);
std::pair<torch::Tensor, torch::Tensor> wcs_hpx_deproject(
    const torch::Tensor& phi_deg,
    const torch::Tensor& theta_deg,
    double H,
    double K
);
std::pair<torch::Tensor, torch::Tensor> wcs_tpv_invert(
    const torch::Tensor& xi_target,
    const torch::Tensor& eta_target,
    const torch::Tensor& idx1,
    const torch::Tensor& c1,
    const torch::Tensor& idx2,
    const torch::Tensor& c2,
    int64_t max_iter,
    double tol
);
std::pair<torch::Tensor, torch::Tensor> wcs_tpv_distort(
    const torch::Tensor& u_in,
    const torch::Tensor& v_in,
    const torch::Tensor& idx1,
    const torch::Tensor& c1,
    const torch::Tensor& idx2,
    const torch::Tensor& c2
);
std::pair<torch::Tensor, torch::Tensor> wcs_sip_quadratic_distort(
    const torch::Tensor& u_in,
    const torch::Tensor& v_in,
    double a20,
    double a11,
    double a02,
    double b20,
    double b11,
    double b02
);
std::pair<torch::Tensor, torch::Tensor> wcs_csc_x2s(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg
);
std::pair<torch::Tensor, torch::Tensor> wcs_csc_s2x(
    const torch::Tensor& phi,
    const torch::Tensor& theta
);

void wcs_pixel_to_world_fused_cpu(
    const torch::Tensor& x_tensor,
    const torch::Tensor& y_tensor,
    torch::Tensor& ra_tensor,
    torch::Tensor& dec_tensor,
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
void wcs_world_to_pixel_fused_cpu(
    const torch::Tensor& ra_tensor,
    const torch::Tensor& dec_tensor,
    torch::Tensor& x_tensor,
    torch::Tensor& y_tensor,
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

}
