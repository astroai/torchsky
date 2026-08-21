#include <ATen/ATen.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace torchsky {

constexpr double WCS_PI = 3.14159265358979323846;
constexpr double D2R = WCS_PI / 180.0;
constexpr double R2D = 180.0 / WCS_PI;
constexpr double SQRT2 = 1.4142135623730950488;

namespace {

inline double wrap_lon360_scalar(double a) {
    double w = std::fmod(a, 360.0);
    if (w < 0.0) w += 360.0;
    return w;
}

inline double wrap_lon180_scalar(double a) {
    double w = std::fmod(a + 180.0, 360.0);
    if (w < 0.0) w += 360.0;
    return w - 180.0;
}

struct TpvAffineSeedParams {
    double b1 = 0.0;
    double b2 = 0.0;
    double a11 = 0.0;
    double a12 = 0.0;
    double a21 = 0.0;
    double a22 = 0.0;
    double inv11 = 0.0;
    double inv12 = 0.0;
    double inv21 = 0.0;
    double inv22 = 0.0;
    bool valid = false;
    bool affine_only = true;
};

using TpvPowCache = std::array<torch::Tensor, 8>;

TpvPowCache tpv_make_pow_cache(const torch::Tensor& base) {
    TpvPowCache pows;
    pows[0] = torch::ones_like(base);
    pows[1] = base;
    for (int i = 2; i <= 7; ++i) {
        pows[i] = pows[i - 1] * base;
    }
    return pows;
}

TpvAffineSeedParams tpv_extract_affine_seed_params(
    const torch::Tensor& idx1,
    const torch::Tensor& c1,
    const torch::Tensor& idx2,
    const torch::Tensor& c2
) {
    TpvAffineSeedParams p;
    auto idx1_acc = idx1.accessor<int64_t, 2>();
    auto c1_acc = c1.accessor<double, 1>();
    const int64_t n1 = c1.size(0);
    for (int64_t k = 0; k < n1; ++k) {
        const int64_t px = idx1_acc[k][0];
        const int64_t py = idx1_acc[k][1];
        const int64_t pr = idx1_acc[k][2];
        const double coeff = c1_acc[k];
        if (coeff == 0.0) continue;
        if (pr != 0) {
            p.affine_only = false;
            continue;
        }
        if (px == 0 && py == 0) p.b1 += coeff;
        else if (px == 1 && py == 0) p.a11 += coeff;
        else if (px == 0 && py == 1) p.a12 += coeff;
        else p.affine_only = false;
    }

    // Axis-2 TPV convention is swapped in evaluation: idx2 (px, py) corresponds to
    // eta(v, u), so px is power of v and py is power of u.
    auto idx2_acc = idx2.accessor<int64_t, 2>();
    auto c2_acc = c2.accessor<double, 1>();
    const int64_t n2 = c2.size(0);
    for (int64_t k = 0; k < n2; ++k) {
        const int64_t px = idx2_acc[k][0];
        const int64_t py = idx2_acc[k][1];
        const int64_t pr = idx2_acc[k][2];
        const double coeff = c2_acc[k];
        if (coeff == 0.0) continue;
        if (pr != 0) {
            p.affine_only = false;
            continue;
        }
        if (px == 0 && py == 0) p.b2 += coeff;
        else if (px == 0 && py == 1) p.a21 += coeff;  // u coefficient
        else if (px == 1 && py == 0) p.a22 += coeff;  // v coefficient
        else p.affine_only = false;
    }

    const double det = p.a11 * p.a22 - p.a12 * p.a21;
    p.valid = std::isfinite(det) && std::abs(det) > 1e-18;
    if (p.valid) {
        p.inv11 = p.a22 / det;
        p.inv12 = -p.a12 / det;
        p.inv21 = -p.a21 / det;
        p.inv22 = p.a11 / det;
    }
    return p;
}

std::pair<torch::Tensor, torch::Tensor> tpv_initial_guess_affine(
    const torch::Tensor& xi_target,
    const torch::Tensor& eta_target,
    const torch::Tensor& idx1,
    const torch::Tensor& c1,
    const torch::Tensor& idx2,
    const torch::Tensor& c2
) {
    auto p = tpv_extract_affine_seed_params(idx1, c1, idx2, c2);
    if (!p.valid) {
        return {xi_target.clone(), eta_target.clone()};
    }

    const double det = p.a11 * p.a22 - p.a12 * p.a21;
    const double inv11 = p.a22 / det;
    const double inv12 = -p.a12 / det;
    const double inv21 = -p.a21 / det;
    const double inv22 = p.a11 / det;

    auto dx = xi_target - p.b1;
    auto dy = eta_target - p.b2;
    auto u = dx * inv11 + dy * inv12;
    auto v = dx * inv21 + dy * inv22;
    return {u, v};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tpv_eval_axis_with_jacobian(
    const torch::Tensor& base_u,
    const torch::Tensor& idx,
    const torch::Tensor& coeffs,
    const TpvPowCache& xc,
    const TpvPowCache& yc,
    const TpvPowCache& r_p,
    const torch::Tensor& dr_dx,
    const torch::Tensor& dr_dy
) {
    auto out = torch::zeros_like(base_u);
    auto d_out_dx = torch::zeros_like(base_u);
    auto d_out_dy = torch::zeros_like(base_u);
    if (coeffs.numel() == 0) {
        return std::make_tuple(out, d_out_dx, d_out_dy);
    }

    auto radial_pref = torch::zeros_like(base_u);
    bool has_radial = false;

    auto idx_acc = idx.accessor<int64_t, 2>();
    auto c_acc = coeffs.accessor<double, 1>();
    const int64_t n_terms = coeffs.size(0);
    for (int64_t k = 0; k < n_terms; ++k) {
        const int64_t px = idx_acc[k][0];
        const int64_t py = idx_acc[k][1];
        const int64_t pr = idx_acc[k][2];
        const double coeff = c_acc[k];

        auto xy = xc[px] * yc[py];
        out.add_(xy * r_p[pr], coeff);

        if (px > 0) {
            d_out_dx.add_(xc[px - 1] * yc[py] * r_p[pr], coeff * static_cast<double>(px));
        }
        if (py > 0) {
            d_out_dy.add_(xc[px] * yc[py - 1] * r_p[pr], coeff * static_cast<double>(py));
        }
        if (pr > 0) {
            radial_pref.add_(xy * r_p[pr - 1], coeff * static_cast<double>(pr));
            has_radial = true;
        }
    }

    if (has_radial) {
        d_out_dx.add_(radial_pref * dr_dx);
        d_out_dy.add_(radial_pref * dr_dy);
    }

    return std::make_tuple(out, d_out_dx, d_out_dy);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tpv_distort_and_jacobian_impl(
    const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& idx1,
    const torch::Tensor& c1,
    const torch::Tensor& idx2,
    const torch::Tensor& c2
) {
    auto r = (u * u + v * v).sqrt();
    auto x_p = tpv_make_pow_cache(u);
    auto y_p = tpv_make_pow_cache(v);
    auto r_p = tpv_make_pow_cache(r);

    auto r_safe = torch::where(r > 0, r, torch::ones_like(r));
    auto dr_du = torch::where(r > 0, u / r_safe, torch::zeros_like(u));
    auto dr_dv = torch::where(r > 0, v / r_safe, torch::zeros_like(v));

    auto [xi, dxi_du, dxi_dv] = tpv_eval_axis_with_jacobian(
        u, idx1, c1, x_p, y_p, r_p, dr_du, dr_dv
    );
    auto [eta, deta_dv, deta_du] = tpv_eval_axis_with_jacobian(
        u, idx2, c2, y_p, x_p, r_p, dr_dv, dr_du
    );

    return std::make_tuple(xi, eta, dxi_du, dxi_dv, deta_du, deta_dv);
}

inline std::tuple<double, double, double, double, double, double> eval_tpv_jac_scalar_raw(
    double u,
    double v,
    const int64_t* idx1_p,
    const double* c1_p,
    int64_t n1,
    const int64_t* idx2_p,
    const double* c2_p,
    int64_t n2)
{
    if (n1 == 0 && n2 == 0) return {u, v, 1.0, 0.0, 0.0, 1.0};
    double r = std::sqrt(u * u + v * v);
    double xp[8] = {1.0, u, u*u, u*u*u, u*u*u*u, u*u*u*u*u, u*u*u*u*u*u, u*u*u*u*u*u*u};
    double yp[8] = {1.0, v, v*v, v*v*v, v*v*v*v, v*v*v*v*v, v*v*v*v*v*v, v*v*v*v*v*v*v};
    double rp[8] = {1.0, r, r*r, r*r*r, r*r*r*r, r*r*r*r*r, r*r*r*r*r*r, r*r*r*r*r*r*r};

    double dr_du = (r > 0) ? u / r : 0.0;
    double dr_dv = (r > 0) ? v / r : 0.0;

    auto eval_axis = [&](const int64_t* idx_p, const double* c_p, int64_t n_terms, const double* a_p, const double* b_p, double dr_da, double dr_db) {
        double out = 0, da = 0, db = 0, dr_pref = 0;
        for (int64_t k = 0; k < n_terms; ++k) {
            int64_t px = idx_p[k*3], py = idx_p[k*3+1], pr = idx_p[k*3+2];
            double coeff = c_p[k];
            out += a_p[px] * b_p[py] * rp[pr] * coeff;
            if (px > 0) da += a_p[px-1] * b_p[py] * rp[pr] * coeff * px;
            if (py > 0) db += a_p[px] * b_p[py-1] * rp[pr] * coeff * py;
            if (pr > 0) dr_pref += a_p[px] * b_p[py] * rp[pr-1] * coeff * pr;
        }
        da += dr_pref * dr_da;
        db += dr_pref * dr_db;
        return std::make_tuple(out, da, db);
    };

    auto [xi, j11, j12] = eval_axis(idx1_p, c1_p, n1, xp, yp, dr_du, dr_dv);
    auto [eta, j22, j21] = eval_axis(idx2_p, c2_p, n2, yp, xp, dr_dv, dr_du);
    return {xi, eta, j11, j12, j21, j22};
}

inline std::pair<double, double> invert_tpv_scalar_with_seed(
    double xi_target, double eta_target,
    const TpvAffineSeedParams& p,
    const int64_t* idx1_p,
    const double* c1_p,
    int64_t n1,
    const int64_t* idx2_p,
    const double* c2_p,
    int64_t n2)
{
    if (n1 == 0 && n2 == 0) return {xi_target, eta_target};
    double u, v;
    if (p.valid) {
        double dx = xi_target - p.b1, dy = eta_target - p.b2;
        u = dx * p.inv11 + dy * p.inv12;
        v = dx * p.inv21 + dy * p.inv22;
    } else {
        u = xi_target; v = eta_target;
    }
    if (p.valid && p.affine_only) {
        return {u, v};
    }

    double tol2 = 1e-22;
    for (int i = 0; i < 20; ++i) {
        auto [xi, eta, j11, j12, j21, j22] = eval_tpv_jac_scalar_raw(
            u,
            v,
            idx1_p,
            c1_p,
            n1,
            idx2_p,
            c2_p,
            n2
        );
        double rx = xi - xi_target, ry = eta - eta_target;
        if (rx*rx + ry*ry < tol2) break;
        double det = j11 * j22 - j12 * j21;
        if (std::abs(det) < 1e-18) det = (det >= 0) ? 1e-18 : -1e-18;
        double du = -(j22 * rx - j12 * ry) / det;
        double dv = -(-j21 * rx + j11 * ry) / det;
        u += std::max(-1.0, std::min(1.0, du));
        v += std::max(-1.0, std::min(1.0, dv));
    }
    return {u, v};
}

inline std::pair<double, double> invert_tpv_scalar(
    double xi_target, double eta_target,
    const torch::Tensor& idx1, const torch::Tensor& c1,
    const torch::Tensor& idx2, const torch::Tensor& c2)
{
    return invert_tpv_scalar_with_seed(
        xi_target,
        eta_target,
        tpv_extract_affine_seed_params(idx1, c1, idx2, c2),
        idx1.data_ptr<int64_t>(),
        c1.data_ptr<double>(),
        c1.size(0),
        idx2.data_ptr<int64_t>(),
        c2.data_ptr<double>(),
        c2.size(0)
    );
}

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
) {
    auto ra_rad = ra_deg * D2R;
    auto dec_rad = dec_deg * D2R;
    auto sin_dec = dec_rad.sin();
    auto cos_dec = dec_rad.cos();
    auto sin_ra = ra_rad.sin();
    auto cos_ra = ra_rad.cos();

    auto X = cos_dec * cos_ra;
    auto Y = cos_dec * sin_ra;
    auto Z = sin_dec;

    auto u = X * east_x + Y * east_y;
    auto n = X * north_x + Y * north_y + Z * north_z;
    auto w = X * radial_x + Y * radial_y + Z * radial_z;
    w = w.clamp(-1.0, 1.0);

    auto theta = w.asin() * R2D;
    auto phi = (torch::atan2(u, n) + phi_p_rad) * R2D;
    phi = torch::remainder(phi, 360.0);
    return {phi, theta};
}

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
) {
    auto phi_rad = phi_deg * D2R - phi_p_rad;
    auto theta_rad = theta_deg * D2R;

    auto sin_theta = theta_rad.sin();
    auto cos_theta = theta_rad.cos();
    auto sin_phi = phi_rad.sin();
    auto cos_phi = phi_rad.cos();

    auto u = cos_theta * sin_phi;
    auto v = cos_theta * cos_phi;
    auto w = sin_theta;

    auto X = u * east_x + v * north_x + w * radial_x;
    auto Y = u * east_y + v * north_y + w * radial_y;
    auto Z = v * north_z + w * radial_z;
    Z = Z.clamp(-1.0, 1.0);

    auto ra = torch::remainder(torch::atan2(Y, X) * R2D, 360.0);
    auto dec = Z.asin() * R2D;
    return {ra, dec};
}

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
) {
    auto ra_rad = ra_deg * D2R;
    auto dec_rad = dec_deg * D2R;
    auto sin_dec = dec_rad.sin();
    auto cos_dec = dec_rad.cos();
    auto sin_ra = ra_rad.sin();
    auto cos_ra = ra_rad.cos();

    auto X = cos_dec * cos_ra;
    auto Y = cos_dec * sin_ra;
    auto Z = sin_dec;

    auto u = X * east_x + Y * east_y;
    auto v = X * north_x + Y * north_y + Z * north_z;
    auto w = X * radial_x + Y * radial_y + Z * radial_z;
    auto scale = R2D / torch::clamp(w, 1e-12);
    return {u * scale, v * scale};
}
}

std::pair<torch::Tensor, torch::Tensor> wcs_zenithal_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg,
    const std::string& proj_code
) {
    auto r = (xi_deg * xi_deg + eta_deg * eta_deg).sqrt();
    auto r_rad = r * D2R;
    auto phi = torch::atan2(xi_deg, -eta_deg) * R2D;

    torch::Tensor theta;

    if (proj_code == "TAN") {
        theta = torch::atan2(torch::ones_like(r_rad), r_rad) * R2D;
    } else if (proj_code == "SIN") {
        auto invalid = r_rad > 1.0;
        auto r_dim = r_rad.clamp(-1.0, 1.0);
        theta = r_dim.acos() * R2D;
        theta = torch::where(invalid, torch::full_like(theta, std::numeric_limits<double>::quiet_NaN()), theta);
    } else if (proj_code == "ARC") {
        theta = 90.0 - r;
    } else if (proj_code == "ZEA") {
        auto val = (r_rad / 2.0).clamp(-1.0, 1.0);
        theta = 90.0 - 2.0 * val.asin() * R2D;
    } else if (proj_code == "STG") {
        theta = 90.0 - 2.0 * (r_rad / 2.0).atan() * R2D;
    } else {
        throw std::runtime_error("Unknown zenithal projection: " + proj_code);
    }

    return {phi, theta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_zpn_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg,
    const torch::Tensor& coeffs
) {
    if (
        xi_deg.device().is_cpu() && eta_deg.device().is_cpu() &&
        xi_deg.scalar_type() == torch::kFloat64 && eta_deg.scalar_type() == torch::kFloat64 &&
        coeffs.device().is_cpu() && coeffs.scalar_type() == torch::kFloat64
    ) {
        auto xi_flat = xi_deg.contiguous().view({-1});
        auto eta_flat = eta_deg.contiguous().view({-1});
        auto phi_flat = torch::empty_like(xi_flat);
        auto theta_flat = torch::empty_like(eta_flat);

        auto coeffs_c = coeffs.contiguous().view({-1});
        const auto coeff_n = coeffs_c.numel();
        const double* coeff_ptr = coeffs_c.data_ptr<double>();
        int64_t max_m = -1;
        for (int64_t i = coeff_n - 1; i >= 0; --i) {
            if (std::abs(coeff_ptr[i]) > 1e-15) {
                max_m = i;
                break;
            }
        }
        const bool linear_case = (max_m == 1 && std::abs(coeff_ptr[1]) > 1e-15);
        const double c0 = (coeff_n >= 1) ? coeff_ptr[0] : 0.0;
        const double c1 = (coeff_n >= 2) ? coeff_ptr[1] : 1.0;

        const auto n = xi_flat.numel();
        const double* xi_ptr = xi_flat.data_ptr<double>();
        const double* eta_ptr = eta_flat.data_ptr<double>();
        double* phi_ptr = phi_flat.data_ptr<double>();
        double* theta_ptr = theta_flat.data_ptr<double>();

        for (int64_t i = 0; i < n; ++i) {
            const double xi = xi_ptr[i];
            const double eta = eta_ptr[i];
            const double r = std::hypot(xi, eta);
            const double phi = std::atan2(-xi, -eta) * R2D;
            const double r_rad_target = r * D2R;

            double w = r_rad_target;
            if (linear_case) {
                w = (r_rad_target - c0) / c1;
            } else if (max_m >= 0) {
                for (int iter = 0; iter < 12; ++iter) {
                    double val = coeff_ptr[max_m];
                    double der = 0.0;
                    for (int64_t j = max_m - 1; j >= 0; --j) {
                        der = der * w + val;
                        val = val * w + coeff_ptr[j];
                    }
                    const double diff = val - r_rad_target;
                    if (std::abs(diff) < 1e-12) {
                        break;
                    }
                    if (std::abs(der) < 1e-15) {
                        break;
                    }
                    w -= diff / der;
                }
            }

            bool valid = std::isfinite(w) && w >= 0.0 && w <= WCS_PI;
            if (valid && max_m >= 0) {
                double val = coeff_ptr[max_m];
                for (int64_t j = max_m - 1; j >= 0; --j) {
                    val = val * w + coeff_ptr[j];
                }
                valid = std::abs(val - r_rad_target) <= 1e-10;
            }

            phi_ptr[i] = phi;
            theta_ptr[i] = valid ? (90.0 - w * R2D) : std::numeric_limits<double>::quiet_NaN();
        }

        return {phi_flat.view(xi_deg.sizes()), theta_flat.view(eta_deg.sizes())};
    }

    auto p_coeffs = coeffs.contiguous().view({-1});
    auto mask_valid = p_coeffs.abs() > 1e-15;
    if (!mask_valid.any().item<bool>()) {
        auto r = (xi_deg * xi_deg + eta_deg * eta_deg).sqrt();
        auto phi = torch::atan2(-xi_deg, -eta_deg) * R2D;
        auto theta = 90.0 - (r * D2R) * R2D;
        return {phi, theta};
    }

    auto valid_idx = torch::nonzero(mask_valid).squeeze(1);
    auto max_idx = valid_idx.max().item<int64_t>();
    auto coeff_used = p_coeffs.narrow(0, 0, max_idx + 1);
    auto r = (xi_deg * xi_deg + eta_deg * eta_deg).sqrt();
    auto phi = torch::atan2(-xi_deg, -eta_deg) * R2D;
    auto r_rad_target = r * D2R;
    if (max_idx == 1) {
        auto c0 = coeff_used[0];
        auto c1 = coeff_used[1];
        auto c1_safe = torch::where(
            c1.abs() < 1e-15,
            torch::ones_like(c1),
            c1
        );
        auto w = (r_rad_target - c0) / c1_safe;
        auto valid = torch::isfinite(w) & (w >= 0.0) & (w <= WCS_PI);
        auto val = c0 + c1 * w;
        valid = valid & ((val - r_rad_target).abs() <= 1e-10);
        w = torch::where(valid, w, torch::full_like(w, std::numeric_limits<double>::quiet_NaN()));
        auto theta = 90.0 - w * R2D;
        return {phi, theta};
    }
    auto w = r_rad_target.clone();
    for (int iter = 0; iter < 12; ++iter) {
        auto val = torch::full_like(w, coeff_used[max_idx].item<double>());
        auto der = torch::zeros_like(w);
        for (int64_t j = max_idx - 1; j >= 0; --j) {
            der = der * w + val;
            val = val * w + coeff_used[j];
        }
        auto diff = val - r_rad_target;
        if (diff.abs().max().item<double>() < 1e-12) {
            break;
        }
        auto der_safe = torch::where(der.abs() < 1e-15, torch::ones_like(der), der);
        w = w - diff / der_safe;
    }
    auto val = torch::full_like(w, coeff_used[max_idx].item<double>());
    for (int64_t j = max_idx - 1; j >= 0; --j) {
        val = val * w + coeff_used[j];
    }
    auto valid = torch::isfinite(w) & (w >= 0.0) & (w <= WCS_PI) & ((val - r_rad_target).abs() <= 1e-10);
    w = torch::where(valid, w, torch::full_like(w, std::numeric_limits<double>::quiet_NaN()));
    auto theta = 90.0 - w * R2D;
    return {phi, theta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_zenithal_deproject(
    const torch::Tensor& phi_deg,
    const torch::Tensor& theta_deg,
    const std::string& proj_code
) {
    torch::Tensor r;
    auto theta_rad = theta_deg * D2R;

    if (proj_code == "TAN") {
        auto tan_theta = theta_rad.tan();
        r = R2D / (tan_theta + 1e-12);
    } else if (proj_code == "SIN") {
        r = theta_rad.cos() * R2D;
    } else if (proj_code == "ARC") {
        r = 90.0 - theta_deg;
    } else if (proj_code == "ZEA") {
        r = 2.0 * ((90.0 - theta_deg) * D2R / 2.0).sin() * R2D;
    } else if (proj_code == "STG") {
        r = 2.0 * ((90.0 - theta_deg) * D2R / 2.0).tan() * R2D;
    } else {
        throw std::runtime_error("Unknown zenithal projection: " + proj_code);
    }

    auto phi_rad = phi_deg * D2R;
    auto xi = r * phi_rad.sin();
    auto eta = -r * phi_rad.cos();

    return {xi, eta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_cylindrical_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg,
    const std::string& proj_code,
    double lambda_param
) {
    torch::Tensor phi, theta;

    if (proj_code == "CEA") {
        phi = xi_deg;
        auto val = (lambda_param * eta_deg * D2R).clamp(-1.0, 1.0);
        theta = val.asin() * R2D;
    } else if (proj_code == "MER") {
        phi = xi_deg;
        auto arg = (eta_deg * D2R).clamp(-20.0, 20.0);
        auto exp_val = arg.exp();
        theta = 2.0 * (exp_val.atan() * R2D - 45.0);
    } else {
        throw std::runtime_error("Unknown cylindrical projection: " + proj_code);
    }

    return {phi, theta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_cylindrical_deproject(
    const torch::Tensor& phi_deg,
    const torch::Tensor& theta_deg,
    const std::string& proj_code,
    double lambda_param
) {
    torch::Tensor xi, eta;

    if (proj_code == "CEA") {
        xi = phi_deg;
        eta = (R2D / lambda_param) * (theta_deg * D2R).sin();
    } else if (proj_code == "MER") {
        xi = phi_deg;
        auto half_theta = theta_deg / 2.0;
        auto tan_arg = ((45.0 + half_theta) * D2R).tan().clamp_min(1e-12);
        eta = R2D * tan_arg.log();
    } else {
        throw std::runtime_error("Unknown cylindrical projection: " + proj_code);
    }

    return {xi, eta};
}

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
) {
    auto x_flat = x_in.contiguous().view({-1});
    auto y_flat = y_in.contiguous().view({-1});
    auto ra_flat = torch::empty_like(x_flat);
    auto dec_flat = torch::empty_like(y_flat);

    const auto n = x_flat.numel();
    const double* x_ptr = x_flat.data_ptr<double>();
    const double* y_ptr = y_flat.data_ptr<double>();
    double* ra_ptr = ra_flat.data_ptr<double>();
    double* dec_ptr = dec_flat.data_ptr<double>();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double pix_shift = 1.0 - origin;
    const double hpx_eta_scale = 90.0 * (hpx_K / hpx_H);
    const double hpx_eta_boundary = hpx_eta_scale * (2.0 / 3.0);
    const double hpx_eta_pole = 90.0;
    const double hpx_polar_denom = hpx_eta_pole - hpx_eta_boundary;

    for (int64_t i = 0; i < n; ++i) {
            const double rel_x = x_ptr[i] + pix_shift - crpix0;
            const double rel_y = y_ptr[i] + pix_shift - crpix1;
            const double xi = cd00 * rel_x + cd01 * rel_y;
            const double eta = cd10 * rel_x + cd11 * rel_y;

            if (proj_code == "CAR") {
                ra_ptr[i] = wrap_lon360_scalar(xi + alpha0);
                dec_ptr[i] = eta;
            } else if (proj_code == "SFL") {
                const double dec = eta;
                double c = std::cos(dec * D2R);
                if (std::abs(c) < 1e-12) c = (c < 0.0) ? -1e-12 : 1e-12;
                ra_ptr[i] = wrap_lon360_scalar((xi / c) + alpha0);
                dec_ptr[i] = dec;
            } else if (proj_code == "CEA") {
                const double s = std::clamp(eta / cea_eta_scale, -1.0, 1.0);
                ra_ptr[i] = wrap_lon360_scalar(xi + alpha0);
                dec_ptr[i] = std::asin(s) * R2D;
            } else if (proj_code == "MER") {
                const double arg = std::clamp(eta * D2R, -20.0, 20.0);
                ra_ptr[i] = wrap_lon360_scalar(xi + alpha0);
                dec_ptr[i] = 2.0 * (std::atan(std::exp(arg)) * R2D - 45.0);
            } else if (proj_code == "AIT") {
                const double X = xi * D2R;
                const double Y = eta * D2R;
                const double xq = X * 0.25;
                const double yq = Y * 0.5;
                const double r2 = xq * xq + yq * yq;
                if (r2 > 1.0) {
                    ra_ptr[i] = nan;
                    dec_ptr[i] = nan;
                    continue;
                }
                const double z = std::sqrt(std::max(0.0, 1.0 - r2));
                const double z2 = z * z;
                const double phi = 2.0 * std::atan2(0.5 * z * X, 2.0 * z2 - 1.0) * R2D;
                const double s = std::clamp(z * Y, -1.0, 1.0);
                const double theta = std::asin(s) * R2D;
                ra_ptr[i] = wrap_lon360_scalar(phi + alpha0);
                dec_ptr[i] = theta;
            } else if (proj_code == "MOL") {
                const double X = xi * D2R;
                const double Y = eta * D2R;
                if (std::abs(Y) > SQRT2) {
                    ra_ptr[i] = nan;
                    dec_ptr[i] = nan;
                    continue;
                }
                const double sin_gamma = std::clamp(Y / SQRT2, -1.0, 1.0);
                const double gamma = std::asin(sin_gamma);
                const double cos_gamma = std::sqrt(std::max(0.0, 1.0 - sin_gamma * sin_gamma));
                const double t_val = std::clamp(
                    (2.0 * gamma + 2.0 * sin_gamma * cos_gamma) / WCS_PI, -1.0, 1.0
                );
                const double theta = std::asin(t_val) * R2D;
                const double denom = 2.0 * SQRT2 * cos_gamma;
                double phi = 0.0;
                if (std::abs(denom) >= 1e-12) {
                    phi = WCS_PI * X / denom * R2D;
                }
                ra_ptr[i] = wrap_lon360_scalar(phi + alpha0);
                dec_ptr[i] = theta;
            } else if (proj_code == "HPX") {
                const double abs_eta = std::abs(eta);
                if (abs_eta > hpx_eta_pole) {
                    ra_ptr[i] = nan;
                    dec_ptr[i] = nan;
                    continue;
                }
                double phi = xi;
                double theta = 0.0;
                if (abs_eta <= hpx_eta_boundary) {
                    const double s = std::clamp(eta / hpx_eta_scale, -1.0, 1.0);
                    theta = std::asin(s) * R2D;
                } else {
                    const double sigma = std::max((hpx_eta_pole - abs_eta) / hpx_polar_denom, 0.0);
                    const double s = std::clamp(std::copysign(1.0, eta) * (1.0 - (sigma * sigma) / 3.0), -1.0, 1.0);
                    theta = std::asin(s) * R2D;
                    const double xc = std::round((xi - 45.0) / 90.0) * 90.0 + 45.0;
                    const double dx = xi - xc;
                    if (std::abs(dx) > (45.0 * sigma + 1e-8)) {
                        ra_ptr[i] = nan;
                        dec_ptr[i] = nan;
                        continue;
                    }
                    const double sigma_safe = (std::abs(sigma) < 1e-12) ? 1.0 : sigma;
                    phi = xc + dx / sigma_safe;
                }
                ra_ptr[i] = wrap_lon360_scalar(phi + alpha0);
                dec_ptr[i] = theta;
            } else {
                ra_ptr[i] = wrap_lon360_scalar(xi + alpha0);
                dec_ptr[i] = eta;
            }
    }

    return {ra_flat.view(x_in.sizes()), dec_flat.view(y_in.sizes())};
}

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
) {
    auto ra_flat = ra_deg.contiguous().view({-1});
    auto dec_flat = dec_deg.contiguous().view({-1});
    auto px_flat = torch::empty_like(ra_flat);
    auto py_flat = torch::empty_like(dec_flat);

    const auto n = ra_flat.numel();
    const double* ra_ptr = ra_flat.data_ptr<double>();
    const double* dec_ptr = dec_flat.data_ptr<double>();
    double* px_ptr = px_flat.data_ptr<double>();
    double* py_ptr = py_flat.data_ptr<double>();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double hpx_eta_scale = 90.0 * (hpx_K / hpx_H);
    const double hpx_eta_boundary = hpx_eta_scale * (2.0 / 3.0);

    for (int64_t i = 0; i < n; ++i) {
        const double theta = dec_ptr[i];
        const double phi = wrap_lon180_scalar(ra_ptr[i] - alpha0);

        double xi = phi;
        double eta = theta;
        if (proj_code == "SFL") {
            xi = phi * std::cos(theta * D2R);
            eta = theta;
        } else if (proj_code == "CEA") {
            xi = phi;
            eta = cea_eta_scale * std::sin(theta * D2R);
        } else if (proj_code == "MER") {
            xi = phi;
            double tan_arg = std::tan((45.0 + theta * 0.5) * D2R);
            if (tan_arg < 1e-12) tan_arg = 1e-12;
            eta = R2D * std::log(tan_arg);
        } else if (proj_code == "AIT") {
            const double phi_rad = phi * D2R;
            const double theta_rad = theta * D2R;
            const double half_phi = 0.5 * phi_rad;
            const double cos_theta = std::cos(theta_rad);
            const double sin_theta = std::sin(theta_rad);
            const double denom2 = 0.5 * (1.0 + cos_theta * std::cos(half_phi));
            if (denom2 <= 0.0) {
                px_ptr[i] = nan;
                py_ptr[i] = nan;
                continue;
            }
            const double denom = std::sqrt(denom2);
            xi = 2.0 * cos_theta * std::sin(half_phi) / denom * R2D;
            eta = sin_theta / denom * R2D;
        } else if (proj_code == "HPX") {
            const double s = std::sin(theta * D2R);
            const double abs_s = std::abs(s);
            if (abs_s <= (2.0 / 3.0)) {
                xi = phi;
                eta = hpx_eta_scale * s;
            } else {
                const double sigma = std::sqrt(std::max(3.0 * (1.0 - abs_s), 0.0));
                const double xc = std::round((phi - 45.0) / 90.0) * 90.0 + 45.0;
                xi = xc + sigma * (phi - xc);
                eta = std::copysign(1.0, s) * (90.0 - (90.0 - hpx_eta_boundary) * sigma);
            }
        }

        const double rel_x = cdi00 * xi + cdi01 * eta;
        const double rel_y = cdi10 * xi + cdi11 * eta;
        px_ptr[i] = rel_x + crpix0 - (1.0 - origin);
        py_ptr[i] = rel_y + crpix1 - (1.0 - origin);
    }

    return {px_flat.view(ra_deg.sizes()), py_flat.view(dec_deg.sizes())};
}

std::pair<torch::Tensor, torch::Tensor> wcs_ait_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg
) {
    if (
        xi_deg.device().is_cpu() && eta_deg.device().is_cpu() &&
        xi_deg.scalar_type() == torch::kFloat64 && eta_deg.scalar_type() == torch::kFloat64
    ) {
        auto xi_flat = xi_deg.contiguous().view({-1});
        auto eta_flat = eta_deg.contiguous().view({-1});
        auto phi_flat = torch::empty_like(xi_flat);
        auto theta_flat = torch::empty_like(eta_flat);

        const auto n = xi_flat.numel();
        const double* xi_ptr = xi_flat.data_ptr<double>();
        const double* eta_ptr = eta_flat.data_ptr<double>();
        double* phi_ptr = phi_flat.data_ptr<double>();
        double* theta_ptr = theta_flat.data_ptr<double>();
        const double nan = std::numeric_limits<double>::quiet_NaN();

        for (int64_t i = 0; i < n; ++i) {
            const double X = xi_ptr[i] * D2R;
            const double Y = eta_ptr[i] * D2R;
            const double xq = X * 0.25;
            const double yq = Y * 0.5;
            const double r2 = xq * xq + yq * yq;
            if (r2 > 1.0) {
                phi_ptr[i] = nan;
                theta_ptr[i] = nan;
                continue;
            }
            const double z = std::sqrt(std::max(0.0, 1.0 - r2));
            const double z2 = z * z;
            const double phi_rad = 2.0 * std::atan2(0.5 * z * X, 2.0 * z2 - 1.0);
            const double sin_theta = std::clamp(z * Y, -1.0, 1.0);
            const double theta_rad = std::asin(sin_theta);

            phi_ptr[i] = phi_rad * R2D;
            theta_ptr[i] = theta_rad * R2D;
        }

        return {phi_flat.view(xi_deg.sizes()), theta_flat.view(eta_deg.sizes())};
    }

    auto X = xi_deg * D2R;
    auto Y = eta_deg * D2R;

    auto xq = X * 0.25;
    auto yq = Y * 0.5;
    auto r2 = xq * xq + yq * yq;
    auto valid = r2 <= 1.0;

    auto z = (1.0 - r2.clamp_max(1.0)).sqrt();
    auto z2 = z * z;
    auto phi_rad = 2.0 * torch::atan2(0.5 * z * X, 2.0 * z2 - 1.0);
    auto sin_theta = (z * Y).clamp(-1.0, 1.0);
    auto theta_rad = sin_theta.asin();

    auto phi = phi_rad * R2D;
    auto theta = theta_rad * R2D;

    auto invalid = ~valid;
    phi = phi.masked_fill(invalid, std::nan(""));
    theta = theta.masked_fill(invalid, std::nan(""));

    return {phi, theta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_ait_deproject(
    const torch::Tensor& phi_deg,
    const torch::Tensor& theta_deg
) {
    if (
        phi_deg.device().is_cpu() && theta_deg.device().is_cpu() &&
        phi_deg.scalar_type() == torch::kFloat64 && theta_deg.scalar_type() == torch::kFloat64
    ) {
        auto phi_flat = phi_deg.contiguous().view({-1});
        auto theta_flat = theta_deg.contiguous().view({-1});
        auto xi_flat = torch::empty_like(phi_flat);
        auto eta_flat = torch::empty_like(theta_flat);

        const auto n = phi_flat.numel();
        const double* phi_ptr = phi_flat.data_ptr<double>();
        const double* theta_ptr = theta_flat.data_ptr<double>();
        double* xi_ptr = xi_flat.data_ptr<double>();
        double* eta_ptr = eta_flat.data_ptr<double>();
        const double nan = std::numeric_limits<double>::quiet_NaN();

        for (int64_t i = 0; i < n; ++i) {
            const double phi_rad = phi_ptr[i] * D2R;
            const double theta_rad = theta_ptr[i] * D2R;
            const double half_phi = 0.5 * phi_rad;
            const double cos_theta = std::cos(theta_rad);
            const double sin_theta = std::sin(theta_rad);
            const double cos_half_phi = std::cos(half_phi);
            const double denom2 = 0.5 * (1.0 + cos_theta * cos_half_phi);
            if (denom2 <= 0.0) {
                xi_ptr[i] = nan;
                eta_ptr[i] = nan;
                continue;
            }
            const double denom = std::sqrt(denom2);
            xi_ptr[i] = 2.0 * cos_theta * std::sin(half_phi) / denom * R2D;
            eta_ptr[i] = sin_theta / denom * R2D;
        }

        return {xi_flat.view(phi_deg.sizes()), eta_flat.view(theta_deg.sizes())};
    }

    auto phi_rad = phi_deg * D2R;
    auto theta_rad = theta_deg * D2R;

    auto half_phi = phi_rad / 2.0;
    auto cos_theta = theta_rad.cos();
    auto sin_theta = theta_rad.sin();
    auto cos_half_phi = half_phi.cos();

    auto denom = (0.5 * (1.0 + cos_theta * cos_half_phi)).sqrt();

    auto xi = 2.0 * cos_theta * half_phi.sin() / denom * R2D;
    auto eta = sin_theta / denom * R2D;

    return {xi, eta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_mol_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg
) {
    if (
        xi_deg.device().is_cpu() && eta_deg.device().is_cpu() &&
        xi_deg.scalar_type() == torch::kFloat64 && eta_deg.scalar_type() == torch::kFloat64
    ) {
        auto xi_flat = xi_deg.contiguous().view({-1});
        auto eta_flat = eta_deg.contiguous().view({-1});
        auto phi_flat = torch::empty_like(xi_flat);
        auto theta_flat = torch::empty_like(eta_flat);

        const auto n = xi_flat.numel();
        const double* xi_ptr = xi_flat.data_ptr<double>();
        const double* eta_ptr = eta_flat.data_ptr<double>();
        double* phi_ptr = phi_flat.data_ptr<double>();
        double* theta_ptr = theta_flat.data_ptr<double>();
        const double nan = std::numeric_limits<double>::quiet_NaN();

        for (int64_t i = 0; i < n; ++i) {
            const double X = xi_ptr[i] * D2R;
            const double Y = eta_ptr[i] * D2R;
            if (std::abs(Y) > SQRT2) {
                phi_ptr[i] = nan;
                theta_ptr[i] = nan;
                continue;
            }

            const double sin_gamma = std::clamp(Y / SQRT2, -1.0, 1.0);
            const double gamma = std::asin(sin_gamma);
            const double cos_gamma = std::sqrt(std::max(0.0, 1.0 - sin_gamma * sin_gamma));
            const double t_val = std::clamp(
                (2.0 * gamma + 2.0 * sin_gamma * cos_gamma) / WCS_PI, -1.0, 1.0
            );
            const double theta_rad = std::asin(t_val);
            const double denom = 2.0 * SQRT2 * cos_gamma;

            double phi_rad = 0.0;
            if (std::abs(denom) >= 1e-12) {
                phi_rad = WCS_PI * X / denom;
            }

            phi_ptr[i] = phi_rad * R2D;
            theta_ptr[i] = theta_rad * R2D;
        }

        return {phi_flat.view(xi_deg.sizes()), theta_flat.view(eta_deg.sizes())};
    }

    auto X = xi_deg * D2R;
    auto Y = eta_deg * D2R;

    auto valid = Y.abs() <= SQRT2;

    auto sin_gamma = (Y / SQRT2).clamp(-1.0, 1.0);
    auto gamma = sin_gamma.asin();
    auto cos_gamma = (1.0 - sin_gamma * sin_gamma).clamp_min(0.0).sqrt();

    auto t_val = (2.0 * gamma + 2.0 * sin_gamma * cos_gamma) / WCS_PI;
    t_val = t_val.clamp(-1.0, 1.0);
    auto theta_rad = t_val.asin();

    auto denom = 2.0 * SQRT2 * cos_gamma;
    auto good = denom.abs() >= 1e-12;
    auto denom_safe = torch::where(good, denom, torch::ones_like(denom));
    auto phi_rad = WCS_PI * X / denom_safe;
    phi_rad = torch::where(good, phi_rad, torch::zeros_like(phi_rad));

    auto phi = phi_rad * R2D;
    auto theta = theta_rad * R2D;

    auto invalid = ~valid;
    phi = phi.masked_fill(invalid, std::nan(""));
    theta = theta.masked_fill(invalid, std::nan(""));

    return {phi, theta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_mol_deproject(
    const torch::Tensor& phi_deg,
    const torch::Tensor& theta_deg
) {
    auto phi_rad = phi_deg * D2R;
    auto theta_rad = theta_deg * D2R;

    auto sin_theta = theta_rad.sin();

    auto gamma = torch::zeros_like(theta_rad);
    for (int i = 0; i < 10; ++i) {
        auto f = 2.0 * gamma + (2.0 * gamma).sin() - WCS_PI * sin_theta;
        auto fp = 2.0 + 2.0 * (2.0 * gamma).cos();
        gamma = gamma - f / (fp + 1e-12);
    }

    auto cos_gamma = gamma.cos();
    auto sin_gamma = gamma.sin();

    auto xi = 2.0 * SQRT2 * phi_rad * cos_gamma / WCS_PI * R2D;
    auto eta = SQRT2 * sin_gamma * R2D;

    return {xi, eta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_hpx_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg,
    double H,
    double K
) {
    if (
        xi_deg.device().is_cpu() && eta_deg.device().is_cpu() &&
        xi_deg.scalar_type() == torch::kFloat64 && eta_deg.scalar_type() == torch::kFloat64
    ) {
        auto xi_flat = xi_deg.contiguous().view({-1});
        auto eta_flat = eta_deg.contiguous().view({-1});
        auto phi_flat = torch::empty_like(xi_flat);
        auto theta_flat = torch::empty_like(eta_flat);

        const double eta_scale = 90.0 * (K / H);
        const double eta_boundary = eta_scale * (2.0 / 3.0);
        const double eta_pole = 90.0;
        const double polar_denom = eta_pole - eta_boundary;
        const double nan = std::numeric_limits<double>::quiet_NaN();

        const auto n = xi_flat.numel();
        const double* xi_ptr = xi_flat.data_ptr<double>();
        const double* eta_ptr = eta_flat.data_ptr<double>();
        double* phi_ptr = phi_flat.data_ptr<double>();
        double* theta_ptr = theta_flat.data_ptr<double>();

        for (int64_t i = 0; i < n; ++i) {
            const double xi = xi_ptr[i];
            const double eta = eta_ptr[i];
            const double abs_eta = std::abs(eta);
            if (abs_eta > eta_pole) {
                phi_ptr[i] = nan;
                theta_ptr[i] = nan;
                continue;
            }
            if (abs_eta <= eta_boundary) {
                const double s = std::clamp(eta / eta_scale, -1.0, 1.0);
                phi_ptr[i] = xi;
                theta_ptr[i] = std::asin(s) * R2D;
                continue;
            }

            const double sigma = std::max((eta_pole - abs_eta) / polar_denom, 0.0);
            const double s = std::clamp(std::copysign(1.0, eta) * (1.0 - (sigma * sigma) / 3.0), -1.0, 1.0);
            const double theta = std::asin(s) * R2D;
            const double xc = std::round((xi - 45.0) / 90.0) * 90.0 + 45.0;
            const double dx = xi - xc;

            if (std::abs(dx) > (45.0 * sigma + 1e-8)) {
                phi_ptr[i] = nan;
                theta_ptr[i] = nan;
                continue;
            }

            const double sigma_safe = (std::abs(sigma) < 1e-12) ? 1.0 : sigma;
            phi_ptr[i] = xc + dx / sigma_safe;
            theta_ptr[i] = theta;
        }

        return {phi_flat.view(xi_deg.sizes()), theta_flat.view(eta_deg.sizes())};
    }

    double eta_scale = 90.0 * (K / H);
    double eta_boundary = eta_scale * (2.0 / 3.0);
    double eta_pole = 90.0;
    double polar_denom = eta_pole - eta_boundary;

    auto abs_eta = eta_deg.abs();
    auto invalid_eta = abs_eta > eta_pole;
    auto mask_eq = abs_eta <= eta_boundary;
    auto mask_pol = (~mask_eq) & (~invalid_eta);

    auto s_theta_eq = (eta_deg / eta_scale).clamp(-1.0, 1.0);
    auto theta_eq = s_theta_eq.asin() * R2D;

    auto sigma = ((eta_pole - abs_eta) / polar_denom).clamp_min(0.0);
    auto s_theta_pol = (eta_deg.sign() * (1.0 - (sigma * sigma) / 3.0)).clamp(-1.0, 1.0);
    auto theta_pol = s_theta_pol.asin() * R2D;

    auto xc = ((xi_deg - 45.0) / 90.0).round() * 90.0 + 45.0;
    auto dx = xi_deg - xc;
    auto sigma_safe = torch::where(sigma.abs() < 1e-12, torch::ones_like(sigma), sigma);
    auto phi_pol = xc + dx / sigma_safe;

    auto invalid_x = dx.abs() > (45.0 * sigma + 1e-8);
    auto invalid = invalid_eta | (mask_pol & invalid_x);

    auto phi = torch::where(mask_eq, xi_deg, phi_pol);
    auto theta = torch::where(mask_eq, theta_eq, theta_pol);
    phi = phi.masked_fill(invalid, std::nan(""));
    theta = theta.masked_fill(invalid, std::nan(""));

    return {phi, theta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_hpx_deproject(
    const torch::Tensor& phi_deg,
    const torch::Tensor& theta_deg,
    double H,
    double K
) {
    if (
        phi_deg.device().is_cpu() && theta_deg.device().is_cpu() &&
        phi_deg.scalar_type() == torch::kFloat64 && theta_deg.scalar_type() == torch::kFloat64
    ) {
        auto phi_flat = phi_deg.contiguous().view({-1});
        auto theta_flat = theta_deg.contiguous().view({-1});
        auto xi_flat = torch::empty_like(phi_flat);
        auto eta_flat = torch::empty_like(theta_flat);

        const double eta_scale = 90.0 * (K / H);
        const double eta_boundary = eta_scale * (2.0 / 3.0);

        const auto n = phi_flat.numel();
        const double* phi_ptr = phi_flat.data_ptr<double>();
        const double* theta_ptr = theta_flat.data_ptr<double>();
        double* xi_ptr = xi_flat.data_ptr<double>();
        double* eta_ptr = eta_flat.data_ptr<double>();

        for (int64_t i = 0; i < n; ++i) {
            const double phi = phi_ptr[i];
            const double theta = theta_ptr[i];
            const double s = std::sin(theta * D2R);
            const double abs_s = std::abs(s);
            if (abs_s <= (2.0 / 3.0)) {
                xi_ptr[i] = phi;
                eta_ptr[i] = eta_scale * s;
                continue;
            }

            const double sigma = std::sqrt(std::max(3.0 * (1.0 - abs_s), 0.0));
            const double xc = std::round((phi - 45.0) / 90.0) * 90.0 + 45.0;
            xi_ptr[i] = xc + sigma * (phi - xc);
            eta_ptr[i] = std::copysign(1.0, s) * (90.0 - (90.0 - eta_boundary) * sigma);
        }

        return {xi_flat.view(phi_deg.sizes()), eta_flat.view(theta_deg.sizes())};
    }

    auto s_theta = (theta_deg * D2R).sin();
    auto abs_s = s_theta.abs();

    double eta_scale = 90.0 * (K / H);
    double eta_boundary = eta_scale * (2.0 / 3.0);

    auto mask_eq = abs_s <= (2.0 / 3.0);
    auto sigma = (3.0 * (1.0 - abs_s)).clamp_min(0.0).sqrt();
    auto eta_eq = eta_scale * s_theta;
    auto eta_pol = s_theta.sign() * (90.0 - (90.0 - eta_boundary) * sigma);

    auto xc = ((phi_deg - 45.0) / 90.0).round() * 90.0 + 45.0;
    auto xi_pol = xc + sigma * (phi_deg - xc);

    auto xi = torch::where(mask_eq, phi_deg, xi_pol);
    auto eta = torch::where(mask_eq, eta_eq, eta_pol);

    return {xi, eta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_tpv_invert(
    const torch::Tensor& xi_target,
    const torch::Tensor& eta_target,
    const torch::Tensor& idx1,
    const torch::Tensor& c1,
    const torch::Tensor& idx2,
    const torch::Tensor& c2,
    int64_t max_iter,
    double tol
) {
    auto shape = xi_target.sizes();
    auto xi_flat = xi_target.contiguous().view({-1});
    auto eta_flat = eta_target.contiguous().view({-1});
    auto [u, v] = tpv_initial_guess_affine(xi_flat, eta_flat, idx1, c1, idx2, c2);
    torch::Tensor active_idx;
    bool has_active = false;
    const double tol2 = tol * tol;

    for (int64_t iter = 0; iter < max_iter; ++iter) {
        torch::Tensor u_a, v_a, xi_t_a, eta_t_a;
        if (!has_active) {
            u_a = u;
            v_a = v;
            xi_t_a = xi_flat;
            eta_t_a = eta_flat;
        } else {
            if (active_idx.numel() == 0) {
                break;
            }
            u_a = u.index_select(0, active_idx);
            v_a = v.index_select(0, active_idx);
            xi_t_a = xi_flat.index_select(0, active_idx);
            eta_t_a = eta_flat.index_select(0, active_idx);
        }

        auto [xi, eta, j11, j12, j21, j22] = tpv_distort_and_jacobian_impl(
            u_a, v_a, idx1, c1, idx2, c2
        );
        auto rx = xi - xi_t_a;
        auto ry = eta - eta_t_a;

        auto dist2 = rx * rx + ry * ry;
        auto keep_active = dist2 >= tol2;
        if (!keep_active.any().item<bool>()) {
            break;
        }

        auto det = j11 * j22 - j12 * j21;
        auto det_sign = torch::where(det >= 0, torch::ones_like(det), -torch::ones_like(det));
        det = torch::where(det.abs() < 1e-18, det_sign * 1e-18, det);

        auto du = -(j22 * rx - j12 * ry) / det;
        auto dv = -(-j21 * rx + j11 * ry) / det;
        du = du.clamp(-1.0, 1.0);
        dv = dv.clamp(-1.0, 1.0);
        auto u_next = u_a + du;
        auto v_next = v_a + dv;

        const bool keep_all = keep_active.all().item<bool>();
        if (!has_active) {
            if (keep_all) {
                u = u_next;
                v = v_next;
            } else {
                auto next_local_idx = torch::nonzero(keep_active).squeeze(1);
                u = u_next;
                v = v_next;
                active_idx = next_local_idx;
                has_active = true;
            }
        } else {
            u.index_copy_(0, active_idx, u_next);
            v.index_copy_(0, active_idx, v_next);
            if (!keep_all) {
                auto next_local_idx = torch::nonzero(keep_active).squeeze(1);
                active_idx = active_idx.index_select(0, next_local_idx);
            }
        }
    }

    return {u.view(shape), v.view(shape)};
}

std::pair<torch::Tensor, torch::Tensor> wcs_tpv_distort(
    const torch::Tensor& u_in,
    const torch::Tensor& v_in,
    const torch::Tensor& idx1,
    const torch::Tensor& c1,
    const torch::Tensor& idx2,
    const torch::Tensor& c2
) {
    if (
        u_in.device().is_cpu() && v_in.device().is_cpu() &&
        idx1.device().is_cpu() && c1.device().is_cpu() &&
        idx2.device().is_cpu() && c2.device().is_cpu() &&
        u_in.scalar_type() == torch::kFloat64 && v_in.scalar_type() == torch::kFloat64 &&
        c1.scalar_type() == torch::kFloat64 && c2.scalar_type() == torch::kFloat64 &&
        idx1.scalar_type() == torch::kInt64 && idx2.scalar_type() == torch::kInt64
    ) {
        auto shape = u_in.sizes();
        auto u = u_in.contiguous().view({-1});
        auto v = v_in.contiguous().view({-1});
        auto xi = torch::empty_like(u);
        auto eta = torch::empty_like(v);

        auto idx1c = idx1.contiguous();
        auto idx2c = idx2.contiguous();
        auto c1c = c1.contiguous();
        auto c2c = c2.contiguous();

        const auto n = u.numel();
        const auto n1 = idx1c.size(0);
        const auto n2 = idx2c.size(0);

        const double* u_ptr = u.data_ptr<double>();
        const double* v_ptr = v.data_ptr<double>();
        double* xi_ptr = xi.data_ptr<double>();
        double* eta_ptr = eta.data_ptr<double>();
        const int64_t* idx1_ptr = idx1c.data_ptr<int64_t>();
        const int64_t* idx2_ptr = idx2c.data_ptr<int64_t>();
        const double* c1_ptr = c1c.data_ptr<double>();
        const double* c2_ptr = c2c.data_ptr<double>();

        for (int64_t i = 0; i < n; ++i) {
            const double uu = u_ptr[i];
            const double vv = v_ptr[i];
            const double rr = std::sqrt(uu * uu + vv * vv);

            double xp[8], yp[8], rp[8];
            xp[0] = 1.0;
            yp[0] = 1.0;
            rp[0] = 1.0;
            xp[1] = uu;
            yp[1] = vv;
            rp[1] = rr;
            for (int j = 2; j < 8; ++j) {
                xp[j] = xp[j - 1] * uu;
                yp[j] = yp[j - 1] * vv;
                rp[j] = rp[j - 1] * rr;
            }

            double out_x = 0.0;
            for (int64_t k = 0; k < n1; ++k) {
                const int64_t px = idx1_ptr[3 * k + 0];
                const int64_t py = idx1_ptr[3 * k + 1];
                const int64_t pr = idx1_ptr[3 * k + 2];
                out_x += c1_ptr[k] * xp[px] * yp[py] * rp[pr];
            }

            double out_y = 0.0;
            for (int64_t k = 0; k < n2; ++k) {
                const int64_t px = idx2_ptr[3 * k + 0];
                const int64_t py = idx2_ptr[3 * k + 1];
                const int64_t pr = idx2_ptr[3 * k + 2];
                out_y += c2_ptr[k] * yp[px] * xp[py] * rp[pr];
            }

            xi_ptr[i] = out_x;
            eta_ptr[i] = out_y;
        }

        return {xi.view(shape), eta.view(shape)};
    }

    auto r = (u_in * u_in + v_in * v_in).sqrt();
    auto x_p = tpv_make_pow_cache(u_in);
    auto y_p = tpv_make_pow_cache(v_in);
    auto r_p = tpv_make_pow_cache(r);
    auto [xi, _, __] = tpv_eval_axis_with_jacobian(
        u_in, idx1, c1, x_p, y_p, r_p, torch::zeros_like(u_in), torch::zeros_like(u_in)
    );
    auto [eta, ___, ____] = tpv_eval_axis_with_jacobian(
        u_in, idx2, c2, y_p, x_p, r_p, torch::zeros_like(u_in), torch::zeros_like(u_in)
    );
    return {xi, eta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_sip_quadratic_distort(
    const torch::Tensor& u_in,
    const torch::Tensor& v_in,
    double a20,
    double a11,
    double a02,
    double b20,
    double b11,
    double b02
) {
    if (
        u_in.device().is_cpu() && v_in.device().is_cpu() &&
        u_in.scalar_type() == torch::kFloat64 && v_in.scalar_type() == torch::kFloat64
    ) {
        auto shape = u_in.sizes();
        auto u = u_in.contiguous().view({-1});
        auto v = v_in.contiguous().view({-1});
        auto out_u = torch::empty_like(u);
        auto out_v = torch::empty_like(v);

        const auto n = u.numel();
        const double* up = u.data_ptr<double>();
        const double* vp = v.data_ptr<double>();
        double* uo = out_u.data_ptr<double>();
        double* vo = out_v.data_ptr<double>();

        for (int64_t i = 0; i < n; ++i) {
            const double uu = up[i];
            const double vv = vp[i];
            const double uv = uu * vv;
            const double u2 = uu * uu;
            const double v2 = vv * vv;
            const double fu = a20 * u2 + a11 * uv + a02 * v2;
            const double gv = b20 * u2 + b11 * uv + b02 * v2;
            uo[i] = uu + fu;
            vo[i] = vv + gv;
        }

        return {out_u.view(shape), out_v.view(shape)};
    }

    auto uv = u_in * v_in;
    auto u2 = u_in * u_in;
    auto v2 = v_in * v_in;
    auto f_uv = a20 * u2 + a11 * uv + a02 * v2;
    auto g_uv = b20 * u2 + b11 * uv + b02 * v2;
    return {u_in + f_uv, v_in + g_uv};
}

std::tuple<torch::Tensor, torch::Tensor, int64_t, std::vector<int64_t>, int64_t> wcs_tpv_invert_trace(
    const torch::Tensor& xi_target,
    const torch::Tensor& eta_target,
    const torch::Tensor& idx1,
    const torch::Tensor& c1,
    const torch::Tensor& idx2,
    const torch::Tensor& c2,
    int64_t max_iter,
    double tol
) {
    auto shape = xi_target.sizes();
    auto xi_flat = xi_target.contiguous().view({-1});
    auto eta_flat = eta_target.contiguous().view({-1});
    auto [u, v] = tpv_initial_guess_affine(xi_flat, eta_flat, idx1, c1, idx2, c2);
    torch::Tensor active_idx;
    bool has_active = false;
    const double tol2 = tol * tol;
    std::vector<int64_t> active_counts;
    active_counts.reserve(static_cast<size_t>(std::max<int64_t>(max_iter, 0)));
    int64_t iterations_used = 0;
    int64_t final_active = 0;

    for (int64_t iter = 0; iter < max_iter; ++iter) {
        torch::Tensor u_a, v_a, xi_t_a, eta_t_a;
        if (!has_active) {
            u_a = u;
            v_a = v;
            xi_t_a = xi_flat;
            eta_t_a = eta_flat;
        } else {
            if (active_idx.numel() == 0) {
                final_active = 0;
                break;
            }
            u_a = u.index_select(0, active_idx);
            v_a = v.index_select(0, active_idx);
            xi_t_a = xi_flat.index_select(0, active_idx);
            eta_t_a = eta_flat.index_select(0, active_idx);
        }

        active_counts.push_back(static_cast<int64_t>(u_a.numel()));
        iterations_used += 1;

        auto [xi, eta, j11, j12, j21, j22] = tpv_distort_and_jacobian_impl(
            u_a, v_a, idx1, c1, idx2, c2
        );
        auto rx = xi - xi_t_a;
        auto ry = eta - eta_t_a;

        auto dist2 = rx * rx + ry * ry;
        auto keep_active = dist2 >= tol2;
        if (!keep_active.any().item<bool>()) {
            final_active = 0;
            break;
        }

        auto det = j11 * j22 - j12 * j21;
        auto det_sign = torch::where(det >= 0, torch::ones_like(det), -torch::ones_like(det));
        det = torch::where(det.abs() < 1e-18, det_sign * 1e-18, det);

        auto du = -(j22 * rx - j12 * ry) / det;
        auto dv = -(-j21 * rx + j11 * ry) / det;
        du = du.clamp(-1.0, 1.0);
        dv = dv.clamp(-1.0, 1.0);
        auto u_next = u_a + du;
        auto v_next = v_a + dv;

        const bool keep_all = keep_active.all().item<bool>();
        if (!has_active) {
            if (keep_all) {
                u = u_next;
                v = v_next;
                final_active = static_cast<int64_t>(u.numel());
            } else {
                auto next_local_idx = torch::nonzero(keep_active).squeeze(1);
                u = u_next;
                v = v_next;
                active_idx = next_local_idx;
                has_active = true;
                final_active = static_cast<int64_t>(active_idx.numel());
            }
        } else {
            u.index_copy_(0, active_idx, u_next);
            v.index_copy_(0, active_idx, v_next);
            if (!keep_all) {
                auto next_local_idx = torch::nonzero(keep_active).squeeze(1);
                active_idx = active_idx.index_select(0, next_local_idx);
            }
            final_active = static_cast<int64_t>(active_idx.numel());
        }
    }

    if (iterations_used == max_iter) {
        if (!has_active) {
            final_active = static_cast<int64_t>(u.numel());
        } else if (active_idx.defined()) {
            final_active = static_cast<int64_t>(active_idx.numel());
        }
    }

    return std::make_tuple(u.view(shape), v.view(shape), iterations_used, active_counts, final_active);
}

inline std::pair<double, double> eval_tpv_scalar(
    double u, double v,
    const torch::Tensor& idx1, const torch::Tensor& c1,
    const torch::Tensor& idx2, const torch::Tensor& c2)
{
    if (c1.numel() == 0 && c2.numel() == 0) return {u, v};
    double r = std::sqrt(u * u + v * v);
    double xp[8] = {1.0, u, u*u, u*u*u, u*u*u*u, u*u*u*u*u, u*u*u*u*u*u, u*u*u*u*u*u*u};
    double yp[8] = {1.0, v, v*v, v*v*v, v*v*v*v, v*v*v*v*v, v*v*v*v*v*v, v*v*v*v*v*v*v};
    double rp[8] = {1.0, r, r*r, r*r*r, r*r*r*r, r*r*r*r*r, r*r*r*r*r*r, r*r*r*r*r*r*r};

    double xi = 0.0;
    const int64_t* idx1_p = idx1.data_ptr<int64_t>();
    const double* c1_p = c1.data_ptr<double>();
    for (int k = 0; k < c1.size(0); ++k) {
        xi += xp[idx1_p[k*3]] * yp[idx1_p[k*3+1]] * rp[idx1_p[k*3+2]] * c1_p[k];
    }

    double eta = 0.0;
    const int64_t* idx2_p = idx2.data_ptr<int64_t>();
    const double* c2_p = c2.data_ptr<double>();
    for (int k = 0; k < c2.size(0); ++k) {
        eta += yp[idx2_p[k*3]] * xp[idx2_p[k*3+1]] * rp[idx2_p[k*3+2]] * c2_p[k];
    }
    return {xi, eta};
}

bool quadcube_xy_to_face(double x, double y, int* face, double* xf, double* yf) {
    static const double cx[6] = {0.0, 0.0, 90.0, 180.0, 270.0, 0.0};
    static const double cy[6] = {90.0, 0.0, 0.0, 0.0, 0.0, -90.0};
    for (int f = 0; f < 6; ++f) {
        double fx = (x - cx[f]) / 45.0;
        double fy = (y - cy[f]) / 45.0;
        if (std::abs(fx) <= 1.0 + 1e-9 && std::abs(fy) <= 1.0 + 1e-9) {
            *face = f;
            *xf = std::max(-1.0, std::min(1.0, fx));
            *yf = std::max(-1.0, std::min(1.0, fy));
            return true;
        }
    }
    return false;
}

void quadcube_face_to_xy(int face, double xf, double yf, double* x, double* y) {
    static const double cx[6] = {0.0, 0.0, 90.0, 180.0, 270.0, 0.0};
    static const double cy[6] = {90.0, 0.0, 0.0, 0.0, 0.0, -90.0};
    *x = cx[face] + 45.0 * xf;
    *y = cy[face] + 45.0 * yf;
}

// WCSLIB cscs2x / cscx2s evaluate Chan–O'Neill polynomials in float32.
float csc_forward_poly(float chi, float psi) {
    const float gstar = 1.37484847732f;
    const float mm = 0.004869491981f;
    const float gamma = -0.13161671474f;
    const float omega1 = -0.159596235474f;
    const float d0 = 0.0759196200467f;
    const float d1 = -0.0217762490699f;
    const float c00 = 0.141189631152f;
    const float c10 = 0.0809701286525f;
    const float c01 = -0.281528535557f;
    const float c11 = 0.15384112876f;
    const float c20 = -0.178251207466f;
    const float c02 = 0.106959469314f;
    float chi2 = chi * chi;
    float psi2 = psi * psi;
    float chi2co = 1.0f - chi2;
    float psi2co = 1.0f - psi2;
    float chi4 = (chi2 > 1.0e-16f) ? chi2 * chi2 : 0.0f;
    float psi4 = (psi2 > 1.0e-16f) ? psi2 * psi2 : 0.0f;
    float chipsi = std::fabs(chi * psi);
    float chi2psi2 = (chipsi > 1.0e-16f) ? chi2 * psi2 : 0.0f;
    return chi * (chi2 + chi2co * (gstar + psi2 * (gamma * chi2co + mm * chi2 +
        psi2co * (c00 + c10 * chi2 + c01 * psi2 + c11 * chi2psi2 + c20 * chi4 +
        c02 * psi4)) + chi2 * (omega1 - chi2co * (d0 + d1 * chi2))));
}

float csc_inverse_poly(float xf, float yf) {
    const float p00 = -0.27292696f;
    const float p10 = -0.07629969f;
    const float p20 = -0.22797056f;
    const float p30 = 0.54852384f;
    const float p40 = -0.62930065f;
    const float p50 = 0.25795794f;
    const float p60 = 0.02584375f;
    const float p01 = -0.02819452f;
    const float p11 = -0.01471565f;
    const float p21 = 0.48051509f;
    const float p31 = -1.74114454f;
    const float p41 = 1.71547508f;
    const float p51 = -0.53022337f;
    const float p02 = 0.27058160f;
    const float p12 = -0.56800938f;
    const float p22 = 0.30803317f;
    const float p32 = 0.98938102f;
    const float p42 = -0.83180469f;
    const float p03 = -0.60441560f;
    const float p13 = 1.50880086f;
    const float p23 = -0.93678576f;
    const float p33 = 0.08693841f;
    const float p04 = 0.93412077f;
    const float p14 = -1.41601920f;
    const float p24 = 0.33887446f;
    const float p05 = -0.63915306f;
    const float p15 = 0.52032238f;
    const float p06 = 0.14381585f;
    float xx = xf * xf;
    float yy = yf * yf;
    float z0 = p00 + xx * (p10 + xx * (p20 + xx * (p30 + xx * (p40 + xx * (p50 + xx * p60)))));
    float z1 = p01 + xx * (p11 + xx * (p21 + xx * (p31 + xx * (p41 + xx * p51))));
    float z2 = p02 + xx * (p12 + xx * (p22 + xx * (p32 + xx * p42)));
    float z3 = p03 + xx * (p13 + xx * (p23 + xx * p33));
    float z4 = p04 + xx * (p14 + xx * p24);
    float z5 = p05 + xx * p15;
    float chi = z0 + yy * (z1 + yy * (z2 + yy * (z3 + yy * (z4 + yy * (z5 + yy * p06)))));
    return xf + xf * (1.0f - xx) * chi;
}

void csc_s2x(double phi_w, double theta, double* xi, double* eta) {
    double p = phi_w * D2R;
    double t = theta * D2R;
    double ct = std::cos(t);
    double l = ct * std::cos(p);
    double m = ct * std::sin(p);
    double n = std::sin(t);
    int face = 0;
    double zeta = n;
    if (l > zeta) { face = 1; zeta = l; }
    if (m > zeta) { face = 2; zeta = m; }
    if (-l > zeta) { face = 3; zeta = -l; }
    if (-m > zeta) { face = 4; zeta = -m; }
    if (-n > zeta) { face = 5; zeta = -n; }
    double chi_d, psi_d;
    if (face == 1) { chi_d = m / zeta; psi_d = n / zeta; }
    else if (face == 2) { chi_d = -l / zeta; psi_d = n / zeta; }
    else if (face == 3) { chi_d = -m / zeta; psi_d = n / zeta; }
    else if (face == 4) { chi_d = l / zeta; psi_d = n / zeta; }
    else if (face == 5) { chi_d = m / zeta; psi_d = l / zeta; }
    else { chi_d = m / zeta; psi_d = -l / zeta; }
    float xf = csc_forward_poly(static_cast<float>(chi_d), static_cast<float>(psi_d));
    float yf = csc_forward_poly(static_cast<float>(psi_d), static_cast<float>(chi_d));
    xf = std::max(-1.0f, std::min(1.0f, xf));
    yf = std::max(-1.0f, std::min(1.0f, yf));
    quadcube_face_to_xy(face, xf, yf, xi, eta);
}

bool csc_x2s(double xi_deg, double eta_deg, double* phi, double* theta) {
    int face;
    double xf, yf;
    if (!quadcube_xy_to_face(xi_deg, eta_deg, &face, &xf, &yf)) {
        return false;
    }
    float chi = csc_inverse_poly(static_cast<float>(xf), static_cast<float>(yf));
    float psi = csc_inverse_poly(static_cast<float>(yf), static_cast<float>(xf));
    double t = 1.0 / std::sqrt(static_cast<double>(chi * chi + psi * psi) + 1.0);
    double l, m, n;
    if (face == 0) { l = -psi * t; m = chi * t; n = t; }
    else if (face == 1) { l = t; m = chi * t; n = psi * t; }
    else if (face == 2) { l = -chi * t; m = t; n = psi * t; }
    else if (face == 3) { l = -t; m = -chi * t; n = psi * t; }
    else if (face == 4) { l = chi * t; m = -t; n = psi * t; }
    else { l = psi * t; m = chi * t; n = -t; }
    *theta = std::asin(std::max(-1.0, std::min(1.0, n))) * R2D;
    *phi = (l == 0.0 && m == 0.0) ? 0.0 : (std::atan2(m, l) * R2D);
    return true;
}

void qsc_s2x(double phi_w, double theta, double* xi, double* eta) {
    if (std::abs(theta - 90.0) < 1e-12) { *xi = 0.0; *eta = 90.0; return; }
    if (std::abs(theta + 90.0) < 1e-12) { *xi = 0.0; *eta = -90.0; return; }
    double p = phi_w * D2R;
    double t = theta * D2R;
    double ct = std::cos(t);
    double l = ct * std::cos(p);
    double m = ct * std::sin(p);
    double n = std::sin(t);
    int face = 0;
    double zeta = n;
    if (l > zeta) { face = 1; zeta = l; }
    if (m > zeta) { face = 2; zeta = m; }
    if (-l > zeta) { face = 3; zeta = -l; }
    if (-m > zeta) { face = 4; zeta = -m; }
    if (-n > zeta) { face = 5; zeta = -n; }
    double xi_f, eta_f, zeta_f;
    if (face == 1) { xi_f = m; eta_f = n; zeta_f = l; }
    else if (face == 2) { xi_f = -l; eta_f = n; zeta_f = m; }
    else if (face == 3) { xi_f = -m; eta_f = n; zeta_f = -l; }
    else if (face == 4) { xi_f = l; eta_f = n; zeta_f = -m; }
    else if (face == 5) { xi_f = m; eta_f = l; zeta_f = -n; }
    else { xi_f = m; eta_f = -l; zeta_f = n; }
    double zeco = 1.0 - zeta_f;
    double xf, yf;
    if (xi_f == 0.0 && eta_f == 0.0) {
        xf = 0.0;
        yf = 0.0;
    } else if (std::abs(xi_f) >= std::abs(eta_f)) {
        double omega = (xi_f == 0.0) ? 0.0 : eta_f / xi_f;
        double tau = 1.0 + omega * omega;
        double mag = std::sqrt(std::max(0.0, zeco / (1.0 - 1.0 / std::sqrt(1.0 + tau))));
        xf = std::copysign(mag, xi_f);
        if (xi_f == 0.0) xf = 0.0;
        yf = (xf / 15.0) * (std::atan(omega) - std::asin(omega / std::sqrt(tau + tau))) * R2D;
    } else {
        double omega = (eta_f == 0.0) ? 0.0 : xi_f / eta_f;
        double tau = 1.0 + omega * omega;
        double mag = std::sqrt(std::max(0.0, zeco / (1.0 - 1.0 / std::sqrt(1.0 + tau))));
        yf = std::copysign(mag, eta_f);
        if (eta_f == 0.0) yf = 0.0;
        xf = (yf / 15.0) * (std::atan(omega) - std::asin(omega / std::sqrt(tau + tau))) * R2D;
    }
    xf = std::max(-1.0, std::min(1.0, xf));
    yf = std::max(-1.0, std::min(1.0, yf));
    quadcube_face_to_xy(face, xf, yf, xi, eta);
}

bool qsc_x2s(double xi_deg, double eta_deg, double* phi, double* theta) {
    int face;
    double xf, yf;
    if (!quadcube_xy_to_face(xi_deg, eta_deg, &face, &xf, &yf)) {
        return false;
    }
    bool direct = std::abs(xf) > std::abs(yf);
    auto omega_tau_zeta = [](double u, double v, double* o, double* t, double* z) {
        double w = (15.0 * v / (u == 0.0 ? 1.0 : u)) * D2R;
        double sw = std::sin(w);
        double cw = std::cos(w);
        *o = sw / (cw - 0.7071067811865476);
        *t = 1.0 + (*o) * (*o);
        *z = 1.0 - u * u * (1.0 - 1.0 / std::sqrt(1.0 + *t));
    };
    double omega, tau, zeta;
    if (xf == 0.0 && yf == 0.0) {
        omega = 0.0;
        tau = 1.0;
        zeta = 1.0;
    } else if (direct && xf != 0.0) {
        omega_tau_zeta(xf, yf, &omega, &tau, &zeta);
    } else if ((!direct) && yf != 0.0) {
        omega_tau_zeta(yf, xf, &omega, &tau, &zeta);
    } else {
        omega = 0.0;
        tau = 1.0;
        zeta = 1.0;
    }
    zeta = std::max(-1.0, std::min(1.0, zeta));
    double zeco = 1.0 - zeta;
    double wmag = std::sqrt(std::max(0.0, zeco * (2.0 - zeco) / tau));
    double l, m, n;
    if (face == 0) {
        if (direct) {
            m = (xf < 0.0) ? -wmag : wmag;
            l = -m * omega;
        } else {
            l = (yf > 0.0) ? -wmag : wmag;
            m = -l * omega;
        }
        n = zeta;
    } else if (face == 1) {
        l = zeta;
        if (direct) {
            double s = (xf < 0.0) ? -wmag : wmag;
            m = s;
            n = s * omega;
        } else {
            double s = (yf < 0.0) ? -wmag : wmag;
            m = s * omega;
            n = s;
        }
    } else if (face == 2) {
        m = zeta;
        if (direct) {
            double s = (xf > 0.0) ? -wmag : wmag;
            l = s;
            n = -s * omega;
        } else {
            double s = (yf < 0.0) ? -wmag : wmag;
            l = -s * omega;
            n = s;
        }
    } else if (face == 3) {
        l = -zeta;
        if (direct) {
            double s = (xf > 0.0) ? -wmag : wmag;
            m = s;
            n = -s * omega;
        } else {
            double s = (yf < 0.0) ? -wmag : wmag;
            m = -s * omega;
            n = s;
        }
    } else if (face == 4) {
        m = -zeta;
        if (direct) {
            double s = (xf < 0.0) ? -wmag : wmag;
            l = s;
            n = s * omega;
        } else {
            double s = (yf < 0.0) ? -wmag : wmag;
            l = s * omega;
            n = s;
        }
    } else {
        n = -zeta;
        if (direct) {
            double s = (xf < 0.0) ? -wmag : wmag;
            m = s;
            l = s * omega;
        } else {
            double s = (yf < 0.0) ? -wmag : wmag;
            m = s * omega;
            l = s;
        }
    }
    *theta = std::asin(std::max(-1.0, std::min(1.0, n))) * R2D;
    *phi = (l == 0.0 && m == 0.0) ? 0.0 : (std::atan2(m, l) * R2D);
    return true;
}

void wcs_pixel_to_world_fused_cpu(
    const torch::Tensor& x_tensor,
    const torch::Tensor& y_tensor,
    torch::Tensor& ra_tensor,
    torch::Tensor& dec_tensor,
    double crpix0, double crpix1,
    double cd00, double cd01,
    double cd10, double cd11,
    const std::string& proj_code,
    double lambda_param,
    double hpx_H,
    double hpx_K,
    double east_x, double east_y,
    double north_x, double north_y, double north_z,
    double radial_x, double radial_y, double radial_z,
    double ra_p_rad, double dec_p_rad, double phi_p_rad,
    const std::string& native_type,
    double alpha0, double delta0,
    bool is_tpv,
    const torch::Tensor& tpv_idx1,
    const torch::Tensor& tpv_c1,
    const torch::Tensor& tpv_idx2,
    const torch::Tensor& tpv_c2
) {
    const int64_t n = x_tensor.numel();
    const double* x_ptr = x_tensor.data_ptr<double>();
    const double* y_ptr = y_tensor.data_ptr<double>();
    double* ra_ptr = ra_tensor.data_ptr<double>();
    double* dec_ptr = dec_tensor.data_ptr<double>();

    const bool native_is_pole = (native_type == "pole");
    bool use_center_equator_fast = (native_type == "center" && std::abs(delta0) < 1e-12 && std::abs(std::fmod(phi_p_rad * R2D, 360.0)) < 1e-12);

    // CPU-only path: use libtorch intra-op parallel (not OpenMP pragmas) so behavior
    // stays consistent with other torchsky CPU kernels and does not assume GPU.
    // Skip parallel_for setup below the grain; it loses at N<=~8k vs a serial loop.
    auto fused_p2w_range = [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
        // 1. CD Matrix application
        double dx = x_ptr[i] - crpix0;
        double dy = y_ptr[i] - crpix1;
        double u_deg = cd00 * dx + cd01 * dy;
        double v_deg = cd10 * dx + cd11 * dy;

        double xi_deg = u_deg;
        double eta_deg = v_deg;

        if (is_tpv) {
            auto [tpv_xi, tpv_eta] = eval_tpv_scalar(u_deg, v_deg, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2);
            xi_deg = tpv_xi;
            eta_deg = tpv_eta;
        }

        // 2. Projection (xi, eta) -> (phi, theta)
        double phi, theta;
        if (proj_code == "TAN") {
            double r = std::sqrt(xi_deg * xi_deg + eta_deg * eta_deg);
            double r_rad = r * D2R;
            phi = std::atan2(-xi_deg, -eta_deg) * R2D;
            theta = std::atan2(1.0, std::max(r_rad, 1e-15)) * R2D;
        } else if (proj_code == "SIN") {
            double r = std::sqrt(xi_deg * xi_deg + eta_deg * eta_deg);
            double r_rad = std::min(r * D2R, 1.0);
            phi = std::atan2(-xi_deg, -eta_deg) * R2D;
            theta = std::acos(r_rad) * R2D;
        } else if (proj_code == "ARC") {
            // Zenithal equidistant: R = hypot(xi, eta) in degrees; theta = 90 - R.
            double r = std::sqrt(xi_deg * xi_deg + eta_deg * eta_deg);
            phi = std::atan2(-xi_deg, -eta_deg) * R2D;
            theta = 90.0 - r;
        } else if (proj_code == "CEA") {
            phi = xi_deg;
            double val = std::max(-1.0, std::min(1.0, lambda_param * eta_deg * D2R));
            theta = std::asin(val) * R2D;
        } else if (proj_code == "AIT") {
            double X = xi_deg * D2R;
            double Y = eta_deg * D2R;
            double xq = X * 0.25;
            double yq = Y * 0.5;
            double r2 = xq * xq + yq * yq;
            if (r2 > 1.0) {
                phi = std::nan("");
                theta = std::nan("");
            } else {
                double z = std::sqrt(1.0 - r2);
                phi = 2.0 * std::atan2(0.5 * z * X, 2.0 * z * z - 1.0) * R2D;
                theta = std::asin(std::max(-1.0, std::min(1.0, z * Y))) * R2D;
            }
        } else if (proj_code == "HPX") {
            double eta_scale = 90.0 * (hpx_K / hpx_H);
            double eta_boundary = eta_scale * (2.0 / 3.0);
            double abs_eta = std::abs(eta_deg);
            if (abs_eta <= eta_boundary) {
                phi = xi_deg;
                theta = std::asin(std::max(-1.0, std::min(1.0, eta_deg / eta_scale))) * R2D;
            } else {
                double sigma = std::max(0.0, (90.0 - abs_eta) / (90.0 - eta_boundary));
                double xc = std::round((xi_deg - 45.0) / 90.0) * 90.0 + 45.0;
                double dx_hpx = xi_deg - xc;
                phi = xc + (sigma > 1e-12 ? dx_hpx / sigma : 0.0);
                double s_theta = (eta_deg >= 0 ? 1.0 : -1.0) * (1.0 - (sigma * sigma) / 3.0);
                theta = std::asin(std::max(-1.0, std::min(1.0, s_theta))) * R2D;
            }
        } else if (proj_code == "CAR") {
            phi = xi_deg;
            theta = eta_deg;
        } else if (proj_code == "SFL") {
            theta = eta_deg;
            double cos_theta = std::cos(theta * D2R);
            phi = (std::abs(cos_theta) > 1e-12) ? (xi_deg / cos_theta) : 0.0;
        } else if (proj_code == "MOL") {
            double sqrt2 = std::sqrt(2.0);
            double sin_gamma = std::max(-1.0, std::min(1.0, (eta_deg * D2R) / sqrt2));
            double gamma = std::asin(sin_gamma);
            double cos_gamma = std::cos(gamma);
            double t_val = (2.0 * gamma + std::sin(2.0 * gamma)) / WCS_PI;
            theta = std::asin(std::max(-1.0, std::min(1.0, t_val))) * R2D;
            phi = (std::abs(cos_gamma) > 1e-12) ? (xi_deg * WCS_PI / (2.0 * sqrt2 * cos_gamma)) : 0.0;
        } else if (proj_code == "PCO") {
            // Inverse PCO: Newton on G(theta) in radians.
            double big_x = xi_deg * D2R;
            double big_y = eta_deg * D2R;
            if (std::abs(big_x) < 1e-15 && std::abs(big_y) < 1e-15) {
                phi = 0.0;
                theta = 0.0;
            } else {
                const double half_pi = 0.5 * WCS_PI;
                double th = std::max(-half_pi + 1e-3, std::min(half_pi - 1e-3, big_y));
                for (int j = 0; j < 64; ++j) {
                    double dy = big_y - th;
                    double r2 = big_x * big_x + dy * dy;
                    double cos_t = std::cos(th);
                    if (std::abs(cos_t) < 1e-15) break;
                    double tan_t = std::sin(th) / cos_t;
                    double g = tan_t * r2 - 2.0 * dy;
                    double gp = r2 / (cos_t * cos_t) - 2.0 * tan_t * dy + 2.0;
                    if (std::abs(gp) < 1e-15) break;
                    double dt = g / gp;
                    th -= dt;
                    th = std::max(-half_pi + 1e-9, std::min(half_pi - 1e-9, th));
                    if (std::abs(dt) < 1e-13) break;
                }
                theta = th * R2D;
                if (std::abs(th) < 1e-12) {
                    phi = xi_deg;
                } else {
                    double tan_t = std::tan(th);
                    double sin_e = big_x * tan_t;
                    double cos_e = 1.0 - (big_y - th) * tan_t;
                    double e = std::atan2(sin_e, cos_e);
                    phi = (e / std::sin(th)) * R2D;
                }
            }
        } else if (proj_code == "XPH") {
            const double s = 0.7071067811865476;
            double xr = xi_deg / s;
            double yr = eta_deg / s;
            double xi1, eta1, phi0;
            if (xr <= 0.0 && yr > 0.0) {
                xi1 = (-xr - yr) * 0.5; eta1 = (xr - yr) * 0.5; phi0 = -180.0;
            } else if (xr < 0.0 && yr <= 0.0) {
                xi1 = (xr - yr) * 0.5; eta1 = (xr + yr) * 0.5; phi0 = -90.0;
            } else if (xr >= 0.0 && yr < 0.0) {
                xi1 = (xr + yr) * 0.5; eta1 = (-xr + yr) * 0.5; phi0 = 0.0;
            } else {
                xi1 = (-xr + yr) * 0.5; eta1 = (-xr - yr) * 0.5; phi0 = 90.0;
            }
            double xi_f = xi1 + 45.0;
            double eta_f = eta1 + 90.0;
            double abseta = std::abs(eta_f);
            if (abseta > 90.0) {
                phi = std::nan("");
                theta = std::nan("");
            } else if (abseta <= 45.0) {
                phi = -(phi0 + xi_f);  // torchsky pole-phi convention
                theta = std::asin(std::max(-1.0, std::min(1.0, eta_f / 67.5))) * R2D;
            } else {
                double sigma = (90.0 - abseta) / 45.0;
                double ph;
                if (xr == 0.0) ph = (yr <= 0.0) ? 0.0 : 180.0;
                else if (yr == 0.0) ph = (xr < 0.0) ? -90.0 : 90.0;
                else ph = phi0 + 45.0 + xi1 / std::max(sigma, 1e-12);
                phi = -ph;
                double th = (sigma < 1e-4)
                    ? (90.0 - sigma * (std::sqrt(2.0 / 3.0) * R2D))
                    : (std::asin(std::max(-1.0, std::min(1.0, 1.0 - sigma * sigma / 3.0))) * R2D);
                if (eta_f < 0.0) th = -th;
                theta = th;
            }
            // wrap phi to [-180,180)
            phi = std::fmod(phi + 180.0, 360.0);
            if (phi < 0) phi += 360.0;
            phi -= 180.0;
        } else if (proj_code == "TSC") {
            // Inverse TSC: locate face then recover (l,m,n).
            static const double cx[6] = {0.0, 0.0, 90.0, 180.0, 270.0, 0.0};
            static const double cy[6] = {90.0, 0.0, 0.0, 0.0, 0.0, -90.0};
            int face = -1;
            double xf = 0.0, yf = 0.0;
            for (int f = 0; f < 6; ++f) {
                double fx = (xi_deg - cx[f]) / 45.0;
                double fy = (eta_deg - cy[f]) / 45.0;
                if (std::abs(fx) <= 1.0 + 1e-9 && std::abs(fy) <= 1.0 + 1e-9) {
                    face = f;
                    xf = std::max(-1.0, std::min(1.0, fx));
                    yf = std::max(-1.0, std::min(1.0, fy));
                    break;
                }
            }
            if (face < 0) {
                phi = std::nan("");
                theta = std::nan("");
            } else {
                double norm = std::sqrt(1.0 + xf * xf + yf * yf);
                double l, m, ncomp;
                if (face == 0) { l = -yf / norm; m = xf / norm; ncomp = 1.0 / norm; }
                else if (face == 1) { l = 1.0 / norm; m = xf / norm; ncomp = yf / norm; }
                else if (face == 2) { l = -xf / norm; m = 1.0 / norm; ncomp = yf / norm; }
                else if (face == 3) { l = -1.0 / norm; m = -xf / norm; ncomp = yf / norm; }
                else if (face == 4) { l = xf / norm; m = -1.0 / norm; ncomp = yf / norm; }
                else { l = yf / norm; m = xf / norm; ncomp = -1.0 / norm; }
                theta = std::asin(std::max(-1.0, std::min(1.0, ncomp))) * R2D;
                phi = (l == 0.0 && m == 0.0) ? 0.0 : (std::atan2(m, l) * R2D);
            }
        } else if (proj_code == "CSC") {
            if (!csc_x2s(xi_deg, eta_deg, &phi, &theta)) {
                phi = std::nan("");
                theta = std::nan("");
            }
        } else if (proj_code == "QSC") {
            if (!qsc_x2s(xi_deg, eta_deg, &phi, &theta)) {
                phi = std::nan("");
                theta = std::nan("");
            }
        } else {
            phi = std::nan("");
            theta = std::nan("");
        }

        // 3. Spherical Rotation (phi, theta) -> (ra, dec)
        if (use_center_equator_fast) {
            ra_ptr[i] = std::fmod(phi + alpha0 + 360.0, 360.0);
            dec_ptr[i] = theta;
        } else if (native_is_pole) {
            double phi_rad = phi * D2R;
            double theta_rad = theta * D2R;
            double cos_theta = std::cos(theta_rad);
            double sin_theta = std::sin(theta_rad);
            double cos_phi_diff = std::cos(phi_rad - phi_p_rad);

            double costhe_cosphi = cos_theta * cos_phi_diff;
            double arg = std::max(-1.0, std::min(1.0, sin_theta * std::sin(dec_p_rad) + costhe_cosphi * std::cos(dec_p_rad)));
            dec_ptr[i] = std::asin(arg) * R2D;

            double y_rot = cos_theta * std::sin(phi_rad - phi_p_rad);
            double x_rot = sin_theta * std::cos(dec_p_rad) - costhe_cosphi * std::sin(dec_p_rad);
            ra_ptr[i] = std::fmod(std::atan2(y_rot, x_rot) * R2D + ra_p_rad * R2D + 360.0, 360.0);
        } else {
            const double phi_rad = phi * D2R;
            const double theta_rad = theta * D2R;
            const double cos_theta = std::cos(theta_rad);
            const double sin_theta = std::sin(theta_rad);
            const double sin_phi = std::sin(phi_rad);
            const double cos_phi = std::cos(phi_rad);

            const double u = cos_theta * sin_phi;
            const double v = -cos_theta * cos_phi;
            const double w = sin_theta;

            const double X = u * east_x + v * (-radial_x) + w * north_x;
            const double Y = u * east_y + v * (-radial_y) + w * north_y;
            const double Z = std::max(
                -1.0,
                std::min(1.0, v * (-radial_z) + w * north_z)
            );

            dec_ptr[i] = std::asin(Z) * R2D;
            ra_ptr[i] = std::fmod(std::atan2(Y, X) * R2D + 360.0, 360.0);
        }
    }
    };
    if (n <= 8192) {
        fused_p2w_range(0, n);
    } else {
        at::parallel_for(0, n, 4096, fused_p2w_range);
    }
}

void wcs_world_to_pixel_fused_cpu(
    const torch::Tensor& ra_tensor,
    const torch::Tensor& dec_tensor,
    torch::Tensor& x_tensor,
    torch::Tensor& y_tensor,
    double crpix0, double crpix1,
    double cdi00, double cdi01,
    double cdi10, double cdi11,
    const std::string& proj_code,
    double lambda_param,
    double hpx_H,
    double hpx_K,
    double east_x, double east_y,
    double north_x, double north_y, double north_z,
    double radial_x, double radial_y, double radial_z,
    double ra_p_rad, double dec_p_rad, double phi_p_rad,
    const std::string& native_type,
    double alpha0, double delta0,
    bool is_tpv,
    const torch::Tensor& tpv_idx1,
    const torch::Tensor& tpv_c1,
    const torch::Tensor& tpv_idx2,
    const torch::Tensor& tpv_c2
) {
    enum class FusedProjKind {
        TAN,
        SIN,
        ARC,
        CEA,
        AIT,
        HPX,
        CAR,
        SFL,
        MOL,
        PCO,
        XPH,
        TSC,
        CSC,
        QSC,
        OTHER,
    };

    const int64_t n = ra_tensor.numel();
    const double* ra_ptr = ra_tensor.data_ptr<double>();
    const double* dec_ptr = dec_tensor.data_ptr<double>();
    double* x_ptr = x_tensor.data_ptr<double>();
    double* y_ptr = y_tensor.data_ptr<double>();

    FusedProjKind proj_kind = FusedProjKind::OTHER;
    if (proj_code == "TAN") proj_kind = FusedProjKind::TAN;
    else if (proj_code == "SIN") proj_kind = FusedProjKind::SIN;
    else if (proj_code == "ARC") proj_kind = FusedProjKind::ARC;
    else if (proj_code == "CEA") proj_kind = FusedProjKind::CEA;
    else if (proj_code == "AIT") proj_kind = FusedProjKind::AIT;
    else if (proj_code == "HPX") proj_kind = FusedProjKind::HPX;
    else if (proj_code == "CAR") proj_kind = FusedProjKind::CAR;
    else if (proj_code == "SFL") proj_kind = FusedProjKind::SFL;
    else if (proj_code == "MOL") proj_kind = FusedProjKind::MOL;
    else if (proj_code == "PCO") proj_kind = FusedProjKind::PCO;
    else if (proj_code == "XPH") proj_kind = FusedProjKind::XPH;
    else if (proj_code == "TSC") proj_kind = FusedProjKind::TSC;
    else if (proj_code == "CSC") proj_kind = FusedProjKind::CSC;
    else if (proj_code == "QSC") proj_kind = FusedProjKind::QSC;

    const bool native_is_pole = (native_type == "pole");
    bool use_center_equator_fast = (native_type == "center" && std::abs(delta0) < 1e-12 && std::abs(std::fmod(phi_p_rad * R2D, 360.0)) < 1e-12);
    TpvAffineSeedParams tpv_seed;
    const int64_t* tpv_idx1_ptr = nullptr;
    const double* tpv_c1_ptr = nullptr;
    int64_t tpv_n1 = 0;
    const int64_t* tpv_idx2_ptr = nullptr;
    const double* tpv_c2_ptr = nullptr;
    int64_t tpv_n2 = 0;
    if (is_tpv) {
        tpv_seed = tpv_extract_affine_seed_params(tpv_idx1, tpv_c1, tpv_idx2, tpv_c2);
        tpv_idx1_ptr = tpv_idx1.data_ptr<int64_t>();
        tpv_c1_ptr = tpv_c1.data_ptr<double>();
        tpv_n1 = tpv_c1.size(0);
        tpv_idx2_ptr = tpv_idx2.data_ptr<int64_t>();
        tpv_c2_ptr = tpv_c2.data_ptr<double>();
        tpv_n2 = tpv_c2.size(0);
    }

    auto fused_w2p_range = [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
        // 1. Inverse Spherical Rotation (ra, dec) -> (phi, theta)
        double ra_deg = ra_ptr[i];
        double dec_deg = dec_ptr[i];

        double phi, theta;
        if (use_center_equator_fast) {
            double phi_tmp = std::fmod(ra_deg - alpha0 + 180.0, 360.0);
            if (phi_tmp < 0) phi_tmp += 360.0;
            phi = phi_tmp - 180.0;
            theta = dec_deg;
        } else {
            double ra_rad = ra_deg * D2R;
            double dec_rad = dec_deg * D2R;
            double sin_dec = std::sin(dec_rad);
            double cos_dec = std::cos(dec_rad);
            double sin_ra = std::sin(ra_rad);
            double cos_ra = std::cos(ra_rad);

            double X = cos_dec * cos_ra;
            double Y = cos_dec * sin_ra;
            double Z = sin_dec;

            if (native_is_pole) {
                double u_rot = X * east_x + Y * east_y;
                double n_rot = X * north_x + Y * north_y + Z * north_z;
                double w_rot = std::max(-1.0, std::min(1.0, X * radial_x + Y * radial_y + Z * radial_z));
                theta = std::asin(w_rot) * R2D;
                phi = std::atan2(u_rot, n_rot) * R2D + phi_p_rad * R2D;
                phi = std::fmod(phi + 360.0, 360.0);
            } else {
                double u_rot = X * east_x + Y * east_y;
                double v_rot = -(X * radial_x + Y * radial_y + Z * radial_z);
                double w_rot = std::max(-1.0, std::min(1.0, X * north_x + Y * north_y + Z * north_z));
                theta = std::asin(w_rot) * R2D;
                phi = std::atan2(u_rot, -v_rot) * R2D;
                phi = std::fmod(phi + 360.0, 360.0);
            }
        }

        // 2. Deprojection (phi, theta) -> (xi, eta)
        double xi, eta;
        double phi_w = std::fmod(phi + 180.0, 360.0);
        if (phi_w < 0) phi_w += 360.0;
        phi_w -= 180.0;

        switch (proj_kind) {
        case FusedProjKind::TAN: {
            double theta_rad = theta * D2R;
            double r = R2D / std::tan(theta_rad);
            double phi_rad = phi * D2R;
            xi = -r * std::sin(phi_rad);
            eta = -r * std::cos(phi_rad);
            break;
        }
        case FusedProjKind::SIN: {
            double theta_rad = theta * D2R;
            double r = std::cos(theta_rad) * R2D;
            double phi_rad = phi * D2R;
            xi = -r * std::sin(phi_rad);
            eta = -r * std::cos(phi_rad);
            break;
        }
        case FusedProjKind::ARC: {
            double r = 90.0 - theta;
            double phi_rad = phi * D2R;
            xi = -r * std::sin(phi_rad);
            eta = -r * std::cos(phi_rad);
            break;
        }
        case FusedProjKind::CEA:
            xi = phi_w;
            eta = (R2D / lambda_param) * std::sin(theta * D2R);
            break;
        case FusedProjKind::AIT: {
            double phi_rad = phi_w * D2R;
            double theta_rad = theta * D2R;
            double half_phi = phi_rad * 0.5;
            double cos_theta = std::cos(theta_rad);
            double sin_theta = std::sin(theta_rad);
            double cos_half_phi = std::cos(half_phi);
            double denom = std::sqrt(0.5 * (1.0 + cos_theta * cos_half_phi));
            xi = 2.0 * cos_theta * std::sin(half_phi) / denom * R2D;
            eta = sin_theta / denom * R2D;
            break;
        }
        case FusedProjKind::HPX: {
            double s_theta = std::sin(theta * D2R);
            double abs_s = std::abs(s_theta);
            double eta_scale = 90.0 * (hpx_K / hpx_H);
            double eta_boundary = eta_scale * (2.0 / 3.0);
            if (abs_s <= (2.0 / 3.0)) {
                xi = phi_w;
                eta = eta_scale * s_theta;
            } else {
                double sigma = std::sqrt(std::max(0.0, 3.0 * (1.0 - abs_s)));
                eta = (s_theta >= 0 ? 1.0 : -1.0) * (90.0 - (90.0 - eta_boundary) * sigma);
                double xc = std::round((phi_w - 45.0) / 90.0) * 90.0 + 45.0;
                xi = xc + sigma * (phi_w - xc);
            }
            break;
        }
        case FusedProjKind::CAR:
            xi = phi_w;
            eta = theta;
            break;
        case FusedProjKind::SFL:
            xi = phi_w * std::cos(theta * D2R);
            eta = theta;
            break;
        case FusedProjKind::MOL: {
            double phi_rad = phi_w * D2R;
            double sin_theta = std::sin(theta * D2R);
            double gamma = 0.0;
            for (int j = 0; j < 10; ++j) {
                double f = 2.0 * gamma + std::sin(2.0 * gamma) - WCS_PI * sin_theta;
                double fp = 2.0 + 2.0 * std::cos(2.0 * gamma);
                gamma -= f / (fp + 1e-12);
            }
            double cos_gamma = std::cos(gamma);
            double sqrt2 = std::sqrt(2.0);
            xi = 2.0 * sqrt2 * phi_rad * cos_gamma / WCS_PI * R2D;
            eta = sqrt2 * std::sin(gamma) * R2D;
            break;
        }
        case FusedProjKind::PCO: {
            // Paper II PCO forward; phi_w in degrees.
            double t = theta * D2R;
            if (std::abs(t) < 1e-12) {
                xi = phi_w;
                eta = 0.0;
            } else {
                double w = std::sin(t);
                double e = phi_w * D2R * w;
                double tan_t = std::tan(t);
                xi = R2D * std::sin(e) / tan_t;
                eta = theta + R2D * (1.0 - std::cos(e)) / tan_t;
            }
            break;
        }
        case FusedProjKind::XPH: {
            // Match Python: negate torchsky pole-phi into WCSLIB convention.
            double phi_n = -phi_w;
            while (phi_n < -180.0) phi_n += 360.0;
            while (phi_n >= 180.0) phi_n -= 360.0;
            double chi = phi_n + 180.0;
            double psi = std::fmod(chi, 90.0);
            if (psi < 0) psi += 90.0;
            double sinthe = std::sin(theta * D2R);
            double abssin = std::abs(sinthe);
            double xi0, eta0;
            if (abssin <= (2.0 / 3.0)) {
                xi0 = psi;
                eta0 = 67.5 * sinthe;
            } else {
                double sigma = std::sqrt(std::max(0.0, 3.0 * (1.0 - abssin)));
                xi0 = 45.0 + (psi - 45.0) * sigma;
                eta0 = 45.0 * (2.0 - sigma);
                if (theta < 0.0) eta0 = -eta0;
            }
            xi0 -= 45.0;
            eta0 -= 90.0;
            const double s = 0.7071067811865476;
            if (phi_n < -90.0) {
                xi = s * (-xi0 + eta0);
                eta = s * (-xi0 - eta0);
            } else if (phi_n < 0.0) {
                xi = s * (xi0 + eta0);
                eta = s * (-xi0 + eta0);
            } else if (phi_n < 90.0) {
                xi = s * (xi0 - eta0);
                eta = s * (xi0 + eta0);
            } else {
                xi = s * (-xi0 - eta0);
                eta = s * (xi0 - eta0);
            }
            break;
        }
        case FusedProjKind::TSC: {
            // Tangential spherical cube forward (face layout Paper II).
            double p = phi_w * D2R;
            double t = theta * D2R;
            double ct = std::cos(t);
            double l = ct * std::cos(p);
            double m = ct * std::sin(p);
            double nn = std::sin(t);
            double abs_l = std::abs(l), abs_m = std::abs(m), abs_n = std::abs(nn);
            int face = 0;
            double xf, yf;
            if (nn >= abs_l && nn >= abs_m) {
                face = 0; xf = m / nn; yf = -l / nn;
            } else if (-nn >= abs_l && -nn >= abs_m) {
                face = 5; xf = m / (-nn); yf = l / (-nn);
            } else if (l >= abs_m && l >= abs_n) {
                face = 1; xf = m / l; yf = nn / l;
            } else if (m >= abs_l && m >= abs_n) {
                face = 2; xf = -l / m; yf = nn / m;
            } else if (-l >= abs_m && -l >= abs_n) {
                face = 3; xf = -m / (-l); yf = nn / (-l);
            } else {
                face = 4; xf = l / (-m); yf = nn / (-m);
            }
            static const double cx[6] = {0.0, 0.0, 90.0, 180.0, 270.0, 0.0};
            static const double cy[6] = {90.0, 0.0, 0.0, 0.0, 0.0, -90.0};
            xi = cx[face] + 45.0 * xf;
            eta = cy[face] + 45.0 * yf;
            break;
        }
        case FusedProjKind::CSC: {
            csc_s2x(phi_w, theta, &xi, &eta);
            break;
        }
        case FusedProjKind::QSC: {
            qsc_s2x(phi_w, theta, &xi, &eta);
            break;
        }
        case FusedProjKind::OTHER:
            xi = std::nan("");
            eta = std::nan("");
        }

        double u_deg = xi;
        double v_deg = eta;

        if (is_tpv) {
            auto [tpv_u, tpv_v] = invert_tpv_scalar_with_seed(
                xi,
                eta,
                tpv_seed,
                tpv_idx1_ptr,
                tpv_c1_ptr,
                tpv_n1,
                tpv_idx2_ptr,
                tpv_c2_ptr,
                tpv_n2
            );
            u_deg = tpv_u;
            v_deg = tpv_v;
        }

        // 3. Inverse CD Matrix
        x_ptr[i] = cdi00 * u_deg + cdi01 * v_deg + crpix0;
        y_ptr[i] = cdi10 * u_deg + cdi11 * v_deg + crpix1;
    }
    };
    if (n <= 8192) {
        fused_w2p_range(0, n);
    } else {
        at::parallel_for(0, n, 4096, fused_w2p_range);
    }
}
}
