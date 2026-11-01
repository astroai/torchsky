#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "torchsky/wcs.h"

namespace torchsky {

constexpr double WCS_PI = 3.14159265358979323846;
constexpr double D2R = WCS_PI / 180.0;
constexpr double R2D = 180.0 / WCS_PI;
constexpr double SQRT2 = 1.4142135623730950488;

namespace {

constexpr double CEA_DOMAIN_TOL = 1e-12;
constexpr double MOL_DOMAIN_TOL = 1e-13;
constexpr double TAN_HORIZON_TOL = 64.0 * std::numeric_limits<double>::epsilon();
const double ARC_RADIUS_MAX = std::nextafter(180.0, std::numeric_limits<double>::infinity());

struct HpxParams {
    double H;
    double K;
    double transition_sin;
    double equatorial_scale;
    double transition_eta;
    double pole_eta;
    double facet_spacing;
    double facet_half_width;

    double facet_center(double coordinate, bool south) const {
        const double offset = south && std::fmod(K, 2.0) == 0.0 ? 0.0 : facet_half_width;
        return std::floor(
            (coordinate + 180.0 - offset + facet_half_width) / facet_spacing
        ) * facet_spacing - 180.0 + offset;
    }
};

inline HpxParams make_hpx_params(double H, double K) {
    TORCH_CHECK(
        std::isfinite(H) && H > 0.0 && std::floor(H) == H,
        "HPX H must be a finite positive integer"
    );
    TORCH_CHECK(
        std::isfinite(K) && K > 0.0 && std::floor(K) == K,
        "HPX K must be a finite positive integer"
    );
    const double facet_spacing = 360.0 / H;
    const double facet_half_width = 0.5 * facet_spacing;
    return {
        H,
        K,
        (K - 1.0) / K,
        90.0 * K / H,
        90.0 * (K - 1.0) / H,
        90.0 * (K + 1.0) / H,
        facet_spacing,
        facet_half_width,
    };
}

inline bool hpx_plane_to_native_scalar(
    double xi,
    double eta,
    const HpxParams& p,
    double* phi,
    double* theta
) {
    if (!std::isfinite(xi) || !std::isfinite(eta)) return false;
    const double abs_eta = std::abs(eta);
    if (abs_eta <= p.transition_eta) {
        *phi = xi;
        *theta = std::asin(std::clamp(eta / p.equatorial_scale, -1.0, 1.0)) * R2D;
        return true;
    }
    if (abs_eta > p.pole_eta) return false;
    const double sigma = std::max((p.pole_eta - abs_eta) / p.facet_half_width, 0.0);
    const double xc = p.facet_center(xi, eta < 0.0);
    const double dx = xi - xc;
    if (std::abs(dx) > p.facet_half_width * sigma + 1e-8) return false;
    *phi = xc + (sigma > 1e-12 ? dx / sigma : 0.0);
    const double s = std::copysign(1.0, eta) * (1.0 - sigma * sigma / p.K);
    *theta = std::asin(std::clamp(s, -1.0, 1.0)) * R2D;
    return true;
}

inline bool hpx_native_to_plane_scalar(
    double phi,
    double theta,
    const HpxParams& p,
    double* xi,
    double* eta
) {
    if (
        !std::isfinite(phi) || !std::isfinite(theta) ||
        std::abs(theta) > 90.0 + CEA_DOMAIN_TOL
    ) return false;
    const double s = std::sin(std::clamp(theta, -90.0, 90.0) * D2R);
    const double abs_s = std::abs(s);
    if (abs_s <= p.transition_sin) {
        *xi = phi;
        *eta = p.equatorial_scale * s;
        return true;
    }
    const double sigma = std::sqrt(std::max(p.K * (1.0 - abs_s), 0.0));
    const double xc = p.facet_center(phi, s < 0.0);
    *xi = xc + sigma * (phi - xc);
    *eta = std::copysign(p.pole_eta - p.facet_half_width * sigma, s);
    return true;
}

inline bool mol_plane_to_native_scalar(
    double xi_deg,
    double eta_deg,
    double* phi_deg,
    double* theta_deg
) {
    const double X = xi_deg * D2R;
    const double Y = eta_deg * D2R;
    const double ellipse = X * X / 8.0 + Y * Y / 2.0;
    if (!std::isfinite(ellipse) || ellipse > 1.0 + MOL_DOMAIN_TOL) {
        return false;
    }

    const double sin_gamma = std::clamp(Y / SQRT2, -1.0, 1.0);
    const double gamma = std::asin(sin_gamma);
    const double cos_gamma = std::sqrt(std::max(0.0, 1.0 - sin_gamma * sin_gamma));
    const double sin_theta = (
        2.0 * gamma + 2.0 * sin_gamma * cos_gamma
    ) / WCS_PI;
    if (!std::isfinite(sin_theta) || std::abs(sin_theta) > 1.0 + MOL_DOMAIN_TOL) {
        return false;
    }

    const double denom = 2.0 * SQRT2 * cos_gamma;
    double phi_rad = 0.0;
    if (std::abs(denom) > 0.0) {
        phi_rad = WCS_PI * X / denom;
    } else if (X != 0.0) {
        return false;
    }
    if (!std::isfinite(phi_rad) || std::abs(phi_rad) > WCS_PI + MOL_DOMAIN_TOL) {
        return false;
    }

    *phi_deg = std::clamp(phi_rad, -WCS_PI, WCS_PI) * R2D;
    *theta_deg = std::asin(std::clamp(sin_theta, -1.0, 1.0)) * R2D;
    return true;
}

inline bool mol_native_to_plane_scalar(
    double phi_deg,
    double theta_deg,
    double* xi_deg,
    double* eta_deg
) {
    if (
        !std::isfinite(phi_deg) || !std::isfinite(theta_deg) ||
        std::abs(theta_deg) > 90.0 + CEA_DOMAIN_TOL
    ) {
        return false;
    }

    const double theta = std::clamp(theta_deg, -90.0, 90.0);
    if (std::abs(theta) == 90.0) {
        *xi_deg = 0.0;
        *eta_deg = std::copysign(SQRT2 * R2D, theta);
        return true;
    }

    const double target = WCS_PI * std::sin(theta * D2R);
    double lo = -0.5 * WCS_PI;
    double hi = 0.5 * WCS_PI;
    double gamma = theta * D2R;
    for (int iter = 0; iter < 64; ++iter) {
        const double value = 2.0 * gamma + std::sin(2.0 * gamma) - target;
        if (value < 0.0) {
            lo = gamma;
        } else {
            hi = gamma;
        }
        const double derivative = 2.0 + 2.0 * std::cos(2.0 * gamma);
        const double newton = gamma - value / derivative;
        gamma = (
            std::isfinite(newton) && newton > lo && newton < hi
        ) ? newton : 0.5 * (lo + hi);
    }

    const double residual = 2.0 * gamma + std::sin(2.0 * gamma) - target;
    if (!std::isfinite(residual) || std::abs(residual) > 1e-12) {
        return false;
    }
    *xi_deg = 2.0 * SQRT2 * (phi_deg * D2R) * std::cos(gamma) / WCS_PI * R2D;
    *eta_deg = SQRT2 * std::sin(gamma) * R2D;
    return std::isfinite(*xi_deg) && std::isfinite(*eta_deg);
}

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

constexpr double TPV_CONDITION_EPS = 64.0 * std::numeric_limits<double>::epsilon();
constexpr int64_t TPV_FUSED_MAX_ITER = 20;
constexpr double TPV_FUSED_TOL = 1e-11;

inline bool tpv_jacobian_is_well_conditioned(
    double j11,
    double j12,
    double j21,
    double j22
) {
    const double scale = std::max({
        std::abs(j11), std::abs(j12), std::abs(j21), std::abs(j22)
    });
    const double det = j11 * j22 - j12 * j21;
    return std::isfinite(scale) && std::isfinite(det) && scale > 0.0 &&
        std::abs(det) > TPV_CONDITION_EPS * scale * scale;
}

torch::Tensor tpv_jacobian_is_well_conditioned(
    const torch::Tensor& j11,
    const torch::Tensor& j12,
    const torch::Tensor& j21,
    const torch::Tensor& j22
) {
    auto scale = torch::maximum(
        torch::maximum(j11.abs(), j12.abs()),
        torch::maximum(j21.abs(), j22.abs())
    );
    auto det = j11 * j22 - j12 * j21;
    return torch::isfinite(scale) & torch::isfinite(det) & (scale > 0.0) &
        (det.abs() > TPV_CONDITION_EPS * scale * scale);
}

void validate_tpv_axis(
    const torch::Tensor& idx,
    const char* idx_name,
    const torch::Tensor& coeff,
    const char* coeff_name,
    bool require_contiguous
) {
    TORCH_CHECK(idx.defined(), idx_name, " must be defined");
    TORCH_CHECK(idx.device().is_cpu(), idx_name, " must be a CPU tensor");
    TORCH_CHECK(idx.scalar_type() == torch::kInt64, idx_name, " must have dtype int64");
    TORCH_CHECK(!require_contiguous || idx.is_contiguous(), idx_name, " must be contiguous");
    TORCH_CHECK(idx.dim() == 2 && idx.size(1) == 3, idx_name, " must have shape [N, 3]");
    TORCH_CHECK(coeff.defined(), coeff_name, " must be defined");
    TORCH_CHECK(coeff.device().is_cpu(), coeff_name, " must be a CPU tensor");
    TORCH_CHECK(coeff.scalar_type() == torch::kFloat64, coeff_name, " must have dtype float64");
    TORCH_CHECK(!require_contiguous || coeff.is_contiguous(), coeff_name, " must be contiguous");
    TORCH_CHECK(coeff.dim() == 1, coeff_name, " must have shape [N]");
    TORCH_CHECK(idx.size(0) == coeff.size(0), idx_name, " and ", coeff_name, " lengths must match");
    auto idx_contiguous = idx.contiguous();
    const int64_t* idx_ptr = idx_contiguous.data_ptr<int64_t>();
    for (int64_t i = 0; i < idx_contiguous.numel(); ++i) {
        TORCH_CHECK(
            idx_ptr[i] >= 0 && idx_ptr[i] <= 7,
            idx_name, " exponents must be in [0, 7]"
        );
    }
    TORCH_CHECK(torch::isfinite(coeff).all().item<bool>(), coeff_name, " must be finite");
}

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
    p.valid = tpv_jacobian_is_well_conditioned(p.a11, p.a12, p.a21, p.a22);
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
    if (c1.numel() == 0 && c2.numel() == 0) {
        return std::make_tuple(
            u, v,
            torch::ones_like(u), torch::zeros_like(u),
            torch::zeros_like(u), torch::ones_like(u)
        );
    }
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
    int64_t n2,
    int64_t max_iter,
    double tol)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(xi_target) || !std::isfinite(eta_target)) return {nan, nan};
    if (n1 == 0 && n2 == 0) return {xi_target, eta_target};
    double u, v;
    if (p.valid) {
        double dx = xi_target - p.b1, dy = eta_target - p.b2;
        u = dx * p.inv11 + dy * p.inv12;
        v = dx * p.inv21 + dy * p.inv22;
    } else {
        u = xi_target; v = eta_target;
    }
    bool viable = std::isfinite(u) && std::isfinite(v);
    for (int64_t i = 0; i < max_iter && viable; ++i) {
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
        viable = std::isfinite(rx) && std::isfinite(ry) &&
            tpv_jacobian_is_well_conditioned(j11, j12, j21, j22);
        if (!viable || std::hypot(rx, ry) <= tol) break;
        double det = j11 * j22 - j12 * j21;
        double du = -(j22 * rx - j12 * ry) / det;
        double dv = -(-j21 * rx + j11 * ry) / det;
        u += std::max(-1.0, std::min(1.0, du));
        v += std::max(-1.0, std::min(1.0, dv));
        viable = std::isfinite(u) && std::isfinite(v);
    }
    if (!viable) return {nan, nan};
    auto [xi, eta, j11, j12, j21, j22] = eval_tpv_jac_scalar_raw(
        u, v, idx1_p, c1_p, n1, idx2_p, c2_p, n2
    );
    const double residual = std::hypot(xi - xi_target, eta - eta_target);
    if (
        !std::isfinite(residual) || residual > tol ||
        !tpv_jacobian_is_well_conditioned(j11, j12, j21, j22)
    ) return {nan, nan};
    return {u, v};
}

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
    auto theta = torch::atan2(w, torch::hypot(u, n)) * R2D;
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
    auto valid = torch::isfinite(w) & (w > TAN_HORIZON_TOL);
    auto w_safe = torch::where(valid, w, torch::ones_like(w));
    auto scale = R2D / w_safe;
    auto nan = torch::full_like(w, std::numeric_limits<double>::quiet_NaN());
    return {
        torch::where(valid, u * scale, nan),
        torch::where(valid, v * scale, nan),
    };
}
std::pair<torch::Tensor, torch::Tensor> wcs_zenithal_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg,
    const std::string& proj_code
) {
    auto r = (xi_deg * xi_deg + eta_deg * eta_deg).sqrt();
    auto r_rad = r * D2R;
    auto phi = torch::atan2(-xi_deg, -eta_deg) * R2D;

    torch::Tensor theta;

    if (proj_code == "TAN") {
        theta = torch::atan2(torch::ones_like(r_rad), r_rad) * R2D;
    } else if (proj_code == "SIN") {
        auto invalid = r_rad > 1.0;
        auto r_dim = r_rad.clamp(-1.0, 1.0);
        theta = r_dim.acos() * R2D;
        theta = torch::where(invalid, torch::full_like(theta, std::numeric_limits<double>::quiet_NaN()), theta);
    } else if (proj_code == "ARC") {
        auto valid = torch::isfinite(r) & (r <= ARC_RADIUS_MAX);
        theta = 90.0 - r.clamp_max(180.0);
        auto nan = torch::full_like(theta, std::numeric_limits<double>::quiet_NaN());
        phi = torch::where(valid, phi, nan);
        theta = torch::where(valid, theta, nan);
    } else if (proj_code == "ZEA") {
        auto invalid = r_rad > 2.0;
        auto val = (r_rad / 2.0).clamp(-1.0, 1.0);
        theta = 90.0 - 2.0 * val.asin() * R2D;
        theta = torch::where(invalid, torch::full_like(theta, std::numeric_limits<double>::quiet_NaN()), theta);
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
        auto valid = torch::isfinite(theta_deg) &
            (theta_deg > TAN_HORIZON_TOL * R2D) &
            (theta_deg <= 90.0 + CEA_DOMAIN_TOL);
        auto tan_safe = torch::where(valid, tan_theta, torch::ones_like(tan_theta));
        r = torch::where(
            valid,
            R2D / tan_safe,
            torch::full_like(theta_deg, std::numeric_limits<double>::quiet_NaN())
        );
    } else if (proj_code == "SIN") {
        r = theta_rad.cos() * R2D;
    } else if (proj_code == "ARC") {
        auto valid = torch::isfinite(theta_deg) &
            (theta_deg >= -90.0) & (theta_deg <= 90.0);
        r = torch::where(
            valid,
            90.0 - theta_deg,
            torch::full_like(theta_deg, std::numeric_limits<double>::quiet_NaN())
        );
    } else if (proj_code == "ZEA") {
        r = 2.0 * ((90.0 - theta_deg) * D2R / 2.0).sin() * R2D;
    } else if (proj_code == "STG") {
        r = 2.0 * ((90.0 - theta_deg) * D2R / 2.0).tan() * R2D;
    } else {
        throw std::runtime_error("Unknown zenithal projection: " + proj_code);
    }

    auto phi_rad = phi_deg * D2R;
    auto xi = -r * phi_rad.sin();
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
        TORCH_CHECK(
            std::isfinite(lambda_param) && lambda_param != 0.0,
            "CEA lambda must be finite and non-zero"
        );
        phi = xi_deg;
        auto val = lambda_param * eta_deg * D2R;
        auto valid = torch::isfinite(val) & (val.abs() <= 1.0 + CEA_DOMAIN_TOL);
        theta = val.clamp(-1.0, 1.0).asin() * R2D;
        auto nan = torch::full_like(theta, std::numeric_limits<double>::quiet_NaN());
        phi = torch::where(valid, phi, nan);
        theta = torch::where(valid, theta, nan);
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
        TORCH_CHECK(
            std::isfinite(lambda_param) && lambda_param != 0.0,
            "CEA lambda must be finite and non-zero"
        );
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
    TORCH_CHECK(
        proj_code != "CEA" || (std::isfinite(cea_eta_scale) && cea_eta_scale != 0.0),
        "CEA scale must be finite and non-zero"
    );
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
    const HpxParams hpx = make_hpx_params(hpx_H, hpx_K);

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
                const double s = eta / cea_eta_scale;
                if (!std::isfinite(s) || std::abs(s) > 1.0 + CEA_DOMAIN_TOL) {
                    ra_ptr[i] = nan;
                    dec_ptr[i] = nan;
                    continue;
                }
                ra_ptr[i] = wrap_lon360_scalar(xi + alpha0);
                dec_ptr[i] = std::asin(std::clamp(s, -1.0, 1.0)) * R2D;
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
                double phi = 0.0;
                double theta = 0.0;
                if (!mol_plane_to_native_scalar(xi, eta, &phi, &theta)) {
                    ra_ptr[i] = nan;
                    dec_ptr[i] = nan;
                    continue;
                }
                ra_ptr[i] = wrap_lon360_scalar(phi + alpha0);
                dec_ptr[i] = theta;
            } else if (proj_code == "HPX") {
                double phi = 0.0;
                double theta = 0.0;
                if (!hpx_plane_to_native_scalar(xi, eta, hpx, &phi, &theta)) {
                    ra_ptr[i] = nan;
                    dec_ptr[i] = nan;
                    continue;
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
    TORCH_CHECK(
        proj_code != "CEA" || (std::isfinite(cea_eta_scale) && cea_eta_scale != 0.0),
        "CEA scale must be finite and non-zero"
    );
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
    const HpxParams hpx = make_hpx_params(hpx_H, hpx_K);

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
        } else if (proj_code == "MOL") {
            if (!mol_native_to_plane_scalar(phi, theta, &xi, &eta)) {
                px_ptr[i] = nan;
                py_ptr[i] = nan;
                continue;
            }
        } else if (proj_code == "HPX") {
            if (!hpx_native_to_plane_scalar(phi, theta, hpx, &xi, &eta)) {
                px_ptr[i] = nan;
                py_ptr[i] = nan;
                continue;
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
            double phi = 0.0;
            double theta = 0.0;
            if (!mol_plane_to_native_scalar(
                    xi_ptr[i], eta_ptr[i], &phi, &theta
                )) {
                phi_ptr[i] = nan;
                theta_ptr[i] = nan;
                continue;
            }
            phi_ptr[i] = phi;
            theta_ptr[i] = theta;
        }

        return {phi_flat.view(xi_deg.sizes()), theta_flat.view(eta_deg.sizes())};
    }

    auto X = xi_deg * D2R;
    auto Y = eta_deg * D2R;

    auto ellipse = X * X / 8.0 + Y * Y / 2.0;
    auto valid = torch::isfinite(ellipse) & (ellipse <= 1.0 + MOL_DOMAIN_TOL);

    auto sin_gamma = (Y / SQRT2).clamp(-1.0, 1.0);
    auto gamma = sin_gamma.asin();
    auto cos_gamma = (1.0 - sin_gamma * sin_gamma).clamp_min(0.0).sqrt();

    auto t_val = (2.0 * gamma + 2.0 * sin_gamma * cos_gamma) / WCS_PI;
    valid = valid & torch::isfinite(t_val) & (t_val.abs() <= 1.0 + MOL_DOMAIN_TOL);
    auto theta_rad = t_val.clamp(-1.0, 1.0).asin();

    auto denom = 2.0 * SQRT2 * cos_gamma;
    auto good = (denom.abs() > 0.0) | (X == 0.0);
    valid = valid & good;
    auto denom_safe = torch::where(good, denom, torch::ones_like(denom));
    auto phi_rad = WCS_PI * X / denom_safe;
    phi_rad = torch::where(denom.abs() > 0.0, phi_rad, torch::zeros_like(phi_rad));
    valid = valid & torch::isfinite(phi_rad) &
        (phi_rad.abs() <= WCS_PI + MOL_DOMAIN_TOL);
    phi_rad = phi_rad.clamp(-WCS_PI, WCS_PI);

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
    if (
        phi_deg.device().is_cpu() && theta_deg.device().is_cpu() &&
        phi_deg.scalar_type() == torch::kFloat64 && theta_deg.scalar_type() == torch::kFloat64
    ) {
        auto phi_flat = phi_deg.contiguous().view({-1});
        auto theta_flat = theta_deg.contiguous().view({-1});
        auto xi_flat = torch::empty_like(phi_flat);
        auto eta_flat = torch::empty_like(theta_flat);
        const double* phi_ptr = phi_flat.data_ptr<double>();
        const double* theta_ptr = theta_flat.data_ptr<double>();
        double* xi_ptr = xi_flat.data_ptr<double>();
        double* eta_ptr = eta_flat.data_ptr<double>();
        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (int64_t i = 0; i < phi_flat.numel(); ++i) {
            if (!mol_native_to_plane_scalar(
                    phi_ptr[i], theta_ptr[i], &xi_ptr[i], &eta_ptr[i]
                )) {
                xi_ptr[i] = nan;
                eta_ptr[i] = nan;
            }
        }
        return {xi_flat.view(phi_deg.sizes()), eta_flat.view(theta_deg.sizes())};
    }

    auto phi_rad = phi_deg * D2R;
    auto valid = torch::isfinite(phi_deg) & torch::isfinite(theta_deg) &
        (theta_deg.abs() <= 90.0 + CEA_DOMAIN_TOL);
    auto theta = theta_deg.clamp(-90.0, 90.0);
    auto theta_rad = theta * D2R;
    auto target = WCS_PI * theta_rad.sin();
    auto pole = theta.abs() == 90.0;
    auto lo = torch::full_like(theta_rad, -0.5 * WCS_PI);
    auto hi = torch::full_like(theta_rad, 0.5 * WCS_PI);
    auto gamma = theta_rad.clone();
    for (int i = 0; i < 64; ++i) {
        auto value = 2.0 * gamma + (2.0 * gamma).sin() - target;
        lo = torch::where(value < 0.0, gamma, lo);
        hi = torch::where(value >= 0.0, gamma, hi);
        auto derivative = 2.0 + 2.0 * (2.0 * gamma).cos();
        auto derivative_safe = torch::where(
            derivative > 0.0, derivative, torch::ones_like(derivative)
        );
        auto newton = gamma - value / derivative_safe;
        auto use_newton = torch::isfinite(newton) & (newton > lo) & (newton < hi);
        gamma = torch::where(use_newton, newton, 0.5 * (lo + hi));
    }
    gamma = torch::where(
        pole,
        torch::copysign(torch::full_like(gamma, 0.5 * WCS_PI), theta),
        gamma
    );
    auto residual = 2.0 * gamma + (2.0 * gamma).sin() - target;
    valid = valid & (pole | (residual.abs() <= 1e-12));

    auto cos_gamma = gamma.cos();
    auto sin_gamma = gamma.sin();

    auto xi = 2.0 * SQRT2 * phi_rad * cos_gamma / WCS_PI * R2D;
    auto eta = SQRT2 * sin_gamma * R2D;
    xi = torch::where(pole, torch::zeros_like(xi), xi);
    auto nan = torch::full_like(xi, std::numeric_limits<double>::quiet_NaN());
    xi = torch::where(valid, xi, nan);
    eta = torch::where(valid, eta, nan);

    return {xi, eta};
}

std::pair<torch::Tensor, torch::Tensor> wcs_hpx_project(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg,
    double H,
    double K
) {
    const HpxParams hpx = make_hpx_params(H, K);
    if (
        xi_deg.device().is_cpu() && eta_deg.device().is_cpu() &&
        xi_deg.scalar_type() == torch::kFloat64 && eta_deg.scalar_type() == torch::kFloat64
    ) {
        auto xi_flat = xi_deg.contiguous().view({-1});
        auto eta_flat = eta_deg.contiguous().view({-1});
        auto phi_flat = torch::empty_like(xi_flat);
        auto theta_flat = torch::empty_like(eta_flat);

        const double nan = std::numeric_limits<double>::quiet_NaN();

        const auto n = xi_flat.numel();
        const double* xi_ptr = xi_flat.data_ptr<double>();
        const double* eta_ptr = eta_flat.data_ptr<double>();
        double* phi_ptr = phi_flat.data_ptr<double>();
        double* theta_ptr = theta_flat.data_ptr<double>();

        for (int64_t i = 0; i < n; ++i) {
            if (!hpx_plane_to_native_scalar(
                xi_ptr[i], eta_ptr[i], hpx, &phi_ptr[i], &theta_ptr[i]
            )) {
                phi_ptr[i] = nan;
                theta_ptr[i] = nan;
            }
        }

        return {phi_flat.view(xi_deg.sizes()), theta_flat.view(eta_deg.sizes())};
    }

    auto abs_eta = eta_deg.abs();
    auto invalid_eta = (~torch::isfinite(xi_deg)) | (~torch::isfinite(eta_deg)) | (abs_eta > hpx.pole_eta);
    auto mask_eq = abs_eta <= hpx.transition_eta;
    auto mask_pol = (~mask_eq) & (~invalid_eta);

    auto s_theta_eq = (eta_deg / hpx.equatorial_scale).clamp(-1.0, 1.0);
    auto theta_eq = s_theta_eq.asin() * R2D;

    auto sigma = ((hpx.pole_eta - abs_eta) / hpx.facet_half_width).clamp_min(0.0);
    auto s_theta_pol = (eta_deg.sign() * (1.0 - (sigma * sigma) / hpx.K)).clamp(-1.0, 1.0);
    auto theta_pol = s_theta_pol.asin() * R2D;

    auto offset = torch::where(
        (eta_deg < 0.0) & (std::fmod(hpx.K, 2.0) == 0.0),
        torch::zeros_like(eta_deg),
        torch::full_like(eta_deg, hpx.facet_half_width)
    );
    auto xc = ((xi_deg + 180.0 - offset + hpx.facet_half_width) / hpx.facet_spacing).floor() * hpx.facet_spacing - 180.0 + offset;
    auto dx = xi_deg - xc;
    auto sigma_safe = torch::where(sigma.abs() < 1e-12, torch::ones_like(sigma), sigma);
    auto phi_pol = xc + dx / sigma_safe;

    auto invalid_x = dx.abs() > (hpx.facet_half_width * sigma + 1e-8);
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
    const HpxParams hpx = make_hpx_params(H, K);
    if (
        phi_deg.device().is_cpu() && theta_deg.device().is_cpu() &&
        phi_deg.scalar_type() == torch::kFloat64 && theta_deg.scalar_type() == torch::kFloat64
    ) {
        auto phi_flat = phi_deg.contiguous().view({-1});
        auto theta_flat = theta_deg.contiguous().view({-1});
        auto xi_flat = torch::empty_like(phi_flat);
        auto eta_flat = torch::empty_like(theta_flat);

        const double nan = std::numeric_limits<double>::quiet_NaN();

        const auto n = phi_flat.numel();
        const double* phi_ptr = phi_flat.data_ptr<double>();
        const double* theta_ptr = theta_flat.data_ptr<double>();
        double* xi_ptr = xi_flat.data_ptr<double>();
        double* eta_ptr = eta_flat.data_ptr<double>();

        for (int64_t i = 0; i < n; ++i) {
            if (!hpx_native_to_plane_scalar(
                phi_ptr[i], theta_ptr[i], hpx, &xi_ptr[i], &eta_ptr[i]
            )) {
                xi_ptr[i] = nan;
                eta_ptr[i] = nan;
            }
        }

        return {xi_flat.view(phi_deg.sizes()), eta_flat.view(theta_deg.sizes())};
    }

    auto s_theta = (theta_deg * D2R).sin();
    auto abs_s = s_theta.abs();

    auto valid = torch::isfinite(phi_deg) & torch::isfinite(theta_deg) & (theta_deg.abs() <= 90.0 + CEA_DOMAIN_TOL);
    auto mask_eq = abs_s <= hpx.transition_sin;
    auto sigma = (hpx.K * (1.0 - abs_s)).clamp_min(0.0).sqrt();
    auto eta_eq = hpx.equatorial_scale * s_theta;
    auto eta_pol = s_theta.sign() * (hpx.pole_eta - hpx.facet_half_width * sigma);

    auto offset = torch::where(
        (s_theta < 0.0) & (std::fmod(hpx.K, 2.0) == 0.0),
        torch::zeros_like(phi_deg),
        torch::full_like(phi_deg, hpx.facet_half_width)
    );
    auto xc = ((phi_deg + 180.0 - offset + hpx.facet_half_width) / hpx.facet_spacing).floor() * hpx.facet_spacing - 180.0 + offset;
    auto xi_pol = xc + sigma * (phi_deg - xc);

    auto xi = torch::where(mask_eq, phi_deg, xi_pol);
    auto eta = torch::where(mask_eq, eta_eq, eta_pol);
    auto nan = torch::full_like(xi, std::numeric_limits<double>::quiet_NaN());
    xi = torch::where(valid, xi, nan);
    eta = torch::where(valid, eta, nan);

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
    TORCH_CHECK(xi_target.sizes() == eta_target.sizes(), "TPV inverse inputs must have identical shapes");
    TORCH_CHECK(max_iter > 0, "TPV max_iter must be positive");
    TORCH_CHECK(std::isfinite(tol) && tol > 0.0, "TPV tolerance must be finite and positive");
    validate_tpv_axis(idx1, "idx1", c1, "c1", false);
    validate_tpv_axis(idx2, "idx2", c2, "c2", false);
    auto shape = xi_target.sizes();
    auto xi_flat = xi_target.contiguous().view({-1});
    auto eta_flat = eta_target.contiguous().view({-1});
    auto finite_target = torch::isfinite(xi_flat) & torch::isfinite(eta_flat);
    auto nan = torch::full_like(xi_flat, std::numeric_limits<double>::quiet_NaN());
    if (c1.numel() == 0 && c2.numel() == 0) {
        return {
            torch::where(finite_target, xi_flat, nan).view(shape),
            torch::where(finite_target, eta_flat, nan).view(shape),
        };
    }
    auto [u, v] = tpv_initial_guess_affine(xi_flat, eta_flat, idx1, c1, idx2, c2);
    auto viable = finite_target & torch::isfinite(u) & torch::isfinite(v);
    auto active = viable.clone();

    for (int64_t iter = 0; iter < max_iter; ++iter) {
        if (!active.any().item<bool>()) break;
        auto [xi, eta, j11, j12, j21, j22] = tpv_distort_and_jacobian_impl(
            u, v, idx1, c1, idx2, c2
        );
        auto rx = xi - xi_flat;
        auto ry = eta - eta_flat;
        auto residual = torch::hypot(rx, ry);
        auto det = j11 * j22 - j12 * j21;
        auto conditioned = tpv_jacobian_is_well_conditioned(j11, j12, j21, j22);
        auto finite_state = torch::isfinite(residual) & torch::isfinite(u) & torch::isfinite(v);
        auto good = finite_state & conditioned;
        viable = viable & (~active | good);
        auto update = active & good & (residual > tol);
        auto det_safe = torch::where(update, det, torch::ones_like(det));
        auto du = (-(j22 * rx - j12 * ry) / det_safe).clamp(-1.0, 1.0);
        auto dv = (-(-j21 * rx + j11 * ry) / det_safe).clamp(-1.0, 1.0);
        u = torch::where(update, u + du, u);
        v = torch::where(update, v + dv, v);
        active = update & torch::isfinite(u) & torch::isfinite(v);
        viable = viable & (~update | active);
    }

    auto [xi, eta, j11, j12, j21, j22] = tpv_distort_and_jacobian_impl(
        u, v, idx1, c1, idx2, c2
    );
    auto residual = torch::hypot(xi - xi_flat, eta - eta_flat);
    auto valid = viable & torch::isfinite(residual) & (residual <= tol) &
        tpv_jacobian_is_well_conditioned(j11, j12, j21, j22);
    return {torch::where(valid, u, nan).view(shape), torch::where(valid, v, nan).view(shape)};
}

std::pair<torch::Tensor, torch::Tensor> wcs_tpv_distort(
    const torch::Tensor& u_in,
    const torch::Tensor& v_in,
    const torch::Tensor& idx1,
    const torch::Tensor& c1,
    const torch::Tensor& idx2,
    const torch::Tensor& c2
) {
    TORCH_CHECK(u_in.sizes() == v_in.sizes(), "TPV distortion inputs must have identical shapes");
    validate_tpv_axis(idx1, "idx1", c1, "c1", false);
    validate_tpv_axis(idx2, "idx2", c2, "c2", false);
    auto finite_input = torch::isfinite(u_in) & torch::isfinite(v_in);
    auto nan_input = torch::full_like(u_in, std::numeric_limits<double>::quiet_NaN());
    if (c1.numel() == 0 && c2.numel() == 0) {
        return {
            torch::where(finite_input, u_in, nan_input),
            torch::where(finite_input, v_in, nan_input),
        };
    }
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

        auto xi_view = xi.view(shape);
        auto eta_view = eta.view(shape);
        auto valid = finite_input & torch::isfinite(xi_view) & torch::isfinite(eta_view);
        return {torch::where(valid, xi_view, nan_input), torch::where(valid, eta_view, nan_input)};
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
    auto valid = finite_input & torch::isfinite(xi) & torch::isfinite(eta);
    return {torch::where(valid, xi, nan_input), torch::where(valid, eta, nan_input)};
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
    if (x < -45.0) x += 360.0;  // cross layout spans x in [-45, 315] (WCSLIB wrap)
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

std::pair<torch::Tensor, torch::Tensor> wcs_csc_x2s(
    const torch::Tensor& xi_deg,
    const torch::Tensor& eta_deg
) {
    auto xi_flat = xi_deg.contiguous().view({-1});
    auto eta_flat = eta_deg.contiguous().view({-1});
    auto phi_flat = torch::empty_like(xi_flat);
    auto theta_flat = torch::empty_like(eta_flat);
    const int64_t n = xi_flat.numel();
    const double* xi_p = xi_flat.data_ptr<double>();
    const double* eta_p = eta_flat.data_ptr<double>();
    double* phi_p = phi_flat.data_ptr<double>();
    double* theta_p = theta_flat.data_ptr<double>();
    auto body = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            if (!csc_x2s(xi_p[i], eta_p[i], phi_p + i, theta_p + i)) {
                phi_p[i] = std::numeric_limits<double>::quiet_NaN();
                theta_p[i] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    };
    if (n <= 8192) {
        body(0, n);
    } else {
        at::parallel_for(0, n, 4096, body);
    }
    return {phi_flat.view(xi_deg.sizes()), theta_flat.view(eta_deg.sizes())};
}

std::pair<torch::Tensor, torch::Tensor> wcs_csc_s2x(
    const torch::Tensor& phi,
    const torch::Tensor& theta
) {
    auto phi_flat = phi.contiguous().view({-1});
    auto theta_flat = theta.contiguous().view({-1});
    auto xi_flat = torch::empty_like(phi_flat);
    auto eta_flat = torch::empty_like(theta_flat);
    const int64_t n = phi_flat.numel();
    const double* phi_p = phi_flat.data_ptr<double>();
    const double* theta_p = theta_flat.data_ptr<double>();
    double* xi_p = xi_flat.data_ptr<double>();
    double* eta_p = eta_flat.data_ptr<double>();
    auto body = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            csc_s2x(phi_p[i], theta_p[i], xi_p + i, eta_p + i);
        }
    };
    if (n <= 8192) {
        body(0, n);
    } else {
        at::parallel_for(0, n, 4096, body);
    }
    return {xi_flat.view(phi.sizes()), eta_flat.view(theta.sizes())};
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

namespace {

void validate_fused_tensor(
    const torch::Tensor& tensor,
    const char* name
) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat64, name, " must have dtype float64");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void validate_fused_output_non_overlap(
    const torch::Tensor& output,
    const char* output_name,
    const torch::Tensor& other,
    const char* other_name
) {
    if (output.numel() == 0 || other.numel() == 0) {
        return;
    }
    const auto output_begin = reinterpret_cast<std::uintptr_t>(output.const_data_ptr());
    const auto other_begin = reinterpret_cast<std::uintptr_t>(other.const_data_ptr());
    const auto output_bytes = static_cast<std::uintptr_t>(output.nbytes());
    const auto other_bytes = static_cast<std::uintptr_t>(other.nbytes());
    TORCH_CHECK(
        output_begin <= std::numeric_limits<std::uintptr_t>::max() - output_bytes &&
        other_begin <= std::numeric_limits<std::uintptr_t>::max() - other_bytes,
        "fused WCS tensor address range overflow"
    );
    const auto output_end = output_begin + output_bytes;
    const auto other_end = other_begin + other_bytes;
    TORCH_CHECK(
        output_end <= other_begin || other_end <= output_begin,
        output_name, " must not overlap ", other_name
    );
}

void validate_fused_buffers(
    const torch::Tensor& input1,
    const char* input1_name,
    const torch::Tensor& input2,
    const char* input2_name,
    const torch::Tensor& output1,
    const char* output1_name,
    const torch::Tensor& output2,
    const char* output2_name
) {
    validate_fused_tensor(input1, input1_name);
    validate_fused_tensor(input2, input2_name);
    validate_fused_tensor(output1, output1_name);
    validate_fused_tensor(output2, output2_name);
    TORCH_CHECK(
        input1.sizes().equals(input2.sizes()) &&
        input1.sizes().equals(output1.sizes()) &&
        input1.sizes().equals(output2.sizes()),
        "fused WCS input and output tensors must have identical shapes"
    );
    TORCH_CHECK(
        input1.numel() == input2.numel() &&
        input1.numel() == output1.numel() &&
        input1.numel() == output2.numel(),
        "fused WCS input and output tensors must have identical numel"
    );
    TORCH_CHECK(
        at::has_internal_overlap(output1) == at::MemOverlap::No,
        output1_name, " must not have internal overlap"
    );
    TORCH_CHECK(
        at::has_internal_overlap(output2) == at::MemOverlap::No,
        output2_name, " must not have internal overlap"
    );
    validate_fused_output_non_overlap(output1, output1_name, input1, input1_name);
    validate_fused_output_non_overlap(output1, output1_name, input2, input2_name);
    validate_fused_output_non_overlap(output1, output1_name, output2, output2_name);
    validate_fused_output_non_overlap(output2, output2_name, input1, input1_name);
    validate_fused_output_non_overlap(output2, output2_name, input2, input2_name);
}

void validate_fused_projection_and_native_type(
    const std::string& proj_code,
    const std::string& native_type
) {
    const bool supported_projection =
        proj_code == "TAN" || proj_code == "SIN" || proj_code == "ARC" ||
        proj_code == "CEA" || proj_code == "AIT" || proj_code == "HPX" ||
        proj_code == "CAR" || proj_code == "SFL" || proj_code == "MOL" ||
        proj_code == "PCO" || proj_code == "XPH" || proj_code == "TSC" ||
        proj_code == "CSC" || proj_code == "QSC";
    TORCH_CHECK(supported_projection, "unsupported fused WCS projection: ", proj_code);
    TORCH_CHECK(
        native_type == "center" || native_type == "pole",
        "unsupported fused WCS native_type: ", native_type
    );
}

void validate_fused_scalar(double value, const char* name) {
    TORCH_CHECK(std::isfinite(value), name, " must be finite");
}

void validate_fused_tpv_axis(
    const torch::Tensor& idx,
    const char* idx_name,
    const torch::Tensor& coeff,
    const char* coeff_name
) {
    validate_tpv_axis(idx, idx_name, coeff, coeff_name, true);
}

void validate_fused_common(
    const std::string& proj_code,
    const std::string& native_type,
    bool is_tpv,
    const torch::Tensor& tpv_idx1,
    const torch::Tensor& tpv_c1,
    const torch::Tensor& tpv_idx2,
    const torch::Tensor& tpv_c2,
    std::initializer_list<std::pair<const char*, double>> scalars
) {
    validate_fused_projection_and_native_type(proj_code, native_type);
    for (const auto& [name, value] : scalars) {
        validate_fused_scalar(value, name);
    }
    if (is_tpv) {
        validate_fused_tpv_axis(tpv_idx1, "tpv_idx1", tpv_c1, "tpv_c1");
        validate_fused_tpv_axis(tpv_idx2, "tpv_idx2", tpv_c2, "tpv_c2");
    }
}

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
    validate_fused_buffers(
        x_tensor, "x_tensor", y_tensor, "y_tensor",
        ra_tensor, "ra_tensor", dec_tensor, "dec_tensor"
    );
    validate_fused_common(
        proj_code, native_type, is_tpv, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2,
        {
            {"crpix0", crpix0}, {"crpix1", crpix1},
            {"cd00", cd00}, {"cd01", cd01}, {"cd10", cd10}, {"cd11", cd11},
            {"lambda_param", lambda_param}, {"hpx_H", hpx_H}, {"hpx_K", hpx_K},
            {"east_x", east_x}, {"east_y", east_y},
            {"north_x", north_x}, {"north_y", north_y}, {"north_z", north_z},
            {"radial_x", radial_x}, {"radial_y", radial_y}, {"radial_z", radial_z},
            {"ra_p_rad", ra_p_rad}, {"dec_p_rad", dec_p_rad}, {"phi_p_rad", phi_p_rad},
            {"alpha0", alpha0}, {"delta0", delta0},
        }
    );
    TORCH_CHECK(
        proj_code != "CEA" || lambda_param != 0.0,
        "CEA lambda must be finite and non-zero"
    );
    const HpxParams hpx = make_hpx_params(hpx_H, hpx_K);
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
            double r = std::hypot(xi_deg, eta_deg);
            double r_rad = r * D2R;
            phi = std::atan2(-xi_deg, -eta_deg) * R2D;
            theta = std::atan2(1.0, std::max(r_rad, 1e-15)) * R2D;
        } else if (proj_code == "SIN") {
            double r = std::hypot(xi_deg, eta_deg);
            double r_rad = r * D2R;
            if (r_rad > 1.0) {
                phi = std::nan("");
                theta = std::nan("");
            } else {
                phi = std::atan2(-xi_deg, -eta_deg) * R2D;
                theta = std::acos(r_rad) * R2D;
            }
        } else if (proj_code == "ARC") {
            double r = std::hypot(xi_deg, eta_deg);
            if (!std::isfinite(r) || r > ARC_RADIUS_MAX) {
                phi = std::nan("");
                theta = std::nan("");
            } else {
                phi = std::atan2(-xi_deg, -eta_deg) * R2D;
                theta = 90.0 - std::min(r, 180.0);
            }
        } else if (proj_code == "CEA") {
            phi = xi_deg;
            double val = lambda_param * eta_deg * D2R;
            if (!std::isfinite(val) || std::abs(val) > 1.0 + CEA_DOMAIN_TOL) {
                phi = std::nan("");
                theta = std::nan("");
            } else {
                theta = std::asin(std::clamp(val, -1.0, 1.0)) * R2D;
            }
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
            if (!hpx_plane_to_native_scalar(xi_deg, eta_deg, hpx, &phi, &theta)) {
                phi = std::nan("");
                theta = std::nan("");
            }
        } else if (proj_code == "CAR") {
            phi = xi_deg;
            theta = eta_deg;
        } else if (proj_code == "SFL") {
            theta = eta_deg;
            double cos_theta = std::cos(theta * D2R);
            phi = (std::abs(cos_theta) > 1e-12) ? (xi_deg / cos_theta) : 0.0;
        } else if (proj_code == "MOL") {
            if (!mol_plane_to_native_scalar(xi_deg, eta_deg, &phi, &theta)) {
                phi = std::nan("");
                theta = std::nan("");
            }
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
            const double xi_wrapped = xi_deg < -45.0 ? xi_deg + 360.0 : xi_deg;
            for (int f = 0; f < 6; ++f) {
                double fx = (xi_wrapped - cx[f]) / 45.0;
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

        if (!std::isfinite(phi) || !std::isfinite(theta)) {
            ra_ptr[i] = std::nan("");
            dec_ptr[i] = std::nan("");
            continue;
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
    validate_fused_buffers(
        ra_tensor, "ra_tensor", dec_tensor, "dec_tensor",
        x_tensor, "x_tensor", y_tensor, "y_tensor"
    );
    validate_fused_common(
        proj_code, native_type, is_tpv, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2,
        {
            {"crpix0", crpix0}, {"crpix1", crpix1},
            {"cdi00", cdi00}, {"cdi01", cdi01}, {"cdi10", cdi10}, {"cdi11", cdi11},
            {"lambda_param", lambda_param}, {"hpx_H", hpx_H}, {"hpx_K", hpx_K},
            {"east_x", east_x}, {"east_y", east_y},
            {"north_x", north_x}, {"north_y", north_y}, {"north_z", north_z},
            {"radial_x", radial_x}, {"radial_y", radial_y}, {"radial_z", radial_z},
            {"ra_p_rad", ra_p_rad}, {"dec_p_rad", dec_p_rad}, {"phi_p_rad", phi_p_rad},
            {"alpha0", alpha0}, {"delta0", delta0},
        }
    );
    TORCH_CHECK(
        proj_code != "CEA" || lambda_param != 0.0,
        "CEA lambda must be finite and non-zero"
    );
    const HpxParams hpx = make_hpx_params(hpx_H, hpx_K);
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
                double w_rot = X * radial_x + Y * radial_y + Z * radial_z;
                theta = std::atan2(w_rot, std::hypot(u_rot, n_rot)) * R2D;
                phi = std::atan2(u_rot, n_rot) * R2D + phi_p_rad * R2D;
                phi = std::fmod(phi + 360.0, 360.0);
            } else {
                double u_rot = X * east_x + Y * east_y;
                double v_rot = -(X * radial_x + Y * radial_y + Z * radial_z);
                double w_rot = X * north_x + Y * north_y + Z * north_z;
                theta = std::atan2(w_rot, std::hypot(u_rot, v_rot)) * R2D;
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
            if (
                !std::isfinite(theta) || theta <= TAN_HORIZON_TOL * R2D ||
                theta > 90.0 + CEA_DOMAIN_TOL
            ) {
                xi = std::nan("");
                eta = std::nan("");
            } else {
                double theta_rad = theta * D2R;
                double r = R2D / std::tan(theta_rad);
                double phi_rad = phi * D2R;
                xi = -r * std::sin(phi_rad);
                eta = -r * std::cos(phi_rad);
            }
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
            if (!std::isfinite(theta) || theta < -90.0 || theta > 90.0) {
                xi = std::nan("");
                eta = std::nan("");
            } else {
                double r = 90.0 - theta;
                double phi_rad = phi * D2R;
                xi = -r * std::sin(phi_rad);
                eta = -r * std::cos(phi_rad);
            }
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
            if (!hpx_native_to_plane_scalar(phi_w, theta, hpx, &xi, &eta)) {
                xi = std::nan("");
                eta = std::nan("");
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
            if (!mol_native_to_plane_scalar(phi_w, theta, &xi, &eta)) {
                xi = std::nan("");
                eta = std::nan("");
            }
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
                tpv_n2,
                TPV_FUSED_MAX_ITER,
                TPV_FUSED_TOL
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
