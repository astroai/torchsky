// TPV (tangent-plane polynomial) inverse on the CPU: per-point Newton.
//
// The pure-torch TPV.invert runs the same Newton iteration as whole-array
// sweeps; at small and medium N its cost is kernel-launch overhead (hundreds of
// tensor ops per sweep). This kernel solves each point independently with the
// identical contract, so results agree with the torch path to the Newton
// tolerance:
//   * affine seed (inverse of the polynomial's linear part when invertible,
//     else the target itself);
//   * steps clamped to [-1, 1] per coordinate;
//   * a point stops when its residual is <= tol, and fails (NaN) when its
//     Jacobian is non-finite or ill-conditioned (|det| <= 64 eps scale^2),
//     when an iterate becomes non-finite, or when it has not converged after
//     max_iter steps.
// It is used for CPU float64 inputs that need no gradient; the torch path
// stays the reference for gradients and other devices.
#include "torchsky/native_nb.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace torchsky {
namespace {

constexpr double kConditionEps = 64.0 * std::numeric_limits<double>::epsilon();

struct Term {
    int px;
    int py;
    int pr;
    double c;
};

std::vector<Term> read_terms(const native::Tensor& terms, const char* name) {
    TS_CHECK(terms.dtype() == native::ScalarType::Double, name, " must be float64");
    TS_CHECK(terms.dim() == 2 && terms.size(1) == 4, name, " must have shape [K, 4]");
    const double* p = terms.data_ptr<double>();
    std::vector<Term> out;
    out.reserve(static_cast<size_t>(terms.size(0)));
    for (int64_t k = 0; k < terms.size(0); ++k) {
        const double* row = p + 4 * k;
        for (int j = 0; j < 3; ++j) {
            TS_CHECK(
                row[j] >= 0.0 && row[j] <= 7.0 && row[j] == std::floor(row[j]),
                name, " exponents must be integers in [0, 7]"
            );
        }
        TS_CHECK(std::isfinite(row[3]), name, " coefficients must be finite");
        out.push_back({static_cast<int>(row[0]), static_cast<int>(row[1]),
                       static_cast<int>(row[2]), row[3]});
    }
    return out;
}

inline bool well_conditioned(double j11, double j12, double j21, double j22) {
    const double scale = std::max(std::max(std::abs(j11), std::abs(j12)),
                                  std::max(std::abs(j21), std::abs(j22)));
    const double det = j11 * j22 - j12 * j21;
    return std::isfinite(scale) && std::isfinite(det) && scale > 0.0 &&
           std::abs(det) > kConditionEps * scale * scale;
}

// One polynomial and its partial derivatives in its own (a, b) arguments.
// Axis 1 is evaluated in (u, v); axis 2 in (v, u), as in the FITS convention.
inline void eval_axis(
    const std::vector<Term>& terms,
    const std::array<double, 8>& ap,
    const std::array<double, 8>& bp,
    const std::array<double, 8>& rp,
    double dr_da,
    double dr_db,
    double& value,
    double& d_da,
    double& d_db
) {
    value = 0.0;
    d_da = 0.0;
    d_db = 0.0;
    for (const Term& t : terms) {
        const double base = ap[t.px] * bp[t.py];
        value += t.c * base * rp[t.pr];
        if (t.px > 0) d_da += t.c * t.px * ap[t.px - 1] * bp[t.py] * rp[t.pr];
        if (t.py > 0) d_db += t.c * t.py * ap[t.px] * bp[t.py - 1] * rp[t.pr];
        if (t.pr > 0) {
            const double radial = t.c * t.pr * base * rp[t.pr - 1];
            d_da += radial * dr_da;
            d_db += radial * dr_db;
        }
    }
}

inline void powers(double x, std::array<double, 8>& p) {
    p[0] = 1.0;
    for (int k = 1; k < 8; ++k) p[k] = p[k - 1] * x;
}

struct Eval {
    double xi, eta, j11, j12, j21, j22;  // j11 = dxi/du, j12 = dxi/dv, ...
};

inline Eval evaluate(const std::vector<Term>& t1, const std::vector<Term>& t2,
                     double u, double v) {
    std::array<double, 8> up{}, vp{}, rp{};
    const double r = std::hypot(u, v);
    powers(u, up);
    powers(v, vp);
    powers(r, rp);
    const double dr_du = r > 0.0 ? u / r : 0.0;
    const double dr_dv = r > 0.0 ? v / r : 0.0;
    Eval e{};
    eval_axis(t1, up, vp, rp, dr_du, dr_dv, e.xi, e.j11, e.j12);
    double deta_dv = 0.0, deta_du = 0.0;
    eval_axis(t2, vp, up, rp, dr_dv, dr_du, e.eta, deta_dv, deta_du);
    e.j21 = deta_du;
    e.j22 = deta_dv;
    return e;
}

std::pair<native::Tensor, native::Tensor> wcs_tpv_invert(
    const native::Tensor& xi_target,
    const native::Tensor& eta_target,
    const native::Tensor& terms1,
    const native::Tensor& terms2,
    const native::Tensor& seed,
    int64_t max_iter,
    double tol
) {
    TS_CHECK(xi_target.dtype() == native::ScalarType::Double &&
                 eta_target.dtype() == native::ScalarType::Double,
             "TPV inverse inputs must be float64");
    TS_CHECK(xi_target.sizes() == eta_target.sizes(),
             "TPV inverse inputs must have identical shapes");
    TS_CHECK(max_iter > 0, "TPV max_iter must be positive");
    TS_CHECK(std::isfinite(tol) && tol > 0.0, "TPV tolerance must be finite and positive");
    TS_CHECK(seed.dtype() == native::ScalarType::Double && seed.dim() == 1 &&
                 seed.size(0) == 7,
             "TPV seed must be float64 [b1, b2, inv00, inv01, inv10, inv11, has_inverse]");
    const std::vector<Term> t1 = read_terms(terms1, "terms1");
    const std::vector<Term> t2 = read_terms(terms2, "terms2");
    const double* s = seed.data_ptr<double>();
    const double b1 = s[0], b2 = s[1], i00 = s[2], i01 = s[3], i10 = s[4], i11 = s[5];
    const bool has_seed = s[6] != 0.0;

    const int64_t n = xi_target.numel();
    auto u_out = native::empty({n}, native::TensorOptions().dtype(native::ScalarType::Double));
    auto v_out = native::empty({n}, native::TensorOptions().dtype(native::ScalarType::Double));
    const double* xi_p = xi_target.data_ptr<double>();
    const double* eta_p = eta_target.data_ptr<double>();
    double* uo = u_out.data_ptr<double>();
    double* vo = v_out.data_ptr<double>();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    native::parallel_for(0, n, 256, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const double xt = xi_p[i], et = eta_p[i];
            double u = xt, v = et;
            if (has_seed) {
                const double x0 = xt - b1, e0 = et - b2;
                u = i00 * x0 + i01 * e0;
                v = i10 * x0 + i11 * e0;
            }
            bool viable = std::isfinite(xt) && std::isfinite(et) &&
                          std::isfinite(u) && std::isfinite(v);
            for (int64_t it = 0; viable && it < max_iter; ++it) {
                const Eval e = evaluate(t1, t2, u, v);
                const double f0 = e.xi - xt, f1 = e.eta - et;
                const double res = std::hypot(f0, f1);
                if (!std::isfinite(res) || !well_conditioned(e.j11, e.j12, e.j21, e.j22)) {
                    viable = false;
                    break;
                }
                if (res <= tol) break;
                const double det = e.j11 * e.j22 - e.j12 * e.j21;
                const double du = (e.j22 * f0 - e.j12 * f1) / det;
                const double dv = (-e.j21 * f0 + e.j11 * f1) / det;
                u -= std::clamp(du, -1.0, 1.0);
                v -= std::clamp(dv, -1.0, 1.0);
                if (!std::isfinite(u) || !std::isfinite(v)) viable = false;
            }
            bool valid = false;
            if (viable) {
                const Eval e = evaluate(t1, t2, u, v);
                const double res = std::hypot(e.xi - xt, e.eta - et);
                valid = std::isfinite(res) && res <= tol &&
                        well_conditioned(e.j11, e.j12, e.j21, e.j22);
            }
            uo[i] = valid ? u : nan;
            vo[i] = valid ? v : nan;
        }
    });
    return {u_out.reshape(xi_target.sizes().vec()), v_out.reshape(xi_target.sizes().vec())};
}

}  // namespace

void register_wcs_tpv(nanobind::module_& m) {
    native::def(m, "wcs_tpv_invert", &wcs_tpv_invert);
}

}  // namespace torchsky
