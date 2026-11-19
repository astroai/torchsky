// Zone-index sky crossmatch kernels. Torch-free: raw CPU buffers through
// nanobind ndarray (DLPack, zero copy) and a local std::thread pool.
//
// Mirrors torchsky/catalogs/_zones.py exactly in window arithmetic: zones of
// height h, keys zone << 32 | floor(ra * 2^32 / 360), RA half-width
// asin(sin W / cos dec) padded by 8 ulp, full circle when the cap holds a pole,
// two RA pieces across 0/360. Candidates pass a padded chord prefilter and are
// accepted on the atan2 separation, so the pair set equals the torch path's
// outside float64 ties. Policy (radii, find modes, ordering of the left rows)
// stays in Python.

#include "torchsky/crossmatch.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace torchsky::xmatch {
namespace {

constexpr int kRaBits = 32;
constexpr double kRaScale = 4294967296.0;
constexpr int64_t kRaTop = (int64_t(1) << kRaBits) - 1;
constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg2Rad = kPi / 180.0;
constexpr double kArcsecPerRad = 180.0 * 3600.0 / kPi;
constexpr double kEps = DBL_EPSILON;

enum Combine : int { kSum = 0, kQuadrature = 1, kMax = 2 };

template <typename T>
using In1 = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
template <typename T>
using In2 = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
template <typename T>
using Out1 = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

struct Index {
    const int64_t* key;
    const double* xyz;     // [M, 3] sorted
    const int64_t* order;  // sorted -> original
    const double* rr;      // sorted right radii (arcsec) or nullptr
    int64_t m;
    double h;
    int64_t n_zones;
    double rr_max;
};

struct Queries {
    const double* ra;
    const double* dec;
    const double* rl;    // per-row radius (arcsec)
    const int64_t* perm; // visiting order (or nullptr)
    int64_t n;
    int64_t row_offset;  // global index of row 0 (self-upper test)
};

inline double combine(double a, double b, int how) {
    switch (how) {
        case kQuadrature: return std::sqrt(a * a + b * b);
        case kMax: return std::max(a, b);
        default: return a + b;
    }
}

inline int64_t zone_of(double dec, const Index& z) {
    double v = std::floor((dec + 90.0) / z.h);
    int64_t zi = static_cast<int64_t>(v);
    return std::clamp<int64_t>(zi, 0, z.n_zones - 1);
}

inline double remainder360(double ra) {
    double r = std::fmod(ra, 360.0);
    if (r < 0.0) r += 360.0;
    if (r >= 360.0) r -= 360.0;
    return r;
}

inline int64_t ra_key_of_piece(double ra_piece) {
    // Pieces are already inside [0, 360]; same floor and clamp as torch.
    double v = std::floor(ra_piece * (kRaScale / 360.0));
    if (!(v >= 0.0)) return 0;
    if (v > static_cast<double>(kRaTop)) return kRaTop;
    return static_cast<int64_t>(v);
}

// Calls fn(j_sorted, sep_arcsec, thr_arcsec) for every right row within
// combine(rl, rr[j]) of the query, scanning only zone/RA candidates.
template <typename Fn>
inline void for_each_pair(const Index& z, double ra_in, double dec, const double* q, double rl,
                          double window_arcsec, int how, Fn&& fn) {
    const double window_deg = window_arcsec / 3600.0;
    const double pad_dec = 8.0 * kEps * 90.0;
    const double pad_ra = 8.0 * kEps * 360.0;
    const double w = window_deg + pad_dec;
    const int64_t zlo = zone_of(dec - w, z);
    const int64_t zhi = zone_of(dec + w, z);
    const double sin_w = std::sin(std::min(window_deg, 90.0) * kDeg2Rad);
    const double cos_d = std::cos(dec * kDeg2Rad);
    const double ratio = sin_w / std::max(cos_d, DBL_MIN);
    const double dra = std::asin(std::min(ratio, 1.0)) / kDeg2Rad + pad_ra;
    const bool full = (ratio * (1.0 + 8.0 * kEps) >= 1.0) || (w >= 90.0 - std::fabs(dec)) || (dra >= 180.0);
    const double ra = remainder360(ra_in);
    const double lo = full ? 0.0 : ra - dra;
    const double hi = full ? 360.0 : ra + dra;
    double plo[2], phi[2];
    int pieces = 1;
    plo[0] = std::max(lo, 0.0);
    phi[0] = std::min(hi, 360.0);
    if (lo < 0.0) {
        plo[1] = lo + 360.0; phi[1] = 360.0; pieces = 2;
    } else if (hi > 360.0) {
        plo[1] = 0.0; phi[1] = hi - 360.0; pieces = 2;
    }
    for (int p = 0; p < pieces; ++p) {
        const int64_t klo = ra_key_of_piece(plo[p]);
        const int64_t khi = ra_key_of_piece(phi[p]);
        for (int64_t zone = zlo; zone <= zhi; ++zone) {
            const int64_t base = zone << kRaBits;
            const int64_t* b = std::lower_bound(z.key, z.key + z.m, base | klo);
            const int64_t* e = std::upper_bound(b, z.key + z.m, base | khi);
            for (const int64_t* it = b; it < e; ++it) {
                const int64_t j = it - z.key;
                const double* r = z.xyz + 3 * j;
                const double rr = z.rr ? z.rr[j] : 0.0;
                const double thr = z.rr ? combine(rl, rr, how) : rl;
                const double half = std::min(thr / 3600.0 * kDeg2Rad, kPi) * 0.5;
                const double s = 2.0 * std::sin(half);
                const double dx = q[0] - r[0], dy = q[1] - r[1], dz = q[2] - r[2];
                const double chord2 = dx * dx + dy * dy + dz * dz;
                if (!(chord2 <= s * s * (1.0 + 64.0 * kEps) + 1e-300)) continue;
                const double cx = q[1] * r[2] - q[2] * r[1];
                const double cy = q[2] * r[0] - q[0] * r[2];
                const double cz = q[0] * r[1] - q[1] * r[0];
                const double dot = std::clamp(q[0] * r[0] + q[1] * r[1] + q[2] * r[2], -1.0, 1.0);
                const double sep = std::atan2(std::sqrt(cx * cx + cy * cy + cz * cz), dot) * kArcsecPerRad;
                if (sep <= thr) fn(j, sep);
            }
        }
    }
}

// Dynamic block scheduling; each row's output is independent, so results do
// not depend on the thread count.
template <typename Fn>
void parallel_rows(int64_t n, int threads, Fn&& fn) {
    constexpr int64_t kBlock = 1024;
    if (threads <= 1 || n <= kBlock) {
        fn(int64_t(0), n);
        return;
    }
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        for (;;) {
            int64_t b = next.fetch_add(kBlock);
            if (b >= n) return;
            fn(b, std::min(n, b + kBlock));
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    for (int t = 1; t < threads; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

Index make_index(In1<int64_t> key, In2<double> xyz, In1<int64_t> order, In1<double> rr, double h,
                 int64_t n_zones, double rr_max) {
    Index z{key.data(), xyz.data(), order.data(), rr.shape(0) ? rr.data() : nullptr,
            static_cast<int64_t>(key.shape(0)), h, n_zones, rr_max};
    if (xyz.shape(0) != key.shape(0) || xyz.shape(1) != 3 || order.shape(0) != key.shape(0) ||
        (rr.shape(0) && rr.shape(0) != key.shape(0)))
        throw std::invalid_argument("zone index arrays have inconsistent shapes");
    return z;
}

Queries make_queries(In1<double> ra, In1<double> dec, In1<double> rl, In1<int64_t> perm, int64_t row_offset,
                     bool rl_optional = false) {
    const int64_t n = static_cast<int64_t>(ra.shape(0));
    if (dec.shape(0) != size_t(n) || (!rl_optional && rl.shape(0) != size_t(n)) ||
        (rl_optional && rl.shape(0) && rl.shape(0) != size_t(n)) || (perm.shape(0) && perm.shape(0) != size_t(n)))
        throw std::invalid_argument("query arrays have inconsistent shapes");
    return Queries{ra.data(), dec.data(), rl.shape(0) ? rl.data() : nullptr, perm.shape(0) ? perm.data() : nullptr,
                   n, row_offset};
}

// Unit vector of a query row (same arithmetic as torch unit_vectors).
inline void unit_of(double ra_deg, double dec_deg, double* q) {
    const double ra = ra_deg * kDeg2Rad, dec = dec_deg * kDeg2Rad;
    const double cd = std::cos(dec);
    q[0] = cd * std::cos(ra);
    q[1] = cd * std::sin(ra);
    q[2] = std::sin(dec);
}

// Right rows per square degree around a query: three zones, +-4h in RA on
// the sky (cost heuristic for the kNN start radius, mirrors local_density).
inline double local_density(const Index& z, double ra_in, double dec) {
    const double h = z.h;
    const int64_t zc = zone_of(dec, z);
    const double cos_d = std::max(std::cos(dec * kDeg2Rad), 1e-6);
    const double half = std::min(4.0 * h / cos_d, 180.0);
    const bool whole = half >= 180.0;
    const double ra = remainder360(ra_in);
    int64_t klo = whole ? 0 : ra_key_of_piece(remainder360(ra - half));
    int64_t khi = whole ? kRaTop : ra_key_of_piece(remainder360(ra + half));
    if (klo > khi) std::swap(klo, khi);
    int64_t count = 0;
    for (int64_t dz = -1; dz <= 1; ++dz) {
        const int64_t zz = zc + dz;
        if (zz < 0 || zz >= z.n_zones) continue;
        const int64_t base = zz << kRaBits;
        const int64_t* b = std::lower_bound(z.key, z.key + z.m, base | klo);
        const int64_t* e = std::upper_bound(b, z.key + z.m, base | khi);
        count += e - b;
    }
    const double area = 3.0 * h * (whole ? 360.0 * cos_d : 8.0 * h);
    return static_cast<double>(count) / area;
}

inline int64_t row_at(const Queries& q, int64_t t) { return q.perm ? q.perm[t] : t; }

}  // namespace

void count_pairs(In1<int64_t> key, In2<double> rxyz, In1<int64_t> order, In1<double> rr, double h, int64_t n_zones,
                 double rr_max, In1<double> ra, In1<double> dec, In1<double> rl, In1<int64_t> perm,
                 int how, bool self_upper, int64_t row_offset, Out1<int64_t> counts, int threads) {
    const Index z = make_index(key, rxyz, order, rr, h, n_zones, rr_max);
    const Queries q = make_queries(ra, dec, rl, perm, row_offset);
    int64_t* out = counts.data();
    nb::gil_scoped_release release;
    parallel_rows(q.n, threads, [&](int64_t b, int64_t e) {
        for (int64_t t = b; t < e; ++t) {
            const int64_t i = row_at(q, t);
            const double window = combine(q.rl[i], z.rr_max, how);
            double qv[3];
            unit_of(q.ra[i], q.dec[i], qv);
            int64_t c = 0;
            const int64_t gi = q.row_offset + i;
            for_each_pair(z, q.ra[i], q.dec[i], qv, q.rl[i], window, how,
                          [&](int64_t j, double) { c += (!self_upper || z.order[j] > gi); });
            out[i] = c;
        }
    });
}

void fill_pairs(In1<int64_t> key, In2<double> rxyz, In1<int64_t> order, In1<double> rr, double h, int64_t n_zones,
                double rr_max, In1<double> ra, In1<double> dec, In1<double> rl, In1<int64_t> perm,
                int how, bool self_upper, int64_t row_offset, In1<int64_t> offsets, Out1<int64_t> out_j,
                Out1<double> out_sep, int threads) {
    const Index z = make_index(key, rxyz, order, rr, h, n_zones, rr_max);
    const Queries q = make_queries(ra, dec, rl, perm, row_offset);
    const int64_t* off = offsets.data();
    int64_t* oj = out_j.data();
    double* os = out_sep.data();
    const int64_t total = static_cast<int64_t>(out_j.shape(0));
    nb::gil_scoped_release release;
    parallel_rows(q.n, threads, [&](int64_t b, int64_t e) {
        std::vector<std::pair<double, int64_t>> buf;
        for (int64_t t = b; t < e; ++t) {
            const int64_t i = row_at(q, t);
            const double window = combine(q.rl[i], z.rr_max, how);
            const int64_t gi = q.row_offset + i;
            double qv[3];
            unit_of(q.ra[i], q.dec[i], qv);
            buf.clear();
            for_each_pair(z, q.ra[i], q.dec[i], qv, q.rl[i], window, how, [&](int64_t j, double sep) {
                const int64_t jo = z.order[j];
                if (!self_upper || jo > gi) buf.emplace_back(sep, jo);
            });
            std::sort(buf.begin(), buf.end());
            int64_t o = off[i];
            for (const auto& [sep, jo] : buf) {
                if (o >= total) break;  // guarded: counts and fill disagree only on misuse
                oj[o] = jo;
                os[o] = sep;
                ++o;
            }
        }
    });
}

void best_pairs(In1<int64_t> key, In2<double> rxyz, In1<int64_t> order, In1<double> rr, double h, int64_t n_zones,
                double rr_max, In1<double> ra, In1<double> dec, In1<double> rl, In1<int64_t> perm,
                int how, Out1<int64_t> out_j, Out1<double> out_sep, int threads) {
    const Index z = make_index(key, rxyz, order, rr, h, n_zones, rr_max);
    const Queries q = make_queries(ra, dec, rl, perm, 0);
    int64_t* oj = out_j.data();
    double* os = out_sep.data();
    nb::gil_scoped_release release;
    parallel_rows(q.n, threads, [&](int64_t b, int64_t e) {
        for (int64_t t = b; t < e; ++t) {
            const int64_t i = row_at(q, t);
            const double window = combine(q.rl[i], z.rr_max, how);
            double qv[3];
            unit_of(q.ra[i], q.dec[i], qv);
            double best = std::numeric_limits<double>::infinity();
            int64_t bj = -1;
            for_each_pair(z, q.ra[i], q.dec[i], qv, q.rl[i], window, how, [&](int64_t j, double sep) {
                const int64_t jo = z.order[j];
                if (sep < best || (sep == best && jo < bj)) {
                    best = sep;
                    bj = jo;
                }
            });
            oj[i] = bj;
            os[i] = best;
        }
    });
}

void knn_pairs(In1<int64_t> key, In2<double> rxyz, In1<int64_t> order, double h, int64_t n_zones, In1<double> ra,
               In1<double> dec, In1<double> r0, In1<int64_t> perm, int64_t k, double cap, Out1<int64_t> out_j,
               Out1<double> out_sep, int threads) {
    const Index z{key.data(), rxyz.data(), order.data(), nullptr, static_cast<int64_t>(key.shape(0)), h, n_zones, 0.0};
    // r0 empty: start radius from the local density in kernel (expected k rows).
    const Queries q = make_queries(ra, dec, r0, perm, 0, /*rl_optional=*/true);
    int64_t* oj = out_j.data();
    double* os = out_sep.data();
    if (out_j.shape(0) != size_t(q.n * k)) throw std::invalid_argument("knn output has the wrong size");
    nb::gil_scoped_release release;
    parallel_rows(q.n, threads, [&](int64_t b, int64_t e) {
        std::vector<std::pair<double, int64_t>> buf;
        for (int64_t t = b; t < e; ++t) {
            const int64_t i = row_at(q, t);
            double qv[3];
            unit_of(q.ra[i], q.dec[i], qv);
            double r;
            if (q.rl) {
                r = q.rl[i];
            } else {
                const double rho = local_density(z, q.ra[i], q.dec[i]);  // per deg^2
                r = rho > 0.0 ? std::sqrt(static_cast<double>(k) / (kPi * rho)) * 3600.0 : 4.0 * z.h * 3600.0;
            }
            r = std::min(r, cap);
            for (;;) {
                buf.clear();
                for_each_pair(z, q.ra[i], q.dec[i], qv, r, r, kSum,
                              [&](int64_t j, double sep) { buf.emplace_back(sep, z.order[j]); });
                if (static_cast<int64_t>(buf.size()) >= k || r >= cap) break;
                r = std::min(2.0 * r, cap);
            }
            const int64_t take = std::min<int64_t>(k, buf.size());
            std::partial_sort(buf.begin(), buf.begin() + take, buf.end());
            for (int64_t c = 0; c < k; ++c) {
                oj[i * k + c] = c < take ? buf[c].second : -1;
                os[i * k + c] = c < take ? buf[c].first : std::numeric_limits<double>::infinity();
            }
        }
    });
}

void register_crossmatch(nb::module_& m) {
    m.def("xmatch_count", &count_pairs);
    m.def("xmatch_fill", &fill_pairs);
    m.def("xmatch_best", &best_pairs);
    m.def("xmatch_knn", &knn_pairs);
}

}  // namespace torchsky::xmatch
