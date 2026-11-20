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
//
// Also the belief-propagation kernels of catalogs/_exclusive.py (exclusive
// association): segmented log-sum-exp excluding self, one pairwise message
// sweep, and the whole Anderson-accelerated pairwise fixed-point solve.

#include "torchsky/crossmatch.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
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

// Calls fn(j_sorted, sep_arcsec) for every right row within
// combine(rl, rr[j]) of the query, scanning only zone/RA candidates. With
// kChord the test is on the chord alone and fn receives chord^2 (monotone in
// the angle, used to rank nearest neighbours without a trig call per candidate).
template <bool kChord = false, typename Fn>
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
                if constexpr (kChord) {
                    if (chord2 <= s * s) fn(j, chord2);
                } else {
                    if (!(chord2 <= s * s * (1.0 + 64.0 * kEps) + 1e-300)) continue;
                    const double cx = q[1] * r[2] - q[2] * r[1];
                    const double cy = q[2] * r[0] - q[0] * r[2];
                    const double cz = q[0] * r[1] - q[1] * r[0];
                    const double dot = std::clamp(q[0] * r[0] + q[1] * r[1] + q[2] * r[2], -1.0, 1.0);
                    const double sep = std::atan2(std::sqrt(cx * cx + cy * cy + cz * cz), dot) * kArcsecPerRad;
                    if (sep <= thr) fn(j, sep, chord2);
                }
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
                          [&](int64_t j, double, double) { c += (!self_upper || z.order[j] > gi); });
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
        // (chord^2, original right row, separation): order by chord^2 (exactly
        // rounded, same as the torch path), ties by the lowest right row.
        std::vector<std::tuple<double, int64_t, double>> buf;
        for (int64_t t = b; t < e; ++t) {
            const int64_t i = row_at(q, t);
            const double window = combine(q.rl[i], z.rr_max, how);
            const int64_t gi = q.row_offset + i;
            double qv[3];
            unit_of(q.ra[i], q.dec[i], qv);
            buf.clear();
            for_each_pair(z, q.ra[i], q.dec[i], qv, q.rl[i], window, how, [&](int64_t j, double sep, double c2) {
                const int64_t jo = z.order[j];
                if (!self_upper || jo > gi) buf.emplace_back(c2, jo, sep);
            });
            std::sort(buf.begin(), buf.end());
            int64_t o = off[i];
            for (const auto& [c2, jo, sep] : buf) {
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
            double best_sep = best;
            int64_t bj = -1;
            for_each_pair(z, q.ra[i], q.dec[i], qv, q.rl[i], window, how, [&](int64_t j, double sep, double c2) {
                const int64_t jo = z.order[j];
                if (c2 < best || (c2 == best && jo < bj)) {
                    best = c2;
                    best_sep = sep;
                    bj = jo;
                }
            });
            oj[i] = bj;
            os[i] = best_sep;
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
                for_each_pair<true>(z, q.ra[i], q.dec[i], qv, r, r, kSum,
                                    [&](int64_t j, double c2) { buf.emplace_back(c2, j); });
                if (static_cast<int64_t>(buf.size()) >= k || r >= cap) break;
                r = std::min(2.0 * r, cap);
            }
            // Rank by (chord^2, original index); atan2 only for the k kept.
            for (auto& item : buf) item.second = z.order[item.second];
            const int64_t take = std::min<int64_t>(k, buf.size());
            std::partial_sort(buf.begin(), buf.begin() + take, buf.end());
            for (int64_t c = 0; c < k; ++c) {
                oj[i * k + c] = c < take ? buf[c].second : -1;
                os[i * k + c] = std::numeric_limits<double>::infinity();
            }
        }
    });
}

// Segmented log-sum-exp excluding self, for belief propagation in
// catalogs/_exclusive.py: out[e] = log(exp(slack[s]) + sum_{f in s, f != e}
// exp(x[f])) for e in segment s = [offsets[s], offsets[s + 1]). The sum without
// the element is taken directly for the segment's first largest element and as
// total - own for the others (never a cancellation: the total then also holds
// the largest term). Same arithmetic as the torch reference
// (lse_excluding_self); an empty slack means no slack term.
void segment_lse_excluding_self(In1<int64_t> offsets, In1<double> x, In1<double> slack, Out1<double> out,
                                int threads) {
    const int64_t n_seg = static_cast<int64_t>(offsets.shape(0)) - 1;
    if (n_seg < 0 || out.shape(0) != x.shape(0) || (slack.shape(0) && slack.shape(0) != size_t(n_seg)))
        throw std::invalid_argument("segment_lse_excluding_self: shape mismatch");
    const int64_t* off = offsets.data();
    const double* xv = x.data();
    const double* sv = slack.shape(0) ? slack.data() : nullptr;
    double* ov = out.data();
    if (n_seg > 0 && (off[0] != 0 || off[n_seg] != int64_t(x.shape(0))))
        throw std::invalid_argument("segment_lse_excluding_self: offsets do not cover x");
    const double ninf = -std::numeric_limits<double>::infinity();
    parallel_rows(n_seg, threads, [&](int64_t s0, int64_t s1) {
        for (int64_t s = s0; s < s1; ++s) {
            const int64_t a = off[s], b = off[s + 1];
            double m = ninf;
            int64_t arg = -1;
            for (int64_t e = a; e < b; ++e)
                if (xv[e] > m || (arg < 0 && xv[e] == m)) {
                    m = xv[e];
                    arg = e;
                }
            double big = sv ? std::max(m, sv[s]) : m;
            if (!std::isfinite(big)) big = 0.0;
            const double base = sv ? std::exp(sv[s] - big) : 0.0;
            double total = base, wo_arg = base;
            for (int64_t e = a; e < b; ++e) {
                const double ex = std::exp(xv[e] - big);
                total += ex;
                if (e != arg) wo_arg += ex;
            }
            for (int64_t e = a; e < b; ++e) {
                const double rest = e == arg ? wo_arg : total - std::exp(xv[e] - big);
                ov[e] = big + std::log(rest);
            }
        }
    });
}

namespace {

// Worker team for one call: threads are spawned once and reused across
// phases (condition-variable hand-off, robust when cores are oversubscribed).
// run(n, fn) calls fn(b, e) on fixed blocks of kTeamBlock rows with dynamic
// scheduling; outputs per row (or per block) never depend on the thread count.
constexpr int64_t kTeamBlock = 2048;

class Team {
   public:
    explicit Team(int threads) : threads_(std::max(1, threads)) {
        for (int t = 1; t < threads_; ++t) pool_.emplace_back([this] { loop(); });
    }
    ~Team() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (auto& th : pool_) th.join();
    }
    template <typename Fn>
    void run(int64_t n, Fn&& fn) {
        if (threads_ <= 1 || n <= kTeamBlock) {
            for (int64_t b = 0; b < n; b += kTeamBlock) fn(b, std::min(n, b + kTeamBlock));
            return;
        }
        std::function<void(int64_t, int64_t)> job(std::ref(fn));
        {
            std::lock_guard<std::mutex> lk(mu_);
            job_ = &job;
            n_ = n;
            next_.store(0);
            busy_ = threads_ - 1;
            ++gen_;
        }
        cv_.notify_all();
        work();
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [this] { return busy_ == 0; });
        job_ = nullptr;
    }

   private:
    void work() {
        for (;;) {
            const int64_t b = next_.fetch_add(kTeamBlock);
            if (b >= n_) return;
            (*job_)(b, std::min(n_, b + kTeamBlock));
        }
    }
    void loop() {
        int64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return gen_ != seen; });
                seen = gen_;
                if (stop_) return;
            }
            work();
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (--busy_ == 0) done_cv_.notify_one();
            }
        }
    }
    int threads_;
    std::vector<std::thread> pool_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::function<void(int64_t, int64_t)>* job_ = nullptr;
    int64_t n_ = 0;
    std::atomic<int64_t> next_{0};
    int busy_ = 0;
    int64_t gen_ = 0;
    bool stop_ = false;
};

// Pairwise belief-propagation layout (see bp_pairwise_sweep).
struct BpGraph {
    const int64_t* goff;
    const double* lw;
    const int64_t* sm;
    const int64_t* soff;
    int64_t n_group, n_src, n_slot, n_msg;
};

BpGraph make_bp_graph(In1<int64_t> group_off, In1<double> logw, In1<int64_t> slot_msg, In1<int64_t> src_off,
                      int64_t n_msg) {
    BpGraph g{group_off.data(), logw.data(), slot_msg.data(), src_off.data(),
              static_cast<int64_t>(group_off.shape(0)) - 1, static_cast<int64_t>(src_off.shape(0)) - 1,
              static_cast<int64_t>(logw.shape(0)), n_msg};
    if (g.n_group < 0 || g.n_src < 0 || slot_msg.shape(0) != size_t(g.n_slot))
        throw std::invalid_argument("bp_pairwise: shape mismatch");
    if ((g.n_group > 0 && (g.goff[0] != 0 || g.goff[g.n_group] != g.n_slot)) ||
        (g.n_src > 0 && (g.soff[0] != 0 || g.soff[g.n_src] != n_msg)))
        throw std::invalid_argument("bp_pairwise: offsets do not cover the arrays");
    for (int64_t e = 0; e < g.n_slot; ++e)
        if (g.sm[e] >= n_msg) throw std::invalid_argument("bp_pairwise: message index out of range");
    return g;
}

// out = g(x), with log_r as scratch (both of length n_msg).
void bp_sweep(const BpGraph& g, const double* xv, double* lr, double* ov, Team& team) {
    const double ninf = -std::numeric_limits<double>::infinity();
    team.run(g.n_group, [&](int64_t g0, int64_t g1) {
        thread_local std::vector<double> u, ex;
        for (int64_t q = g0; q < g1; ++q) {
            const int64_t a = g.goff[q], b = g.goff[q + 1];
            u.resize(static_cast<size_t>(b - a));
            ex.resize(static_cast<size_t>(b - a));
            double m = ninf;
            int64_t arg = -1;
            for (int64_t e = a; e < b; ++e) {
                const double rho = g.sm[e] >= 0 ? xv[g.sm[e]] : 0.0;
                const double v = g.lw[e] + rho;
                u[e - a] = v;
                if (v > m || (arg < 0 && v == m)) {
                    m = v;
                    arg = e;
                }
            }
            const double big = std::isfinite(m) ? m : 0.0;
            double total = 0.0, wo_arg = 0.0;
            for (int64_t e = a; e < b; ++e) {
                ex[e - a] = std::exp(u[e - a] - big);
                total += ex[e - a];
                if (e != arg) wo_arg += ex[e - a];
            }
            for (int64_t e = a; e < b; ++e) {
                if (g.sm[e] < 0) continue;
                const double rho = xv[g.sm[e]];
                const double rest = e == arg ? wo_arg : total - ex[e - a];
                const double v = (u[e - a] - rho) - (big + std::log(rest));
                lr[g.sm[e]] = std::isnan(v) ? ninf : v;
            }
        }
    });
    team.run(g.n_src, [&](int64_t s0, int64_t s1) {
        thread_local std::vector<double> ex;
        for (int64_t s = s0; s < s1; ++s) {
            const int64_t a = g.soff[s], b = g.soff[s + 1];
            ex.resize(static_cast<size_t>(b - a));
            double m = ninf;
            int64_t arg = -1;
            for (int64_t e = a; e < b; ++e)
                if (lr[e] > m || (arg < 0 && lr[e] == m)) {
                    m = lr[e];
                    arg = e;
                }
            const double big = std::max(m, 0.0);  // slack log 1 = 0
            const double base = std::exp(-big);
            double total = base, wo_arg = base;
            for (int64_t e = a; e < b; ++e) {
                ex[e - a] = std::exp(lr[e] - big);
                total += ex[e - a];
                if (e != arg) wo_arg += ex[e - a];
            }
            for (int64_t e = a; e < b; ++e) {
                const double rest = e == arg ? wo_arg : total - ex[e - a];
                ov[e] = -(big + std::log(rest));
            }
        }
    });
}

// Solve (A + reg I) y = r for a small dense symmetric A (row-major, k x k) by
// Gaussian elimination with partial pivoting.
bool small_solve(std::vector<double> a, std::vector<double> r, int k, double* y) {
    for (int c = 0; c < k; ++c) {
        int p = c;
        for (int i = c + 1; i < k; ++i)
            if (std::fabs(a[i * k + c]) > std::fabs(a[p * k + c])) p = i;
        if (a[p * k + c] == 0.0 || !std::isfinite(a[p * k + c])) return false;
        if (p != c) {
            for (int j = 0; j < k; ++j) std::swap(a[c * k + j], a[p * k + j]);
            std::swap(r[c], r[p]);
        }
        for (int i = c + 1; i < k; ++i) {
            const double f = a[i * k + c] / a[c * k + c];
            if (f == 0.0) continue;
            for (int j = c; j < k; ++j) a[i * k + j] -= f * a[c * k + j];
            r[i] -= f * r[c];
        }
    }
    for (int i = k - 1; i >= 0; --i) {
        double s = r[i];
        for (int j = i + 1; j < k; ++j) s -= a[i * k + j] * y[j];
        y[i] = s / a[i * k + i];
    }
    return true;
}

}  // namespace
// One belief-propagation message sweep of the pairwise exclusive model
// (catalogs/_exclusive.py, exclusive_log_rho's update with one hypothesis per
// slot), torch-free. Active slots are in owner order, owner g holding
// [group_off[g], group_off[g + 1]); slot_msg[e] is the message of slot e (or -1:
// rho = 0). Messages are in source order, source s holding
// [src_off[s], src_off[s + 1]); x is their current log rho. Per slot,
// u = logw + rho, and per message
//   log r = (u - rho) - log sum_{f in owner, f != e} exp(u_f)   (NaN -> -inf)
//   out   = -log(1 + sum_{other messages of the source} exp(log r)).
// Both segmented log-sum-exps excluding self use the arithmetic of
// segment_lse_excluding_self (and of the torch reference).
void bp_pairwise_sweep(In1<int64_t> group_off, In1<double> logw, In1<int64_t> slot_msg, In1<int64_t> src_off,
                       In1<double> x, Out1<double> out, int threads) {
    const int64_t n_msg = static_cast<int64_t>(x.shape(0));
    if (out.shape(0) != size_t(n_msg)) throw std::invalid_argument("bp_pairwise_sweep: shape mismatch");
    const BpGraph g = make_bp_graph(group_off, logw, slot_msg, src_off, n_msg);
    std::vector<double> lr(static_cast<size_t>(n_msg), -std::numeric_limits<double>::infinity());
    nb::gil_scoped_release release;
    Team team(threads);
    bp_sweep(g, x.data(), lr.data(), out.data(), team);
}

// Fixed point x = g(x) of the sweep above in one call, by the type-II Anderson
// acceleration of catalogs/_exclusive.py (_anderson, Walker & Ni 2011): the
// last `memory` differences in a ring buffer, the Gram system regularised by
// 1e-12 of its largest diagonal, the extrapolated messages clamped at 0
// (log rho <= 0), and a restart when the residual grows past twice the best one
// or stalls for `memory` steps. x holds the start on input and g(x*) on
// output. Stops when max |g(x) - x| <= tol or after max_iter steps. Returns
// (sweeps, final max |g(x) - x|); the caller decides whether that converged.
// Dot products and maxima are reduced over fixed blocks in block order, so the
// result does not depend on the thread count.
std::pair<int64_t, double> bp_pairwise_solve(In1<int64_t> group_off, In1<double> logw, In1<int64_t> slot_msg,
                                             In1<int64_t> src_off, Out1<double> x, double tol, int64_t max_iter,
                                             int memory, int threads) {
    const int64_t n = static_cast<int64_t>(x.shape(0));
    const BpGraph g = make_bp_graph(group_off, logw, slot_msg, src_off, n);
    if (memory < 1) throw std::invalid_argument("bp_pairwise_solve: memory must be >= 1");
    if (n == 0) return {0, 0.0};
    nb::gil_scoped_release release;
    Team team(threads);
    const size_t un = static_cast<size_t>(n);
    const int mem = memory;
    double* xv = x.data();
    std::vector<double> lr(un), gx(un), f(un), gx_new(un), f_new(un), xs(xv, xv + n);
    std::vector<double> df(un * mem), dg(un * mem);
    const int64_t n_blocks = (n + kTeamBlock - 1) / kTeamBlock;
    // Per block: max |f_new|, then (mem) dots df_j . df_slot and (mem) dots df_j . f_new.
    const int stride = 1 + 2 * mem;
    std::vector<double> part(static_cast<size_t>(n_blocks * stride));
    std::vector<double> gram(static_cast<size_t>(mem * mem), 0.0), rhs(static_cast<size_t>(mem), 0.0);
    int64_t sweeps = 0;

    auto max_abs = [&](const std::vector<double>& v) {
        team.run(n, [&](int64_t b, int64_t e) {
            double m = 0.0;
            for (int64_t i = b; i < e; ++i) m = std::max(m, std::fabs(v[i]));
            part[(b / kTeamBlock) * stride] = m;
        });
        double m = 0.0;
        for (int64_t k = 0; k < n_blocks; ++k) m = std::max(m, part[k * stride]);
        return m;
    };

    bp_sweep(g, xs.data(), lr.data(), gx.data(), team);
    ++sweeps;
    for (int64_t i = 0; i < n; ++i) f[i] = gx[i] - xs[i];
    double res = max_abs(f);
    double best = res;
    int used = 0, slot = 0, stale = 0;
    std::vector<double> gamma(static_cast<size_t>(mem));
    for (int64_t it = 0; it < max_iter; ++it) {
        if (res <= tol) break;
        stale = res < 0.99 * best ? 0 : stale + 1;
        if (res > 2.0 * best || stale > mem) {
            used = 0;  // restart a diverging or stagnating history
            slot = 0;
            stale = 0;
        }
        best = std::min(best, res);
        bool extrapolate = used > 0;
        if (extrapolate) {
            double dmax = 0.0;
            for (int i = 0; i < used; ++i) dmax = std::max(dmax, gram[i * mem + i]);
            const double reg = 1e-12 * std::max(dmax, 1e-300);
            std::vector<double> a(static_cast<size_t>(used * used)), r(static_cast<size_t>(used));
            for (int i = 0; i < used; ++i) {
                for (int j = 0; j < used; ++j) a[i * used + j] = gram[i * mem + j];
                a[i * used + i] += reg;
                r[i] = rhs[i];
            }
            extrapolate = small_solve(std::move(a), std::move(r), used, gamma.data());
        }
        if (extrapolate) {
            team.run(n, [&](int64_t b, int64_t e) {
                for (int64_t i = b; i < e; ++i) {
                    double v = gx[i];
                    for (int j = 0; j < used; ++j) v -= gamma[j] * dg[static_cast<size_t>(j) * un + i];
                    xs[i] = std::min(v, 0.0);
                }
            });
        } else {
            std::copy(gx.begin(), gx.end(), xs.begin());
        }
        bp_sweep(g, xs.data(), lr.data(), gx_new.data(), team);
        ++sweeps;
        // New differences into the ring, the residual, and the dots for the
        // next Gram system (row `slot` against every valid row, and df . f_new).
        const int new_used = std::min(used + 1, mem);
        double* dfs = df.data() + static_cast<size_t>(slot) * un;
        double* dgs = dg.data() + static_cast<size_t>(slot) * un;
        team.run(n, [&](int64_t b, int64_t e) {
            double m = 0.0;
            for (int64_t i = b; i < e; ++i) {
                const double fn = gx_new[i] - xs[i];
                f_new[i] = fn;
                dfs[i] = fn - f[i];
                dgs[i] = gx_new[i] - gx[i];
                m = std::max(m, std::fabs(fn));
            }
            double* p = part.data() + (b / kTeamBlock) * stride;
            p[0] = m;
            for (int j = 0; j < new_used; ++j) {
                const double* dj = df.data() + static_cast<size_t>(j) * un;
                double s1 = 0.0, s2 = 0.0;
                for (int64_t i = b; i < e; ++i) {
                    s1 += dj[i] * dfs[i];
                    s2 += dj[i] * f_new[i];
                }
                p[1 + j] = s1;
                p[1 + mem + j] = s2;
            }
        });
        res = 0.0;
        for (int j = 0; j < new_used; ++j) {
            double s1 = 0.0, s2 = 0.0;
            for (int64_t k = 0; k < n_blocks; ++k) {
                s1 += part[k * stride + 1 + j];
                s2 += part[k * stride + 1 + mem + j];
            }
            gram[j * mem + slot] = gram[slot * mem + j] = s1;
            rhs[j] = s2;
        }
        for (int64_t k = 0; k < n_blocks; ++k) res = std::max(res, part[k * stride]);
        slot = (slot + 1) % mem;
        used = new_used;
        gx.swap(gx_new);
        f.swap(f_new);
    }
    std::copy(gx.begin(), gx.end(), xv);
    return {sweeps, res};
}

void register_crossmatch(nb::module_& m) {
    m.def("xmatch_count", &count_pairs);
    m.def("xmatch_fill", &fill_pairs);
    m.def("xmatch_best", &best_pairs);
    m.def("xmatch_knn", &knn_pairs);
    m.def("xmatch_segment_lse_excl", &segment_lse_excluding_self);
    m.def("xmatch_bp_pairwise_sweep", &bp_pairwise_sweep);
    m.def("xmatch_bp_pairwise_solve", &bp_pairwise_solve);
}

}  // namespace torchsky::xmatch
