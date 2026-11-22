// Native CPU kernels for HEALPix range / MOC set algebra.
//
// Ranges are sorted, disjoint, half-open [start, end) NESTED pixel ranges at a
// fixed order. Everything is O(n log n) in the number of ranges/cells with no
// per-pixel work; the torch implementations in torchsky.sphere.moc stay as the
// device-generic fallback and the parity oracle.

#include "torchsky/native_nb.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "torchsky/healpix.h"

namespace torchsky {
namespace {

constexpr int kMaxOrder = 29;

using Range = std::pair<int64_t, int64_t>;

native::Tensor to_i64_cpu(const native::Tensor& t, const char* name) {
    if (!t.device().is_cpu() || t.scalar_type() != native::kInt64 || t.dim() != 1) {
        throw std::runtime_error(std::string(name) + " must be a 1-D int64 CPU tensor");
    }
    return t.contiguous();
}

std::tuple<native::Tensor, native::Tensor> ranges_to_tensors(const std::vector<Range>& r) {
    const int64_t n = static_cast<int64_t>(r.size());
    auto opts = native::TensorOptions().dtype(native::kInt64);
    auto s = native::empty({n}, opts);
    auto e = native::empty({n}, opts);
    auto* sp = s.data_ptr<int64_t>();
    auto* ep = e.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
        sp[i] = r[static_cast<size_t>(i)].first;
        ep[i] = r[static_cast<size_t>(i)].second;
    }
    return {s, e};
}

void merge_sorted_inplace(std::vector<Range>& r) {
    size_t w = 0;
    for (size_t i = 0; i < r.size(); ++i) {
        if (r[i].second <= r[i].first) {
            continue;
        }
        if (w > 0 && r[i].first <= r[w - 1].second) {
            r[w - 1].second = std::max(r[w - 1].second, r[i].second);
        } else {
            r[w++] = r[i];
        }
    }
    r.resize(w);
}

inline int floor_log2_u64(uint64_t v) {
    return 63 - __builtin_clzll(v);
}

std::tuple<native::Tensor, native::Tensor> moc_uniq_to_ranges_cpu(
    const native::Tensor& uniq_in, int64_t max_order
) {
    const auto uniq = to_i64_cpu(uniq_in, "uniq");
    if (max_order < 0 || max_order > kMaxOrder) {
        throw std::runtime_error("max_order out of range");
    }
    const int64_t n = uniq.numel();
    const int64_t* u = uniq.data_ptr<int64_t>();
    std::vector<Range> r;
    r.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const int64_t v = u[i];
        if (v < 4) {
            throw std::runtime_error("UNIQ values must be >= 4");
        }
        const int order = (floor_log2_u64(static_cast<uint64_t>(v)) - 2) >> 1;
        if (order > kMaxOrder) {
            throw std::runtime_error("UNIQ order too large");
        }
        const int64_t pix = v - (int64_t{4} << (2 * order));
        if (pix < 0 || pix >= (int64_t{12} << (2 * order))) {
            throw std::runtime_error("non-canonical UNIQ");
        }
        const int64_t delta = 2 * (max_order - order);
        if (delta >= 0) {
            r.emplace_back(pix << delta, (pix + 1) << delta);
        } else {
            const int64_t p = pix >> (-delta);
            r.emplace_back(p, p + 1);
        }
    }
    std::sort(r.begin(), r.end());
    merge_sorted_inplace(r);
    return ranges_to_tensors(r);
}

native::Tensor moc_ranges_to_uniq_cpu(
    const native::Tensor& starts_in, const native::Tensor& ends_in, int64_t max_order
) {
    const auto starts = to_i64_cpu(starts_in, "starts");
    const auto ends = to_i64_cpu(ends_in, "ends");
    if (starts.numel() != ends.numel()) {
        throw std::runtime_error("starts/ends size mismatch");
    }
    if (max_order < 0 || max_order > kMaxOrder) {
        throw std::runtime_error("max_order out of range");
    }
    const int64_t n = starts.numel();
    const int64_t* sp = starts.data_ptr<int64_t>();
    const int64_t* ep = ends.data_ptr<int64_t>();
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(std::min<int64_t>(n * 4 + 16, int64_t{1} << 26)));
    for (int64_t i = 0; i < n; ++i) {
        int64_t s = sp[i];
        const int64_t e = ep[i];
        while (s < e) {
            // Largest k with 4^k | s, s + 4^k <= e, k <= max_order.
            int k = (s == 0) ? static_cast<int>(max_order) : std::min<int>(
                static_cast<int>(max_order), __builtin_ctzll(static_cast<uint64_t>(s)) >> 1);
            while (k > 0 && s + (int64_t{1} << (2 * k)) > e) {
                --k;
            }
            const int order = static_cast<int>(max_order) - k;
            out.push_back((int64_t{4} << (2 * order)) + (s >> (2 * k)));
            s += int64_t{1} << (2 * k);
        }
    }
    auto t = native::empty({static_cast<int64_t>(out.size())}, native::TensorOptions().dtype(native::kInt64));
    std::copy(out.begin(), out.end(), t.data_ptr<int64_t>());
    return t;
}

// op: 0 union, 1 intersection, 2 difference (a - b); inputs sorted disjoint.
std::tuple<native::Tensor, native::Tensor> moc_combine_ranges_cpu(
    const native::Tensor& a_s_in,
    const native::Tensor& a_e_in,
    const native::Tensor& b_s_in,
    const native::Tensor& b_e_in,
    int64_t op
) {
    const auto a_s = to_i64_cpu(a_s_in, "a_starts");
    const auto a_e = to_i64_cpu(a_e_in, "a_ends");
    const auto b_s = to_i64_cpu(b_s_in, "b_starts");
    const auto b_e = to_i64_cpu(b_e_in, "b_ends");
    if (a_s.numel() != a_e.numel() || b_s.numel() != b_e.numel()) {
        throw std::runtime_error("starts/ends size mismatch");
    }
    if (op < 0 || op > 2) {
        throw std::runtime_error("op must be 0 (union), 1 (intersection) or 2 (difference)");
    }
    const int64_t na = a_s.numel();
    const int64_t nb = b_s.numel();
    const int64_t* as = a_s.data_ptr<int64_t>();
    const int64_t* ae = a_e.data_ptr<int64_t>();
    const int64_t* bs = b_s.data_ptr<int64_t>();
    const int64_t* be = b_e.data_ptr<int64_t>();
    std::vector<Range> out;
    out.reserve(static_cast<size_t>(na + nb));
    auto push = [&](int64_t s, int64_t e) {
        if (e <= s) {
            return;
        }
        if (!out.empty() && s <= out.back().second) {
            out.back().second = std::max(out.back().second, e);
        } else {
            out.emplace_back(s, e);
        }
    };
    int64_t i = 0;
    int64_t j = 0;
    if (op == 0) {
        while (i < na || j < nb) {
            if (j >= nb || (i < na && as[i] <= bs[j])) {
                push(as[i], ae[i]);
                ++i;
            } else {
                push(bs[j], be[j]);
                ++j;
            }
        }
    } else if (op == 1) {
        while (i < na && j < nb) {
            const int64_t lo = std::max(as[i], bs[j]);
            const int64_t hi = std::min(ae[i], be[j]);
            if (lo < hi) {
                push(lo, hi);
            }
            if (ae[i] < be[j]) {
                ++i;
            } else {
                ++j;
            }
        }
    } else {
        for (i = 0, j = 0; i < na; ++i) {
            int64_t cur = as[i];
            const int64_t end = ae[i];
            while (j < nb && be[j] <= cur) {
                ++j;
            }
            int64_t jj = j;
            while (cur < end && jj < nb && bs[jj] < end) {
                if (bs[jj] > cur) {
                    push(cur, bs[jj]);
                }
                cur = std::max(cur, be[jj]);
                ++jj;
            }
            if (cur < end) {
                push(cur, end);
            }
        }
    }
    return ranges_to_tensors(out);
}

// Complement within [0, total).
std::tuple<native::Tensor, native::Tensor> moc_complement_ranges_cpu(
    const native::Tensor& s_in, const native::Tensor& e_in, int64_t total
) {
    const auto s = to_i64_cpu(s_in, "starts");
    const auto e = to_i64_cpu(e_in, "ends");
    if (s.numel() != e.numel()) {
        throw std::runtime_error("starts/ends size mismatch");
    }
    const int64_t* sp = s.data_ptr<int64_t>();
    const int64_t* ep = e.data_ptr<int64_t>();
    std::vector<Range> out;
    int64_t cur = 0;
    for (int64_t i = 0; i < s.numel(); ++i) {
        if (sp[i] > cur) {
            out.emplace_back(cur, std::min(sp[i], total));
        }
        cur = std::max(cur, ep[i]);
    }
    if (cur < total) {
        out.emplace_back(cur, total);
    }
    return ranges_to_tensors(out);
}

// Cells finer than `order` are replaced by their parent at `order` (not deduplicated).
native::Tensor moc_degrade_uniq_cpu(const native::Tensor& uniq_in, int64_t order) {
    const auto uniq = to_i64_cpu(uniq_in, "uniq");
    if (order < 0 || order > kMaxOrder) {
        throw std::runtime_error("order out of range");
    }
    const int64_t n = uniq.numel();
    const int64_t* u = uniq.data_ptr<int64_t>();
    auto out = native::empty({n}, native::TensorOptions().dtype(native::kInt64));
    int64_t* o = out.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
        const int64_t v = u[i];
        if (v < 4) {
            throw std::runtime_error("UNIQ values must be >= 4");
        }
        const int cell_order = (floor_log2_u64(static_cast<uint64_t>(v)) - 2) >> 1;
        if (cell_order > kMaxOrder) {
            throw std::runtime_error("UNIQ order too large");
        }
        const int64_t pix = v - (int64_t{4} << (2 * cell_order));
        if (pix < 0 || pix >= (int64_t{12} << (2 * cell_order))) {
            throw std::runtime_error("non-canonical UNIQ");
        }
        if (cell_order > order) {
            o[i] = (int64_t{4} << (2 * order)) + (pix >> (2 * (cell_order - order)));
        } else {
            o[i] = v;
        }
    }
    return out;
}

native::Tensor sphere_query_disc_nest_ranges_dispatch_cpu(
    int64_t nside, const native::Tensor& vec, double cos_lim
) {
    return healpix_query_disc_nest_ranges_cpu(nside, vec, cos_lim);
}

}  // namespace
void register_sphere_ranges(nanobind::module_& m) {
    using native::def;
    def(m, "moc_uniq_to_ranges", &moc_uniq_to_ranges_cpu);
    def(m, "moc_ranges_to_uniq", &moc_ranges_to_uniq_cpu);
    def(m, "moc_combine_ranges", &moc_combine_ranges_cpu);
    def(m, "moc_complement_ranges", &moc_complement_ranges_cpu);
    def(m, "moc_degrade_uniq", &moc_degrade_uniq_cpu);
    def(m, "sphere_query_disc_nest_ranges", &sphere_query_disc_nest_ranges_dispatch_cpu);
}

}  // namespace torchsky
