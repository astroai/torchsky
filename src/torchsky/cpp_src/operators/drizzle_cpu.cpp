// Exact square-drizzle overlap accumulation (CPU).
//
// Each input-pixel footprint is a convex quadrilateral in target-pixel coordinates; the
// fractional overlap with every target pixel box [i - 1/2, i + 1/2] x [j - 1/2, j + 1/2]
// is accumulated into the flat target arrays.  The overlap with an axis-aligned box is the
// Green's-theorem boundary integral  oint clamp(x, a, a + 1) dy  over the quad edges clipped
// to the box row (the specialised axis-aligned form of STScI drizzle's boxer/sgarea), so no
// intermediate polygon is built.  Within one target row the integral is evaluated once per
// vertical box line and the box overlap is the difference of two neighbouring lines.
//
// Parallelism: target rows are split into bands; every (quad, box) pair is handled by the
// thread owning the box row, quads in ascending index order, so each target pixel receives its
// contributions in the same order for any thread count (deterministic, no atomics).

#include <ATen/Parallel.h>
#include <torch/library.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "torchsky/drizzle_operators.h"
#include "torchsky_torch.h"

TORCH_LIBRARY_FRAGMENT(torchsky, m) {
    m.def(
        "drizzle_square_accumulate(Tensor quad_x, Tensor quad_y, Tensor numer, Tensor weight, "
        "int width, int height, Tensor(a!) out_numer, Tensor(b!) out_weight, "
        "Tensor(c!) out_variance_weight, bool with_variance) -> int"
    );
}

namespace torchsky {

void ensure_drizzle_operators_linked() {}

namespace {

constexpr int64_t QUAD_PARALLEL_GRAIN = 2048;

enum : uint8_t { QUAD_SKIP = 0, QUAD_OK = 1, QUAD_INVALID = 2 };

struct QuadInfo {
    int32_t min_x;
    int32_t max_x;
    int32_t min_y;
    int32_t max_y;
    double area;
    double resolution;   // overlap area below which a box is indistinguishable from touching
    double orientation;  // +1 counter-clockwise, -1 clockwise
};

// Mean of max(u, 0) along a segment with u = lo at one end and u = hi at the other.
inline double positive_part_mean(double lo, double hi) {
    if (lo <= 0.0 && hi <= 0.0) {
        return 0.0;
    }
    if (lo >= 0.0 && hi >= 0.0) {
        return 0.5 * (lo + hi);
    }
    const double top = lo > hi ? lo : hi;
    return top * top / (2.0 * std::fabs(hi - lo));
}

inline int32_t clamp_to_int(double v, int32_t lo, int32_t hi) {
    if (!(v > static_cast<double>(lo))) {
        return lo;
    }
    if (v > static_cast<double>(hi)) {
        return hi;
    }
    return static_cast<int32_t>(v);
}

struct EdgeClip {
    double xa;
    double xb;
    double dy;
};

}  // namespace

int64_t drizzle_square_accumulate_cpu(
    const torch::Tensor& quad_x,
    const torch::Tensor& quad_y,
    const torch::Tensor& numer,
    const torch::Tensor& weight,
    int64_t width,
    int64_t height,
    const torch::Tensor& out_numer,
    const torch::Tensor& out_weight,
    const torch::Tensor& out_variance_weight,
    bool with_variance
) {
    TORCH_CHECK(quad_x.dim() == 2 && quad_x.size(1) == 4, "quad_x must have shape (n, 4)");
    TORCH_CHECK(quad_y.sizes() == quad_x.sizes(), "quad_y must have the same shape as quad_x");
    const int64_t n_quads = quad_x.size(0);
    TORCH_CHECK(
        quad_x.scalar_type() == torch::kFloat64 && quad_y.scalar_type() == torch::kFloat64,
        "quad_x and quad_y must be float64");
    TORCH_CHECK(
        numer.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32 &&
            out_numer.scalar_type() == torch::kFloat32 &&
            out_weight.scalar_type() == torch::kFloat32,
        "numer, weight and the output buffers must be float32");
    TORCH_CHECK(
        quad_x.is_contiguous() && quad_y.is_contiguous() && numer.is_contiguous() &&
            weight.is_contiguous() && out_numer.is_contiguous() && out_weight.is_contiguous(),
        "drizzle_square_accumulate requires contiguous tensors");
    TORCH_CHECK(
        numer.dim() == 1 && numer.size(0) == n_quads && weight.dim() == 1 &&
            weight.size(0) == n_quads,
        "numer and weight must be 1-D with one entry per quad");
    if (n_quads == 0 || width <= 0 || height <= 0) {
        return 0;
    }
    TORCH_CHECK(
        width <= std::numeric_limits<int64_t>::max() / height &&
            out_numer.numel() == width * height && out_weight.numel() == width * height,
        "output buffers must hold width * height elements");
    if (with_variance) {
        TORCH_CHECK(
            out_variance_weight.scalar_type() == torch::kFloat32 &&
                out_variance_weight.is_contiguous() &&
                out_variance_weight.numel() == width * height,
            "out_variance_weight must be a contiguous float32 buffer of width * height elements");
    }
    const double* qx = quad_x.data_ptr<double>();
    const double* qy = quad_y.data_ptr<double>();
    const float* numer_ptr = numer.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    float* out_n = out_numer.data_ptr<float>();
    float* out_w = out_weight.data_ptr<float>();
    float* out_v = with_variance ? out_variance_weight.data_ptr<float>() : nullptr;

    std::vector<uint8_t> status(static_cast<size_t>(n_quads), QUAD_SKIP);
    std::vector<QuadInfo> info(static_cast<size_t>(n_quads));

    // Pass 1: finiteness, strict convexity, area, candidate box range (clamped to the target).
    at::parallel_for(0, n_quads, QUAD_PARALLEL_GRAIN, [&](int64_t begin, int64_t end) {
        for (int64_t q = begin; q < end; ++q) {
            const double* x = qx + 4 * q;
            const double* y = qy + 4 * q;
            if (!(std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]) &&
                  std::isfinite(x[3]) && std::isfinite(y[0]) && std::isfinite(y[1]) &&
                  std::isfinite(y[2]) && std::isfinite(y[3]))) {
                continue;
            }
            double ex[4];
            double ey[4];
            for (int k = 0; k < 4; ++k) {
                ex[k] = x[(k + 1) & 3] - x[k];
                ey[k] = y[(k + 1) & 3] - y[k];
            }
            bool all_pos = true;
            bool all_neg = true;
            for (int k = 0; k < 4; ++k) {
                const double turn = ex[k] * ey[(k + 1) & 3] - ey[k] * ex[(k + 1) & 3];
                all_pos = all_pos && (turn > 0.0);
                all_neg = all_neg && (turn < 0.0);
            }
            if (!(all_pos || all_neg)) {
                status[q] = QUAD_INVALID;
                continue;
            }
            const double area =
                0.5 * std::fabs((x[2] - x[0]) * (y[3] - y[1]) - (y[2] - y[0]) * (x[3] - x[1]));
            const double min_xv = std::min(std::min(x[0], x[1]), std::min(x[2], x[3]));
            const double max_xv = std::max(std::max(x[0], x[1]), std::max(x[2], x[3]));
            const double min_yv = std::min(std::min(y[0], y[1]), std::min(y[2], y[3]));
            const double max_yv = std::max(std::max(y[0], y[1]), std::max(y[2], y[3]));
            // Pixel i covers [i - 1/2, i + 1/2]: open overlap with [min, max].
            const double lo_x = std::floor(min_xv - 0.5) + 1.0;
            const double hi_x = std::ceil(max_xv + 0.5) - 1.0;
            const double lo_y = std::floor(min_yv - 0.5) + 1.0;
            const double hi_y = std::ceil(max_yv + 0.5) - 1.0;
            QuadInfo& qi = info[q];
            qi.min_x = clamp_to_int(lo_x, 0, static_cast<int32_t>(width));
            qi.max_x = clamp_to_int(hi_x, -1, static_cast<int32_t>(width - 1));
            qi.min_y = clamp_to_int(lo_y, 0, static_cast<int32_t>(height));
            qi.max_y = clamp_to_int(hi_y, -1, static_cast<int32_t>(height - 1));
            qi.area = area;
            // The input vertices carry a representation error of ~eps * |coordinate|; an edge
            // displaced by that much changes an overlap by at most (error) x (chord <= quad
            // extent).  Overlaps below a few multiples of it are rounding-level grazing contacts.
            const double scale = std::max(
                1.0,
                std::max(
                    std::max(std::fabs(min_xv), std::fabs(max_xv)),
                    std::max(std::fabs(min_yv), std::fabs(max_yv))
                )
            );
            qi.resolution = 16.0 * DBL_EPSILON * scale * std::max(max_xv - min_xv, max_yv - min_yv);
            qi.orientation = all_pos ? 1.0 : -1.0;
            status[q] = (area > 0.0 && qi.max_x >= qi.min_x && qi.max_y >= qi.min_y)
                            ? QUAD_OK
                            : QUAD_SKIP;
        }
    });

    int64_t invalid = 0;
    int64_t row_lo = height;
    int64_t row_hi = -1;
    for (int64_t q = 0; q < n_quads; ++q) {
        if (status[q] == QUAD_INVALID) {
            ++invalid;
        } else if (status[q] == QUAD_OK) {
            row_lo = std::min<int64_t>(row_lo, info[q].min_y);
            row_hi = std::max<int64_t>(row_hi, info[q].max_y);
        }
    }
    if (invalid > 0 || row_hi < row_lo) {
        return invalid;
    }

    // Pass 2: bucket quads by target-row band (ascending quad order inside each band).
    const int64_t rows = row_hi - row_lo + 1;
    const int64_t n_threads = std::max<int64_t>(1, at::get_num_threads());
    const int64_t n_bands = std::min<int64_t>(rows, 4 * n_threads);
    const int64_t band_height = (rows + n_bands - 1) / n_bands;
    const int64_t n_used_bands = (rows + band_height - 1) / band_height;
    std::vector<int64_t> band_start(static_cast<size_t>(n_used_bands + 1), 0);
    for (int64_t q = 0; q < n_quads; ++q) {
        if (status[q] != QUAD_OK) continue;
        const int64_t b0 = (info[q].min_y - row_lo) / band_height;
        const int64_t b1 = (info[q].max_y - row_lo) / band_height;
        for (int64_t b = b0; b <= b1; ++b) ++band_start[b + 1];
    }
    for (int64_t b = 0; b < n_used_bands; ++b) band_start[b + 1] += band_start[b];
    std::vector<int64_t> band_fill(band_start.begin(), band_start.end() - 1);
    std::vector<int64_t> band_quads(static_cast<size_t>(band_start[n_used_bands]));
    for (int64_t q = 0; q < n_quads; ++q) {
        if (status[q] != QUAD_OK) continue;
        const int64_t b0 = (info[q].min_y - row_lo) / band_height;
        const int64_t b1 = (info[q].max_y - row_lo) / band_height;
        for (int64_t b = b0; b <= b1; ++b) band_quads[band_fill[b]++] = q;
    }

    // Pass 3: per band, exact overlaps of every quad with the boxes of the band's rows.
    at::parallel_for(0, n_used_bands, 1, [&](int64_t band_begin, int64_t band_end) {
        for (int64_t band = band_begin; band < band_end; ++band) {
            const int64_t band_row_lo = row_lo + band * band_height;
            const int64_t band_row_hi = std::min(band_row_lo + band_height - 1, row_hi);
            for (int64_t slot = band_start[band]; slot < band_start[band + 1]; ++slot) {
                const int64_t q = band_quads[slot];
                const QuadInfo& qi = info[q];
                // Quad vertices relative to the lower-left corner of the first candidate box.
                const double origin_x = static_cast<double>(qi.min_x) - 0.5;
                const double origin_y = static_cast<double>(qi.min_y) - 0.5;
                double vx[4];
                double vy[4];
                double edge_dx[4];
                double edge_inv_dy[4];
                for (int k = 0; k < 4; ++k) {
                    vx[k] = qx[4 * q + k] - origin_x;
                    vy[k] = qy[4 * q + k] - origin_y;
                }
                for (int k = 0; k < 4; ++k) {
                    edge_dx[k] = vx[(k + 1) & 3] - vx[k];
                    const double dy = vy[(k + 1) & 3] - vy[k];
                    edge_inv_dy[k] = dy == 0.0 ? 0.0 : 1.0 / dy;
                }
                const int64_t nx = qi.max_x - qi.min_x + 1;
                const int64_t row_first = std::max<int64_t>(qi.min_y, band_row_lo);
                const int64_t row_last = std::min<int64_t>(qi.max_y, band_row_hi);
                const double inv_area = 1.0 / qi.area;
                const float numer_q = numer_ptr[q];
                const float weight_q = weight_ptr[q];
                for (int64_t row = row_first; row <= row_last; ++row) {
                    const double r = static_cast<double>(row - qi.min_y);
                    EdgeClip clip[4];
                    double left = INFINITY;
                    double right = -INFINITY;
                    for (int k = 0; k < 4; ++k) {
                        const double y0 = vy[k];
                        const double y1 = vy[(k + 1) & 3];
                        const double ya = std::min(std::max(y0, r), r + 1.0);
                        const double yb = std::min(std::max(y1, r), r + 1.0);
                        EdgeClip& c = clip[k];
                        c.dy = yb - ya;
                        const double inv = edge_inv_dy[k];
                        c.xa = vx[k] + edge_dx[k] * ((ya - y0) * inv);
                        c.xb = vx[k] + edge_dx[k] * ((yb - y0) * inv);
                        if (c.dy != 0.0) {
                            left = std::min(left, std::min(c.xa, c.xb));
                            right = std::max(right, std::max(c.xa, c.xb));
                        }
                    }
                    if (!(right > left)) continue;
                    const int64_t k_lo = std::max<int64_t>(0, static_cast<int64_t>(std::floor(left)));
                    const int64_t k_hi =
                        std::min<int64_t>(nx - 1, static_cast<int64_t>(std::ceil(right)) - 1);
                    if (k_hi < k_lo) continue;
                    auto line_integral = [&](double line) {
                        double total = 0.0;
                        for (int k = 0; k < 4; ++k) {
                            const EdgeClip& c = clip[k];
                            if (c.dy != 0.0) {
                                total += c.dy * positive_part_mean(c.xa - line, c.xb - line);
                            }
                        }
                        return total;
                    };
                    double t_prev = line_integral(static_cast<double>(k_lo));
                    float* row_n = out_n + row * width + qi.min_x;
                    float* row_w = out_w + row * width + qi.min_x;
                    float* row_v = out_v != nullptr ? out_v + row * width + qi.min_x : nullptr;
                    for (int64_t col = k_lo; col <= k_hi; ++col) {
                        const double t_next = line_integral(static_cast<double>(col + 1));
                        const double overlap = qi.orientation * (t_prev - t_next);
                        t_prev = t_next;
                        if (!(overlap > qi.resolution)) continue;
                        const float coefficient = static_cast<float>(overlap * inv_area);
                        row_n[col] += numer_q * coefficient;
                        row_w[col] += weight_q * coefficient;
                        if (row_v != nullptr) {
                            row_v[col] += weight_q * (coefficient * coefficient);
                        }
                    }
                }
            }
        }
    });
    return 0;
}

}  // namespace torchsky

TORCH_LIBRARY_IMPL(torchsky, CPU, m) {
    m.impl("drizzle_square_accumulate", TORCH_FN(torchsky::drizzle_square_accumulate_cpu));
}
