#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "torchsky_torch.h"
#include "torchsky/geom.h"

namespace torchsky {
namespace {

constexpr double GEOM_NORM_EPS = 1e-15;
constexpr double GEOM_PI = 3.141592653589793238462643383279502884;
constexpr int64_t GEOM_PARALLEL_MIN = 1024;

struct Vec3 {
    double x;
    double y;
    double z;
};

inline double clamp_unit(double v) {
    if (v < -1.0) {
        return -1.0;
    }
    if (v > 1.0) {
        return 1.0;
    }
    return v;
}

inline double dot3(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline double dot3(const double* a, const Vec3& b) {
    return a[0] * b.x + a[1] * b.y + a[2] * b.z;
}

inline double dot3(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross3(const double* a, const double* b) {
    return Vec3{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

inline Vec3 cross3(const Vec3& a, const Vec3& b) {
    return Vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

inline Vec3 cross3(const double* a, const Vec3& b) {
    return Vec3{
        a[1] * b.z - a[2] * b.y,
        a[2] * b.x - a[0] * b.z,
        a[0] * b.y - a[1] * b.x,
    };
}

inline double norm3(const Vec3& v) {
    return std::sqrt(dot3(v, v));
}

struct ArcData {
    Vec3 normal;
    double length;
    bool proper;
    bool point;
};

inline ArcData make_arc_data(const double* a, const double* b) {
    Vec3 normal = cross3(a, b);
    const double normal_norm = norm3(normal);
    const double endpoint_dot = clamp_unit(dot3(a, b));
    if (!(normal_norm > GEOM_NORM_EPS)) {
        return ArcData{Vec3{0.0, 0.0, 0.0}, 0.0, false, endpoint_dot >= 0.0};
    }
    const double inv_norm = 1.0 / normal_norm;
    normal = Vec3{
        normal.x * inv_norm,
        normal.y * inv_norm,
        normal.z * inv_norm,
    };
    return ArcData{
        normal,
        std::atan2(normal_norm, endpoint_dot),
        true,
        false,
    };
}

inline double angular_distance(const double* a, const Vec3& p) {
    return std::atan2(norm3(cross3(a, p)), clamp_unit(dot3(a, p)));
}

inline bool arc_contains_point(
    const double* a,
    const ArcData& arc,
    const Vec3& p,
    double tol_rad
) {
    if (arc.point) {
        return angular_distance(a, p) <= tol_rad;
    }
    if (!arc.proper) {
        return false;
    }

    const double plane_slack = 8.0 * std::numeric_limits<double>::epsilon();
    const double plane_tol =
        std::sin(std::min(tol_rad, GEOM_PI / 2.0)) + plane_slack;
    if (std::abs(dot3(arc.normal, p)) > plane_tol) {
        return false;
    }

    double angle = std::atan2(
        dot3(arc.normal, cross3(a, p)),
        clamp_unit(dot3(a, p))
    );
    if (angle < 0.0) {
        angle += 2.0 * GEOM_PI;
    }
    return angle <= arc.length + tol_rad || angle >= 2.0 * GEOM_PI - tol_rad;
}

inline void check_matrix_shape(
    const torch::Tensor& t,
    int64_t width,
    const char* name
) {
    if (!t.device().is_cpu()) {
        throw std::runtime_error(std::string(name) + " must be a CPU tensor");
    }
    if (t.scalar_type() != torch::kFloat64) {
        throw std::runtime_error(std::string(name) + " must have dtype float64");
    }
    if (t.dim() != 2 || t.size(1) != width) {
        throw std::runtime_error(
            std::string(name) + " must have shape [N," + std::to_string(width) + "]"
        );
    }
    if (!t.is_contiguous()) {
        throw std::runtime_error(std::string(name) + " must be contiguous");
    }
}

}  // namespace

torch::Tensor sphere_planar_point_in_polygon_cpu(
    const torch::Tensor& points_xy,
    const torch::Tensor& polygon_xy,
    double eps
) {
    check_matrix_shape(points_xy, 2, "points_xy");
    check_matrix_shape(polygon_xy, 2, "polygon_xy");
    if (!std::isfinite(eps) || eps < 0.0) {
        throw std::runtime_error("eps must be finite and nonnegative");
    }
    if (polygon_xy.size(0) < 3) {
        throw std::runtime_error("polygon_xy must contain at least three vertices");
    }

    const int64_t num_points = points_xy.size(0);
    const int64_t num_vertices = polygon_xy.size(0);
    auto out = torch::zeros(
        {num_points}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU)
    );

    const double* points_ptr = points_xy.data_ptr<double>();
    const double* poly_ptr = polygon_xy.data_ptr<double>();
    bool* out_ptr = out.data_ptr<bool>();

    at::parallel_for(
        0, num_points, GEOM_PARALLEL_MIN, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                const double x = points_ptr[2 * i];
                const double y = points_ptr[2 * i + 1];
                bool inside = false;
                for (int64_t j = 0; j < num_vertices; ++j) {
                    const int64_t k = (j + 1 == num_vertices) ? 0 : (j + 1);
                    const double xi = poly_ptr[2 * j];
                    const double yi = poly_ptr[2 * j + 1];
                    const double xj = poly_ptr[2 * k];
                    const double yj = poly_ptr[2 * k + 1];
                    const bool y_between = (yi > y) != (yj > y);
                    if (!y_between) {
                        continue;
                    }
                    const double x_inter =
                        xi + (xj - xi) * ((y - yi) / (yj - yi));
                    if (x < x_inter) {
                        inside = !inside;
                    }
                }
                out_ptr[i] = inside;
            }
        }
    );

    return out;
}

torch::Tensor sphere_arc_intersections_matrix_cpu(
    const torch::Tensor& a0,
    const torch::Tensor& a1,
    const torch::Tensor& b0,
    const torch::Tensor& b1,
    double tol_rad
) {
    check_matrix_shape(a0, 3, "a0");
    check_matrix_shape(a1, 3, "a1");
    check_matrix_shape(b0, 3, "b0");
    check_matrix_shape(b1, 3, "b1");
    if (!std::isfinite(tol_rad) || tol_rad < 0.0) {
        throw std::runtime_error("tol_rad must be finite and nonnegative");
    }
    if (a0.size(0) != a1.size(0)) {
        throw std::runtime_error("a0 and a1 must have matching shapes");
    }
    if (b0.size(0) != b1.size(0)) {
        throw std::runtime_error("b0 and b1 must have matching shapes");
    }

    const int64_t e1 = a0.size(0);
    const int64_t e2 = b0.size(0);
    auto out = torch::zeros(
        {e1, e2}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU)
    );
    if (e1 == 0 || e2 == 0) {
        return out;
    }

    const double* a0_ptr = a0.data_ptr<double>();
    const double* a1_ptr = a1.data_ptr<double>();
    const double* b0_ptr = b0.data_ptr<double>();
    const double* b1_ptr = b1.data_ptr<double>();
    bool* out_ptr = out.data_ptr<bool>();

    std::vector<ArcData> arcs1(static_cast<size_t>(e1));
    std::vector<ArcData> arcs2(static_cast<size_t>(e2));

    for (int64_t i = 0; i < e1; ++i) {
        const double* a0_i = a0_ptr + (3 * i);
        const double* a1_i = a1_ptr + (3 * i);
        arcs1[static_cast<size_t>(i)] = make_arc_data(a0_i, a1_i);
    }
    for (int64_t j = 0; j < e2; ++j) {
        const double* b0_j = b0_ptr + (3 * j);
        const double* b1_j = b1_ptr + (3 * j);
        arcs2[static_cast<size_t>(j)] = make_arc_data(b0_j, b1_j);
    }

    at::parallel_for(0, e1, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const double* a0_i = a0_ptr + (3 * i);
            const double* a1_i = a1_ptr + (3 * i);
            const ArcData& arc1 = arcs1[static_cast<size_t>(i)];
            for (int64_t j = 0; j < e2; ++j) {
                const double* b0_j = b0_ptr + (3 * j);
                const double* b1_j = b1_ptr + (3 * j);
                const ArcData& arc2 = arcs2[static_cast<size_t>(j)];

                if (!arc1.proper || !arc2.proper) {
                    bool hit = false;
                    if (arc1.point && arc2.point) {
                        const Vec3 b0_v{b0_j[0], b0_j[1], b0_j[2]};
                        hit = angular_distance(a0_i, b0_v) <= tol_rad;
                    } else if (arc1.point && arc2.proper) {
                        const Vec3 a0_v{a0_i[0], a0_i[1], a0_i[2]};
                        hit = arc_contains_point(b0_j, arc2, a0_v, tol_rad);
                    } else if (arc1.proper && arc2.point) {
                        const Vec3 b0_v{b0_j[0], b0_j[1], b0_j[2]};
                        hit = arc_contains_point(a0_i, arc1, b0_v, tol_rad);
                    }
                    out_ptr[i * e2 + j] = hit;
                    continue;
                }

                const Vec3 p = cross3(
                    arc1.normal,
                    arc2.normal
                );
                const double p_norm = norm3(p);
                if (p_norm <= GEOM_NORM_EPS) {
                    const Vec3 a0_v{a0_i[0], a0_i[1], a0_i[2]};
                    const Vec3 a1_v{a1_i[0], a1_i[1], a1_i[2]};
                    const Vec3 b0_v{b0_j[0], b0_j[1], b0_j[2]};
                    const Vec3 b1_v{b1_j[0], b1_j[1], b1_j[2]};
                    out_ptr[i * e2 + j] =
                        arc_contains_point(a0_i, arc1, b0_v, tol_rad)
                        || arc_contains_point(a0_i, arc1, b1_v, tol_rad)
                        || arc_contains_point(b0_j, arc2, a0_v, tol_rad)
                        || arc_contains_point(b0_j, arc2, a1_v, tol_rad);
                    continue;
                }

                const double inv_norm = 1.0 / p_norm;
                const Vec3 i1{p.x * inv_norm, p.y * inv_norm, p.z * inv_norm};
                const Vec3 i2{-i1.x, -i1.y, -i1.z};

                const bool hit =
                    (arc_contains_point(a0_i, arc1, i1, tol_rad)
                     && arc_contains_point(b0_j, arc2, i1, tol_rad))
                    || (arc_contains_point(a0_i, arc1, i2, tol_rad)
                        && arc_contains_point(b0_j, arc2, i2, tol_rad));
                out_ptr[i * e2 + j] = hit;
            }
        }
    });

    return out;
}

}  // namespace torchsky
