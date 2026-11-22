#pragma once

#include "torchsky/native.h"

namespace torchsky {

native::Tensor sphere_planar_point_in_polygon_cpu(
    const native::Tensor& points_xy,
    const native::Tensor& polygon_xy,
    double eps
);

native::Tensor sphere_arc_intersections_matrix_cpu(
    const native::Tensor& a0,
    const native::Tensor& a1,
    const native::Tensor& b0,
    const native::Tensor& b1,
    double tol_rad
);

}
