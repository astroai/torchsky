#pragma once

#include "torchsky_torch.h"

namespace torchsky {

torch::Tensor sphere_planar_point_in_polygon_cpu(
    const torch::Tensor& points_xy,
    const torch::Tensor& polygon_xy,
    double eps
);

torch::Tensor sphere_arc_intersections_matrix_cpu(
    const torch::Tensor& a0,
    const torch::Tensor& a1,
    const torch::Tensor& b0,
    const torch::Tensor& b1,
    double tol_rad
);

}
