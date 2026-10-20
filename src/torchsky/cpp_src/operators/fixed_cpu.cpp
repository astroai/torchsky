#include <torch/library.h>

#include <tuple>

#include "torchsky/fixed_operators.h"
#include "torchsky/geom.h"
#include "torchsky/healpix.h"

namespace torchsky {

torch::Tensor sphere_planar_point_in_polygon_dispatch_cpu(
    const torch::Tensor& points_xy,
    const torch::Tensor& polygon_xy,
    double eps
) {
    return sphere_planar_point_in_polygon_cpu(
        points_xy.contiguous(), polygon_xy.contiguous(), eps
    );
}

torch::Tensor sphere_arc_intersections_matrix_dispatch_cpu(
    const torch::Tensor& a0,
    const torch::Tensor& a1,
    const torch::Tensor& b0,
    const torch::Tensor& b1,
    double tol_rad
) {
    return sphere_arc_intersections_matrix_cpu(
        a0.contiguous(), a1.contiguous(), b0.contiguous(), b1.contiguous(), tol_rad
    );
}

std::tuple<torch::Tensor, torch::Tensor> reshape_interp_weights(
    std::pair<torch::Tensor, torch::Tensor> output,
    const torch::Tensor& coordinates
) {
    auto shape = coordinates.sizes().vec();
    shape.insert(shape.begin(), 4);
    return {output.first.reshape(shape), output.second.reshape(shape)};
}

std::tuple<torch::Tensor, torch::Tensor> healpix_pix2ang_ring_dispatch_cpu(
    int64_t nside,
    const torch::Tensor& pixels
) {
    auto output = healpix_pix2ang_ring_cpu(nside, pixels);
    return {output.first, output.second};
}

std::tuple<torch::Tensor, torch::Tensor> healpix_pix2ang_nested_dispatch_cpu(
    int64_t nside,
    const torch::Tensor& pixels
) {
    auto output = healpix_pix2ang_nested_cpu(nside, pixels);
    return {output.first, output.second};
}

std::tuple<torch::Tensor, torch::Tensor> healpix_get_interp_weights_ring_dispatch_cpu(
    int64_t nside,
    const torch::Tensor& lon_deg,
    const torch::Tensor& lat_deg
) {
    return reshape_interp_weights(
        healpix_get_interp_weights_ring_cpu(nside, lon_deg, lat_deg), lon_deg
    );
}

std::tuple<torch::Tensor, torch::Tensor> healpix_get_interp_weights_nested_dispatch_cpu(
    int64_t nside,
    const torch::Tensor& lon_deg,
    const torch::Tensor& lat_deg
) {
    return reshape_interp_weights(
        healpix_get_interp_weights_nested_cpu(nside, lon_deg, lat_deg), lon_deg
    );
}

}

TORCH_LIBRARY_IMPL(torchsky, CPU, m) {
    m.impl("sphere_planar_point_in_polygon", TORCH_FN(torchsky::sphere_planar_point_in_polygon_dispatch_cpu));
    m.impl("sphere_arc_intersections_matrix", TORCH_FN(torchsky::sphere_arc_intersections_matrix_dispatch_cpu));
    m.impl("healpix_ring2nest", TORCH_FN(torchsky::healpix_ring2nest_cpu));
    m.impl("healpix_nest2ring", TORCH_FN(torchsky::healpix_nest2ring_cpu));
    m.impl("healpix_ang2pix_ring", TORCH_FN(torchsky::healpix_ang2pix_ring_cpu));
    m.impl("healpix_ang2pix_nested", TORCH_FN(torchsky::healpix_ang2pix_nested_cpu));
    m.impl("healpix_pix2ang_ring", TORCH_FN(torchsky::healpix_pix2ang_ring_dispatch_cpu));
    m.impl("healpix_pix2ang_nested", TORCH_FN(torchsky::healpix_pix2ang_nested_dispatch_cpu));
    m.impl("healpix_neighbors_ring", TORCH_FN(torchsky::healpix_neighbors_ring_cpu));
    m.impl("healpix_neighbors_nested", TORCH_FN(torchsky::healpix_neighbors_nested_cpu));
    m.impl("healpix_get_interp_weights_ring", TORCH_FN(torchsky::healpix_get_interp_weights_ring_dispatch_cpu));
    m.impl("healpix_get_interp_weights_nested", TORCH_FN(torchsky::healpix_get_interp_weights_nested_dispatch_cpu));
    m.impl("healpix_get_interp_val_ring", TORCH_FN(torchsky::healpix_get_interp_val_ring_cpu));
    m.impl("healpix_get_interp_val_nested", TORCH_FN(torchsky::healpix_get_interp_val_nested_cpu));
    m.impl("sparse_get_values_blocks", TORCH_FN(torchsky::sparse_get_values_blocks_cpu));
}

namespace torchsky {

void ensure_fixed_cpu_operators_linked() {}

}
