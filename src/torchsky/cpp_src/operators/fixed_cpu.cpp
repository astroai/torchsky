#include "torchsky/native_nb.h"

#include <tuple>

#include "torchsky/geom.h"
#include "torchsky/healpix.h"

namespace torchsky {

native::Tensor sphere_planar_point_in_polygon_dispatch_cpu(
    const native::Tensor& points_xy,
    const native::Tensor& polygon_xy,
    double eps
) {
    return sphere_planar_point_in_polygon_cpu(
        points_xy.contiguous(), polygon_xy.contiguous(), eps
    );
}

native::Tensor sphere_arc_intersections_matrix_dispatch_cpu(
    const native::Tensor& a0,
    const native::Tensor& a1,
    const native::Tensor& b0,
    const native::Tensor& b1,
    double tol_rad
) {
    return sphere_arc_intersections_matrix_cpu(
        a0.contiguous(), a1.contiguous(), b0.contiguous(), b1.contiguous(), tol_rad
    );
}

std::tuple<native::Tensor, native::Tensor> reshape_interp_weights(
    std::pair<native::Tensor, native::Tensor> output,
    const native::Tensor& coordinates
) {
    auto shape = coordinates.sizes().vec();
    shape.insert(shape.begin(), 4);
    return {output.first.reshape(shape), output.second.reshape(shape)};
}

std::tuple<native::Tensor, native::Tensor> healpix_pix2ang_ring_dispatch_cpu(
    int64_t nside,
    const native::Tensor& pixels
) {
    auto output = healpix_pix2ang_ring_cpu(nside, pixels);
    return {output.first, output.second};
}

std::tuple<native::Tensor, native::Tensor> healpix_pix2ang_nested_dispatch_cpu(
    int64_t nside,
    const native::Tensor& pixels
) {
    auto output = healpix_pix2ang_nested_cpu(nside, pixels);
    return {output.first, output.second};
}

std::tuple<native::Tensor, native::Tensor> healpix_get_interp_weights_ring_dispatch_cpu(
    int64_t nside,
    const native::Tensor& lon_deg,
    const native::Tensor& lat_deg
) {
    return reshape_interp_weights(
        healpix_get_interp_weights_ring_cpu(nside, lon_deg, lat_deg), lon_deg
    );
}

std::tuple<native::Tensor, native::Tensor> healpix_get_interp_weights_nested_dispatch_cpu(
    int64_t nside,
    const native::Tensor& lon_deg,
    const native::Tensor& lat_deg
) {
    return reshape_interp_weights(
        healpix_get_interp_weights_nested_cpu(nside, lon_deg, lat_deg), lon_deg
    );
}

void register_fixed(nanobind::module_& m) {
    using native::def;
    def(m, "sphere_planar_point_in_polygon", &sphere_planar_point_in_polygon_dispatch_cpu);
    def(m, "sphere_arc_intersections_matrix", &sphere_arc_intersections_matrix_dispatch_cpu);
    def(m, "healpix_ring2nest", &healpix_ring2nest_cpu);
    def(m, "healpix_nest2ring", &healpix_nest2ring_cpu);
    def(m, "healpix_ang2pix_ring", &healpix_ang2pix_ring_cpu);
    def(m, "healpix_ang2pix_nested", &healpix_ang2pix_nested_cpu);
    def(m, "healpix_pix2ang_ring", &healpix_pix2ang_ring_dispatch_cpu);
    def(m, "healpix_pix2ang_nested", &healpix_pix2ang_nested_dispatch_cpu);
    def(m, "healpix_neighbors_ring", &healpix_neighbors_ring_cpu);
    def(m, "healpix_neighbors_nested", &healpix_neighbors_nested_cpu);
    def(m, "healpix_get_interp_weights_ring", &healpix_get_interp_weights_ring_dispatch_cpu);
    def(m, "healpix_get_interp_weights_nested", &healpix_get_interp_weights_nested_dispatch_cpu);
    def(m, "healpix_get_interp_val_ring", &healpix_get_interp_val_ring_cpu);
    def(m, "healpix_get_interp_val_nested", &healpix_get_interp_val_nested_cpu);
    def(m, "healpix_query_disc", &healpix_query_disc_cpu);
    def(m, "sparse_get_values_blocks", &sparse_get_values_blocks_cpu);
}

}  // namespace torchsky
