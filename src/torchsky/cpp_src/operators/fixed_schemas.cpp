#include <torch/library.h>

#include "torchsky/fixed_operators.h"

TORCH_LIBRARY_FRAGMENT(torchsky, m) {
    m.def("sphere_planar_point_in_polygon(Tensor points_xy, Tensor polygon_xy, float eps=1e-15) -> Tensor");
    m.def("sphere_arc_intersections_matrix(Tensor a0, Tensor a1, Tensor b0, Tensor b1, float tol_rad=1e-10) -> Tensor");
    m.def("healpix_ring2nest(int nside, Tensor pix_ring) -> Tensor");
    m.def("healpix_nest2ring(int nside, Tensor pix_nest) -> Tensor");
    m.def("healpix_ang2pix_ring(int nside, Tensor lon_deg, Tensor lat_deg) -> Tensor");
    m.def("healpix_ang2pix_nested(int nside, Tensor lon_deg, Tensor lat_deg) -> Tensor");
    m.def("healpix_pix2ang_ring(int nside, Tensor pix_ring) -> (Tensor lon_deg, Tensor lat_deg)");
    m.def("healpix_pix2ang_nested(int nside, Tensor pix_nest) -> (Tensor lon_deg, Tensor lat_deg)");
    m.def("healpix_neighbors_ring(int nside, Tensor pix_ring) -> Tensor");
    m.def("healpix_neighbors_nested(int nside, Tensor pix_nest) -> Tensor");
    m.def("healpix_get_interp_weights_ring(int nside, Tensor lon_deg, Tensor lat_deg) -> (Tensor pixels, Tensor weights)");
    m.def("healpix_get_interp_weights_nested(int nside, Tensor lon_deg, Tensor lat_deg) -> (Tensor pixels, Tensor weights)");
    m.def("healpix_get_interp_val_ring(int nside, Tensor maps, Tensor lon_deg, Tensor lat_deg) -> Tensor");
    m.def("healpix_get_interp_val_nested(int nside, Tensor maps, Tensor lon_deg, Tensor lat_deg) -> Tensor");
    m.def("healpix_query_disc(int nside, Tensor vec, float cos_lim, bool nest=False) -> Tensor");
    m.def("sparse_get_values_blocks(Tensor query, Tensor block_index, Tensor block_values, Tensor fill) -> Tensor");
}

namespace torchsky {

void ensure_fixed_operator_schemas_linked() {}

}
