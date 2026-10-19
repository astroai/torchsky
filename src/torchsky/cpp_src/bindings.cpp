#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

#include <Python.h>
#include <memory>
#include <cstring>

#include "torchsky_torch.h"
// #include <torch/csrc/autograd/python_variable.h> // Removed to avoid pybind11 conflict
#include <ATen/DLConvertor.h>
#undef READONLY

#include "torch_compat.h"

#include <vector>
#include <vector>
#include <unordered_map>

namespace nb = nanobind;
using namespace nb::literals;

#include "healpix.cpp"
#include "geom.cpp"
#include "wcs.cpp"

namespace torchsky {
void wcs_pixel_to_world_fused_cpu(
    const torch::Tensor& x_tensor,
    const torch::Tensor& y_tensor,
    torch::Tensor& ra_tensor,
    torch::Tensor& dec_tensor,
    double crpix0, double crpix1,
    double cd00, double cd01,
    double cd10, double cd11,
    const std::string& proj_code,
    double lambda_param,
    double hpx_H,
    double hpx_K,
    double east_x, double east_y,
    double north_x, double north_y, double north_z,
    double radial_x, double radial_y, double radial_z,
    double ra_p_rad, double dec_p_rad, double phi_p_rad,
    const std::string& native_type,
    double alpha0, double delta0,
    bool is_tpv,
    const torch::Tensor& tpv_idx1,
    const torch::Tensor& tpv_c1,
    const torch::Tensor& tpv_idx2,
    const torch::Tensor& tpv_c2
);

void wcs_world_to_pixel_fused_cpu(
    const torch::Tensor& ra_tensor,
    const torch::Tensor& dec_tensor,
    torch::Tensor& x_tensor,
    torch::Tensor& y_tensor,
    double crpix0, double crpix1,
    double cdi00, double cdi01,
    double cdi10, double cdi11,
    const std::string& proj_code,
    double lambda_param,
    double hpx_H,
    double hpx_K,
    double east_x, double east_y,
    double north_x, double north_y, double north_z,
    double radial_x, double radial_y, double radial_z,
    double ra_p_rad, double dec_p_rad, double phi_p_rad,
    const std::string& native_type,
    double alpha0, double delta0,
    bool is_tpv,
    const torch::Tensor& tpv_idx1,
    const torch::Tensor& tpv_c1,
    const torch::Tensor& tpv_idx2,
    const torch::Tensor& tpv_c2
);
}
void bind_wcs(nb::module_& m) {
    m.def(
        "wcs_pixel_to_world_fused_cpu",
        [](nb::object x_obj, nb::object y_obj, nb::object ra_obj, nb::object dec_obj,
           double crpix0, double crpix1, double cd00, double cd01, double cd10, double cd11,
           const std::string& proj_code, double lambda_param, double hpx_H, double hpx_K,
           double east_x, double east_y, double north_x, double north_y, double north_z,
           double radial_x, double radial_y, double radial_z,
           double ra_p_rad, double dec_p_rad, double phi_p_rad,
           const std::string& native_type, double alpha0, double delta0,
           bool is_tpv,
           nb::object tpv_idx1_obj, nb::object tpv_c1_obj,
           nb::object tpv_idx2_obj, nb::object tpv_c2_obj) {
            auto x = python_to_tensor(x_obj);
            auto y = python_to_tensor(y_obj);
            auto ra = python_to_tensor(ra_obj);
            auto dec = python_to_tensor(dec_obj);
            auto tpv_idx1 = python_to_tensor(tpv_idx1_obj);
            auto tpv_c1 = python_to_tensor(tpv_c1_obj);
            auto tpv_idx2 = python_to_tensor(tpv_idx2_obj);
            auto tpv_c2 = python_to_tensor(tpv_c2_obj);
            {
                nb::gil_scoped_release release;
                torchsky::wcs_pixel_to_world_fused_cpu(
                    x, y, ra, dec, crpix0, crpix1, cd00, cd01, cd10, cd11,
                    proj_code, lambda_param, hpx_H, hpx_K,
                    east_x, east_y, north_x, north_y, north_z,
                    radial_x, radial_y, radial_z,
                    ra_p_rad, dec_p_rad, phi_p_rad,
                    native_type, alpha0, delta0,
                    is_tpv, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2
                );
            }
        },
        "x_tensor"_a, "y_tensor"_a, "ra_tensor"_a, "dec_tensor"_a,
        "crpix0"_a, "crpix1"_a, "cd00"_a, "cd01"_a, "cd10"_a, "cd11"_a,
        "proj_code"_a, "lambda_param"_a, "hpx_H"_a, "hpx_K"_a,
        "east_x"_a, "east_y"_a, "north_x"_a, "north_y"_a, "north_z"_a,
        "radial_x"_a, "radial_y"_a, "radial_z"_a,
        "ra_p_rad"_a, "dec_p_rad"_a, "phi_p_rad"_a,
        "native_type"_a, "alpha0"_a, "delta0"_a,
        "is_tpv"_a = false,
        "tpv_idx1"_a = nb::none(), "tpv_c1"_a = nb::none(),
        "tpv_idx2"_a = nb::none(), "tpv_c2"_a = nb::none()
    );

    m.def(
        "wcs_world_to_pixel_fused_cpu",
        [](nb::object ra_obj, nb::object dec_obj, nb::object x_obj, nb::object y_obj,
           double crpix0, double crpix1, double cdi00, double cdi01, double cdi10, double cdi11,
           const std::string& proj_code, double lambda_param, double hpx_H, double hpx_K,
           double east_x, double east_y, double north_x, double north_y, double north_z,
           double radial_x, double radial_y, double radial_z,
           double ra_p_rad, double dec_p_rad, double phi_p_rad,
           const std::string& native_type, double alpha0, double delta0,
           bool is_tpv,
           nb::object tpv_idx1_obj, nb::object tpv_c1_obj,
           nb::object tpv_idx2_obj, nb::object tpv_c2_obj) {
            auto ra = python_to_tensor(ra_obj);
            auto dec = python_to_tensor(dec_obj);
            auto x = python_to_tensor(x_obj);
            auto y = python_to_tensor(y_obj);
            auto tpv_idx1 = python_to_tensor(tpv_idx1_obj);
            auto tpv_c1 = python_to_tensor(tpv_c1_obj);
            auto tpv_idx2 = python_to_tensor(tpv_idx2_obj);
            auto tpv_c2 = python_to_tensor(tpv_c2_obj);
            {
                nb::gil_scoped_release release;
                torchsky::wcs_world_to_pixel_fused_cpu(
                    ra, dec, x, y, crpix0, crpix1, cdi00, cdi01, cdi10, cdi11,
                    proj_code, lambda_param, hpx_H, hpx_K,
                    east_x, east_y, north_x, north_y, north_z,
                    radial_x, radial_y, radial_z,
                    ra_p_rad, dec_p_rad, phi_p_rad,
                    native_type, alpha0, delta0,
                    is_tpv, tpv_idx1, tpv_c1, tpv_idx2, tpv_c2
                );
            }
        },
        "ra_tensor"_a, "dec_tensor"_a, "x_tensor"_a, "y_tensor"_a,
        "crpix0"_a, "crpix1"_a, "cdi00"_a, "cdi01"_a, "cdi10"_a, "cdi11"_a,
        "proj_code"_a, "lambda_param"_a, "hpx_H"_a, "hpx_K"_a,
        "east_x"_a, "east_y"_a, "north_x"_a, "north_y"_a, "north_z"_a,
        "radial_x"_a, "radial_y"_a, "radial_z"_a,
        "ra_p_rad"_a, "dec_p_rad"_a, "phi_p_rad"_a,
        "native_type"_a, "alpha0"_a, "delta0"_a,
        "is_tpv"_a = false,
        "tpv_idx1"_a = nb::none(), "tpv_c1"_a = nb::none(),
        "tpv_idx2"_a = nb::none(), "tpv_c2"_a = nb::none()
    );

    m.def("healpix_ring2nest_torch_cpu", [](int64_t nside, nb::object pix_obj) {
        auto pix = python_to_tensor(pix_obj).to(torch::kInt64).contiguous();
        torch::Tensor out;
        {
            nb::gil_scoped_release release;
            out = torchsky::healpix_ring2nest_cpu(nside, pix);
        }
        return tensor_to_python(out);
    }, nb::arg("nside"), nb::arg("pix"));


    m.def("healpix_nest2ring_torch_cpu", [](int64_t nside, nb::object pix_obj) {
        auto pix = python_to_tensor(pix_obj).to(torch::kInt64).contiguous();
        torch::Tensor out;
        {
            nb::gil_scoped_release release;
            out = torchsky::healpix_nest2ring_cpu(nside, pix);
        }
        return tensor_to_python(out);
    }, nb::arg("nside"), nb::arg("pix"));



    m.def("healpix_neighbors_ring_torch_cpu", [](int64_t nside, nb::object pix_obj) {
        auto pix = python_to_tensor(pix_obj).to(torch::kInt64).contiguous();
        torch::Tensor out;
        {
            nb::gil_scoped_release release;
            out = torchsky::healpix_neighbors_ring_cpu(nside, pix);
        }
        return tensor_to_python(out);
    }, nb::arg("nside"), nb::arg("pix"));

    m.def("healpix_neighbors_nested_torch_cpu", [](int64_t nside, nb::object pix_obj) {
        auto pix = python_to_tensor(pix_obj).to(torch::kInt64).contiguous();
        torch::Tensor out;
        {
            nb::gil_scoped_release release;
            out = torchsky::healpix_neighbors_nested_cpu(nside, pix);
        }
        return tensor_to_python(out);
    }, nb::arg("nside"), nb::arg("pix"));



    m.def("healpix_get_interp_weights_ring_torch_cpu",
          [](int64_t nside, nb::object lon_obj, nb::object lat_obj) {
              auto lon = python_to_tensor(lon_obj).to(torch::kFloat64).contiguous();
              auto lat = python_to_tensor(lat_obj).to(torch::kFloat64).contiguous();
              if (!lon.sizes().equals(lat.sizes())) {
                  throw std::runtime_error("lon/lat shape mismatch");
              }
              std::vector<int64_t> out_shape;
              out_shape.reserve(static_cast<size_t>(lon.dim()) + 1);
              out_shape.push_back(4);
              for (const auto s : lon.sizes()) {
                  out_shape.push_back(static_cast<int64_t>(s));
              }
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::healpix_get_interp_weights_ring_cpu(nside, lon, lat);
              }
              out.first = out.first.reshape(out_shape);
              out.second = out.second.reshape(out_shape);
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("nside"), nb::arg("lon"), nb::arg("lat"));

    m.def("healpix_get_interp_weights_nested_torch_cpu",
          [](int64_t nside, nb::object lon_obj, nb::object lat_obj) {
              auto lon = python_to_tensor(lon_obj).to(torch::kFloat64).contiguous();
              auto lat = python_to_tensor(lat_obj).to(torch::kFloat64).contiguous();
              if (!lon.sizes().equals(lat.sizes())) {
                  throw std::runtime_error("lon/lat shape mismatch");
              }
              std::vector<int64_t> out_shape;
              out_shape.reserve(static_cast<size_t>(lon.dim()) + 1);
              out_shape.push_back(4);
              for (const auto s : lon.sizes()) {
                  out_shape.push_back(static_cast<int64_t>(s));
              }
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::healpix_get_interp_weights_nested_cpu(nside, lon, lat);
              }
              out.first = out.first.reshape(out_shape);
              out.second = out.second.reshape(out_shape);
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("nside"), nb::arg("lon"), nb::arg("lat"));

    m.def("healpix_get_interp_val_ring_torch_cpu",
          [](int64_t nside, nb::object maps_obj, nb::object lon_obj, nb::object lat_obj) {
              auto maps = python_to_tensor(maps_obj).contiguous();
              auto lon = python_to_tensor(lon_obj).to(torch::kFloat64).contiguous();
              auto lat = python_to_tensor(lat_obj).to(torch::kFloat64).contiguous();
              if (!lon.sizes().equals(lat.sizes())) {
                  throw std::runtime_error("lon/lat shape mismatch");
              }
              torch::Tensor out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::healpix_get_interp_val_ring_cpu(nside, maps, lon, lat);
              }
              return tensor_to_python(out);
          },
          nb::arg("nside"), nb::arg("maps"), nb::arg("lon"), nb::arg("lat"));

    m.def("healpix_get_interp_val_nested_torch_cpu",
          [](int64_t nside, nb::object maps_obj, nb::object lon_obj, nb::object lat_obj) {
              auto maps = python_to_tensor(maps_obj).contiguous();
              auto lon = python_to_tensor(lon_obj).to(torch::kFloat64).contiguous();
              auto lat = python_to_tensor(lat_obj).to(torch::kFloat64).contiguous();
              if (!lon.sizes().equals(lat.sizes())) {
                  throw std::runtime_error("lon/lat shape mismatch");
              }
              torch::Tensor out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::healpix_get_interp_val_nested_cpu(nside, maps, lon, lat);
              }
              return tensor_to_python(out);
          },
          nb::arg("nside"), nb::arg("maps"), nb::arg("lon"), nb::arg("lat"));

    m.def("healpix_query_disc_torch_cpu",
          [](int64_t nside, nb::object vec_obj, double cos_lim, bool nest) {
              auto vec = python_to_tensor(vec_obj).to(torch::kFloat64).contiguous();
              torch::Tensor out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::healpix_query_disc_cpu(nside, vec, cos_lim, nest);
              }
              return tensor_to_python(out);
          },
          nb::arg("nside"), nb::arg("vec"), nb::arg("cos_lim"), nb::arg("nest") = false);


    m.def("healpix_ang2pix_ring_torch_cpu",
          [](int64_t nside, nb::object ra_obj, nb::object dec_obj) {
              auto ra = python_to_tensor(ra_obj).to(torch::kFloat64).contiguous();
              auto dec = python_to_tensor(dec_obj).to(torch::kFloat64).contiguous();
              if (!ra.sizes().equals(dec.sizes())) {
                  throw std::runtime_error("RA/Dec shape mismatch");
              }
              torch::Tensor out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::healpix_ang2pix_ring_cpu(nside, ra, dec);
              }
              return tensor_to_python(out);
          },
          nb::arg("nside"), nb::arg("ra"), nb::arg("dec"));


    m.def("healpix_ang2pix_nested_torch_cpu",
          [](int64_t nside, nb::object ra_obj, nb::object dec_obj) {
              auto ra = python_to_tensor(ra_obj).to(torch::kFloat64).contiguous();
              auto dec = python_to_tensor(dec_obj).to(torch::kFloat64).contiguous();
              if (!ra.sizes().equals(dec.sizes())) {
                  throw std::runtime_error("RA/Dec shape mismatch");
              }
              torch::Tensor out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::healpix_ang2pix_nested_cpu(nside, ra, dec);
              }
              return tensor_to_python(out);
          },
          nb::arg("nside"), nb::arg("ra"), nb::arg("dec"));


    m.def("healpix_pix2ang_ring_torch_cpu",
          [](int64_t nside, nb::object pix_obj) {
              auto pix = python_to_tensor(pix_obj).to(torch::kInt64).contiguous();
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::healpix_pix2ang_ring_cpu(nside, pix);
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("nside"), nb::arg("pix"));


    m.def("healpix_pix2ang_nested_torch_cpu",
          [](int64_t nside, nb::object pix_obj) {
              auto pix = python_to_tensor(pix_obj).to(torch::kInt64).contiguous();
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::healpix_pix2ang_nested_cpu(nside, pix);
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("nside"), nb::arg("pix"));

    m.def(
        "_sphere_planar_point_in_polygon_cpu",
        [](nb::object points_xy_obj, nb::object polygon_xy_obj, double eps) {
            auto points_xy = as_f64_contig(points_xy_obj);
            auto polygon_xy = as_f64_contig(polygon_xy_obj);
            torch::Tensor out;
            {
                nb::gil_scoped_release release;
                out = torchsky::sphere_planar_point_in_polygon_cpu(
                    points_xy, polygon_xy, eps
                );
            }
            return tensor_to_python(out);
        },
        nb::arg("points_xy"),
        nb::arg("polygon_xy"),
        nb::arg("eps") = 1e-15
    );

    m.def(
        "_sphere_arc_intersections_matrix_cpu",
        [](nb::object a0_obj,
           nb::object a1_obj,
           nb::object b0_obj,
           nb::object b1_obj,
           double tol_rad) {
            auto a0 = as_f64_contig(a0_obj);
            auto a1 = as_f64_contig(a1_obj);
            auto b0 = as_f64_contig(b0_obj);
            auto b1 = as_f64_contig(b1_obj);
            torch::Tensor out;
            {
                nb::gil_scoped_release release;
                out = torchsky::sphere_arc_intersections_matrix_cpu(
                    a0, a1, b0, b1, tol_rad
                );
            }
            return tensor_to_python(out);
        },
        nb::arg("a0"),
        nb::arg("a1"),
        nb::arg("b0"),
        nb::arg("b1"),
        nb::arg("tol_rad") = 1e-10
    );

    m.def("wcs_zenithal_project",
          [](nb::object xi_obj, nb::object eta_obj, std::string proj_code) {
              auto xi = as_f64_contig(xi_obj);
              auto eta = as_f64_contig(eta_obj);
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_zenithal_project(xi, eta, proj_code);
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("xi"), nb::arg("eta"), nb::arg("proj_code"));

    m.def("wcs_csc_x2s",
          [](nb::object xi_obj, nb::object eta_obj) {
              auto xi = as_f64_contig(xi_obj);
              auto eta = as_f64_contig(eta_obj);
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_csc_x2s(xi, eta);
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("xi"), nb::arg("eta"));

    m.def("wcs_csc_s2x",
          [](nb::object phi_obj, nb::object theta_obj) {
              auto phi = as_f64_contig(phi_obj);
              auto theta = as_f64_contig(theta_obj);
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_csc_s2x(phi, theta);
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("phi"), nb::arg("theta"));

    m.def("wcs_zpn_project",
          [](nb::object xi_obj, nb::object eta_obj, nb::object coeffs_obj) {
              auto xi = as_f64_contig(xi_obj);
              auto eta = as_f64_contig(eta_obj);
              auto coeffs = as_f64_contig(coeffs_obj);
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_zpn_project(xi, eta, coeffs);
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("xi"), nb::arg("eta"), nb::arg("coeffs"));

    m.def("wcs_zenithal_deproject",
          [](nb::object phi_obj, nb::object theta_obj, std::string proj_code) {
              auto phi = as_f64_contig(phi_obj);
              auto theta = as_f64_contig(theta_obj);
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_zenithal_deproject(phi, theta, proj_code);
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("phi"), nb::arg("theta"), nb::arg("proj_code"));

    m.def("wcs_cylindrical_project",
          [](nb::object xi_obj, nb::object eta_obj, std::string proj_code, double lambda_param) {
              auto xi = as_f64_contig(xi_obj);
              auto eta = as_f64_contig(eta_obj);
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_cylindrical_project(xi, eta, proj_code, lambda_param);
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("xi"), nb::arg("eta"), nb::arg("proj_code"), nb::arg("lambda_param") = 1.0);

    m.def("wcs_cylindrical_deproject",
          [](nb::object phi_obj, nb::object theta_obj, std::string proj_code, double lambda_param) {
              auto phi = as_f64_contig(phi_obj);
              auto theta = as_f64_contig(theta_obj);
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_cylindrical_deproject(phi, theta, proj_code, lambda_param);
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("phi"), nb::arg("theta"), nb::arg("proj_code"), nb::arg("lambda_param") = 1.0);

    m.def("wcs_center_pixel_to_world",
          [](nb::object x_obj,
             nb::object y_obj,
             double origin,
             double crpix0,
             double crpix1,
             double cd00,
             double cd01,
             double cd10,
             double cd11,
             std::string proj_code,
             double alpha0,
             double cea_eta_scale,
             double hpx_H,
             double hpx_K) {
              auto x = as_f64_contig(x_obj);
              auto y = as_f64_contig(y_obj);
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_center_pixel_to_world(
                      x, y, origin, crpix0, crpix1, cd00, cd01, cd10, cd11, proj_code, alpha0, cea_eta_scale, hpx_H, hpx_K
                  );
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("x"), nb::arg("y"), nb::arg("origin"), nb::arg("crpix0"), nb::arg("crpix1"),
          nb::arg("cd00"), nb::arg("cd01"), nb::arg("cd10"), nb::arg("cd11"),
          nb::arg("proj_code"), nb::arg("alpha0"), nb::arg("cea_eta_scale"), nb::arg("hpx_H"), nb::arg("hpx_K"));

    m.def("wcs_center_world_to_pixel",
          [](nb::object ra_obj,
             nb::object dec_obj,
             double crpix0,
             double crpix1,
             double cdi00,
             double cdi01,
             double cdi10,
             double cdi11,
             std::string proj_code,
             double alpha0,
             double cea_eta_scale,
             double origin,
             double hpx_H,
             double hpx_K) {
              auto ra = as_f64_contig(ra_obj);
              auto dec = as_f64_contig(dec_obj);
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_center_world_to_pixel(
                      ra, dec, crpix0, crpix1, cdi00, cdi01, cdi10, cdi11,
                      proj_code, alpha0, cea_eta_scale, origin, hpx_H, hpx_K
                  );
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("ra"), nb::arg("dec"), nb::arg("crpix0"), nb::arg("crpix1"),
          nb::arg("cdi00"), nb::arg("cdi01"), nb::arg("cdi10"), nb::arg("cdi11"),
          nb::arg("proj_code"), nb::arg("alpha0"), nb::arg("cea_eta_scale"), nb::arg("origin"), nb::arg("hpx_H"), nb::arg("hpx_K"));

    m.def("wcs_ait_project",
          [](nb::object xi_obj, nb::object eta_obj) {
              auto xi = as_f64_contig(xi_obj);
              auto eta = as_f64_contig(eta_obj);
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_ait_project(xi, eta);
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("xi"), nb::arg("eta"));

    m.def("wcs_ait_deproject",
          [](nb::object phi_obj, nb::object theta_obj) {
              auto phi = as_f64_contig(phi_obj);
              auto theta = as_f64_contig(theta_obj);
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_ait_deproject(phi, theta);
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("phi"), nb::arg("theta"));

    m.def("wcs_mol_project",
          [](nb::object xi_obj, nb::object eta_obj) {
              auto xi = as_f64_contig(xi_obj);
              auto eta = as_f64_contig(eta_obj);
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_mol_project(xi, eta);
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("xi"), nb::arg("eta"));

    m.def("wcs_mol_deproject",
          [](nb::object phi_obj, nb::object theta_obj) {
              auto phi = as_f64_contig(phi_obj);
              auto theta = as_f64_contig(theta_obj);
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_mol_deproject(phi, theta);
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("phi"), nb::arg("theta"));

    m.def("wcs_hpx_project",
          [](nb::object xi_obj, nb::object eta_obj, double H, double K) {
              auto xi = as_f64_contig(xi_obj);
              auto eta = as_f64_contig(eta_obj);
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_hpx_project(xi, eta, H, K);
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("xi"), nb::arg("eta"), nb::arg("H") = 4.0, nb::arg("K") = 3.0);

    m.def("wcs_hpx_deproject",
          [](nb::object phi_obj, nb::object theta_obj, double H, double K) {
              auto phi = as_f64_contig(phi_obj);
              auto theta = as_f64_contig(theta_obj);
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_hpx_deproject(phi, theta, H, K);
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("phi"), nb::arg("theta"), nb::arg("H") = 4.0, nb::arg("K") = 3.0);

    m.def("wcs_inverse_spherical_rotation_pole",
          [](nb::object ra_obj,
             nb::object dec_obj,
             double phi_p_rad,
             double east_x,
             double east_y,
             double north_x,
             double north_y,
             double north_z,
             double radial_x,
             double radial_y,
             double radial_z) {
              auto ra = python_to_tensor(ra_obj);
              auto dec = python_to_tensor(dec_obj);
              if (!(ra.device().is_cpu() && dec.device().is_cpu())) {
                  throw std::runtime_error("wcs_inverse_spherical_rotation_pole expects CPU tensors");
              }
              if (!(ra.scalar_type() == torch::kFloat64 && dec.scalar_type() == torch::kFloat64)) {
                  throw std::runtime_error("wcs_inverse_spherical_rotation_pole expects float64 inputs");
              }
              if (!ra.is_contiguous()) ra = ra.contiguous();
              if (!dec.is_contiguous()) dec = dec.contiguous();
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_inverse_spherical_rotation_pole(
                      ra, dec, phi_p_rad,
                      east_x, east_y,
                      north_x, north_y, north_z,
                      radial_x, radial_y, radial_z
                  );
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("ra"), nb::arg("dec"), nb::arg("phi_p_rad"),
          nb::arg("east_x"), nb::arg("east_y"),
          nb::arg("north_x"), nb::arg("north_y"), nb::arg("north_z"),
          nb::arg("radial_x"), nb::arg("radial_y"), nb::arg("radial_z"));

    m.def("wcs_spherical_rotation_pole",
          [](nb::object phi_obj,
             nb::object theta_obj,
             double phi_p_rad,
             double east_x,
             double east_y,
             double north_x,
             double north_y,
             double north_z,
             double radial_x,
             double radial_y,
             double radial_z) {
              auto phi = python_to_tensor(phi_obj);
              auto theta = python_to_tensor(theta_obj);
              if (!(phi.device().is_cpu() && theta.device().is_cpu())) {
                  throw std::runtime_error("wcs_spherical_rotation_pole expects CPU tensors");
              }
              if (!(phi.scalar_type() == torch::kFloat64 && theta.scalar_type() == torch::kFloat64)) {
                  throw std::runtime_error("wcs_spherical_rotation_pole expects float64 inputs");
              }
              if (!phi.is_contiguous()) phi = phi.contiguous();
              if (!theta.is_contiguous()) theta = theta.contiguous();
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_spherical_rotation_pole(
                      phi, theta, phi_p_rad,
                      east_x, east_y,
                      north_x, north_y, north_z,
                      radial_x, radial_y, radial_z
                  );
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("phi"), nb::arg("theta"), nb::arg("phi_p_rad"),
          nb::arg("east_x"), nb::arg("east_y"),
          nb::arg("north_x"), nb::arg("north_y"), nb::arg("north_z"),
          nb::arg("radial_x"), nb::arg("radial_y"), nb::arg("radial_z"));

    m.def("wcs_tan_intermediate_from_radec",
          [](nb::object ra_obj,
             nb::object dec_obj,
             double east_x,
             double east_y,
             double north_x,
             double north_y,
             double north_z,
             double radial_x,
             double radial_y,
             double radial_z) {
              auto ra = python_to_tensor(ra_obj);
              auto dec = python_to_tensor(dec_obj);
              if (!(ra.device().is_cpu() && dec.device().is_cpu())) {
                  throw std::runtime_error("wcs_tan_intermediate_from_radec expects CPU tensors");
              }
              if (!(ra.scalar_type() == torch::kFloat64 && dec.scalar_type() == torch::kFloat64)) {
                  throw std::runtime_error("wcs_tan_intermediate_from_radec expects float64 inputs");
              }
              if (!ra.is_contiguous()) ra = ra.contiguous();
              if (!dec.is_contiguous()) dec = dec.contiguous();
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_tan_intermediate_from_radec(
                      ra, dec,
                      east_x, east_y,
                      north_x, north_y, north_z,
                      radial_x, radial_y, radial_z
                  );
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("ra"), nb::arg("dec"),
          nb::arg("east_x"), nb::arg("east_y"),
          nb::arg("north_x"), nb::arg("north_y"), nb::arg("north_z"),
          nb::arg("radial_x"), nb::arg("radial_y"), nb::arg("radial_z"));

    m.def("wcs_tpv_invert",
          [](nb::object xi_obj,
             nb::object eta_obj,
             nb::object idx1_obj,
             nb::object c1_obj,
             nb::object idx2_obj,
             nb::object c2_obj,
             int64_t max_iter,
             double tol) {
              auto xi = python_to_tensor(xi_obj);
              auto eta = python_to_tensor(eta_obj);
              auto idx1 = python_to_tensor(idx1_obj);
              auto c1 = python_to_tensor(c1_obj);
              auto idx2 = python_to_tensor(idx2_obj);
              auto c2 = python_to_tensor(c2_obj);
              if (!(xi.device().is_cpu() && eta.device().is_cpu() &&
                    idx1.device().is_cpu() && idx2.device().is_cpu() &&
                    c1.device().is_cpu() && c2.device().is_cpu())) {
                  throw std::runtime_error("wcs_tpv_invert expects CPU tensors");
              }
              if (!(xi.scalar_type() == torch::kFloat64 && eta.scalar_type() == torch::kFloat64 &&
                    c1.scalar_type() == torch::kFloat64 && c2.scalar_type() == torch::kFloat64 &&
                    idx1.scalar_type() == torch::kInt64 && idx2.scalar_type() == torch::kInt64)) {
                  throw std::runtime_error("wcs_tpv_invert expects float64 inputs and int64 indices");
              }
              if (!xi.is_contiguous()) xi = xi.contiguous();
              if (!eta.is_contiguous()) eta = eta.contiguous();
              if (!idx1.is_contiguous()) idx1 = idx1.contiguous();
              if (!c1.is_contiguous()) c1 = c1.contiguous();
              if (!idx2.is_contiguous()) idx2 = idx2.contiguous();
              if (!c2.is_contiguous()) c2 = c2.contiguous();
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_tpv_invert(xi, eta, idx1, c1, idx2, c2, max_iter, tol);
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("xi"), nb::arg("eta"), nb::arg("idx1"), nb::arg("c1"), nb::arg("idx2"), nb::arg("c2"),
          nb::arg("max_iter") = 10, nb::arg("tol") = 1e-11);

    m.def("wcs_tpv_distort",
          [](nb::object u_obj,
             nb::object v_obj,
             nb::object idx1_obj,
             nb::object c1_obj,
             nb::object idx2_obj,
             nb::object c2_obj) {
              auto u = python_to_tensor(u_obj);
              auto v = python_to_tensor(v_obj);
              auto idx1 = python_to_tensor(idx1_obj);
              auto c1 = python_to_tensor(c1_obj);
              auto idx2 = python_to_tensor(idx2_obj);
              auto c2 = python_to_tensor(c2_obj);
              if (!(u.device().is_cpu() && v.device().is_cpu() &&
                    idx1.device().is_cpu() && idx2.device().is_cpu() &&
                    c1.device().is_cpu() && c2.device().is_cpu())) {
                  throw std::runtime_error("wcs_tpv_distort expects CPU tensors");
              }
              if (!(u.scalar_type() == torch::kFloat64 && v.scalar_type() == torch::kFloat64 &&
                    c1.scalar_type() == torch::kFloat64 && c2.scalar_type() == torch::kFloat64 &&
                    idx1.scalar_type() == torch::kInt64 && idx2.scalar_type() == torch::kInt64)) {
                  throw std::runtime_error("wcs_tpv_distort expects float64 inputs and int64 indices");
              }
              if (!u.is_contiguous()) u = u.contiguous();
              if (!v.is_contiguous()) v = v.contiguous();
              if (!idx1.is_contiguous()) idx1 = idx1.contiguous();
              if (!c1.is_contiguous()) c1 = c1.contiguous();
              if (!idx2.is_contiguous()) idx2 = idx2.contiguous();
              if (!c2.is_contiguous()) c2 = c2.contiguous();
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_tpv_distort(u, v, idx1, c1, idx2, c2);
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("u"), nb::arg("v"), nb::arg("idx1"), nb::arg("c1"), nb::arg("idx2"), nb::arg("c2"));

    m.def("wcs_sip_quadratic_distort",
          [](nb::object u_obj,
             nb::object v_obj,
             double a20,
             double a11,
             double a02,
             double b20,
             double b11,
             double b02) {
              auto u = python_to_tensor(u_obj);
              auto v = python_to_tensor(v_obj);
              if (!(u.device().is_cpu() && v.device().is_cpu())) {
                  throw std::runtime_error("wcs_sip_quadratic_distort expects CPU tensors");
              }
              if (!(u.scalar_type() == torch::kFloat64 && v.scalar_type() == torch::kFloat64)) {
                  throw std::runtime_error("wcs_sip_quadratic_distort expects float64 inputs");
              }
              if (!u.is_contiguous()) u = u.contiguous();
              if (!v.is_contiguous()) v = v.contiguous();
              std::pair<torch::Tensor, torch::Tensor> out;
              {
                  nb::gil_scoped_release release;
                  out = torchsky::wcs_sip_quadratic_distort(
                      u, v, a20, a11, a02, b20, b11, b02
                  );
              }
              return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
          },
          nb::arg("u"), nb::arg("v"),
          nb::arg("a20"), nb::arg("a11"), nb::arg("a02"),
          nb::arg("b20"), nb::arg("b11"), nb::arg("b02"));


    m.def(
        "_healpix_scalar_alm2map_direct_cpu",
        [](nb::object alm_obj,
           nb::object cos_theta_obj,
           nb::object phi_obj,
           int64_t lmax,
           int64_t mmax) {
            torch::Tensor alm = python_to_tensor(alm_obj).contiguous();
            torch::Tensor cos_theta = python_to_tensor(cos_theta_obj).contiguous();
            torch::Tensor phi = python_to_tensor(phi_obj).contiguous();
            torch::Tensor out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_scalar_alm2map_direct_cpu(alm, cos_theta, phi, lmax, mmax);
            }
            return tensor_to_python(out);
        },
        nb::arg("alm"),
        nb::arg("cos_theta"),
        nb::arg("phi"),
        nb::arg("lmax"),
        nb::arg("mmax"));

    m.def(
        "_healpix_ring_fourier_modes_cpu",
        [](nb::object rows_obj,
           nb::object starts_obj,
           nb::object lengths_obj,
           int64_t mmax) {
            torch::Tensor rows = python_to_tensor(rows_obj).contiguous();
            torch::Tensor starts = python_to_tensor(starts_obj).contiguous();
            torch::Tensor lengths = python_to_tensor(lengths_obj).contiguous();
            torch::Tensor out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_ring_fourier_modes_cpu(rows, starts, lengths, mmax);
            }
            return tensor_to_python(out);
        },
        nb::arg("rows"),
        nb::arg("starts"),
        nb::arg("lengths"),
        nb::arg("mmax"));

    m.def(
        "_healpix_ring_fourier_modes_spin_conj_cpu",
        [](nb::object row_obj,
           nb::object starts_obj,
           nb::object lengths_obj,
           int64_t mmax) {
            torch::Tensor row = python_to_tensor(row_obj).contiguous();
            torch::Tensor starts = python_to_tensor(starts_obj).contiguous();
            torch::Tensor lengths = python_to_tensor(lengths_obj).contiguous();
            torch::Tensor out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_ring_fourier_modes_spin_conj_cpu(row, starts, lengths, mmax);
            }
            return tensor_to_python(out);
        },
        nb::arg("row"),
        nb::arg("starts"),
        nb::arg("lengths"),
        nb::arg("mmax"));

    m.def(
        "_healpix_ring_fourier_synthesis_cpu",
        [](nb::object s_modes_obj,
           nb::object starts_obj,
           nb::object lengths_obj) {
            torch::Tensor s_modes = python_to_tensor(s_modes_obj).contiguous();
            torch::Tensor starts = python_to_tensor(starts_obj).contiguous();
            torch::Tensor lengths = python_to_tensor(lengths_obj).contiguous();
            torch::Tensor out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_ring_fourier_synthesis_cpu(s_modes, starts, lengths);
            }
            return tensor_to_python(out);
        },
        nb::arg("s_modes"),
        nb::arg("starts"),
        nb::arg("lengths"));

    m.def(
        "_healpix_spin_ring_finalize_cpu",
        [](nb::object s_plus_obj,
           nb::object s_minus_obj,
           nb::object starts_obj,
           nb::object lengths_obj) {
            torch::Tensor s_plus = python_to_tensor(s_plus_obj).contiguous();
            torch::Tensor s_minus = python_to_tensor(s_minus_obj).contiguous();
            torch::Tensor starts = python_to_tensor(starts_obj).contiguous();
            torch::Tensor lengths = python_to_tensor(lengths_obj).contiguous();
            torch::Tensor out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_spin_ring_finalize_cpu(s_plus, s_minus, starts, lengths);
            }
            return tensor_to_python(out);
        },
        nb::arg("s_plus"),
        nb::arg("s_minus"),
        nb::arg("starts"),
        nb::arg("lengths"));

    m.def(
        "_healpix_spin_map2alm_ring_concat_cpu",
        [](nb::object p_plus_obj,
           nb::object starts_obj,
           nb::object lengths_obj,
           nb::object phase0_neg_obj,
           nb::object y_plus_obj,
           nb::object y_minus_obj,
           int64_t lmax,
           int64_t mmax,
           double pix_w) {
            torch::Tensor p_plus = python_to_tensor(p_plus_obj).contiguous();
            torch::Tensor starts = python_to_tensor(starts_obj).contiguous();
            torch::Tensor lengths = python_to_tensor(lengths_obj).contiguous();
            torch::Tensor phase0_neg = python_to_tensor(phase0_neg_obj).contiguous();
            torch::Tensor y_plus = python_to_tensor(y_plus_obj).contiguous();
            torch::Tensor y_minus = python_to_tensor(y_minus_obj).contiguous();
            std::pair<torch::Tensor, torch::Tensor> out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_spin_map2alm_ring_concat_cpu(
                    p_plus, starts, lengths, phase0_neg, y_plus, y_minus, lmax, mmax, pix_w
                );
            }
            return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
        },
        nb::arg("p_plus"),
        nb::arg("starts"),
        nb::arg("lengths"),
        nb::arg("phase0_neg"),
        nb::arg("y_plus"),
        nb::arg("y_minus"),
        nb::arg("lmax"),
        nb::arg("mmax"),
        nb::arg("pix_w"));

    // Experimental spin-2 helper: not part of public stable API.
    // Kept for local profiling only while the interface/runtime safety is audited.
    m.def(
        "_healpix_spin_map2alm_from_basis_cpu",
        [](nb::object q_obj,
           nb::object u_obj,
           nb::object ycp_t_obj,
           nb::object ycm_t_obj,
           double pix_w) {
            torch::Tensor q = python_to_tensor(q_obj).contiguous();
            torch::Tensor u = python_to_tensor(u_obj).contiguous();
            torch::Tensor ycp_t = python_to_tensor(ycp_t_obj).contiguous();
            torch::Tensor ycm_t = python_to_tensor(ycm_t_obj).contiguous();
            std::pair<torch::Tensor, torch::Tensor> out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_spin_map2alm_from_basis_cpu(q, u, ycp_t, ycm_t, pix_w);
            }
            return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
        },
        nb::arg("q"),
        nb::arg("u"),
        nb::arg("ycp_t"),
        nb::arg("ycm_t"),
        nb::arg("pix_w"));

    m.def(
        "_healpix_spin_alm2map_from_basis_cpu",
        [](nb::object coeff_plus_obj,
           nb::object coeff_minus_obj,
           nb::object y_plus_obj,
           nb::object y_minus_obj,
           nb::object y_plus_m0_obj,
           nb::object y_minus_m0_obj) {
            torch::Tensor coeff_plus = python_to_tensor(coeff_plus_obj).contiguous();
            torch::Tensor coeff_minus = python_to_tensor(coeff_minus_obj).contiguous();
            torch::Tensor y_plus = python_to_tensor(y_plus_obj).contiguous();
            torch::Tensor y_minus = python_to_tensor(y_minus_obj).contiguous();
            torch::Tensor y_plus_m0 = python_to_tensor(y_plus_m0_obj).contiguous();
            torch::Tensor y_minus_m0 = python_to_tensor(y_minus_m0_obj).contiguous();
            torch::Tensor out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_spin_alm2map_from_basis_cpu(
                    coeff_plus,
                    coeff_minus,
                    y_plus,
                    y_minus,
                    y_plus_m0,
                    y_minus_m0
                );
            }
            return tensor_to_python(out);
        },
        nb::arg("coeff_plus"),
        nb::arg("coeff_minus"),
        nb::arg("y_plus"),
        nb::arg("y_minus"),
        nb::arg("y_plus_m0"),
        nb::arg("y_minus_m0"));

    m.def(
        "_healpix_spin_interpolate_concat_cpu",
        [](nb::object coeff_plus_obj,
           nb::object coeff_minus_obj,
           nb::object y_plus_obj,
           nb::object y_minus_obj,
           int64_t lmax,
           int64_t mmax) {
            torch::Tensor coeff_plus = python_to_tensor(coeff_plus_obj).contiguous();
            torch::Tensor coeff_minus = python_to_tensor(coeff_minus_obj).contiguous();
            torch::Tensor y_plus = python_to_tensor(y_plus_obj).contiguous();
            torch::Tensor y_minus = python_to_tensor(y_minus_obj).contiguous();
            std::pair<torch::Tensor, torch::Tensor> out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_spin_interpolate_concat_cpu(
                    coeff_plus, coeff_minus, y_plus, y_minus, lmax, mmax
                );
            }
            return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
        },
        nb::arg("coeff_plus"),
        nb::arg("coeff_minus"),
        nb::arg("y_plus"),
        nb::arg("y_minus"),
        nb::arg("lmax"),
        nb::arg("mmax"));

    m.def(
        "_healpix_spin_integrate_concat_cpu",
        [](nb::object s_plus_obj,
           nb::object s_minus_obj,
           nb::object y_plus_obj,
           nb::object y_minus_obj,
           int64_t lmax,
           int64_t mmax,
           double pix_w) {
            torch::Tensor s_plus = python_to_tensor(s_plus_obj).contiguous();
            torch::Tensor s_minus = python_to_tensor(s_minus_obj).contiguous();
            torch::Tensor y_plus = python_to_tensor(y_plus_obj).contiguous();
            torch::Tensor y_minus = python_to_tensor(y_minus_obj).contiguous();
            std::pair<torch::Tensor, torch::Tensor> out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_spin_integrate_concat_cpu(
                    s_plus, s_minus, y_plus, y_minus, lmax, mmax, pix_w
                );
            }
            return nb::make_tuple(tensor_to_python(out.first), tensor_to_python(out.second));
        },
        nb::arg("s_plus"),
        nb::arg("s_minus"),
        nb::arg("y_plus"),
        nb::arg("y_minus"),
        nb::arg("lmax"),
        nb::arg("mmax"),
        nb::arg("pix_w"));



    // Experimental recurrence kernels
    m.def(
        "_healpix_spin_integrate_recurrence_cpu",
        [](nb::object s_plus_obj,
           nb::object s_minus_obj,
           nb::object theta_obj,
           nb::object weight_obj,
           int64_t lmax,
           int64_t mmax,
           int64_t spin) {
            torch::Tensor s_plus = python_to_tensor(s_plus_obj).contiguous();
            torch::Tensor s_minus = python_to_tensor(s_minus_obj).contiguous();
            torch::Tensor theta = python_to_tensor(theta_obj).contiguous();
            torch::Tensor weight = python_to_tensor(weight_obj).contiguous();
            torch::Tensor out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_spin_integrate_recurrence_cpu(
                    s_plus, s_minus, theta, weight, lmax, mmax, spin
                );
            }
            return tensor_to_python(out);
        },
        nb::arg("s_plus"),
        nb::arg("s_minus"),
        nb::arg("theta"),
        nb::arg("weight"),
        nb::arg("lmax"),
        nb::arg("mmax"),
        nb::arg("spin"));

    m.def(
        "_healpix_spin_interpolate_recurrence_cpu",
        [](nb::object alms_obj,
           nb::object theta_obj,
           int64_t lmax,
           int64_t mmax,
           int64_t spin) {
            torch::Tensor alms = python_to_tensor(alms_obj).contiguous();
            torch::Tensor theta = python_to_tensor(theta_obj).contiguous();
            torch::Tensor out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_spin_interpolate_recurrence_cpu(
                    alms, theta, lmax, mmax, spin
                );
            }
            return tensor_to_python(out);
        },
        nb::arg("alms"),
        nb::arg("theta"),
        nb::arg("lmax"),
        nb::arg("mmax"),
        nb::arg("spin"));

    m.def(
        "_healpix_spin_map2alm_ring_recurrence_cpu",
        [](nb::object maps_obj,
           nb::object starts_obj,
           nb::object lengths_obj,
           nb::object theta_obj,
           nb::object phase0_neg_obj,
           int64_t lmax,
           int64_t mmax,
           int64_t spin,
           double pix_w) {
            torch::Tensor maps = python_to_tensor(maps_obj);
            torch::Tensor starts = python_to_tensor(starts_obj);
            torch::Tensor lengths = python_to_tensor(lengths_obj);
            torch::Tensor theta = python_to_tensor(theta_obj);
            torch::Tensor phase0_neg = python_to_tensor(phase0_neg_obj);
            torch::Tensor out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_spin_map2alm_ring_recurrence_cpu(
                    maps, starts, lengths, theta, phase0_neg, lmax, mmax, spin, pix_w
                );
            }
            return tensor_to_python(out);
        },
        nb::arg("maps"),
        nb::arg("starts"),
        nb::arg("lengths"),
        nb::arg("theta"),
        nb::arg("phase0_neg"),
        nb::arg("lmax"),
        nb::arg("mmax"),
        nb::arg("spin"),
        nb::arg("pix_w"));

    m.def(
        "_healpix_spin_alm2map_ring_recurrence_cpu",
        [](nb::object alm_plus_obj,
           nb::object starts_obj,
           nb::object lengths_obj,
           nb::object theta_obj,
           nb::object phase0_pos_obj,
           int64_t lmax,
           int64_t mmax,
           int64_t spin,
           nb::object alm_minus_obj) {
            torch::Tensor alm_plus = python_to_tensor(alm_plus_obj);
            torch::Tensor starts = python_to_tensor(starts_obj);
            torch::Tensor lengths = python_to_tensor(lengths_obj);
            torch::Tensor theta = python_to_tensor(theta_obj);
            torch::Tensor phase0_pos = python_to_tensor(phase0_pos_obj);
            torch::Tensor alm_minus = python_to_tensor(alm_minus_obj);
            torch::Tensor out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_spin_alm2map_ring_recurrence_cpu(
                    alm_plus, starts, lengths, theta, phase0_pos, lmax, mmax, spin, alm_minus
                );
            }
            return tensor_to_python(out);
        },
        nb::arg("alm_plus"),
        nb::arg("starts"),
        nb::arg("lengths"),
        nb::arg("theta"),
        nb::arg("phase0_pos"),
        nb::arg("lmax"),
        nb::arg("mmax"),
        nb::arg("spin"),
        nb::arg("alm_minus"));

    // Dedicated scalar (spin=0) recurrence kernels
    m.def(
        "_healpix_scalar_integrate_recurrence_cpu",
        [](nb::object s_modes_obj,
           nb::object theta_obj,
           nb::object weight_obj,
           int64_t lmax,
           int64_t mmax) {
            torch::Tensor s_modes = python_to_tensor(s_modes_obj).contiguous();
            torch::Tensor theta = python_to_tensor(theta_obj).contiguous();
            torch::Tensor weight = python_to_tensor(weight_obj).contiguous();
            torch::Tensor out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_scalar_integrate_recurrence_cpu(
                    s_modes, theta, weight, lmax, mmax
                );
            }
            return tensor_to_python(out);
        },
        nb::arg("s_modes"),
        nb::arg("theta"),
        nb::arg("weight"),
        nb::arg("lmax"),
        nb::arg("mmax"));

    m.def(
        "_healpix_scalar_interpolate_recurrence_cpu",
        [](nb::object alm_obj,
           nb::object theta_obj,
           int64_t lmax,
           int64_t mmax) {
            torch::Tensor alm = python_to_tensor(alm_obj).contiguous();
            torch::Tensor theta = python_to_tensor(theta_obj).contiguous();
            torch::Tensor out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_scalar_interpolate_recurrence_cpu(
                    alm, theta, lmax, mmax
                );
            }
            return tensor_to_python(out);
        },
        nb::arg("alm"),
        nb::arg("theta"),
        nb::arg("lmax"),
        nb::arg("mmax"));

    m.def(
        "_sparse_get_values_blocks_cpu",
        [](nb::object query_obj,
           nb::object block_index_obj,
           nb::object block_values_obj,
           nb::object fill_obj) {
            torch::Tensor query = python_to_tensor(query_obj);
            torch::Tensor block_index = python_to_tensor(block_index_obj);
            torch::Tensor block_values = python_to_tensor(block_values_obj);
            torch::Tensor fill = python_to_tensor(fill_obj);
            torch::Tensor out;
            {
                nb::gil_scoped_release release;
                out = torchsky::sparse_get_values_blocks_cpu(
                    query, block_index, block_values, fill
                );
            }
            return tensor_to_python(out);
        },
        nb::arg("query"),
        nb::arg("block_index"),
        nb::arg("block_values"),
        nb::arg("fill"));

    m.def("echo_tensor", [](nb::object obj) {
        return obj;
    });

}

NB_MODULE(_cpp_ext, m) {
    const nb::str runtime_version_object(
        nb::module_::import_("torch").attr("__version__")
    );
    const std::string runtime_version(runtime_version_object.c_str());
    const std::string required_abi = TORCHSKY_TORCH_ABI;
    const bool matching_abi =
        runtime_version.compare(0, required_abi.size(), required_abi) == 0
        && (runtime_version.size() == required_abi.size()
            || runtime_version[required_abi.size()] == '.'
            || runtime_version[required_abi.size()] == '+');
    if (!matching_abi) {
        PyErr_Format(
            PyExc_ImportError,
            "torchsky was built for PyTorch %s.x but found PyTorch %s",
            required_abi.c_str(),
            runtime_version.c_str()
        );
        throw nb::python_error();
    }
    bind_wcs(m);
}
