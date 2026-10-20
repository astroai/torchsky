#include <nanobind/nanobind.h>

#include <Python.h>
#include <string>

#include "torch_compat.h"
#include "torchsky/fixed_operators.h"
#include "torchsky/healpix.h"
#include "torchsky/ishioka_kernels.h"
#include "torchsky/wcs_operators.h"

namespace nb = nanobind;

namespace {

void bind_deferred_legacy_kernels(nb::module_& m) {
    m.def(
        "healpix_query_disc_torch_cpu",
        [](int64_t nside, nb::object vec_obj, double cos_lim, bool nest) {
            auto vec = python_to_tensor(vec_obj).to(torch::kFloat64).contiguous();
            torch::Tensor out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_query_disc_cpu(nside, vec, cos_lim, nest);
            }
            return tensor_to_python(out);
        },
        nb::arg("nside"),
        nb::arg("vec"),
        nb::arg("cos_lim"),
        nb::arg("nest") = false
    );

    m.def(
        "_healpix_spin_map2alm_from_basis_cpu",
        [](nb::object q_obj,
           nb::object u_obj,
           nb::object ycp_t_obj,
           nb::object ycm_t_obj,
           double pixel_weight) {
            auto q = python_to_tensor(q_obj).contiguous();
            auto u = python_to_tensor(u_obj).contiguous();
            auto ycp_t = python_to_tensor(ycp_t_obj).contiguous();
            auto ycm_t = python_to_tensor(ycm_t_obj).contiguous();
            std::pair<torch::Tensor, torch::Tensor> out;
            {
                nb::gil_scoped_release release;
                out = torchsky::healpix_spin_map2alm_from_basis_cpu(
                    q, u, ycp_t, ycm_t, pixel_weight
                );
            }
            return nb::make_tuple(
                tensor_to_python(out.first), tensor_to_python(out.second)
            );
        },
        nb::arg("q"),
        nb::arg("u"),
        nb::arg("ycp_t"),
        nb::arg("ycm_t"),
        nb::arg("pix_w")
    );

    m.def(
        "_healpix_spin_alm2map_from_basis_cpu",
        [](nb::object coeff_plus_obj,
           nb::object coeff_minus_obj,
           nb::object y_plus_obj,
           nb::object y_minus_obj,
           nb::object y_plus_m0_obj,
           nb::object y_minus_m0_obj) {
            auto coeff_plus = python_to_tensor(coeff_plus_obj).contiguous();
            auto coeff_minus = python_to_tensor(coeff_minus_obj).contiguous();
            auto y_plus = python_to_tensor(y_plus_obj).contiguous();
            auto y_minus = python_to_tensor(y_minus_obj).contiguous();
            auto y_plus_m0 = python_to_tensor(y_plus_m0_obj).contiguous();
            auto y_minus_m0 = python_to_tensor(y_minus_m0_obj).contiguous();
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
        nb::arg("y_minus_m0")
    );
}

}

NB_MODULE(_cpp_ext, m) {
    torchsky::ensure_fixed_operator_schemas_linked();
    torchsky::ensure_fixed_cpu_operators_linked();
    torchsky::ensure_wcs_operator_schemas_linked();
    torchsky::ensure_wcs_cpu_operators_linked();
    torchsky::ensure_spectral_operator_schemas_linked();
    torchsky::ensure_spectral_cpu_operators_linked();

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

    bind_deferred_legacy_kernels(m);
    m.def("selected_cpu_isa", []() { return torchsky::selected_cpu_isa(); });
}
