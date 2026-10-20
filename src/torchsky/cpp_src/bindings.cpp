#include <nanobind/nanobind.h>

#include <Python.h>
#include <string>

#include "torchsky/fixed_operators.h"
#include "torchsky/healpix.h"
#include "torchsky/ishioka_kernels.h"
#include "torchsky/wcs_operators.h"

namespace nb = nanobind;

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

    m.def("selected_cpu_isa", []() { return torchsky::selected_cpu_isa(); });
}
