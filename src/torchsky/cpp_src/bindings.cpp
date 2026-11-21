// Python module for torchsky's torch-free native core. Kernels see raw
// buffers (DLPack, zero copy); operator schemas, fake tensors and autograd are
// registered from Python (`torchsky/_native_ops.py`) with torch.library, so the
// extension does not link libtorch and works with any supported PyTorch.
#include <nanobind/nanobind.h>

#include "torchsky/crossmatch.h"
#include "torchsky/ishioka_kernels.h"

namespace nb = nanobind;

namespace torchsky {
void register_fixed(nb::module_& m);
void register_spectral(nb::module_& m);
void register_sphere_ranges(nb::module_& m);
void register_drizzle(nb::module_& m);
void register_wcs_tpv(nb::module_& m);
}  // namespace torchsky

NB_MODULE(_cpp_ext, m) {
    m.attr("NATIVE_ABI") = 2;  // torch-free core; bumped when kernel signatures change
    m.def("selected_cpu_isa", []() { return torchsky::selected_cpu_isa(); });
    torchsky::xmatch::register_crossmatch(m);
    torchsky::register_fixed(m);
    torchsky::register_spectral(m);
    torchsky::register_sphere_ranges(m);
    torchsky::register_drizzle(m);
    torchsky::register_wcs_tpv(m);
}
