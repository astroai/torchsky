#pragma once

// Torch-free zone-index crossmatch kernels (see crossmatch.cpp).

#include <nanobind/nanobind.h>

namespace torchsky::xmatch {

void register_crossmatch(nanobind::module_& m);

}  // namespace torchsky::xmatch
