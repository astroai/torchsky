#pragma once

// Kernels written against libtorch's spelling (torch::Tensor, at::parallel_for,
// c10::complex, TORCH_CHECK) compile against the torch-free core instead: inside
// namespace torchsky these names resolve to torchsky::native. Nothing here
// includes or links libtorch, so the extension works with any PyTorch version.

#include "torchsky/native.h"

namespace torchsky {
namespace torch = ::torchsky::native;
namespace at = ::torchsky::native;
namespace c10 {
template <typename T>
using complex = std::complex<T>;
}  // namespace c10
}  // namespace torchsky

#define TORCH_CHECK(cond, ...) TS_CHECK(cond, __VA_ARGS__)
