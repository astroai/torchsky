// nanobind boundary for the torch-free core.
//
// Tensor arguments arrive as any CPU, C-contiguous DLPack array (torch
// tensors pass through zero-copy) and are viewed as native::Tensor without a
// copy. Results go back as torch tensors that own the native storage (DLPack,
// zero copy). A result that aliases an input buffer is copied first, so it
// never outlives the caller's tensor.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>

#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "torchsky/native.h"

namespace torchsky::native {

namespace nb = nanobind;

using Array = nb::ndarray<nb::device::cpu, nb::c_contig>;

inline ScalarType dtype_from(const nb::dlpack::dtype& d) {
    using code = nb::dlpack::dtype_code;
    const auto c = static_cast<code>(d.code);
    TS_CHECK(d.lanes == 1, "vector dtypes are not supported");
    if (c == code::Bool) return ScalarType::Bool;
    if (c == code::Float && d.bits == 64) return ScalarType::Double;
    if (c == code::Float && d.bits == 32) return ScalarType::Float;
    if (c == code::Float && d.bits == 16) return ScalarType::Half;
    if (c == code::Bfloat && d.bits == 16) return ScalarType::BFloat16;
    if (c == code::Int && d.bits == 64) return ScalarType::Long;
    if (c == code::Int && d.bits == 32) return ScalarType::Int;
    if (c == code::Int && d.bits == 16) return ScalarType::Short;
    if (c == code::UInt && d.bits == 8) return ScalarType::Byte;
    if (c == code::Complex && d.bits == 128) return ScalarType::ComplexDouble;
    if (c == code::Complex && d.bits == 64) return ScalarType::ComplexFloat;
    fail("unsupported DLPack dtype (code ", static_cast<int>(d.code), ", bits ", static_cast<int>(d.bits), ")");
}

inline nb::dlpack::dtype dtype_to(ScalarType t) {
    switch (t) {
        case ScalarType::Bool: return nb::dtype<bool>();
        case ScalarType::Byte: return nb::dtype<uint8_t>();
        case ScalarType::Short: return nb::dtype<int16_t>();
        case ScalarType::Int: return nb::dtype<int32_t>();
        case ScalarType::Long: return nb::dtype<int64_t>();
        case ScalarType::Float: return nb::dtype<float>();
        case ScalarType::Double: return nb::dtype<double>();
        case ScalarType::ComplexFloat: return nb::dtype<std::complex<float>>();
        case ScalarType::ComplexDouble: return nb::dtype<std::complex<double>>();
        case ScalarType::Half: {
            nb::dlpack::dtype d{};
            d.code = static_cast<uint8_t>(nb::dlpack::dtype_code::Float);
            d.bits = 16;
            d.lanes = 1;
            return d;
        }
        case ScalarType::BFloat16: {
            nb::dlpack::dtype d{};
            d.code = static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat);
            d.bits = 16;
            d.lanes = 1;
            return d;
        }
    }
    fail("unsupported dtype");
}

// A borrowed view of a caller's buffer (valid for the duration of the call).
inline Tensor from_array(const Array& a) {
    std::vector<int64_t> shape(a.ndim());
    for (size_t i = 0; i < a.ndim(); ++i) shape[i] = static_cast<int64_t>(a.shape(i));
    return Tensor::borrow(const_cast<void*>(a.data()), shape, dtype_from(a.dtype()));
}

// An owning torch tensor (zero copy through DLPack; nanobind calls
// torch.utils.dlpack.from_dlpack). Must be called with the GIL held.
inline nb::object to_python(Tensor t) {
    if (t.borrowed()) t = t.clone();
    auto* keep = new std::shared_ptr<void>(t.storage());
    nb::capsule owner(keep, [](void* p) noexcept { delete static_cast<std::shared_ptr<void>*>(p); });
    const std::vector<int64_t> dims = t.sizes().vec();
    std::vector<size_t> shape(dims.begin(), dims.end());
    nb::ndarray<nb::pytorch, nb::device::cpu> out(
        t.raw_data(), shape.size(), shape.data(), owner, nullptr, dtype_to(t.dtype())
    );
    return nb::cast(out, nb::rv_policy::reference);
}

// ------------------------------------------------------------- generic wrapper

template <typename A>
struct arg_traits {
    using py = std::decay_t<A>;
    static py convert(py v) { return v; }
};
template <>
struct arg_traits<const Tensor&> {
    using py = Array;
    static Tensor convert(const Array& a) { return from_array(a); }
};
template <>
struct arg_traits<Tensor> {
    using py = Array;
    static Tensor convert(const Array& a) { return from_array(a); }
};
template <>
struct arg_traits<Tensor&> {
    using py = Array;
    static Tensor convert(const Array& a) { return from_array(a); }
};

inline nb::object result_to_python(Tensor t) { return to_python(std::move(t)); }
inline nb::object result_to_python(int64_t v) { return nb::int_(v); }
inline nb::object result_to_python(double v) { return nb::float_(v); }
inline nb::object result_to_python(bool v) { return nb::bool_(v); }
inline nb::object result_to_python(std::pair<Tensor, Tensor> p) {
    return nb::make_tuple(to_python(std::move(p.first)), to_python(std::move(p.second)));
}
template <typename... Ts>
inline nb::object result_to_python(std::tuple<Ts...> t) {
    return std::apply([](auto&&... v) { return nb::make_tuple(result_to_python(std::move(v))...); }, std::move(t));
}

// Binds `fn` as `name`: tensors arrive as DLPack arrays, the computation runs
// without the GIL on `threads` worker threads, results return as capsules.
template <typename R, typename... A>
void def(nb::module_& m, const char* name, R (*fn)(A...)) {
    m.def(name, [fn](int threads, typename arg_traits<A>::py... args) -> nb::object {
        set_num_threads(threads);
        std::optional<R> out;
        {
            nb::gil_scoped_release release;
            out.emplace(fn(arg_traits<A>::convert(args)...));
        }
        return result_to_python(std::move(*out));
    });
}

}  // namespace torchsky::native
