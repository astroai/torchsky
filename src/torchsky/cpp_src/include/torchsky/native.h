// Torch-free native core: a minimal contiguous tensor, a thread pool and
// error helpers for torchsky's CPU kernels.
//
// The kernels were written against libtorch, which tied every wheel to one
// PyTorch minor (no stable C++ ABI across minors). They only use tensors as
// typed, contiguous, row-major buffers, so this header gives them exactly that:
// `torchsky::native::Tensor` owns (or borrows) a buffer with a dtype and a
// shape. Python passes torch tensors in through nanobind/DLPack (zero copy) and
// receives results as DLPack capsules (`torch.from_dlpack`, zero copy).
// Operators, fake-tensor shapes and autograd are registered from Python with
// `torch.library`, so `torch.compile` and autograd keep working.
#pragma once

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace torchsky::native {

// ------------------------------------------------------------------ errors

template <typename... Args>
[[noreturn]] inline void fail(Args&&... args) {
    std::ostringstream os;
    (os << ... << args);
    throw std::runtime_error(os.str());
}

#define TS_CHECK(cond, ...)                                                    \
    do {                                                                       \
        if (!(cond)) ::torchsky::native::fail(__VA_ARGS__);                    \
    } while (0)

// ------------------------------------------------------------------ dtypes

enum class ScalarType : int8_t {
    Bool,
    Byte,
    Short,
    Int,
    Long,
    Half,
    BFloat16,
    Float,
    Double,
    ComplexFloat,
    ComplexDouble,
};

constexpr ScalarType kBool = ScalarType::Bool;
constexpr ScalarType kByte = ScalarType::Byte;
constexpr ScalarType kUInt8 = ScalarType::Byte;
constexpr ScalarType kShort = ScalarType::Short;
constexpr ScalarType kInt16 = ScalarType::Short;
constexpr ScalarType kInt = ScalarType::Int;
constexpr ScalarType kInt32 = ScalarType::Int;
constexpr ScalarType kLong = ScalarType::Long;
constexpr ScalarType kInt64 = ScalarType::Long;
constexpr ScalarType kHalf = ScalarType::Half;
constexpr ScalarType kBFloat16 = ScalarType::BFloat16;
constexpr ScalarType kFloat = ScalarType::Float;
constexpr ScalarType kFloat32 = ScalarType::Float;
constexpr ScalarType kDouble = ScalarType::Double;
constexpr ScalarType kFloat64 = ScalarType::Double;
constexpr ScalarType kComplexFloat = ScalarType::ComplexFloat;
constexpr ScalarType kComplexDouble = ScalarType::ComplexDouble;

inline size_t element_size(ScalarType t) {
    switch (t) {
        case ScalarType::Bool:
        case ScalarType::Byte:
            return 1;
        case ScalarType::Short:
        case ScalarType::Half:
        case ScalarType::BFloat16:
            return 2;
        case ScalarType::Int:
        case ScalarType::Float:
            return 4;
        case ScalarType::Long:
        case ScalarType::Double:
        case ScalarType::ComplexFloat:
            return 8;
        case ScalarType::ComplexDouble:
            return 16;
    }
    return 0;
}

inline const char* dtype_name(ScalarType t) {
    switch (t) {
        case ScalarType::Bool: return "bool";
        case ScalarType::Byte: return "uint8";
        case ScalarType::Short: return "int16";
        case ScalarType::Int: return "int32";
        case ScalarType::Long: return "int64";
        case ScalarType::Half: return "float16";
        case ScalarType::BFloat16: return "bfloat16";
        case ScalarType::Float: return "float32";
        case ScalarType::Double: return "float64";
        case ScalarType::ComplexFloat: return "complex64";
        case ScalarType::ComplexDouble: return "complex128";
    }
    return "?";
}

template <typename T>
constexpr ScalarType scalar_type_of() {
    if constexpr (std::is_same_v<T, bool>) return ScalarType::Bool;
    else if constexpr (std::is_same_v<T, uint8_t>) return ScalarType::Byte;
    else if constexpr (std::is_same_v<T, int16_t>) return ScalarType::Short;
    else if constexpr (std::is_same_v<T, int32_t>) return ScalarType::Int;
    else if constexpr (std::is_same_v<T, int64_t>) return ScalarType::Long;
    else if constexpr (std::is_same_v<T, float>) return ScalarType::Float;
    else if constexpr (std::is_same_v<T, double>) return ScalarType::Double;
    else if constexpr (std::is_same_v<T, std::complex<float>>) return ScalarType::ComplexFloat;
    else if constexpr (std::is_same_v<T, std::complex<double>>) return ScalarType::ComplexDouble;
    else static_assert(sizeof(T) == 0, "unsupported element type");
}

// ------------------------------------------------------------------ shapes

struct Device {
    bool is_cpu() const { return true; }
};
constexpr Device kCPU{};

class Shape {
  public:
    Shape() = default;
    Shape(std::vector<int64_t> v) : v_(std::move(v)) {}
    Shape(std::initializer_list<int64_t> v) : v_(v) {}
    const std::vector<int64_t>& vec() const { return v_; }
    size_t size() const { return v_.size(); }
    int64_t operator[](size_t i) const { return v_[i]; }
    bool equals(const Shape& o) const { return v_ == o.v_; }
    bool operator==(const Shape& o) const { return v_ == o.v_; }
    bool operator!=(const Shape& o) const { return v_ != o.v_; }
    auto begin() const { return v_.begin(); }
    auto end() const { return v_.end(); }

  private:
    std::vector<int64_t> v_;
};

struct TensorOptions {
    ScalarType dtype_ = ScalarType::Double;
    TensorOptions dtype(ScalarType t) const {
        TensorOptions o = *this;
        o.dtype_ = t;
        return o;
    }
    TensorOptions device(Device) const { return *this; }
};

// ------------------------------------------------------------------ tensor

// A contiguous, row-major tensor. Storage is shared between views (reshape,
// select(0, i), narrow(0, ...)); only dimension-0 views exist, so every
// tensor stays contiguous. `borrowed` storage belongs to a caller's buffer.
class Tensor {
  public:
    Tensor() = default;

    static Tensor empty(const std::vector<int64_t>& shape, ScalarType dtype) {
        Tensor t;
        t.shape_ = shape;
        t.dtype_ = dtype;
        const int64_t n = t.numel();
        TS_CHECK(n >= 0, "negative tensor size");
        const size_t bytes = static_cast<size_t>(std::max<int64_t>(n, 1)) * element_size(dtype);
        std::shared_ptr<std::byte[]> buf(new std::byte[bytes]);
        t.storage_ = std::shared_ptr<void>(buf, buf.get());
        t.base_ = buf.get();
        t.offset_ = 0;
        t.borrowed_ = false;
        return t;
    }

    // A view of memory owned elsewhere (input buffers). `keep_alive` holds the
    // owner for as long as any view exists.
    static Tensor borrow(void* data, const std::vector<int64_t>& shape, ScalarType dtype,
                         std::shared_ptr<void> keep_alive = {}) {
        Tensor t;
        t.shape_ = shape;
        t.dtype_ = dtype;
        t.storage_ = keep_alive ? std::move(keep_alive) : std::shared_ptr<void>(data, [](void*) {});
        t.base_ = static_cast<std::byte*>(data);
        t.offset_ = 0;
        t.borrowed_ = true;
        return t;
    }

    bool defined() const { return base_ != nullptr || numel() == 0; }
    bool borrowed() const { return borrowed_; }
    ScalarType scalar_type() const { return dtype_; }
    ScalarType dtype() const { return dtype_; }
    TensorOptions options() const { return TensorOptions{}.dtype(dtype_); }
    Device device() const { return {}; }
    bool is_contiguous() const { return true; }
    bool is_complex() const {
        return dtype_ == ScalarType::ComplexFloat || dtype_ == ScalarType::ComplexDouble;
    }
    int64_t dim() const { return static_cast<int64_t>(shape_.size()); }
    Shape sizes() const { return Shape(shape_); }
    int64_t size(int64_t d) const { return shape_.at(static_cast<size_t>(wrap(d))); }
    int64_t numel() const {
        return std::accumulate(shape_.begin(), shape_.end(), int64_t{1}, std::multiplies<int64_t>());
    }
    int64_t stride(int64_t d) const {
        d = wrap(d);
        int64_t s = 1;
        for (size_t k = static_cast<size_t>(d) + 1; k < shape_.size(); ++k) s *= shape_[k];
        return s;
    }
    size_t nbytes() const { return static_cast<size_t>(numel()) * element_size(dtype_); }

    void* raw_data() const { return static_cast<void*>(base_ + offset_); }

    template <typename T>
    T* data_ptr() const {
        TS_CHECK(scalar_type_of<std::remove_cv_t<T>>() == dtype_, "expected ",
                 dtype_name(scalar_type_of<std::remove_cv_t<T>>()), " data, tensor is ",
                 dtype_name(dtype_));
        return reinterpret_cast<T*>(base_ + offset_);
    }

    Tensor contiguous() const { return *this; }

    Tensor reshape(std::vector<int64_t> shape) const {
        int64_t known = 1;
        int64_t infer = -1;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] == -1) {
                TS_CHECK(infer < 0, "reshape: only one dimension can be -1");
                infer = static_cast<int64_t>(i);
            } else {
                known *= shape[i];
            }
        }
        if (infer >= 0) shape[static_cast<size_t>(infer)] = known ? numel() / known : 0;
        Tensor t = *this;
        t.shape_ = std::move(shape);
        TS_CHECK(t.numel() == numel(), "reshape: size mismatch");
        return t;
    }
    Tensor reshape(std::initializer_list<int64_t> shape) const {
        return reshape(std::vector<int64_t>(shape));
    }
    Tensor view(std::vector<int64_t> shape) const { return reshape(std::move(shape)); }
    Tensor view(std::initializer_list<int64_t> shape) const {
        return reshape(std::vector<int64_t>(shape));
    }

    Tensor unsqueeze(int64_t d) const {
        TS_CHECK(d == 0, "unsqueeze: only dimension 0 is supported");
        std::vector<int64_t> s = shape_;
        s.insert(s.begin(), 1);
        return reshape(s);
    }

    Tensor select(int64_t d, int64_t i) const {
        TS_CHECK(wrap(d) == 0, "select: only dimension 0 is supported");
        TS_CHECK(dim() >= 1, "select on a 0-d tensor");
        if (i < 0) i += shape_[0];
        TS_CHECK(i >= 0 && i < shape_[0], "select: index out of range");
        Tensor t = *this;
        t.shape_.erase(t.shape_.begin());
        t.offset_ += static_cast<size_t>(i) * static_cast<size_t>(t.numel()) * element_size(dtype_);
        return t;
    }

    Tensor narrow(int64_t d, int64_t start, int64_t length) const {
        TS_CHECK(wrap(d) == 0, "narrow: only dimension 0 is supported");
        TS_CHECK(start >= 0 && length >= 0 && start + length <= shape_[0], "narrow: out of range");
        Tensor t = *this;
        const size_t row = static_cast<size_t>(numel() / std::max<int64_t>(shape_[0], 1));
        t.shape_[0] = length;
        t.offset_ += static_cast<size_t>(start) * row * element_size(dtype_);
        return t;
    }

    Tensor clone() const {
        Tensor t = empty(shape_, dtype_);
        if (nbytes()) std::memcpy(t.raw_data(), raw_data(), nbytes());
        return t;
    }

    Tensor& copy_(const Tensor& src) {
        TS_CHECK(src.numel() == numel(), "copy_: size mismatch");
        if (src.dtype_ == dtype_) {
            if (nbytes()) std::memmove(raw_data(), src.raw_data(), nbytes());
        } else {
            *this = assign_converted(src);
        }
        return *this;
    }

    template <typename S>
    Tensor& fill_(S value) {
        visit([&](auto* p) {
            using T = std::remove_pointer_t<decltype(p)>;
            const T v = static_cast<T>(value);
            std::fill(p, p + numel(), v);
        });
        return *this;
    }
    Tensor& zero_() {
        if (nbytes()) std::memset(raw_data(), 0, nbytes());
        return *this;
    }

    // dtype conversion (a copy unless the dtype already matches).
    Tensor to(ScalarType t) const {
        if (t == dtype_) return *this;
        Tensor out = empty(shape_, t);
        out.assign_converted_into(*this);
        return out;
    }

    template <typename T>
    T item() const {
        TS_CHECK(numel() == 1, "item: tensor has ", numel(), " elements");
        T out{};
        visit([&](auto* p) { out = static_cast<T>(p[0]); });
        return out;
    }

    // Applies `f(T*)` with the element pointer of the tensor's real or integer type.
    template <typename F>
    void visit(F&& f) const {
        switch (dtype_) {
            case ScalarType::Bool: f(reinterpret_cast<bool*>(raw_data())); break;
            case ScalarType::Byte: f(reinterpret_cast<uint8_t*>(raw_data())); break;
            case ScalarType::Short: f(reinterpret_cast<int16_t*>(raw_data())); break;
            case ScalarType::Int: f(reinterpret_cast<int32_t*>(raw_data())); break;
            case ScalarType::Long: f(reinterpret_cast<int64_t*>(raw_data())); break;
            case ScalarType::Float: f(reinterpret_cast<float*>(raw_data())); break;
            case ScalarType::Double: f(reinterpret_cast<double*>(raw_data())); break;
            default: fail("unsupported dtype ", dtype_name(dtype_), " for this operation");
        }
    }

    std::shared_ptr<void> storage() const { return storage_; }

  private:
    int64_t wrap(int64_t d) const {
        const int64_t n = dim();
        if (d < 0) d += n;
        TS_CHECK(d >= 0 && d < std::max<int64_t>(n, 1), "dimension out of range");
        return d;
    }

    Tensor assign_converted(const Tensor& src) {
        assign_converted_into(src);
        return *this;
    }

    void assign_converted_into(const Tensor& src) {
        const int64_t n = numel();
        if (is_complex() || src.is_complex()) {
            TS_CHECK(is_complex(), "cannot convert complex to real");
            auto* dst = data_ptr<std::complex<double>>();
            if (src.dtype_ == ScalarType::ComplexDouble) {
                std::memmove(dst, src.raw_data(), nbytes());
            } else {
                src.visit([&](auto* p) {
                    for (int64_t i = 0; i < n; ++i) dst[i] = std::complex<double>(static_cast<double>(p[i]), 0.0);
                });
            }
            return;
        }
        visit([&](auto* d) {
            using D = std::remove_pointer_t<decltype(d)>;
            src.visit([&](auto* s) {
                for (int64_t i = 0; i < n; ++i) d[i] = static_cast<D>(s[i]);
            });
        });
    }

    std::vector<int64_t> shape_;
    ScalarType dtype_ = ScalarType::Double;
    std::shared_ptr<void> storage_;
    std::byte* base_ = nullptr;
    size_t offset_ = 0;
    bool borrowed_ = false;
};

// ------------------------------------------------------------------ factories

inline Tensor empty(const std::vector<int64_t>& shape, TensorOptions o = {}) {
    return Tensor::empty(shape, o.dtype_);
}
inline Tensor empty(std::initializer_list<int64_t> shape, TensorOptions o = {}) {
    return Tensor::empty(std::vector<int64_t>(shape), o.dtype_);
}
inline Tensor empty(const Shape& shape, TensorOptions o = {}) { return empty(shape.vec(), o); }
inline Tensor zeros(const std::vector<int64_t>& shape, TensorOptions o = {}) {
    return Tensor::empty(shape, o.dtype_).zero_();
}
inline Tensor zeros(std::initializer_list<int64_t> shape, TensorOptions o = {}) {
    return zeros(std::vector<int64_t>(shape), o);
}
inline Tensor zeros(const Shape& shape, TensorOptions o = {}) { return zeros(shape.vec(), o); }
template <typename S>
inline Tensor full(const std::vector<int64_t>& shape, S value, TensorOptions o = {}) {
    Tensor t = Tensor::empty(shape, o.dtype_);
    t.fill_(value);
    return t;
}
template <typename S>
inline Tensor full(std::initializer_list<int64_t> shape, S value, TensorOptions o = {}) {
    return full(std::vector<int64_t>(shape), value, o);
}
inline Tensor empty_like(const Tensor& t) { return Tensor::empty(t.sizes().vec(), t.dtype()); }
inline Tensor empty_like(const Tensor& t, TensorOptions o) { return Tensor::empty(t.sizes().vec(), o.dtype_); }
inline Tensor zeros_like(const Tensor& t) { return empty_like(t).zero_(); }
inline Tensor zeros_like(const Tensor& t, TensorOptions o) { return empty_like(t, o).zero_(); }
template <typename S>
inline Tensor full_like(const Tensor& t, S value) {
    Tensor out = empty_like(t);
    out.fill_(value);
    return out;
}
inline Tensor arange(int64_t n, TensorOptions o = TensorOptions{}.dtype(ScalarType::Long)) {
    Tensor t = Tensor::empty({n}, o.dtype_);
    t.visit([&](auto* p) {
        using T = std::remove_pointer_t<decltype(p)>;
        for (int64_t i = 0; i < n; ++i) p[i] = static_cast<T>(i);
    });
    return t;
}

// Stacks equally shaped tensors along a new leading dimension.
inline Tensor stack(std::initializer_list<Tensor> parts, int64_t dim = 0) {
    TS_CHECK(dim == 0, "stack: only dimension 0 is supported");
    TS_CHECK(parts.size() > 0, "stack: no tensors");
    const Tensor& first = *parts.begin();
    std::vector<int64_t> shape = first.sizes().vec();
    shape.insert(shape.begin(), static_cast<int64_t>(parts.size()));
    Tensor out = Tensor::empty(shape, first.dtype());
    size_t off = 0;
    for (const Tensor& p : parts) {
        TS_CHECK(p.sizes() == first.sizes() && p.dtype() == first.dtype(), "stack: shape or dtype mismatch");
        std::memcpy(static_cast<std::byte*>(out.raw_data()) + off, p.raw_data(), p.nbytes());
        off += p.nbytes();
    }
    return out;
}

// ------------------------------------------------------------------ threads

namespace detail {
inline std::atomic<int>& thread_count() {
    static std::atomic<int> n{std::max(1, static_cast<int>(std::thread::hardware_concurrency()))};
    return n;
}
}  // namespace detail

// Set from Python before each operator call (torch.get_num_threads()).
inline void set_num_threads(int n) { detail::thread_count().store(std::max(1, n)); }
inline int get_num_threads() { return detail::thread_count().load(); }

// Same contract as at::parallel_for: f(begin, end) over chunks of at least
// `grain` items; results must not depend on the chunking.
template <typename F>
inline void parallel_for(int64_t begin, int64_t end, int64_t grain, const F& f) {
    if (end <= begin) return;
    const int64_t n = end - begin;
    const int threads = get_num_threads();
    grain = std::max<int64_t>(grain, 1);
    const int64_t max_chunks = (n + grain - 1) / grain;
    const int workers = static_cast<int>(std::min<int64_t>(threads, max_chunks));
    if (workers <= 1) {
        f(begin, end);
        return;
    }
    const int64_t chunk = std::max<int64_t>(grain, (n + workers * 4 - 1) / (workers * 4));
    std::atomic<int64_t> next{begin};
    std::exception_ptr error;
    std::atomic<bool> failed{false};
    auto work = [&]() {
        try {
            for (;;) {
                const int64_t b = next.fetch_add(chunk);
                if (b >= end || failed.load()) return;
                f(b, std::min(end, b + chunk));
            }
        } catch (...) {
            if (!failed.exchange(true)) error = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(workers - 1));
    for (int t = 1; t < workers; ++t) pool.emplace_back(work);
    work();
    for (auto& th : pool) th.join();
    if (error) std::rethrow_exception(error);
}

}  // namespace torchsky::native
