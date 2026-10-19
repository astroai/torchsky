#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>

#if defined(__linux__)
#include <sched.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#define POCKETFFT_NO_MULTITHREADING
#define POCKETFFT_CACHE_SIZE 128
#include "third_party/pocketfft_hdronly.h"

#include <ATen/Parallel.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/fft_ifft.h>
#include "torchsky_torch.h"

namespace torchsky {
namespace {

constexpr int64_t JRLL[12] = {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
constexpr int64_t JPLL[12] = {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};
constexpr int64_t NB_XOFFSET[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
constexpr int64_t NB_YOFFSET[8] = {0, 1, 1, 1, 0, -1, -1, -1};
constexpr int64_t NB_FACEARRAY[9][12] = {
    {8, 9, 10, 11, -1, -1, -1, -1, 10, 11, 8, 9},
    {5, 6, 7, 4, 8, 9, 10, 11, 9, 10, 11, 8},
    {-1, -1, -1, -1, 5, 6, 7, 4, -1, -1, -1, -1},
    {4, 5, 6, 7, 11, 8, 9, 10, 11, 8, 9, 10},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    {1, 2, 3, 0, 0, 1, 2, 3, 5, 6, 7, 4},
    {-1, -1, -1, -1, 7, 4, 5, 6, -1, -1, -1, -1},
    {3, 0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 7},
    {2, 3, 0, 1, -1, -1, -1, -1, 0, 1, 2, 3},
};
constexpr int64_t NB_SWAPARRAY[9][3] = {
    {0, 0, 3},
    {0, 0, 6},
    {0, 0, 0},
    {0, 0, 5},
    {0, 0, 0},
    {5, 0, 0},
    {0, 0, 0},
    {6, 0, 0},
    {3, 0, 0},
};
constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double TWO_PI = 6.283185307179586476925286766559005768;
constexpr double RAD2DEG = 180.0 / PI;
// Benchmark-relevant arrays are commonly O(1e5-1e6); parallelize earlier.
constexpr int64_t PARALLEL_MIN_ELEMS = 1 << 16;

inline bool is_valid_nside(int64_t nside) {
    return nside > 0 && (nside & (nside - 1)) == 0;
}

inline int64_t floor_div2(int64_t v) {
    return (v >= 0) ? (v / 2) : -(((-v) + 1) / 2);
}

inline int64_t alm_size(int64_t lmax, int64_t mmax) {
    return (mmax + 1) * (lmax + 1) - (mmax * (mmax + 1)) / 2;
}

inline int64_t alm_packed_index(int64_t l, int64_t m, int64_t lmax) {
    return m * (lmax + 1) - (m * (m - 1)) / 2 + (l - m);
}

inline int64_t ilog2_pow2(int64_t x) {
    int64_t s = 0;
    while ((x & 1) == 0) {
        x >>= 1;
        ++s;
    }
    return s;
}

inline const std::array<uint8_t, 65536>& compact_lut16() {
    static const std::array<uint8_t, 65536> lut = []() {
        std::array<uint8_t, 65536> table{};
        for (uint32_t i = 0; i < 65536; ++i) {
            uint8_t v = 0;
            for (uint32_t b = 0; b < 8; ++b) {
                v |= static_cast<uint8_t>(((i >> (2U * b)) & 1U) << b);
            }
            table[i] = v;
        }
        return table;
    }();
    return lut;
}

inline uint64_t isqrt_u64(uint64_t v) {
    uint64_t r = static_cast<uint64_t>(std::sqrt(static_cast<double>(v)));
    while ((r + 1) * (r + 1) <= v) {
        ++r;
    }
    while (r * r > v) {
        --r;
    }
    return r;
}

inline uint64_t spread_bits_u64(uint64_t x) {
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FFULL;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x << 2)) & 0x3333333333333333ULL;
    x = (x | (x << 1)) & 0x5555555555555555ULL;
    return x;
}

inline uint64_t compact_bits_u64(uint64_t x) {
    const auto& lut = compact_lut16();
    return static_cast<uint64_t>(lut[x & 0xFFFFULL])
           | (static_cast<uint64_t>(lut[(x >> 16) & 0xFFFFULL]) << 8)
           | (static_cast<uint64_t>(lut[(x >> 32) & 0xFFFFULL]) << 16)
           | (static_cast<uint64_t>(lut[(x >> 48) & 0xFFFFULL]) << 24);
}

inline double wrap_rad_2pi(double v) {
    double r = std::fmod(v, TWO_PI);
    if (r < 0.0) {
        r += TWO_PI;
    }
    if (r >= TWO_PI) {
        r = 0.0;
    }
    return r;
}

inline int64_t xyf_to_nest(int64_t npface, int64_t ix, int64_t iy, int64_t face_num) {
    return face_num * npface
           + static_cast<int64_t>(spread_bits_u64(static_cast<uint64_t>(ix)))
           + static_cast<int64_t>(spread_bits_u64(static_cast<uint64_t>(iy)) << 1);
}

inline void nest_to_xyf(int64_t npface, int64_t pix_nest, int64_t& ix, int64_t& iy, int64_t& face_num) {
    face_num = pix_nest / npface;
    const uint64_t ipf = static_cast<uint64_t>(pix_nest % npface);
    ix = static_cast<int64_t>(compact_bits_u64(ipf));
    iy = static_cast<int64_t>(compact_bits_u64(ipf >> 1));
}

inline int64_t xyf_to_ring(
    int64_t nside,
    int64_t ncap,
    int64_t npix,
    int64_t nl4,
    int64_t ix,
    int64_t iy,
    int64_t face_num
) {
    const int64_t jr = JRLL[face_num] * nside - ix - iy - 1;
    int64_t nr;
    int64_t n_before;
    int64_t kshift = 0;
    if (jr < nside) {
        nr = jr;
        n_before = 2 * nr * (nr - 1);
    } else if (jr > (3 * nside)) {
        nr = nl4 - jr;
        n_before = npix - 2 * (nr + 1) * nr;
    } else {
        nr = nside;
        n_before = ncap + (jr - nside) * nl4;
        kshift = (jr - nside) & 1;
    }

    int64_t jp = (JPLL[face_num] * nr + ix - iy + 1 + kshift) / 2;
    if (jp > nl4) {
        jp -= nl4;
    }
    if (jp < 1) {
        jp += nl4;
    }
    return n_before + jp - 1;
}

inline void ring_to_xyf(
    int64_t nside,
    int64_t ncap,
    int64_t npix,
    int64_t nl2,
    int64_t nside_shift,
    int64_t four_nside_shift,
    int64_t four_nside_mask,
    int64_t p,
    int64_t& ix,
    int64_t& iy,
    int64_t& face_num
) {
    int64_t iring;
    int64_t iphi;
    int64_t kshift;
    int64_t nr;

    if (p < ncap) {
        iring = static_cast<int64_t>((1 + isqrt_u64(static_cast<uint64_t>(1 + 2 * p))) >> 1);
        iphi = (p + 1) - 2 * iring * (iring - 1);
        kshift = 0;
        nr = iring;
        face_num = (iphi - 1) / iring;
    } else if (p < (npix - ncap)) {
        const int64_t ip = p - ncap;
        iring = (ip >> four_nside_shift) + nside;
        iphi = (ip & four_nside_mask) + 1;
        kshift = (iring + nside) & 1;
        nr = nside;
        const int64_t ire = iring - nside + 1;
        const int64_t irm = nl2 + 2 - ire;
        const int64_t ifm = (iphi - (ire / 2) + nside - 1) >> nside_shift;
        const int64_t ifp = (iphi - (irm / 2) + nside - 1) >> nside_shift;
        if (ifp == ifm) {
            face_num = ifp | 4;
        } else if (ifp < ifm) {
            face_num = ifp;
        } else {
            face_num = ifm + 8;
        }
    } else {
        const int64_t ip = npix - p;
        const int64_t irs = static_cast<int64_t>((1 + isqrt_u64(static_cast<uint64_t>(2 * ip - 1))) >> 1);
        iphi = 4 * irs + 1 - (ip - 2 * irs * (irs - 1));
        iring = 2 * nl2 - irs;
        kshift = 0;
        nr = irs;
        face_num = 8 + (iphi - 1) / irs;
    }

    const int64_t irt = iring - JRLL[face_num] * nside + 1;
    int64_t ipt = 2 * iphi - JPLL[face_num] * nr - kshift - 1;
    if (ipt >= nl2) {
        ipt -= 8 * nside;
    }

    ix = floor_div2(ipt - irt);
    iy = floor_div2(-(ipt + irt));
}

inline int64_t ring_above_scalar(int64_t nside, double z) {
    const double az = std::abs(z);
    if (az <= (2.0 / 3.0)) {
        return static_cast<int64_t>(std::floor(nside * (2.0 - 1.5 * z)));
    }
    const double irp_f = nside * std::sqrt(std::max(0.0, 3.0 * (1.0 - az)));
    const int64_t irp = static_cast<int64_t>(std::floor(irp_f));
    if (z > 0.0) {
        return irp;
    }
    return 4 * nside - irp - 1;
}

inline void get_ring_info2_scalar(
    int64_t nside,
    int64_t ring,
    int64_t& startpix,
    int64_t& ringpix,
    double& theta,
    bool& shifted
) {
    const int64_t npix = 12 * nside * nside;
    const int64_t ncap = 2 * nside * (nside - 1);
    const double fact2 = 4.0 / static_cast<double>(npix);
    const double fact1 = static_cast<double>(2 * nside) * fact2;

    const int64_t northr = (ring > (2 * nside)) ? (4 * nside - ring) : ring;
    if (northr < nside) {
        const double northf = static_cast<double>(northr);
        const double tmp = northf * northf * fact2;
        const double costheta = 1.0 - tmp;
        const double sintheta = std::sqrt(std::max(0.0, tmp * (2.0 - tmp)));
        theta = std::atan2(sintheta, costheta);
        ringpix = 4 * northr;
        shifted = true;
        startpix = 2 * northr * (northr - 1);
    } else {
        double costheta = (2.0 * static_cast<double>(nside) - static_cast<double>(northr)) * fact1;
        if (costheta > 1.0) {
            costheta = 1.0;
        } else if (costheta < -1.0) {
            costheta = -1.0;
        }
        theta = std::acos(costheta);
        ringpix = 4 * nside;
        shifted = (((northr - nside) & 1) == 0);
        startpix = ncap + (northr - nside) * ringpix;
    }

    if (northr != ring) {
        theta = PI - theta;
        startpix = npix - startpix - ringpix;
    }
}

struct RecurrenceCoeffs {
    double c1, c2_p, c3;
};

class RecurrenceCache {
    std::mutex mutex_;
    int64_t last_lmax_ = -1;
    int64_t last_mmax_ = -1;
    int64_t last_spin_ = -1;
    std::vector<RecurrenceCoeffs> coeffs_;
    std::vector<int64_t> m_offsets_;

public:
    const std::vector<RecurrenceCoeffs>& get_coeffs(int64_t lmax, int64_t mmax, int64_t spin, const std::vector<int64_t>& offsets) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (lmax == last_lmax_ && mmax == last_mmax_ && spin == last_spin_) {
            return coeffs_;
        }

        int64_t total_size = 0;
        for (int64_t m = 0; m <= mmax; ++m) {
            int64_t l0 = std::max((int64_t)std::abs(spin), m);
            if (lmax >= l0 + 2) {
                total_size += (lmax - (l0 + 2) + 1);
            }
        }

        coeffs_.assign(total_size, {0, 0, 0});
        int64_t current_off = 0;
        for (int64_t m = 0; m <= mmax; ++m) {
            int64_t l0 = std::max((int64_t)std::abs(spin), m);
            for (int64_t l = l0 + 2; l <= lmax; ++l) {
                double kl = std::sqrt((double)(l * l - m * m) * (l * l - spin * spin));
                double preFactor = (double)l / kl;
                double c1 = preFactor * (2.0 * l - 1.0);
                double c3 = -preFactor * (std::sqrt((double)((l - 1) * (l - 1) - m * m) * ((l - 1) * (l - 1) - spin * spin)) / (l - 1));
                double shift = (double)(m * (-spin)) / (double)(l * (l - 1));
                double c2_p = -c1 * shift;
                coeffs_[current_off++] = {c1, c2_p, c3};
            }
        }
        last_lmax_ = lmax;
        last_mmax_ = mmax;
        last_spin_ = spin;
        return coeffs_;
    }
};

static RecurrenceCache g_recurrence_cache;

inline uint64_t ring_alias_cache_key(int64_t nph, int64_t mcount) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(nph)) << 32)
           | static_cast<uint64_t>(static_cast<uint32_t>(mcount));
}

class RingAliasIndexCache {
    std::mutex mutex_;
    std::unordered_map<uint64_t, torch::Tensor> plus_idx_;
    std::unordered_map<uint64_t, torch::Tensor> minus_idx_;
    std::unordered_map<uint64_t, torch::Tensor> both_idx_;
    static constexpr size_t kMaxEntries = 256;

    static torch::Tensor make_plus_idx(int64_t nph, int64_t mcount) {
        auto idx = torch::empty({mcount}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto* ptr = idx.data_ptr<int64_t>();
        for (int64_t m = 0; m < mcount; ++m) {
            ptr[m] = m % nph;
        }
        return idx;
    }

    static torch::Tensor make_minus_idx(int64_t nph, int64_t mcount) {
        auto idx = torch::empty({mcount}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto* ptr = idx.data_ptr<int64_t>();
        for (int64_t m = 0; m < mcount; ++m) {
            const int64_t k = m % nph;
            ptr[m] = (k == 0) ? 0 : (nph - k);
        }
        return idx;
    }

    static torch::Tensor make_both_idx(int64_t nph, int64_t mcount) {
        auto idx = torch::empty({2 * mcount}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto* ptr = idx.data_ptr<int64_t>();
        for (int64_t m = 0; m < mcount; ++m) {
            const int64_t k = m % nph;
            ptr[m] = k;
            ptr[mcount + m] = (k == 0) ? 0 : (nph - k);
        }
        return idx;
    }

public:
    const torch::Tensor& plus(int64_t nph, int64_t mcount) {
        const uint64_t key = ring_alias_cache_key(nph, mcount);
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = plus_idx_.find(key);
        if (it != plus_idx_.end()) {
            return it->second;
        }
        if (plus_idx_.size() > kMaxEntries) {
            plus_idx_.clear();
        }
        return plus_idx_.emplace(key, make_plus_idx(nph, mcount)).first->second;
    }

    const torch::Tensor& minus(int64_t nph, int64_t mcount) {
        const uint64_t key = ring_alias_cache_key(nph, mcount);
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = minus_idx_.find(key);
        if (it != minus_idx_.end()) {
            return it->second;
        }
        if (minus_idx_.size() > kMaxEntries) {
            minus_idx_.clear();
        }
        return minus_idx_.emplace(key, make_minus_idx(nph, mcount)).first->second;
    }

    const torch::Tensor& both(int64_t nph, int64_t mcount) {
        const uint64_t key = ring_alias_cache_key(nph, mcount);
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = both_idx_.find(key);
        if (it != both_idx_.end()) {
            return it->second;
        }
        if (both_idx_.size() > kMaxEntries) {
            both_idx_.clear();
        }
        return both_idx_.emplace(key, make_both_idx(nph, mcount)).first->second;
    }
};

static RingAliasIndexCache g_ring_alias_idx_cache;

inline bool spin_mlim_prune_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("TORCHSKY_SPIN_MLIM_PRUNE");
        if (!env) {
            return true;
        }
        return env[0] != '0';
    }();
    return enabled;
}

// Read at call time so tests can toggle without restarting the process.
inline bool spin_ishioka_enabled() {
    const char* env = std::getenv("TORCHSKY_SPIN_ISHIOKA");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    return env[0] != '0';
}

inline int64_t sharp_mlim_estimate(int64_t lmax, int64_t spin_abs, double sth, double cth) {
    double ofs = static_cast<double>(lmax) * 0.01;
    if (ofs < 100.0) {
        ofs = 100.0;
    }
    const double b = -2.0 * static_cast<double>(spin_abs) * std::abs(cth);
    const double t1 = static_cast<double>(lmax) * sth + ofs;
    const double c = static_cast<double>(spin_abs * spin_abs) - t1 * t1;
    const double discr = b * b - 4.0 * c;
    if (discr <= 0.0) {
        return lmax;
    }
    double res = (-b + std::sqrt(discr)) * 0.5;
    if (res < 0.0) {
        res = 0.0;
    }
    if (res > static_cast<double>(lmax)) {
        res = static_cast<double>(lmax);
    }
    return static_cast<int64_t>(res + 0.5);
}

}  // namespace

torch::Tensor healpix_ang2pix_ring_cpu(int64_t nside, const torch::Tensor& ra_deg, const torch::Tensor& dec_deg) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!ra_deg.device().is_cpu() || !dec_deg.device().is_cpu()) {
        throw std::runtime_error("healpix_ang2pix_ring_cpu expects CPU tensors");
    }
    if (ra_deg.scalar_type() != torch::kDouble || dec_deg.scalar_type() != torch::kDouble) {
        throw std::runtime_error("healpix_ang2pix_ring_cpu expects float64 tensors");
    }
    if (!ra_deg.sizes().equals(dec_deg.sizes())) {
        throw std::runtime_error("RA/Dec shape mismatch");
    }

    auto ra = ra_deg.contiguous().reshape({-1});
    auto dec = dec_deg.contiguous().reshape({-1});
    auto out = torch::empty_like(ra, torch::TensorOptions().dtype(torch::kInt64));
    const auto* ra_ptr = ra.data_ptr<double>();
    const auto* dec_ptr = dec.data_ptr<double>();
    auto* dst = out.data_ptr<int64_t>();
    const int64_t n = ra.numel();
    const int64_t npix = 12 * nside * nside;

    for (int64_t i = 0; i < n; ++i) {
        const double z = std::sin(dec_ptr[i] * PI / 180.0);
        const double za = std::abs(z);
        const double tt = wrap_rad_2pi(ra_ptr[i] * PI / 180.0) * (2.0 / PI);

        int64_t pix;
        if (za <= (2.0 / 3.0)) {
            const double temp1 = nside * (0.5 + tt);
            const double temp2 = nside * z * 0.75;
            const int64_t jp = static_cast<int64_t>(temp1 - temp2);
            const int64_t jm = static_cast<int64_t>(temp1 + temp2);
            const int64_t ir = nside + 1 + jp - jm;
            const int64_t kshift = 1 - (ir & 1);
            int64_t ip = ((jp + jm - nside + kshift + 1) / 2) + 1;
            if (ip > (4 * nside)) {
                ip -= (4 * nside);
            }
            if (ip < 1) {
                ip += (4 * nside);
            }
            pix = 2 * nside * (nside - 1) + (ir - 1) * (4 * nside) + (ip - 1);
        } else {
            const int64_t ntt = static_cast<int64_t>(tt);
            const double tp = tt - static_cast<double>(ntt);
            const double tmp = nside * std::sqrt(3.0 * (1.0 - std::abs(z)));
            const int64_t jp = static_cast<int64_t>(tp * tmp);
            const int64_t jm = static_cast<int64_t>((1.0 - tp) * tmp);
            const int64_t ir = jp + jm + 1;
            const int64_t ip = static_cast<int64_t>(tt * static_cast<double>(ir)) % (4 * ir);
            if (z > 0.0) {
                pix = 2 * ir * (ir - 1) + ip;
            } else {
                pix = npix - 2 * ir * (ir + 1) + ip;
            }
        }
        dst[i] = pix;
    }
    return out.reshape(ra_deg.sizes().vec());
}

torch::Tensor healpix_ang2pix_nested_cpu(int64_t nside, const torch::Tensor& ra_deg, const torch::Tensor& dec_deg) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!ra_deg.device().is_cpu() || !dec_deg.device().is_cpu()) {
        throw std::runtime_error("healpix_ang2pix_nested_cpu expects CPU tensors");
    }
    if (ra_deg.scalar_type() != torch::kDouble || dec_deg.scalar_type() != torch::kDouble) {
        throw std::runtime_error("healpix_ang2pix_nested_cpu expects float64 tensors");
    }
    if (!ra_deg.sizes().equals(dec_deg.sizes())) {
        throw std::runtime_error("RA/Dec shape mismatch");
    }

    auto ra = ra_deg.contiguous().reshape({-1});
    auto dec = dec_deg.contiguous().reshape({-1});
    auto out = torch::empty_like(ra, torch::TensorOptions().dtype(torch::kInt64));
    const auto* ra_ptr = ra.data_ptr<double>();
    const auto* dec_ptr = dec.data_ptr<double>();
    auto* dst = out.data_ptr<int64_t>();
    const int64_t n = ra.numel();
    const int64_t npface = nside * nside;
    const int64_t nmask = nside - 1;

    for (int64_t i = 0; i < n; ++i) {
        const double z = std::sin(dec_ptr[i] * PI / 180.0);
        const double za = std::abs(z);
        const double tt = wrap_rad_2pi(ra_ptr[i] * PI / 180.0) * (2.0 / PI);

        int64_t face_num;
        int64_t ix;
        int64_t iy;

        if (za <= (2.0 / 3.0)) {
            const double temp1 = nside * (0.5 + tt);
            const double temp2 = nside * (z * 0.75);
            const int64_t jp = static_cast<int64_t>(temp1 - temp2);
            const int64_t jm = static_cast<int64_t>(temp1 + temp2);
            const int64_t ifp = jp / nside;
            const int64_t ifm = jm / nside;
            if (ifp == ifm) {
                face_num = ifp | 4;
            } else if (ifp < ifm) {
                face_num = ifp;
            } else {
                face_num = ifm + 8;
            }
            ix = jm & nmask;
            iy = nside - (jp & nmask) - 1;
        } else {
            int64_t ntt = static_cast<int64_t>(tt);
            if (ntt >= 4) {
                ntt = 3;
            }
            const double tp = tt - static_cast<double>(ntt);
            const double tmp = nside * std::sqrt(3.0 * (1.0 - std::abs(z)));
            int64_t jp = static_cast<int64_t>(tp * tmp);
            int64_t jm = static_cast<int64_t>((1.0 - tp) * tmp);
            if (jp > (nside - 1)) {
                jp = nside - 1;
            }
            if (jm > (nside - 1)) {
                jm = nside - 1;
            }
            if (z >= 0.0) {
                face_num = ntt;
                ix = nside - jm - 1;
                iy = nside - jp - 1;
            } else {
                face_num = ntt + 8;
                ix = jp;
                iy = jm;
            }
        }

        dst[i] = face_num * npface
                 + static_cast<int64_t>(spread_bits_u64(static_cast<uint64_t>(ix)))
                 + static_cast<int64_t>(spread_bits_u64(static_cast<uint64_t>(iy)) << 1);
    }
    return out.reshape(ra_deg.sizes().vec());
}

std::pair<torch::Tensor, torch::Tensor> healpix_pix2ang_ring_cpu(int64_t nside, const torch::Tensor& pix_ring) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!pix_ring.device().is_cpu()) {
        throw std::runtime_error("healpix_pix2ang_ring_cpu expects CPU tensor");
    }
    if (pix_ring.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_pix2ang_ring_cpu expects int64 tensor");
    }

    auto pix = pix_ring.contiguous().reshape({-1});
    auto ra = torch::empty_like(pix, torch::TensorOptions().dtype(torch::kDouble));
    auto dec = torch::empty_like(pix, torch::TensorOptions().dtype(torch::kDouble));

    const int64_t ncap = 2 * nside * (nside - 1);
    const int64_t npix = 12 * nside * nside;
    const double fact2 = 4.0 / static_cast<double>(npix);

    const int64_t* in = pix.data_ptr<int64_t>();
    double* ra_ptr = ra.data_ptr<double>();
    double* dec_ptr = dec.data_ptr<double>();
    const int64_t n = pix.numel();
    std::atomic<bool> invalid{false};
    const double fact1 = static_cast<double>(2 * nside) * fact2;
    auto compute_block = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t p = in[i];
            if (p < 0 || p >= npix) {
                invalid.store(true, std::memory_order_relaxed);
                ra_ptr[i] = 0.0;
                dec_ptr[i] = 0.0;
                continue;
            }

            double z;
            double phi;
            if (p < ncap) {
                const int64_t iring = static_cast<int64_t>((1 + isqrt_u64(static_cast<uint64_t>(1 + 2 * p))) >> 1);
                const int64_t iphi = (p + 1) - 2 * iring * (iring - 1);
                z = 1.0 - static_cast<double>(iring * iring) * fact2;
                phi = (static_cast<double>(iphi) - 0.5) * ((PI / 2.0) / static_cast<double>(iring));
            } else if (p < (npix - ncap)) {
                const int64_t ip = p - ncap;
                const int64_t iring = (ip / (4 * nside)) + nside;
                const int64_t iphi = (ip % (4 * nside)) + 1;
                const double fodd = (((iring + nside) & 1) != 0) ? 1.0 : 0.5;
                z = static_cast<double>(2 * nside - iring) * fact1;
                phi = (static_cast<double>(iphi) - fodd) * (PI / static_cast<double>(2 * nside));
            } else {
                const int64_t ip = npix - p;
                const int64_t iring = static_cast<int64_t>((1 + isqrt_u64(static_cast<uint64_t>(2 * ip - 1))) >> 1);
                const int64_t iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
                z = -1.0 + static_cast<double>(iring * iring) * fact2;
                phi = (static_cast<double>(iphi) - 0.5) * ((PI / 2.0) / static_cast<double>(iring));
            }

            double ra_deg = phi * RAD2DEG;
            if (ra_deg >= 360.0) {
                ra_deg -= 360.0;
            } else if (ra_deg < 0.0) {
                ra_deg += 360.0;
            }
            ra_ptr[i] = ra_deg;
            dec_ptr[i] = std::asin(std::fmax(-1.0, std::fmin(1.0, z))) * RAD2DEG;
        }
    };
    if (n >= PARALLEL_MIN_ELEMS) {
        at::parallel_for(0, n, 4096, compute_block);
    } else {
        compute_block(0, n);
    }
    if (invalid.load(std::memory_order_relaxed)) {
        throw std::runtime_error("pixel index out of range for nside");
    }

    auto shape = pix_ring.sizes().vec();
    return {ra.reshape(shape), dec.reshape(shape)};
}

std::pair<torch::Tensor, torch::Tensor> healpix_pix2ang_nested_cpu(int64_t nside, const torch::Tensor& pix_nest) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!pix_nest.device().is_cpu()) {
        throw std::runtime_error("healpix_pix2ang_nested_cpu expects CPU tensor");
    }
    if (pix_nest.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_pix2ang_nested_cpu expects int64 tensor");
    }

    auto pix = pix_nest.contiguous().reshape({-1});
    auto ra = torch::empty_like(pix, torch::TensorOptions().dtype(torch::kDouble));
    auto dec = torch::empty_like(pix, torch::TensorOptions().dtype(torch::kDouble));

    const int64_t npface = nside * nside;
    const int64_t npix = 12 * nside * nside;
    const int64_t nl4 = 4 * nside;
    const int64_t ncap = 2 * nside * (nside - 1);
    const double fact2 = 4.0 / static_cast<double>(npix);
    const double fact1 = static_cast<double>(2 * nside) * fact2;

    const int64_t* in = pix.data_ptr<int64_t>();
    double* ra_ptr = ra.data_ptr<double>();
    double* dec_ptr = dec.data_ptr<double>();
    const int64_t n = pix.numel();
    std::atomic<bool> invalid{false};
    auto compute_block = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t p = in[i];
            if (p < 0 || p >= npix) {
                invalid.store(true, std::memory_order_relaxed);
                ra_ptr[i] = 0.0;
                dec_ptr[i] = 0.0;
                continue;
            }
            const int64_t face_num = p / npface;
            const uint64_t ipf = static_cast<uint64_t>(p % npface);
            const int64_t ix = static_cast<int64_t>(compact_bits_u64(ipf));
            const int64_t iy = static_cast<int64_t>(compact_bits_u64(ipf >> 1));
            const int64_t jr = JRLL[face_num] * nside - ix - iy - 1;

            int64_t nr;
            int64_t kshift = 0;
            double z;
            if (jr < nside) {
                nr = jr;
                z = 1.0 - static_cast<double>(nr * nr) * fact2;
            } else if (jr > (3 * nside)) {
                nr = nl4 - jr;
                z = static_cast<double>(nr * nr) * fact2 - 1.0;
            } else {
                nr = nside;
                z = static_cast<double>(2 * nside - jr) * fact1;
                kshift = (jr - nside) & 1;
            }

            int64_t jp = (JPLL[face_num] * nr + ix - iy + 1 + kshift) / 2;
            if (jp > nl4) {
                jp -= nl4;
            }
            if (jp < 1) {
                jp += nl4;
            }

            const double phi = (static_cast<double>(jp) - 0.5 * (static_cast<double>(kshift) + 1.0))
                               * ((PI / 2.0) / static_cast<double>(nr));
            double ra = phi * RAD2DEG;
            if (ra >= 360.0) {
                ra -= 360.0;
            } else if (ra < 0.0) {
                ra += 360.0;
            }
            ra_ptr[i] = ra;
            dec_ptr[i] = std::asin(std::fmax(-1.0, std::fmin(1.0, z))) * RAD2DEG;
        }
    };
    if (n >= PARALLEL_MIN_ELEMS) {
        at::parallel_for(0, n, 8192, compute_block);
    } else {
        compute_block(0, n);
    }
    if (invalid.load(std::memory_order_relaxed)) {
        throw std::runtime_error("pixel index out of range for nside");
    }

    auto shape = pix_nest.sizes().vec();
    return {ra.reshape(shape), dec.reshape(shape)};
}

torch::Tensor healpix_ring2nest_cpu(int64_t nside, const torch::Tensor& pix_ring) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!pix_ring.device().is_cpu()) {
        throw std::runtime_error("healpix_ring2nest_cpu expects CPU tensor");
    }
    if (pix_ring.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_ring2nest_cpu expects int64 tensor");
    }

    auto pix = pix_ring.contiguous();
    auto out = torch::empty_like(pix);

    const int64_t npface = nside * nside;
    const int64_t ncap = 2 * nside * (nside - 1);
    const int64_t npix = 12 * nside * nside;
    const int64_t nl2 = 2 * nside;
    const int64_t nside_shift = ilog2_pow2(nside);
    const int64_t four_nside_shift = nside_shift + 2;
    const int64_t four_nside_mask = (4 * nside) - 1;

    const int64_t* in = pix.data_ptr<int64_t>();
    int64_t* dst = out.data_ptr<int64_t>();
    const int64_t n = pix.numel();
    if (n == 0) {
        return out;
    }
    std::vector<uint64_t> spread_x(static_cast<size_t>(nside));
    std::vector<uint64_t> spread_y(static_cast<size_t>(nside));
    for (int64_t i = 0; i < nside; ++i) {
        spread_x[static_cast<size_t>(i)] = spread_bits_u64(static_cast<uint64_t>(i));
        spread_y[static_cast<size_t>(i)] = spread_bits_u64(static_cast<uint64_t>(i)) << 1;
    }
    std::atomic<bool> invalid{false};

    auto compute_block = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t p = in[i];
            if (p < 0 || p >= npix) {
                invalid.store(true, std::memory_order_relaxed);
                dst[i] = -1;
                continue;
            }
            int64_t iring;
            int64_t iphi;
            int64_t kshift;
            int64_t nr;
            int64_t face_num;

            if (p < ncap) {
                iring = static_cast<int64_t>((1 + isqrt_u64(static_cast<uint64_t>(1 + 2 * p))) >> 1);
                iphi = (p + 1) - 2 * iring * (iring - 1);
                kshift = 0;
                nr = iring;
                face_num = (iphi - 1) / iring;
            } else if (p < (npix - ncap)) {
                const int64_t ip = p - ncap;
                iring = (ip >> four_nside_shift) + nside;
                iphi = (ip & four_nside_mask) + 1;
                kshift = (iring + nside) & 1;
                nr = nside;
                const int64_t ire = iring - nside + 1;
                const int64_t irm = nl2 + 2 - ire;
                const int64_t ifm = (iphi - (ire / 2) + nside - 1) >> nside_shift;
                const int64_t ifp = (iphi - (irm / 2) + nside - 1) >> nside_shift;
                if (ifp == ifm) {
                    face_num = ifp | 4;
                } else if (ifp < ifm) {
                    face_num = ifp;
                } else {
                    face_num = ifm + 8;
                }
            } else {
                const int64_t ip = npix - p;
                const int64_t irs = static_cast<int64_t>((1 + isqrt_u64(static_cast<uint64_t>(2 * ip - 1))) >> 1);
                iphi = 4 * irs + 1 - (ip - 2 * irs * (irs - 1));
                iring = 2 * nl2 - irs;
                kshift = 0;
                nr = irs;
                face_num = 8 + (iphi - 1) / irs;
            }

            const int64_t irt = iring - JRLL[face_num] * nside + 1;
            int64_t ipt = 2 * iphi - JPLL[face_num] * nr - kshift - 1;
            if (ipt >= nl2) {
                ipt -= 8 * nside;
            }

            const int64_t ix = floor_div2(ipt - irt);
            const int64_t iy = floor_div2(-(ipt + irt));
            if (ix < 0 || ix >= nside || iy < 0 || iy >= nside) {
                invalid.store(true, std::memory_order_relaxed);
                dst[i] = -1;
                continue;
            }
            dst[i] = face_num * npface
                     + static_cast<int64_t>(spread_x[static_cast<size_t>(ix)])
                     + static_cast<int64_t>(spread_y[static_cast<size_t>(iy)]);
        }
    };
    if (n >= PARALLEL_MIN_ELEMS) {
        at::parallel_for(0, n, 16384, compute_block);
    } else {
        compute_block(0, n);
    }
    if (invalid.load(std::memory_order_relaxed)) {
        throw std::runtime_error("pixel index out of range for nside");
    }

    return out;
}

torch::Tensor healpix_nest2ring_cpu(int64_t nside, const torch::Tensor& pix_nest) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!pix_nest.device().is_cpu()) {
        throw std::runtime_error("healpix_nest2ring_cpu expects CPU tensor");
    }
    if (pix_nest.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_nest2ring_cpu expects int64 tensor");
    }

    auto pix = pix_nest.contiguous();
    auto out = torch::empty_like(pix);

    const int64_t npface = nside * nside;
    const int64_t ncap = 2 * nside * (nside - 1);
    const int64_t npix = 12 * nside * nside;
    const int64_t nl4 = 4 * nside;

    const int64_t* in = pix.data_ptr<int64_t>();
    int64_t* dst = out.data_ptr<int64_t>();
    const int64_t n = pix.numel();
    if (n == 0) {
        return out;
    }
    std::atomic<bool> invalid{false};
    auto compute_block = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t p = in[i];
            if (p < 0 || p >= npix) {
                invalid.store(true, std::memory_order_relaxed);
                dst[i] = -1;
                continue;
            }
            const int64_t face_num = p / npface;
            const uint64_t ipf = static_cast<uint64_t>(p % npface);
            const int64_t ix = static_cast<int64_t>(compact_bits_u64(ipf));
            const int64_t iy = static_cast<int64_t>(compact_bits_u64(ipf >> 1));

            const int64_t jr = JRLL[face_num] * nside - ix - iy - 1;
            int64_t nr;
            int64_t n_before;
            int64_t kshift = 0;
            if (jr < nside) {
                nr = jr;
                n_before = 2 * nr * (nr - 1);
            } else if (jr > (3 * nside)) {
                nr = nl4 - jr;
                n_before = npix - 2 * (nr + 1) * nr;
            } else {
                nr = nside;
                n_before = ncap + (jr - nside) * nl4;
                kshift = (jr - nside) & 1;
            }

            int64_t jp = (JPLL[face_num] * nr + ix - iy + 1 + kshift) / 2;
            if (jp > nl4) {
                jp -= nl4;
            }
            if (jp < 1) {
                jp += nl4;
            }
            dst[i] = n_before + jp - 1;
        }
    };
    if (n >= PARALLEL_MIN_ELEMS) {
        at::parallel_for(0, n, 16384, compute_block);
    } else {
        compute_block(0, n);
    }
    if (invalid.load(std::memory_order_relaxed)) {
        throw std::runtime_error("pixel index out of range for nside");
    }

    return out;
}

static torch::Tensor healpix_neighbors_cpu_impl(
    int64_t nside,
    const torch::Tensor& pix_in,
    bool input_nest,
    bool output_nest
) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!pix_in.device().is_cpu()) {
        throw std::runtime_error("healpix_neighbors_cpu expects CPU tensor");
    }
    if (pix_in.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_neighbors_cpu expects int64 tensor");
    }

    auto pix = pix_in.contiguous().reshape({-1});
    const int64_t n = pix.numel();
    auto out = torch::full({n, 8}, static_cast<int64_t>(-1), torch::TensorOptions().dtype(torch::kInt64));
    if (n == 0) {
        auto shape = pix_in.sizes().vec();
        shape.push_back(8);
        return out.reshape(shape);
    }

    const int64_t npface = nside * nside;
    const int64_t ncap = 2 * nside * (nside - 1);
    const int64_t npix = 12 * nside * nside;
    const int64_t nl2 = 2 * nside;
    const int64_t nl4 = 4 * nside;
    const int64_t nside_shift = ilog2_pow2(nside);
    const int64_t four_nside_shift = nside_shift + 2;
    const int64_t four_nside_mask = (4 * nside) - 1;
    const int64_t nsm1 = nside - 1;
    std::vector<uint64_t> spread_x;
    std::vector<uint64_t> spread_y;
    if (output_nest) {
        spread_x.resize(static_cast<size_t>(nside));
        spread_y.resize(static_cast<size_t>(nside));
        for (int64_t i = 0; i < nside; ++i) {
            spread_x[static_cast<size_t>(i)] = spread_bits_u64(static_cast<uint64_t>(i));
            spread_y[static_cast<size_t>(i)] = spread_bits_u64(static_cast<uint64_t>(i)) << 1;
        }
    }

    const int64_t* in = pix.data_ptr<int64_t>();
    int64_t* dst = out.data_ptr<int64_t>();
    const int64_t min_pix = pix.min().item<int64_t>();
    const int64_t max_pix = pix.max().item<int64_t>();
    if (min_pix < 0 || max_pix >= npix) {
        throw std::runtime_error("pixel index out of range for nside");
    }

    auto compute_block = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t p = in[i];
            int64_t ix;
            int64_t iy;
            int64_t face_num;
            if (input_nest) {
                nest_to_xyf(npface, p, ix, iy, face_num);
            } else {
                ring_to_xyf(
                    nside,
                    ncap,
                    npix,
                    nl2,
                    nside_shift,
                    four_nside_shift,
                    four_nside_mask,
                    p,
                    ix,
                    iy,
                    face_num
                );
            }

            const int64_t row_off = i * 8;
            if (ix > 0 && ix < nsm1 && iy > 0 && iy < nsm1) {
                if (output_nest) {
                    const uint64_t face_base = static_cast<uint64_t>(face_num * npface);
                    const uint64_t sxm = spread_x[static_cast<size_t>(ix - 1)];
                    const uint64_t sx0 = spread_x[static_cast<size_t>(ix)];
                    const uint64_t sxp = spread_x[static_cast<size_t>(ix + 1)];
                    const uint64_t sym = spread_y[static_cast<size_t>(iy - 1)];
                    const uint64_t sy0 = spread_y[static_cast<size_t>(iy)];
                    const uint64_t syp = spread_y[static_cast<size_t>(iy + 1)];

                    dst[row_off + 0] = static_cast<int64_t>(face_base + sxm + sy0); // SW
                    dst[row_off + 1] = static_cast<int64_t>(face_base + sxm + syp); // W
                    dst[row_off + 2] = static_cast<int64_t>(face_base + sx0 + syp); // NW
                    dst[row_off + 3] = static_cast<int64_t>(face_base + sxp + syp); // N
                    dst[row_off + 4] = static_cast<int64_t>(face_base + sxp + sy0); // NE
                    dst[row_off + 5] = static_cast<int64_t>(face_base + sxp + sym); // E
                    dst[row_off + 6] = static_cast<int64_t>(face_base + sx0 + sym); // SE
                    dst[row_off + 7] = static_cast<int64_t>(face_base + sxm + sym); // S
                } else {
                    dst[row_off + 0] = xyf_to_ring(nside, ncap, npix, nl4, ix - 1, iy, face_num);
                    dst[row_off + 1] = xyf_to_ring(nside, ncap, npix, nl4, ix - 1, iy + 1, face_num);
                    dst[row_off + 2] = xyf_to_ring(nside, ncap, npix, nl4, ix, iy + 1, face_num);
                    dst[row_off + 3] = xyf_to_ring(nside, ncap, npix, nl4, ix + 1, iy + 1, face_num);
                    dst[row_off + 4] = xyf_to_ring(nside, ncap, npix, nl4, ix + 1, iy, face_num);
                    dst[row_off + 5] = xyf_to_ring(nside, ncap, npix, nl4, ix + 1, iy - 1, face_num);
                    dst[row_off + 6] = xyf_to_ring(nside, ncap, npix, nl4, ix, iy - 1, face_num);
                    dst[row_off + 7] = xyf_to_ring(nside, ncap, npix, nl4, ix - 1, iy - 1, face_num);
                }
                continue;
            }

            for (int64_t m = 0; m < 8; ++m) {
                int64_t x = ix + NB_XOFFSET[m];
                int64_t y = iy + NB_YOFFSET[m];
                int64_t f = face_num;

                if (x < 0 || x > nsm1 || y < 0 || y > nsm1) {
                    int64_t nbnum = 4;
                    if (x < 0) {
                        x += nside;
                        nbnum -= 1;
                    } else if (x >= nside) {
                        x -= nside;
                        nbnum += 1;
                    }
                    if (y < 0) {
                        y += nside;
                        nbnum -= 3;
                    } else if (y >= nside) {
                        y -= nside;
                        nbnum += 3;
                    }

                    f = NB_FACEARRAY[nbnum][face_num];
                    if (f < 0) {
                        dst[row_off + m] = -1;
                        continue;
                    }

                    const int64_t bits = NB_SWAPARRAY[nbnum][face_num >> 2];
                    if ((bits & 1) != 0) {
                        x = nside - x - 1;
                    }
                    if ((bits & 2) != 0) {
                        y = nside - y - 1;
                    }
                    if ((bits & 4) != 0) {
                        const int64_t t = x;
                        x = y;
                        y = t;
                    }
                }

                if (output_nest) {
                    dst[row_off + m] = static_cast<int64_t>(
                        static_cast<uint64_t>(f * npface) + spread_x[static_cast<size_t>(x)] + spread_y[static_cast<size_t>(y)]
                    );
                } else {
                    dst[row_off + m] = xyf_to_ring(nside, ncap, npix, nl4, x, y, f);
                }
            }
        }
    };

    if (n >= PARALLEL_MIN_ELEMS) {
        at::parallel_for(0, n, 8192, compute_block);
    } else {
        compute_block(0, n);
    }

    auto shape = pix_in.sizes().vec();
    shape.push_back(8);
    return out.reshape(shape);
}

torch::Tensor healpix_neighbors_ring_cpu(int64_t nside, const torch::Tensor& pix_ring) {
    return healpix_neighbors_cpu_impl(nside, pix_ring, false, false);
}

torch::Tensor healpix_neighbors_nested_cpu(int64_t nside, const torch::Tensor& pix_nest) {
    return healpix_neighbors_cpu_impl(nside, pix_nest, true, true);
}

torch::Tensor healpix_scalar_alm2map_direct_cpu(
    const torch::Tensor& alm_values,
    const torch::Tensor& cos_theta,
    const torch::Tensor& phi,
    int64_t lmax,
    int64_t mmax
) {
    if (!alm_values.device().is_cpu() || !cos_theta.device().is_cpu() || !phi.device().is_cpu()) {
        throw std::runtime_error("healpix_scalar_alm2map_direct_cpu expects CPU tensors");
    }
    if (alm_values.scalar_type() != torch::kComplexDouble) {
        throw std::runtime_error("healpix_scalar_alm2map_direct_cpu expects complex128 alm tensor");
    }
    if (cos_theta.scalar_type() != torch::kDouble || phi.scalar_type() != torch::kDouble) {
        throw std::runtime_error("healpix_scalar_alm2map_direct_cpu expects float64 cos_theta/phi tensors");
    }
    if (lmax < 0) {
        throw std::runtime_error("lmax must be non-negative");
    }
    if (mmax < 0 || mmax > lmax) {
        throw std::runtime_error("mmax must satisfy 0 <= mmax <= lmax");
    }

    torch::Tensor rows;
    if (alm_values.dim() == 1) {
        rows = alm_values.contiguous().reshape({1, alm_values.numel()});
    } else if (alm_values.dim() == 2) {
        rows = alm_values.contiguous();
    } else {
        throw std::runtime_error("alm_values must be rank-1 or rank-2");
    }

    auto x = cos_theta.contiguous().reshape({-1});
    auto ph = phi.contiguous().reshape({-1});
    if (x.numel() != ph.numel()) {
        throw std::runtime_error("cos_theta/phi length mismatch");
    }

    const int64_t npix = x.numel();
    const int64_t nmaps = rows.size(0);
    const int64_t nalm = rows.size(1);
    const int64_t expected_nalm = (mmax + 1) * (lmax + 1) - (mmax * (mmax + 1)) / 2;
    if (nalm != expected_nalm) {
        throw std::runtime_error("alm length does not match lmax/mmax");
    }

    std::vector<double> ylm_norms(static_cast<size_t>(nalm), 0.0);
    int64_t idx = 0;
    for (int64_t m = 0; m <= mmax; ++m) {
        for (int64_t ell = m; ell <= lmax; ++ell) {
            const double log_ratio = std::lgamma(static_cast<double>(ell - m + 1))
                                     - std::lgamma(static_cast<double>(ell + m + 1));
            const double pref = (2.0 * static_cast<double>(ell) + 1.0) / (4.0 * PI);
            ylm_norms[static_cast<size_t>(idx)] = std::sqrt(pref * std::exp(log_ratio));
            ++idx;
        }
    }

    auto out = torch::zeros({nmaps, npix}, torch::TensorOptions().dtype(torch::kDouble));
    const auto* alm_ptr = rows.data_ptr<c10::complex<double>>();
    const auto* x_ptr = x.data_ptr<double>();
    const auto* phi_ptr = ph.data_ptr<double>();
    auto* out_ptr = out.data_ptr<double>();

    at::parallel_for(0, npix, 128, [&](int64_t begin, int64_t end) {
        std::vector<double> acc(static_cast<size_t>(nmaps), 0.0);
        for (int64_t p = begin; p < end; ++p) {
            std::fill(acc.begin(), acc.end(), 0.0);

            const double xp = std::max(-1.0, std::min(1.0, x_ptr[p]));
            const double phi_p = phi_ptr[p];
            const double cphi = std::cos(phi_p);
            const double sphi = std::sin(phi_p);
            double cos_m = 1.0;
            double sin_m = 0.0;
            const double somx2 = std::sqrt(std::max(0.0, 1.0 - xp * xp));
            double pmm = 1.0;
            int64_t k = 0;

            for (int64_t m = 0; m <= mmax; ++m) {
                if (m > 0) {
                    const double cos_next = cos_m * cphi - sin_m * sphi;
                    const double sin_next = sin_m * cphi + cos_m * sphi;
                    cos_m = cos_next;
                    sin_m = sin_next;
                    pmm *= -static_cast<double>(2 * m - 1) * somx2;
                }

                double p_lm = pmm;
                double p_lm_prev = 0.0;
                const double m_weight = (m == 0) ? 1.0 : 2.0;
                for (int64_t ell = m; ell <= lmax; ++ell) {
                    const double y_scale = ylm_norms[static_cast<size_t>(k)] * p_lm;
                    const double y_re = y_scale * cos_m;
                    const double y_im = y_scale * sin_m;

                    for (int64_t r = 0; r < nmaps; ++r) {
                        const auto a = alm_ptr[r * nalm + k];
                        const double term = static_cast<double>(a.real()) * y_re
                                            - static_cast<double>(a.imag()) * y_im;
                        acc[static_cast<size_t>(r)] += m_weight * term;
                    }

                    if (ell < lmax) {
                        double p_next;
                        if (ell == m) {
                            p_next = xp * (2.0 * static_cast<double>(m) + 1.0) * pmm;
                        } else {
                            p_next = ((2.0 * static_cast<double>(ell) + 1.0) * xp * p_lm
                                      - static_cast<double>(ell + m) * p_lm_prev)
                                     / static_cast<double>(ell - m + 1);
                        }
                        p_lm_prev = p_lm;
                        p_lm = p_next;
                    }
                    ++k;
                }
            }

            for (int64_t r = 0; r < nmaps; ++r) {
                out_ptr[r * npix + p] = acc[static_cast<size_t>(r)];
            }
        }
    });

    return out;
}

template <typename Fn>
static void parallel_m_ranges(int64_t mmax, Fn&& fn);
static bool sht_profile_enabled();

#if defined(__APPLE__)
static const std::vector<std::complex<double>>& dft_twiddles(int n, int sign) {
    // thread_local: polar FFTs run on SpinMPool workers; a shared mutex map would contend.
    thread_local std::map<uint64_t, std::vector<std::complex<double>>> cache;
    const uint64_t key =
        (static_cast<uint64_t>(static_cast<uint32_t>(n)) << 1)
        | static_cast<uint64_t>(sign < 0 ? 0 : 1);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    std::vector<std::complex<double>> w(static_cast<size_t>(n));
    const double ang0 = static_cast<double>(sign) * 2.0 * PI / static_cast<double>(n);
    for (int j = 0; j < n; ++j) {
        const double th = ang0 * static_cast<double>(j);
        w[static_cast<size_t>(j)] = {std::cos(th), std::sin(th)};
    }
    return cache.emplace(key, std::move(w)).first->second;
}

static void fft_pow2_inplace(std::complex<double>* a, int n, int sign) {
#if defined(__APPLE__)
    int log2n = 0;
    for (int t = n; t > 1; t >>= 1) {
        ++log2n;
    }
    static std::mutex setup_mu;
    static FFTSetupD setups[32] = {};
    FFTSetupD setup = nullptr;
    {
        if (log2n > 0 && log2n < 32) {
            setup = setups[log2n];
        }
        if (setup == nullptr && log2n > 0 && log2n < 32) {
            std::lock_guard<std::mutex> lock(setup_mu);
            if (setups[log2n] == nullptr) {
                setups[log2n] = vDSP_create_fftsetupD(
                    static_cast<vDSP_Length>(log2n), kFFTRadix2
                );
            }
            setup = setups[log2n];
        }
    }
    if (setup != nullptr && n >= 32) {
        thread_local std::vector<double> re;
        thread_local std::vector<double> im;
        if (static_cast<int>(re.size()) < n) {
            re.resize(static_cast<size_t>(n));
            im.resize(static_cast<size_t>(n));
        }
        DSPDoubleSplitComplex split{re.data(), im.data()};
        vDSP_ctozD(
            reinterpret_cast<const DSPDoubleComplex*>(a),
            2,
            &split,
            1,
            static_cast<vDSP_Length>(n)
        );
        vDSP_fft_zipD(
            setup,
            &split,
            1,
            static_cast<vDSP_Length>(log2n),
            sign < 0 ? FFT_FORWARD : FFT_INVERSE
        );
        vDSP_ztocD(
            &split,
            1,
            reinterpret_cast<DSPDoubleComplex*>(a),
            2,
            static_cast<vDSP_Length>(n)
        );
        return;
    }
#endif
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const auto& wtab = dft_twiddles(len, sign);
        const int h = len >> 1;
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < h; ++j) {
                const std::complex<double> u = a[i + j];
                const std::complex<double> v = a[i + j + h] * wtab[static_cast<size_t>(j)];
                a[i + j] = u + v;
                a[i + j + h] = u - v;
            }
        }
    }
}
#endif

// Polar rings have unique nph. Equatorial CPU rings use the same in-process
// FFT (pocketfft; Accelerate radix-2 on Apple). CUDA/MPS stay on ATen.
// Each call is nthreads=1; SpinMPool parallelizes rings.
static void fft_outofplace(
    const std::complex<double>* in,
    int n,
    int in_stride,
    std::complex<double>* out,
    int sign
) {
    if (n <= 1) {
        if (n == 1) {
            out[0] = in[0];
        }
        return;
    }
#if defined(__APPLE__)
    if (in_stride == 1 && (n & (n - 1)) == 0) {
        std::copy(in, in + n, out);
        fft_pow2_inplace(out, n, sign);
        return;
    }
#endif
    const pocketfft::shape_t shape{static_cast<size_t>(n)};
    const pocketfft::stride_t stride_in{
        static_cast<ptrdiff_t>(in_stride)
        * static_cast<ptrdiff_t>(sizeof(std::complex<double>))
    };
    const pocketfft::stride_t stride_out{
        static_cast<ptrdiff_t>(sizeof(std::complex<double>))
    };
    const pocketfft::shape_t axes{0};
    pocketfft::c2c(
        shape,
        stride_in,
        stride_out,
        axes,
        sign < 0,
        in,
        out,
        1.0,
        1
    );
}

torch::Tensor healpix_ring_fourier_modes_cpu(
    const torch::Tensor& rows,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    int64_t mmax
) {
    if (!rows.device().is_cpu() || !starts.device().is_cpu() || !lengths.device().is_cpu()) {
        throw std::runtime_error("healpix_ring_fourier_modes_cpu expects CPU tensors");
    }
    if (rows.scalar_type() != torch::kComplexDouble) {
        throw std::runtime_error("healpix_ring_fourier_modes_cpu expects complex128 rows");
    }
    if (starts.scalar_type() != torch::kInt64 || lengths.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_ring_fourier_modes_cpu expects int64 starts/lengths");
    }
    if (rows.dim() != 2 || starts.dim() != 1 || lengths.dim() != 1) {
        throw std::runtime_error("rows must be rank-2; starts/lengths must be rank-1");
    }
    if (starts.numel() != lengths.numel()) {
        throw std::runtime_error("starts/lengths size mismatch");
    }
    if (mmax < 0) {
        throw std::runtime_error("mmax must be non-negative");
    }
    const int64_t nrows = rows.size(0);
    const int64_t npix = rows.size(1);
    const int64_t nrings = starts.numel();
    const int64_t mcount = mmax + 1;
    if (nrings == 0) {
        return torch::zeros({nrows, mcount, 0}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    }

    auto rows_c = rows.contiguous();
    auto starts_c = starts.contiguous();
    auto lengths_c = lengths.contiguous();
    auto out = torch::zeros({nrows, mcount, nrings}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    const auto* starts_ptr = starts_c.data_ptr<int64_t>();
    const auto* lengths_ptr = lengths_c.data_ptr<int64_t>();
    const auto* rows_ptr = reinterpret_cast<const std::complex<double>*>(
        rows_c.data_ptr<c10::complex<double>>()
    );
    auto* out_ptr = reinterpret_cast<std::complex<double>*>(
        out.data_ptr<c10::complex<double>>()
    );

    int64_t ring = 0;
    std::vector<int64_t> polar_r;
    std::vector<int64_t> polar_pix0;
    std::vector<int> polar_nph;
    int max_polar_nph = 1;
    while (ring < nrings) {
        const int64_t run_start_ring = ring;
        const int64_t nph = lengths_ptr[ring];
        const int64_t run_pix_start = starts_ptr[ring];
        if (nph <= 0 || run_pix_start < 0 || (run_pix_start + nph) > npix) {
            throw std::runtime_error("invalid ring layout in healpix_ring_fourier_modes_cpu");
        }
        int64_t pix_cursor = run_pix_start + nph;
        ++ring;
        while (ring < nrings) {
            const int64_t next_start = starts_ptr[ring];
            const int64_t next_nph = lengths_ptr[ring];
            if (next_nph <= 0 || next_start < 0 || (next_start + next_nph) > npix) {
                throw std::runtime_error("invalid ring layout in healpix_ring_fourier_modes_cpu");
            }
            if (next_nph != nph || next_start != pix_cursor) {
                break;
            }
            pix_cursor += nph;
            ++ring;
        }
        const int64_t run_ring_count = ring - run_start_ring;
        if (run_ring_count == 1) {
            polar_r.push_back(run_start_ring);
            polar_pix0.push_back(run_pix_start);
            polar_nph.push_back(static_cast<int>(nph));
            if (nph > max_polar_nph) {
                max_polar_nph = static_cast<int>(nph);
            }
            continue;
        }

        auto block = rows_c.narrow(1, run_pix_start, run_ring_count * nph).view({nrows, run_ring_count, nph});
        auto fft_block = at::fft_fft(block, std::nullopt, -1, std::nullopt);

        torch::Tensor fft_sel;
        if (nph > mmax) {
            fft_sel = fft_block.narrow(2, 0, mcount);
        } else {
            const auto& idx = g_ring_alias_idx_cache.plus(nph, mcount);
            fft_sel = fft_block.index_select(2, idx);
        }
        out.narrow(2, run_start_ring, run_ring_count).copy_(fft_sel.permute({0, 2, 1}));
    }

    const int64_t npolar = static_cast<int64_t>(polar_r.size());
    if (npolar > 0) {
        const int64_t* polar_r_ptr = polar_r.data();
        const int64_t* polar_pix0_ptr = polar_pix0.data();
        const int* polar_nph_ptr = polar_nph.data();
        parallel_m_ranges(npolar - 1, [&](int64_t i0, int64_t stride) {
            std::vector<std::complex<double>> spec(static_cast<size_t>(max_polar_nph));
            for (int64_t i = i0; i < npolar; i += stride) {
                const int nph = polar_nph_ptr[static_cast<size_t>(i)];
                const int64_t pix0 = polar_pix0_ptr[static_cast<size_t>(i)];
                const int64_t r = polar_r_ptr[static_cast<size_t>(i)];
                for (int64_t row = 0; row < nrows; ++row) {
                    fft_outofplace(rows_ptr + row * npix + pix0, nph, 1, spec.data(), -1);
                    for (int64_t m = 0; m < mcount; ++m) {
                        out_ptr[(row * mcount + m) * nrings + r] = spec[static_cast<size_t>(m % nph)];
                    }
                }
            }
        });
    }

    return out;
}

torch::Tensor healpix_ring_fourier_modes_spin_conj_cpu(
    const torch::Tensor& row,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    int64_t mmax,
    const c10::complex<double>* phase_ptr = nullptr,
    bool from_qu = false
) {
    if (!row.device().is_cpu() || !starts.device().is_cpu() || !lengths.device().is_cpu()) {
        throw std::runtime_error("healpix_ring_fourier_modes_spin_conj_cpu expects CPU tensors");
    }
    if (starts.scalar_type() != torch::kInt64 || lengths.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_ring_fourier_modes_spin_conj_cpu expects int64 starts/lengths");
    }
    if (starts.dim() != 1 || lengths.dim() != 1) {
        throw std::runtime_error("starts/lengths must be rank-1");
    }
    if (starts.numel() != lengths.numel()) {
        throw std::runtime_error("starts/lengths size mismatch");
    }
    if (mmax < 0) {
        throw std::runtime_error("mmax must be non-negative");
    }

    torch::Tensor row_c;
    torch::Tensor q_c;
    torch::Tensor u_c;
    const std::complex<double>* row_ptr = nullptr;
    const double* q_ptr = nullptr;
    const double* u_ptr = nullptr;
    int64_t npix = 0;
    if (from_qu) {
        if (row.dim() != 2 || row.size(0) != 2) {
            throw std::runtime_error("healpix_ring_fourier_modes_spin_conj_cpu from_qu expects maps (2, npix)");
        }
        if (row.scalar_type() != torch::kDouble) {
            throw std::runtime_error("healpix_ring_fourier_modes_spin_conj_cpu from_qu expects float64 maps");
        }
        q_c = row.select(0, 0).contiguous();
        u_c = row.select(0, 1).contiguous();
        q_ptr = q_c.data_ptr<double>();
        u_ptr = u_c.data_ptr<double>();
        npix = q_c.numel();
        if (u_c.numel() != npix) {
            throw std::runtime_error("healpix_ring_fourier_modes_spin_conj_cpu Q/U length mismatch");
        }
    } else {
        if (row.scalar_type() != torch::kComplexDouble) {
            throw std::runtime_error("healpix_ring_fourier_modes_spin_conj_cpu expects complex128 row");
        }
        if (row.dim() != 1) {
            throw std::runtime_error("row must be rank-1; starts/lengths must be rank-1");
        }
        row_c = row.contiguous();
        npix = row_c.size(0);
        row_ptr = reinterpret_cast<const std::complex<double>*>(
            row_c.data_ptr<c10::complex<double>>()
        );
    }
    const int64_t nrings = starts.numel();
    const int64_t mcount = mmax + 1;
    if (nrings == 0) {
        return torch::zeros({2, mcount, 0}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    }
    auto starts_c = starts.contiguous();
    auto lengths_c = lengths.contiguous();
    const auto* starts_ptr = starts_c.data_ptr<int64_t>();
    const auto* lengths_ptr = lengths_c.data_ptr<int64_t>();

    auto out = torch::zeros(
        {2, mcount, nrings},
        torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU)
    );
    auto out_plus = out.select(0, 0);
    auto out_minus = out.select(0, 1);
    auto* plus_ptr = out_plus.data_ptr<c10::complex<double>>();
    auto* minus_ptr = out_minus.data_ptr<c10::complex<double>>();
    const auto cplx_opts = torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU);

    int64_t ring = 0;
    std::vector<int64_t> polar_r;
    std::vector<int64_t> polar_pix0;
    std::vector<int> polar_nph;
    int max_polar_nph = 1;
    while (ring < nrings) {
        const int64_t run_start_ring = ring;
        const int64_t nph = lengths_ptr[ring];
        const int64_t run_pix_start = starts_ptr[ring];
        if (nph <= 0 || run_pix_start < 0 || (run_pix_start + nph) > npix) {
            throw std::runtime_error("invalid ring layout in healpix_ring_fourier_modes_spin_conj_cpu");
        }
        int64_t pix_cursor = run_pix_start + nph;
        ++ring;
        while (ring < nrings) {
            const int64_t next_start = starts_ptr[ring];
            const int64_t next_nph = lengths_ptr[ring];
            if (next_nph <= 0 || next_start < 0 || (next_start + next_nph) > npix) {
                throw std::runtime_error("invalid ring layout in healpix_ring_fourier_modes_spin_conj_cpu");
            }
            if (next_nph != nph || next_start != pix_cursor) {
                break;
            }
            pix_cursor += nph;
            ++ring;
        }
        const int64_t run_ring_count = ring - run_start_ring;
        if (run_ring_count == 1) {
            polar_r.push_back(run_start_ring);
            polar_pix0.push_back(run_pix_start);
            polar_nph.push_back(static_cast<int>(nph));
            if (nph > max_polar_nph) {
                max_polar_nph = static_cast<int>(nph);
            }
            continue;
        }

        torch::Tensor fft_block;
        if (row_ptr != nullptr) {
            auto block = row_c.narrow(0, run_pix_start, run_ring_count * nph).view({run_ring_count, nph});
            fft_block = at::fft_fft(block, std::nullopt, -1, std::nullopt).contiguous();
        } else {
            auto block = torch::empty({run_ring_count, nph}, cplx_opts);
            auto* bp = reinterpret_cast<std::complex<double>*>(block.data_ptr<c10::complex<double>>());
            const int64_t n = run_ring_count * nph;
            for (int64_t j = 0; j < n; ++j) {
                const int64_t p = run_pix_start + j;
                bp[static_cast<size_t>(j)] = {q_ptr[p], u_ptr[p]};
            }
            fft_block = at::fft_fft(block, std::nullopt, -1, std::nullopt).contiguous();
        }
        const auto* fptr = fft_block.data_ptr<c10::complex<double>>();
        for (int64_t ir = 0; ir < run_ring_count; ++ir) {
            const int64_t r = run_start_ring + ir;
            const auto* row_fft = fptr + ir * nph;
            for (int64_t m = 0; m < mcount; ++m) {
                const int64_t k = m % nph;
                const int64_t kneg = (k == 0) ? 0 : (nph - k);
                auto vp = row_fft[k];
                auto vm = row_fft[kneg];
                vm = c10::complex<double>(vm.real(), -vm.imag());
                if (phase_ptr != nullptr) {
                    const auto ph = phase_ptr[m * nrings + r];
                    vp = vp * ph;
                    vm = vm * ph;
                }
                plus_ptr[m * nrings + r] = vp;
                minus_ptr[m * nrings + r] = vm;
            }
        }
    }

    const int64_t npolar = static_cast<int64_t>(polar_r.size());
    if (npolar > 0) {
        const int64_t* polar_r_ptr = polar_r.data();
        const int64_t* polar_pix0_ptr = polar_pix0.data();
        const int* polar_nph_ptr = polar_nph.data();
        const std::complex<double>* row_ptr_cap = row_ptr;
        const double* q_ptr_cap = q_ptr;
        const double* u_ptr_cap = u_ptr;
        parallel_m_ranges(npolar - 1, [&](int64_t i0, int64_t stride) {
            std::vector<std::complex<double>> spec(static_cast<size_t>(max_polar_nph));
            std::vector<std::complex<double>> packed(static_cast<size_t>(max_polar_nph));
            for (int64_t i = i0; i < npolar; i += stride) {
                const int nph = polar_nph_ptr[static_cast<size_t>(i)];
                const int64_t pix0 = polar_pix0_ptr[static_cast<size_t>(i)];
                if (row_ptr_cap != nullptr) {
                    fft_outofplace(row_ptr_cap + pix0, nph, 1, spec.data(), -1);
                } else {
                    for (int j = 0; j < nph; ++j) {
                        const int64_t p = pix0 + j;
                        packed[static_cast<size_t>(j)] = {q_ptr_cap[p], u_ptr_cap[p]};
                    }
                    fft_outofplace(packed.data(), nph, 1, spec.data(), -1);
                }
                const int64_t r = polar_r_ptr[static_cast<size_t>(i)];
                for (int64_t m = 0; m < mcount; ++m) {
                    const int k = static_cast<int>(m % nph);
                    const int kneg = (k == 0) ? 0 : (nph - k);
                    auto vp = spec[static_cast<size_t>(k)];
                    auto vm = spec[static_cast<size_t>(kneg)];
                    vm = std::conj(vm);
                    if (phase_ptr != nullptr) {
                        const auto ph = phase_ptr[m * nrings + r];
                        const std::complex<double> w(ph.real(), ph.imag());
                        vp *= w;
                        vm *= w;
                    }
                    plus_ptr[m * nrings + r] = c10::complex<double>(vp.real(), vp.imag());
                    minus_ptr[m * nrings + r] = c10::complex<double>(vm.real(), vm.imag());
                }
            }
        });
    }

    return out;
}

torch::Tensor healpix_ring_fourier_synthesis_cpu(
    const torch::Tensor& s_modes,
    const torch::Tensor& starts,
    const torch::Tensor& lengths
) {
    if (!s_modes.device().is_cpu() || !starts.device().is_cpu() || !lengths.device().is_cpu()) {
        throw std::runtime_error("healpix_ring_fourier_synthesis_cpu expects CPU tensors");
    }
    if (s_modes.scalar_type() != torch::kComplexDouble) {
        throw std::runtime_error("healpix_ring_fourier_synthesis_cpu expects complex128 s_modes");
    }
    if (starts.scalar_type() != torch::kInt64 || lengths.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_ring_fourier_synthesis_cpu expects int64 starts/lengths");
    }
    if (starts.dim() != 1 || lengths.dim() != 1) {
        throw std::runtime_error("starts/lengths must be rank-1");
    }
    if (starts.numel() != lengths.numel()) {
        throw std::runtime_error("starts/lengths size mismatch");
    }

    torch::Tensor s_batched;
    bool squeeze_out = false;
    if (s_modes.dim() == 2) {
        s_batched = s_modes.contiguous().unsqueeze(0);
        squeeze_out = true;
    } else if (s_modes.dim() == 3) {
        s_batched = s_modes.contiguous();
    } else {
        throw std::runtime_error("s_modes must be rank-2 or rank-3");
    }

    const int64_t nrows = s_batched.size(0);
    const int64_t mcount = s_batched.size(1);
    const int64_t nrings = s_batched.size(2);
    if (nrings != starts.numel()) {
        throw std::runtime_error("s_modes ring dimension does not match starts/lengths");
    }
    if (mcount <= 0) {
        throw std::runtime_error("s_modes m dimension must be positive");
    }

    auto starts_c = starts.contiguous();
    auto lengths_c = lengths.contiguous();
    const auto* starts_ptr = starts_c.data_ptr<int64_t>();
    const auto* lengths_ptr = lengths_c.data_ptr<int64_t>();
    const int64_t npix = starts_ptr[nrings - 1] + lengths_ptr[nrings - 1];
    if (npix < 0) {
        throw std::runtime_error("invalid npix in healpix_ring_fourier_synthesis_cpu");
    }

    auto out = torch::zeros({nrows, npix}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    const auto cplx_opts = torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU);
    const int64_t mmax = mcount - 1;
    const auto* s_ptr = reinterpret_cast<const std::complex<double>*>(
        s_batched.data_ptr<c10::complex<double>>()
    );
    auto* out_ptr = reinterpret_cast<std::complex<double>*>(
        out.data_ptr<c10::complex<double>>()
    );

    int64_t ring = 0;
    std::vector<int64_t> polar_r;
    std::vector<int64_t> polar_pix0;
    std::vector<int> polar_nph;
    int max_polar_nph = 1;
    while (ring < nrings) {
        const int64_t run_start_ring = ring;
        const int64_t nph = lengths_ptr[ring];
        const int64_t run_pix_start = starts_ptr[ring];
        if (nph <= 0 || run_pix_start < 0 || (run_pix_start + nph) > npix) {
            throw std::runtime_error("invalid ring layout in healpix_ring_fourier_synthesis_cpu");
        }
        int64_t pix_cursor = run_pix_start + nph;
        ++ring;
        while (ring < nrings) {
            const int64_t next_start = starts_ptr[ring];
            const int64_t next_nph = lengths_ptr[ring];
            if (next_nph <= 0 || next_start < 0 || (next_start + next_nph) > npix) {
                throw std::runtime_error("invalid ring layout in healpix_ring_fourier_synthesis_cpu");
            }
            if (next_nph != nph || next_start != pix_cursor) {
                break;
            }
            pix_cursor += nph;
            ++ring;
        }
        const int64_t run_ring_count = ring - run_start_ring;
        if (run_ring_count == 1) {
            polar_r.push_back(run_start_ring);
            polar_pix0.push_back(run_pix_start);
            polar_nph.push_back(static_cast<int>(nph));
            if (nph > max_polar_nph) {
                max_polar_nph = static_cast<int>(nph);
            }
            continue;
        }

        auto spec = torch::zeros({nrows, run_ring_count, nph}, cplx_opts);
        auto* spec_ptr = spec.data_ptr<c10::complex<double>>();
        for (int64_t row = 0; row < nrows; ++row) {
            const std::complex<double>* s_row = s_ptr + row * mcount * nrings;
            for (int64_t ir = 0; ir < run_ring_count; ++ir) {
                const int64_t r = run_start_ring + ir;
                c10::complex<double>* dst =
                    spec_ptr + ((row * run_ring_count + ir) * nph);
                if (nph > mmax) {
                    for (int64_t m = 0; m < mcount; ++m) {
                        const auto z = s_row[m * nrings + r];
                        dst[m] = c10::complex<double>(z.real(), z.imag());
                    }
                } else {
                    for (int64_t m = 0; m < mcount; ++m) {
                        const auto z = s_row[m * nrings + r];
                        dst[m % nph] += c10::complex<double>(z.real(), z.imag());
                    }
                }
            }
        }
        auto map_vals = at::fft_ifft(spec, std::nullopt, -1, std::nullopt);
        map_vals.mul_(static_cast<double>(nph));
        out.narrow(1, run_pix_start, run_ring_count * nph).copy_(
            map_vals.reshape({nrows, run_ring_count * nph})
        );
    }

    const int64_t npolar = static_cast<int64_t>(polar_r.size());
    if (npolar > 0) {
        const int64_t* polar_r_ptr = polar_r.data();
        const int64_t* polar_pix0_ptr = polar_pix0.data();
        const int* polar_nph_ptr = polar_nph.data();
        parallel_m_ranges(npolar - 1, [&](int64_t i0, int64_t stride) {
            std::vector<std::complex<double>> spec(static_cast<size_t>(max_polar_nph));
            std::vector<std::complex<double>> pix(static_cast<size_t>(max_polar_nph));
            for (int64_t i = i0; i < npolar; i += stride) {
                const int64_t r = polar_r_ptr[static_cast<size_t>(i)];
                const int64_t pix0 = polar_pix0_ptr[static_cast<size_t>(i)];
                const int nph = polar_nph_ptr[static_cast<size_t>(i)];
                for (int64_t row = 0; row < nrows; ++row) {
                    const std::complex<double>* s_row = s_ptr + row * mcount * nrings;
                    std::fill(spec.begin(), spec.begin() + nph, std::complex<double>(0.0, 0.0));
                    if (nph > mmax) {
                        for (int64_t m = 0; m < mcount; ++m) {
                            spec[static_cast<size_t>(m)] = s_row[m * nrings + r];
                        }
                    } else {
                        for (int64_t m = 0; m < mcount; ++m) {
                            spec[static_cast<size_t>(m % nph)] += s_row[m * nrings + r];
                        }
                    }
                    fft_outofplace(spec.data(), nph, 1, pix.data(), 1);
                    std::complex<double>* dst = out_ptr + row * npix + pix0;
                    std::copy(pix.data(), pix.data() + nph, dst);
                }
            }
        });
    }

    if (squeeze_out) {
        return out.reshape({npix});
    }
    return out;
}

static inline void spin_qu_store_ring(
    const std::complex<double>* __restrict__ a,
    const std::complex<double>* __restrict__ b,
    double* __restrict__ q_ptr,
    double* __restrict__ u_ptr,
    int64_t start,
    int64_t nph,
    std::complex<double> vp0,
    std::complex<double> vm0,
    int64_t mmax
) {
    const std::complex<double> vp0c(vp0.real(), -vp0.imag());
    const std::complex<double> vm0c(vm0.real(), -vm0.imag());
    if (mmax == 0) {
        for (int64_t i = 0; i < nph; ++i) {
            const auto aa = a[i];
            const auto bb = b[i];
            q_ptr[start + i] = -0.5 * (aa.real() + bb.real());
            u_ptr[start + i] = -0.5 * (aa.imag() - bb.imag());
        }
        return;
    }
    for (int64_t i = 0; i < nph; ++i) {
        const auto aa = a[i];
        const auto bb = b[i];
        const std::complex<double> bc(bb.real(), -bb.imag());
        const std::complex<double> ac(aa.real(), -aa.imag());
        const auto p_plus = aa + (bc - vm0c);
        const auto p_minus = bb + (ac - vp0c);
        q_ptr[start + i] = -0.5 * (p_plus.real() + p_minus.real());
        u_ptr[start + i] = -0.5 * (p_plus.imag() - p_minus.imag());
    }
}

static torch::Tensor healpix_spin_synthesize_qu_cpu(
    const torch::Tensor& s_modes,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    torch::Tensor eq_spec,
    int64_t eq_start,
    int64_t n_eq,
    int64_t nph_eq
) {
    auto starts_c = starts.contiguous();
    auto lengths_c = lengths.contiguous();
    const auto* starts_ptr = starts_c.data_ptr<int64_t>();
    const auto* lengths_ptr = lengths_c.data_ptr<int64_t>();
    const int64_t nrings = starts_c.numel();
    const int64_t npix = starts_ptr[nrings - 1] + lengths_ptr[nrings - 1];
    const int64_t mcount = s_modes.size(1);
    const int64_t mmax = mcount - 1;
    const auto* s_ptr = reinterpret_cast<const std::complex<double>*>(
        s_modes.data_ptr<c10::complex<double>>()
    );
    const std::complex<double>* s0 = s_ptr;
    const std::complex<double>* s1 = s_ptr + mcount * nrings;

    auto out = torch::empty({2, npix}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto* q_ptr = out.select(0, 0).data_ptr<double>();
    auto* u_ptr = out.select(0, 1).data_ptr<double>();

    const bool profile = sht_profile_enabled();
    double eq_ms = 0.0;
    double polar_ms = 0.0;
    auto t_mark = std::chrono::steady_clock::now();

    if (n_eq > 0 && eq_spec.defined() && eq_spec.numel() > 0) {
        auto eq_c = eq_spec.contiguous();
        const auto* spec = reinterpret_cast<const std::complex<double>*>(
            eq_c.data_ptr<c10::complex<double>>()
        );
        const int nph = static_cast<int>(nph_eq);
        const int64_t eq_pix0 = starts_ptr[eq_start];
        const int64_t nph64 = nph_eq;
        parallel_m_ranges(n_eq - 1, [&](int64_t i0, int64_t stride) {
            std::vector<std::complex<double>> pix_p(static_cast<size_t>(nph));
            std::vector<std::complex<double>> pix_m(static_cast<size_t>(nph));
            for (int64_t ir = i0; ir < n_eq; ir += stride) {
                const int64_t r = eq_start + ir;
                fft_outofplace(spec + ir * nph64, nph, 1, pix_p.data(), 1);
                fft_outofplace(spec + (n_eq + ir) * nph64, nph, 1, pix_m.data(), 1);
                spin_qu_store_ring(
                    pix_p.data(),
                    pix_m.data(),
                    q_ptr,
                    u_ptr,
                    eq_pix0 + ir * nph64,
                    nph64,
                    s0[r],
                    s1[r],
                    mmax
                );
            }
        });
        if (profile) {
            const auto t1 = std::chrono::steady_clock::now();
            eq_ms = std::chrono::duration<double, std::milli>(t1 - t_mark).count();
            t_mark = t1;
        }
    }

    int64_t ring = 0;
    std::vector<int64_t> polar_r;
    std::vector<int64_t> polar_pix0;
    std::vector<int> polar_nph;
    int max_polar_nph = 1;
    while (ring < nrings) {
        const int64_t run_start_ring = ring;
        const int64_t nph = lengths_ptr[ring];
        const int64_t run_pix_start = starts_ptr[ring];
        int64_t pix_cursor = run_pix_start + nph;
        ++ring;
        while (ring < nrings && lengths_ptr[ring] == nph && starts_ptr[ring] == pix_cursor) {
            pix_cursor += nph;
            ++ring;
        }
        const int64_t run_ring_count = ring - run_start_ring;
        if (run_ring_count == 1) {
            polar_r.push_back(run_start_ring);
            polar_pix0.push_back(run_pix_start);
            polar_nph.push_back(static_cast<int>(nph));
            if (nph > max_polar_nph) {
                max_polar_nph = static_cast<int>(nph);
            }
        }
    }

    const int64_t npolar = static_cast<int64_t>(polar_r.size());
    if (npolar > 0) {
        parallel_m_ranges(npolar - 1, [&](int64_t i0, int64_t stride) {
            std::vector<std::complex<double>> spec_p(static_cast<size_t>(max_polar_nph));
            std::vector<std::complex<double>> spec_m(static_cast<size_t>(max_polar_nph));
            std::vector<std::complex<double>> pix_p(static_cast<size_t>(max_polar_nph));
            std::vector<std::complex<double>> pix_m(static_cast<size_t>(max_polar_nph));
            for (int64_t i = i0; i < npolar; i += stride) {
                const int64_t r = polar_r[static_cast<size_t>(i)];
                const int64_t pix0 = polar_pix0[static_cast<size_t>(i)];
                const int nph = polar_nph[static_cast<size_t>(i)];
                std::fill(spec_p.begin(), spec_p.begin() + nph, std::complex<double>(0.0, 0.0));
                std::fill(spec_m.begin(), spec_m.begin() + nph, std::complex<double>(0.0, 0.0));
                if (nph > mmax) {
                    for (int64_t m = 0; m < mcount; ++m) {
                        spec_p[static_cast<size_t>(m)] = s0[m * nrings + r];
                        spec_m[static_cast<size_t>(m)] = s1[m * nrings + r];
                    }
                } else {
                    for (int64_t m = 0; m < mcount; ++m) {
                        spec_p[static_cast<size_t>(m % nph)] += s0[m * nrings + r];
                        spec_m[static_cast<size_t>(m % nph)] += s1[m * nrings + r];
                    }
                }
                fft_outofplace(spec_p.data(), nph, 1, pix_p.data(), 1);
                fft_outofplace(spec_m.data(), nph, 1, pix_m.data(), 1);
                spin_qu_store_ring(
                    pix_p.data(), pix_m.data(), q_ptr, u_ptr, pix0, nph, s0[r], s1[r], mmax
                );
            }
        });
    }
    if (profile) {
        polar_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_mark
        ).count();
        std::fprintf(
            stderr,
            "TORCHSKY_SHT_PROFILE alm2map_spin eq=%.3f ms polar=%.3f ms\n",
            eq_ms,
            polar_ms
        );
    }
    return out;
}

static torch::Tensor spin_qu_from_ifft_cpu(
    const torch::Tensor& p_pm_pos,
    const torch::Tensor& s_plus,
    const torch::Tensor& s_minus,
    const torch::Tensor& starts,
    const torch::Tensor& lengths
) {
    auto p_plus_pos = p_pm_pos.select(0, 0).contiguous();
    auto p_minus_pos = p_pm_pos.select(0, 1).contiguous();
    auto starts_c = starts.contiguous();
    auto lengths_c = lengths.contiguous();
    const auto* starts_ptr = starts_c.data_ptr<int64_t>();
    const auto* lengths_ptr = lengths_c.data_ptr<int64_t>();
    const int64_t npix = p_plus_pos.size(0);
    const int64_t nrings = starts_c.numel();
    const int64_t mmax = s_plus.size(0) - 1;
    const auto* sp_ptr = s_plus.data_ptr<c10::complex<double>>();
    const auto* sm_ptr = s_minus.data_ptr<c10::complex<double>>();
    const int64_t sp_r_stride = s_plus.stride(1);
    const int64_t sm_r_stride = s_minus.stride(1);

    auto out = torch::empty({2, npix}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    const auto* p_plus_ptr = p_plus_pos.data_ptr<c10::complex<double>>();
    const auto* p_minus_ptr = p_minus_pos.data_ptr<c10::complex<double>>();
    auto* q_ptr = out.select(0, 0).data_ptr<double>();
    auto* u_ptr = out.select(0, 1).data_ptr<double>();

    for (int64_t r = 0; r < nrings; ++r) {
        const int64_t start = starts_ptr[r];
        const int64_t nph = lengths_ptr[r];
        if (nph <= 0 || start < 0 || start + nph > npix) {
            throw std::runtime_error("invalid ring layout in healpix_spin_ring_finalize_cpu");
        }
        const c10::complex<double> v_plus_m0 = sp_ptr[r * sp_r_stride];
        const c10::complex<double> v_minus_m0 = sm_ptr[r * sm_r_stride];
        const c10::complex<double> v_plus_m0_conj(v_plus_m0.real(), -v_plus_m0.imag());
        const c10::complex<double> v_minus_m0_conj(v_minus_m0.real(), -v_minus_m0.imag());

        if (mmax == 0) {
            for (int64_t i = 0; i < nph; ++i) {
                const int64_t p = start + i;
                const c10::complex<double> a = p_plus_ptr[p];
                const c10::complex<double> b = p_minus_ptr[p];
                q_ptr[p] = -0.5 * (a.real() + b.real());
                u_ptr[p] = -0.5 * (a.imag() - b.imag());
            }
            continue;
        }
        for (int64_t i = 0; i < nph; ++i) {
            const int64_t p = start + i;
            const c10::complex<double> a = p_plus_ptr[p];
            const c10::complex<double> b = p_minus_ptr[p];
            const c10::complex<double> b_conj(b.real(), -b.imag());
            const c10::complex<double> a_conj(a.real(), -a.imag());
            const c10::complex<double> p_plus = a + (b_conj - v_minus_m0_conj);
            const c10::complex<double> p_minus = b + (a_conj - v_plus_m0_conj);
            q_ptr[p] = -0.5 * (p_plus.real() + p_minus.real());
            u_ptr[p] = -0.5 * (p_plus.imag() - p_minus.imag());
        }
    }
    return out;
}

torch::Tensor healpix_spin_ring_finalize_cpu(
    const torch::Tensor& s_plus,
    const torch::Tensor& s_minus,
    const torch::Tensor& starts,
    const torch::Tensor& lengths
) {
    if (!s_plus.device().is_cpu() || !s_minus.device().is_cpu() || !starts.device().is_cpu() || !lengths.device().is_cpu()) {
        throw std::runtime_error("healpix_spin_ring_finalize_cpu expects CPU tensors");
    }
    if (s_plus.scalar_type() != torch::kComplexDouble || s_minus.scalar_type() != torch::kComplexDouble) {
        throw std::runtime_error("healpix_spin_ring_finalize_cpu expects complex128 s_plus/s_minus");
    }
    if (starts.scalar_type() != torch::kInt64 || lengths.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_spin_ring_finalize_cpu expects int64 starts/lengths");
    }
    if (s_plus.dim() != 2 || s_minus.dim() != 2 || starts.dim() != 1 || lengths.dim() != 1) {
        throw std::runtime_error("healpix_spin_ring_finalize_cpu expects rank-2 s tensors and rank-1 starts/lengths");
    }
    if (s_plus.sizes() != s_minus.sizes()) {
        throw std::runtime_error("healpix_spin_ring_finalize_cpu s_plus/s_minus shape mismatch");
    }
    if (starts.numel() != lengths.numel()) {
        throw std::runtime_error("healpix_spin_ring_finalize_cpu starts/lengths size mismatch");
    }
    const int64_t nrings = starts.numel();
    if (s_plus.size(1) != nrings) {
        throw std::runtime_error("healpix_spin_ring_finalize_cpu ring dimension mismatch");
    }

    auto sp = s_plus.contiguous();
    auto sm = s_minus.contiguous();
    auto s_stacked = torch::stack({sp, sm}, 0);
    auto p_pm_pos = healpix_ring_fourier_synthesis_cpu(s_stacked, starts, lengths);
    return spin_qu_from_ifft_cpu(p_pm_pos, sp, sm, starts, lengths);
}

std::pair<torch::Tensor, torch::Tensor> healpix_spin_map2alm_from_basis_cpu(
    const torch::Tensor& q,
    const torch::Tensor& u,
    const torch::Tensor& ycp_t,
    const torch::Tensor& ycm_t,
    double pix_w
) {
    if (!q.device().is_cpu() || !u.device().is_cpu() || !ycp_t.device().is_cpu() || !ycm_t.device().is_cpu()) {
        throw std::runtime_error("healpix_spin_map2alm_from_basis_cpu expects CPU tensors");
    }
    if (q.scalar_type() != torch::kDouble || u.scalar_type() != torch::kDouble) {
        throw std::runtime_error("healpix_spin_map2alm_from_basis_cpu expects float64 q/u tensors");
    }
    if (ycp_t.scalar_type() != torch::kComplexDouble || ycm_t.scalar_type() != torch::kComplexDouble) {
        throw std::runtime_error("healpix_spin_map2alm_from_basis_cpu expects complex128 basis tensors");
    }

    auto qv = q.contiguous().reshape({-1});
    auto uv = u.contiguous().reshape({-1});
    auto ycp = ycp_t.contiguous();
    auto ycm = ycm_t.contiguous();

    if (qv.numel() != uv.numel()) {
        throw std::runtime_error("q/u length mismatch");
    }
    if (ycp.dim() != 2 || ycm.dim() != 2) {
        throw std::runtime_error("basis tensors must be rank-2");
    }
    if (ycp.sizes() != ycm.sizes()) {
        throw std::runtime_error("basis tensors shape mismatch");
    }

    const int64_t npix = qv.numel();
    if (ycp.size(0) != npix) {
        throw std::runtime_error("basis first dimension must match q/u length");
    }
    auto p_plus = torch::complex(qv, uv);
    auto p_minus = torch::conj(p_plus);
    auto c_plus = torch::matmul(p_plus, ycp) * pix_w;
    auto c_minus = torch::matmul(p_minus, ycm) * pix_w;
    return {c_plus, c_minus};
}

torch::Tensor healpix_spin_alm2map_from_basis_cpu(
    const torch::Tensor& coeff_plus,
    const torch::Tensor& coeff_minus,
    const torch::Tensor& y_plus,
    const torch::Tensor& y_minus,
    const torch::Tensor& y_plus_m0,
    const torch::Tensor& y_minus_m0
) {
    if (
        !coeff_plus.device().is_cpu() || !coeff_minus.device().is_cpu() || !y_plus.device().is_cpu()
        || !y_minus.device().is_cpu() || !y_plus_m0.device().is_cpu() || !y_minus_m0.device().is_cpu()
    ) {
        throw std::runtime_error("healpix_spin_alm2map_from_basis_cpu expects CPU tensors");
    }
    if (
        coeff_plus.scalar_type() != torch::kComplexDouble || coeff_minus.scalar_type() != torch::kComplexDouble
        || y_plus.scalar_type() != torch::kComplexDouble || y_minus.scalar_type() != torch::kComplexDouble
        || y_plus_m0.scalar_type() != torch::kComplexDouble || y_minus_m0.scalar_type() != torch::kComplexDouble
    ) {
        throw std::runtime_error("healpix_spin_alm2map_from_basis_cpu expects complex128 tensors");
    }

    auto cp = coeff_plus.contiguous().reshape({-1});
    auto cm = coeff_minus.contiguous().reshape({-1});
    auto yp = y_plus.contiguous();
    auto ym = y_minus.contiguous();
    auto yp0 = y_plus_m0.contiguous();
    auto ym0 = y_minus_m0.contiguous();

    if (cp.numel() != cm.numel()) {
        throw std::runtime_error("coeff_plus/coeff_minus length mismatch");
    }
    if (yp.dim() != 2 || ym.dim() != 2 || yp0.dim() != 2 || ym0.dim() != 2) {
        throw std::runtime_error("basis tensors must be rank-2");
    }
    if (yp.sizes() != ym.sizes()) {
        throw std::runtime_error("y_plus/y_minus shape mismatch");
    }
    if (yp0.sizes() != ym0.sizes()) {
        throw std::runtime_error("y_plus_m0/y_minus_m0 shape mismatch");
    }
    const int64_t nalm = cp.numel();
    if (yp.size(0) != nalm) {
        throw std::runtime_error("basis row count must match coefficient length");
    }
    const int64_t npix = yp.size(1);
    if (ym.size(1) != npix || yp0.size(1) != npix || ym0.size(1) != npix) {
        throw std::runtime_error("basis tensors must have the same pixel dimension");
    }
    const int64_t l0 = yp0.size(0);
    if (l0 > nalm) {
        throw std::runtime_error("m0 basis row count cannot exceed coefficient length");
    }

    auto p_plus_pos = torch::matmul(cp, yp);
    auto p_minus_pos = torch::matmul(cm, ym);
    torch::Tensor p_plus;
    torch::Tensor p_minus;
    if (l0 == nalm) {
        p_plus = p_plus_pos;
        p_minus = p_minus_pos;
    } else {
        auto cp0 = cp.narrow(0, 0, l0);
        auto cm0 = cm.narrow(0, 0, l0);
        auto p_plus_m0 = torch::matmul(cp0, yp0);
        auto p_minus_m0 = torch::matmul(cm0, ym0);
        p_plus = p_plus_pos + torch::conj(p_minus_pos - p_minus_m0);
        p_minus = p_minus_pos + torch::conj(p_plus_pos - p_plus_m0);
    }

    auto q = -0.5 * torch::real(p_plus + p_minus);
    auto u = -0.5 * torch::imag(p_plus - p_minus);
    return torch::stack({q.to(torch::kFloat64), u.to(torch::kFloat64)}, 0);
}

std::pair<torch::Tensor, torch::Tensor> healpix_spin_interpolate_concat_cpu(
    const torch::Tensor& coeff_plus,
    const torch::Tensor& coeff_minus,
    const torch::Tensor& y_plus,
    const torch::Tensor& y_minus,
    int64_t lmax,
    int64_t mmax
) {
    if (!coeff_plus.device().is_cpu() || !coeff_minus.device().is_cpu() || !y_plus.device().is_cpu()
        || !y_minus.device().is_cpu()) {
        throw std::runtime_error("healpix_spin_interpolate_concat_cpu expects CPU tensors");
    }
    if (coeff_plus.scalar_type() != torch::kComplexDouble || coeff_minus.scalar_type() != torch::kComplexDouble
        || y_plus.scalar_type() != torch::kComplexDouble || y_minus.scalar_type() != torch::kComplexDouble) {
        throw std::runtime_error("healpix_spin_interpolate_concat_cpu expects complex128 tensors");
    }
    if (coeff_plus.dim() != 1 || coeff_minus.dim() != 1 || y_plus.dim() != 2 || y_minus.dim() != 2) {
        throw std::runtime_error("healpix_spin_interpolate_concat_cpu expects coeff rank-1 and basis rank-2");
    }
    if (coeff_plus.numel() != coeff_minus.numel()) {
        throw std::runtime_error("healpix_spin_interpolate_concat_cpu coeff length mismatch");
    }
    if (y_plus.sizes() != y_minus.sizes()) {
        throw std::runtime_error("healpix_spin_interpolate_concat_cpu basis shape mismatch");
    }
    if (mmax < 0 || mmax > lmax) {
        throw std::runtime_error("healpix_spin_interpolate_concat_cpu invalid lmax/mmax");
    }

    auto cp = coeff_plus.contiguous();
    auto cm = coeff_minus.contiguous();
    auto yp = y_plus.contiguous();
    auto ym = y_minus.contiguous();

    const int64_t nalm = cp.numel();
    const int64_t nrings = yp.size(1);
    const int64_t mcount = mmax + 1;
    if (yp.size(0) != nalm) {
        throw std::runtime_error("healpix_spin_interpolate_concat_cpu basis row count must match coeff length");
    }
    if (alm_size(lmax, mmax) != nalm) {
        throw std::runtime_error("healpix_spin_interpolate_concat_cpu nalm does not match lmax/mmax");
    }

    auto out_plus = torch::zeros({mcount, nrings}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto out_minus = torch::zeros_like(out_plus);

    const auto* cp_ptr = cp.data_ptr<c10::complex<double>>();
    const auto* cm_ptr = cm.data_ptr<c10::complex<double>>();
    const auto* yp_ptr = yp.data_ptr<c10::complex<double>>();
    const auto* ym_ptr = ym.data_ptr<c10::complex<double>>();
    auto* op_ptr = out_plus.data_ptr<c10::complex<double>>();
    auto* om_ptr = out_minus.data_ptr<c10::complex<double>>();

    at::parallel_for(0, mcount, 1, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            const int64_t base = m * (lmax + 1) - (m * (m - 1)) / 2;
            const int64_t l_count = lmax - m + 1;
            auto* op_row = op_ptr + m * nrings;
            auto* om_row = om_ptr + m * nrings;
            for (int64_t r = 0; r < nrings; ++r) {
                c10::complex<double> accp(0.0, 0.0);
                c10::complex<double> accm(0.0, 0.0);
                for (int64_t t = 0; t < l_count; ++t) {
                    const int64_t i = base + t;
                    accp += cp_ptr[i] * yp_ptr[i * nrings + r];
                    accm += cm_ptr[i] * ym_ptr[i * nrings + r];
                }
                op_row[r] = accp;
                om_row[r] = accm;
            }
        }
    });

    return {out_plus, out_minus};
}

std::pair<torch::Tensor, torch::Tensor> healpix_spin_integrate_concat_cpu(
    const torch::Tensor& s_plus,
    const torch::Tensor& s_minus,
    const torch::Tensor& y_plus,
    const torch::Tensor& y_minus,
    int64_t lmax,
    int64_t mmax,
    double pix_w
) {
    if (!s_plus.device().is_cpu() || !s_minus.device().is_cpu() || !y_plus.device().is_cpu()
        || !y_minus.device().is_cpu()) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu expects CPU tensors");
    }
    if (s_plus.scalar_type() != torch::kComplexDouble || s_minus.scalar_type() != torch::kComplexDouble
        || y_plus.scalar_type() != torch::kComplexDouble || y_minus.scalar_type() != torch::kComplexDouble) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu expects complex128 tensors");
    }
    if (s_plus.dim() != 2 || s_minus.dim() != 2 || y_plus.dim() != 2 || y_minus.dim() != 2) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu expects rank-2 tensors");
    }
    if (s_plus.sizes() != s_minus.sizes()) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu s_plus/s_minus shape mismatch");
    }
    if (y_plus.sizes() != y_minus.sizes()) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu y_plus/y_minus shape mismatch");
    }
    if (mmax < 0 || mmax > lmax) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu invalid lmax/mmax");
    }

    auto sp = s_plus.contiguous();
    auto sm = s_minus.contiguous();
    auto yp = y_plus.contiguous();
    auto ym = y_minus.contiguous();

    const int64_t mcount = mmax + 1;
    const int64_t nrings = sp.size(1);
    const int64_t nalm = yp.size(0);
    if (sp.size(0) != mcount || sm.size(0) != mcount) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu s_plus/s_minus first dim must be mmax+1");
    }
    if (yp.size(1) != nrings || ym.size(1) != nrings) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu basis ring dim mismatch");
    }
    if (alm_size(lmax, mmax) != nalm) {
        throw std::runtime_error("healpix_spin_integrate_concat_cpu nalm does not match lmax/mmax");
    }

    auto out_plus = torch::zeros({nalm}, torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU));
    auto out_minus = torch::zeros_like(out_plus);

    const auto* sp_ptr = sp.data_ptr<c10::complex<double>>();
    const auto* sm_ptr = sm.data_ptr<c10::complex<double>>();
    const auto* yp_ptr = yp.data_ptr<c10::complex<double>>();
    const auto* ym_ptr = ym.data_ptr<c10::complex<double>>();
    auto* op_ptr = out_plus.data_ptr<c10::complex<double>>();
    auto* om_ptr = out_minus.data_ptr<c10::complex<double>>();

    at::parallel_for(0, mcount, 1, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            const int64_t base = m * (lmax + 1) - (m * (m - 1)) / 2;
            const int64_t l_count = lmax - m + 1;

            auto yp_block = yp.narrow(0, base, l_count);
            auto ym_block = ym.narrow(0, base, l_count);
            auto sp_vec = sp.select(0, m);
            auto sm_vec = sm.select(0, m);

            out_plus.narrow(0, base, l_count).copy_(at::mv(yp_block, sp_vec) * pix_w);
            out_minus.narrow(0, base, l_count).copy_(at::mv(ym_block, sm_vec) * pix_w);
        }
    });

    return {out_plus, out_minus};
}

std::pair<torch::Tensor, torch::Tensor> healpix_spin_map2alm_ring_concat_cpu(
    const torch::Tensor& p_plus,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    const torch::Tensor& phase0_neg,
    const torch::Tensor& y_plus,
    const torch::Tensor& y_minus,
    int64_t lmax,
    int64_t mmax,
    double pix_w
) {
    if (!p_plus.device().is_cpu() || !starts.device().is_cpu() || !lengths.device().is_cpu()
        || !phase0_neg.device().is_cpu() || !y_plus.device().is_cpu() || !y_minus.device().is_cpu()) {
        throw std::runtime_error("healpix_spin_map2alm_ring_concat_cpu expects CPU tensors");
    }
    if (p_plus.scalar_type() != torch::kComplexDouble || phase0_neg.scalar_type() != torch::kComplexDouble
        || y_plus.scalar_type() != torch::kComplexDouble || y_minus.scalar_type() != torch::kComplexDouble) {
        throw std::runtime_error("healpix_spin_map2alm_ring_concat_cpu expects complex128 tensors");
    }
    if (starts.scalar_type() != torch::kInt64 || lengths.scalar_type() != torch::kInt64) {
        throw std::runtime_error("healpix_spin_map2alm_ring_concat_cpu expects int64 starts/lengths");
    }
    if (p_plus.dim() != 1 || starts.dim() != 1 || lengths.dim() != 1 || phase0_neg.dim() != 2
        || y_plus.dim() != 2 || y_minus.dim() != 2) {
        throw std::runtime_error("healpix_spin_map2alm_ring_concat_cpu invalid tensor ranks");
    }
    if (y_plus.sizes() != y_minus.sizes()) {
        throw std::runtime_error("healpix_spin_map2alm_ring_concat_cpu y_plus/y_minus shape mismatch");
    }
    if (mmax < 0 || mmax > lmax) {
        throw std::runtime_error("healpix_spin_map2alm_ring_concat_cpu invalid lmax/mmax");
    }
    const int64_t mcount = mmax + 1;
    auto s_stacked = healpix_ring_fourier_modes_spin_conj_cpu(p_plus, starts, lengths, mmax);
    auto phase = phase0_neg.contiguous();
    auto s_plus = s_stacked.select(0, 0);
    auto s_minus = s_stacked.select(0, 1);
    if (phase.size(0) != mcount || phase.size(1) != s_plus.size(1)) {
        throw std::runtime_error("healpix_spin_map2alm_ring_concat_cpu phase0_neg shape mismatch");
    }
    s_plus = s_plus * phase;
    s_minus = s_minus * phase;
    return healpix_spin_integrate_concat_cpu(s_plus, s_minus, y_plus, y_minus, lmax, mmax, pix_w);
}

std::pair<torch::Tensor, torch::Tensor> healpix_get_interp_weights_ring_cpu(
    int64_t nside,
    const torch::Tensor& lon_deg,
    const torch::Tensor& lat_deg
) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!lon_deg.device().is_cpu() || !lat_deg.device().is_cpu()) {
        throw std::runtime_error("healpix_get_interp_weights_ring_cpu expects CPU tensors");
    }
    if (lon_deg.scalar_type() != torch::kDouble || lat_deg.scalar_type() != torch::kDouble) {
        throw std::runtime_error("healpix_get_interp_weights_ring_cpu expects float64 tensors");
    }
    if (!lon_deg.sizes().equals(lat_deg.sizes())) {
        throw std::runtime_error("lon/lat shape mismatch");
    }

    auto lon = lon_deg.contiguous().reshape({-1});
    auto lat = lat_deg.contiguous().reshape({-1});
    const int64_t n = lon.numel();
    auto pix = torch::empty({4, n}, torch::TensorOptions().dtype(torch::kInt64));
    auto wgt = torch::zeros({4, n}, torch::TensorOptions().dtype(torch::kDouble));

    const auto* lon_ptr = lon.data_ptr<double>();
    const auto* lat_ptr = lat.data_ptr<double>();
    auto* p_ptr = pix.data_ptr<int64_t>();
    auto* w_ptr = wgt.data_ptr<double>();

    const int64_t npix = 12 * nside * nside;
    const int64_t n4 = 4 * nside;

    auto compute_block = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const double theta = (90.0 - lat_ptr[i]) * PI / 180.0;
            const double phi = wrap_rad_2pi(lon_ptr[i] * PI / 180.0);
            const double z = std::cos(theta);
            const int64_t ir1 = ring_above_scalar(nside, z);
            const int64_t ir2 = ir1 + 1;

            int64_t p0 = -1;
            int64_t p1 = -1;
            int64_t p2 = -1;
            int64_t p3 = -1;
            double w0 = 0.0;
            double w1 = 0.0;
            double w2 = 0.0;
            double w3 = 0.0;
            double theta1 = 0.0;
            double theta2 = 0.0;

            if (ir1 > 0) {
                int64_t sp;
                int64_t nr;
                bool shifted;
                get_ring_info2_scalar(nside, ir1, sp, nr, theta1, shifted);
                const double sh = shifted ? 1.0 : 0.0;
                const double dphi = TWO_PI / static_cast<double>(nr);
                const double tmp = phi / dphi - 0.5 * sh;
                int64_t i1 = static_cast<int64_t>(std::floor(tmp));
                const double wt = (phi - (static_cast<double>(i1) + 0.5 * sh) * dphi) / dphi;
                int64_t i2 = i1 + 1;
                if (i1 < 0) {
                    i1 += nr;
                }
                if (i2 >= nr) {
                    i2 -= nr;
                }
                p0 = sp + i1;
                p1 = sp + i2;
                w0 = 1.0 - wt;
                w1 = wt;
            }

            if (ir2 < n4) {
                int64_t sp;
                int64_t nr;
                bool shifted;
                get_ring_info2_scalar(nside, ir2, sp, nr, theta2, shifted);
                const double sh = shifted ? 1.0 : 0.0;
                const double dphi = TWO_PI / static_cast<double>(nr);
                const double tmp = phi / dphi - 0.5 * sh;
                int64_t i1 = static_cast<int64_t>(std::floor(tmp));
                const double wt = (phi - (static_cast<double>(i1) + 0.5 * sh) * dphi) / dphi;
                int64_t i2 = i1 + 1;
                if (i1 < 0) {
                    i1 += nr;
                }
                if (i2 >= nr) {
                    i2 -= nr;
                }
                p2 = sp + i1;
                p3 = sp + i2;
                w2 = 1.0 - wt;
                w3 = wt;
            }

            if (ir1 == 0) {
                const double wtheta = theta / theta2;
                w2 *= wtheta;
                w3 *= wtheta;
                const double fac = (1.0 - wtheta) * 0.25;
                w0 = fac;
                w1 = fac;
                w2 += fac;
                w3 += fac;
                p0 = (p2 + 2) & 3;
                p1 = (p3 + 2) & 3;
            } else if (ir2 == n4) {
                const double wtheta = (theta - theta1) / (PI - theta1);
                w0 *= (1.0 - wtheta);
                w1 *= (1.0 - wtheta);
                const double fac = wtheta * 0.25;
                w0 += fac;
                w1 += fac;
                w2 = fac;
                w3 = fac;
                p2 = ((p0 + 2) & 3) + npix - 4;
                p3 = ((p1 + 2) & 3) + npix - 4;
            } else {
                const double wtheta = (theta - theta1) / (theta2 - theta1);
                w0 *= (1.0 - wtheta);
                w1 *= (1.0 - wtheta);
                w2 *= wtheta;
                w3 *= wtheta;
            }

            p_ptr[i] = p0;
            p_ptr[n + i] = p1;
            p_ptr[2 * n + i] = p2;
            p_ptr[3 * n + i] = p3;
            w_ptr[i] = w0;
            w_ptr[n + i] = w1;
            w_ptr[2 * n + i] = w2;
            w_ptr[3 * n + i] = w3;
        }
    };
    if (n >= PARALLEL_MIN_ELEMS) {
        at::parallel_for(0, n, 4096, compute_block);
    } else {
        compute_block(0, n);
    }

    return {pix, wgt};
}

std::pair<torch::Tensor, torch::Tensor> healpix_get_interp_weights_nested_cpu(
    int64_t nside,
    const torch::Tensor& lon_deg,
    const torch::Tensor& lat_deg
) {
    auto out = healpix_get_interp_weights_ring_cpu(nside, lon_deg, lat_deg);
    auto pix_n = healpix_ring2nest_cpu(nside, out.first.reshape({-1})).reshape_as(out.first);
    return {pix_n, out.second};
}

static torch::Tensor healpix_interp_val_from_weights_cpu(
    const torch::Tensor& maps_2d,
    const torch::Tensor& pix4n,
    const torch::Tensor& wgt4n
) {
    auto pix_flat = pix4n.contiguous().reshape({-1});
    const int64_t nq = pix4n.size(1);
    auto gathered = maps_2d.index_select(1, pix_flat).reshape({maps_2d.size(0), 4, nq});
    auto w = wgt4n.to(maps_2d.scalar_type()).unsqueeze(0);
    return (gathered * w).sum(1);
}

torch::Tensor healpix_get_interp_val_ring_cpu(
    int64_t nside,
    const torch::Tensor& maps_in,
    const torch::Tensor& lon_deg,
    const torch::Tensor& lat_deg
) {
    if (!maps_in.device().is_cpu() || !lon_deg.device().is_cpu() || !lat_deg.device().is_cpu()) {
        throw std::runtime_error("healpix_get_interp_val_ring_cpu expects CPU tensors");
    }
    if (lon_deg.scalar_type() != torch::kDouble || lat_deg.scalar_type() != torch::kDouble) {
        throw std::runtime_error("healpix_get_interp_val_ring_cpu expects float64 lon/lat tensors");
    }
    if (!lon_deg.sizes().equals(lat_deg.sizes())) {
        throw std::runtime_error("lon/lat shape mismatch");
    }
    if (maps_in.dim() != 1 && maps_in.dim() != 2) {
        throw std::runtime_error("maps must be rank-1 or rank-2");
    }
    if (
        maps_in.scalar_type() != torch::kFloat && maps_in.scalar_type() != torch::kDouble
        && maps_in.scalar_type() != torch::kBFloat16 && maps_in.scalar_type() != torch::kHalf
    ) {
        throw std::runtime_error("maps must be floating point");
    }

    const bool one_map = maps_in.dim() == 1;
    auto maps = one_map ? maps_in.contiguous().reshape({1, -1}) : maps_in.contiguous();
    const int64_t npix = maps.size(1);
    if (npix != 12 * nside * nside) {
        throw std::runtime_error("maps second dimension must be 12*nside*nside");
    }

    auto out_w = healpix_get_interp_weights_ring_cpu(nside, lon_deg, lat_deg);
    auto values = healpix_interp_val_from_weights_cpu(maps, out_w.first, out_w.second);

    std::vector<int64_t> qshape = lon_deg.sizes().vec();
    if (one_map) {
        return values.reshape(qshape);
    }
    std::vector<int64_t> out_shape;
    out_shape.reserve(static_cast<size_t>(qshape.size()) + 1);
    out_shape.push_back(maps.size(0));
    out_shape.insert(out_shape.end(), qshape.begin(), qshape.end());
    return values.reshape(out_shape);
}

torch::Tensor healpix_get_interp_val_nested_cpu(
    int64_t nside,
    const torch::Tensor& maps_in,
    const torch::Tensor& lon_deg,
    const torch::Tensor& lat_deg
) {
    if (!maps_in.device().is_cpu() || !lon_deg.device().is_cpu() || !lat_deg.device().is_cpu()) {
        throw std::runtime_error("healpix_get_interp_val_nested_cpu expects CPU tensors");
    }
    if (lon_deg.scalar_type() != torch::kDouble || lat_deg.scalar_type() != torch::kDouble) {
        throw std::runtime_error("healpix_get_interp_val_nested_cpu expects float64 lon/lat tensors");
    }
    if (!lon_deg.sizes().equals(lat_deg.sizes())) {
        throw std::runtime_error("lon/lat shape mismatch");
    }
    if (maps_in.dim() != 1 && maps_in.dim() != 2) {
        throw std::runtime_error("maps must be rank-1 or rank-2");
    }
    if (
        maps_in.scalar_type() != torch::kFloat && maps_in.scalar_type() != torch::kDouble
        && maps_in.scalar_type() != torch::kBFloat16 && maps_in.scalar_type() != torch::kHalf
    ) {
        throw std::runtime_error("maps must be floating point");
    }

    const bool one_map = maps_in.dim() == 1;
    auto maps = one_map ? maps_in.contiguous().reshape({1, -1}) : maps_in.contiguous();
    const int64_t npix = maps.size(1);
    if (npix != 12 * nside * nside) {
        throw std::runtime_error("maps second dimension must be 12*nside*nside");
    }

    auto out_w = healpix_get_interp_weights_nested_cpu(nside, lon_deg, lat_deg);
    auto values = healpix_interp_val_from_weights_cpu(maps, out_w.first, out_w.second);

    std::vector<int64_t> qshape = lon_deg.sizes().vec();
    if (one_map) {
        return values.reshape(qshape);
    }
    std::vector<int64_t> out_shape;
    out_shape.reserve(static_cast<size_t>(qshape.size()) + 1);
    out_shape.push_back(maps.size(0));
    out_shape.insert(out_shape.end(), qshape.begin(), qshape.end());
    return values.reshape(out_shape);
}

struct WignerDCoeffs {
    int num_terms;
    double coeff[4];
    int p_ct[4];
    int p_st[4];
};

static inline double fast_pow(double base, int exp) {
    if (exp == 0) return 1.0;
    if (exp == 1) return base;
    if (exp == 2) return base * base;

    double res = 1.0;
    while (exp > 0) {
        if (exp & 1) res *= base;
        base *= base;
        exp >>= 1;
    }
    return res;
}

static WignerDCoeffs get_wigner_d_coeffs(int64_t l, int64_t m, int64_t mp) {
    WignerDCoeffs res;
    res.num_terms = 0;

    int64_t k_min = std::max((int64_t)0, m - mp);
    int64_t k_max = std::min(l + m, l - mp);

    if (k_min > k_max) return res;

    double log_pref = 0.5 * (std::lgamma(l + m + 1) + std::lgamma(l - m + 1) +
                             std::lgamma(l + mp + 1) + std::lgamma(l - mp + 1));

    for(int64_t k = k_min; k <= k_max && res.num_terms < 4; ++k) {
        int64_t a = l + m - k;
        int64_t b = k;
        int64_t c = mp - m + k;
        int64_t d = l - mp - k;

        if (a < 0 || b < 0 || c < 0 || d < 0) continue;

        double log_denom = std::lgamma(a + 1) + std::lgamma(b + 1) +
                           std::lgamma(c + 1) + std::lgamma(d + 1);

        double sign = ((k + mp - m) % 2 != 0) ? -1.0 : 1.0;

        res.p_ct[res.num_terms] = (int)(2 * l + m - mp - 2 * k);
        res.p_st[res.num_terms] = (int)(mp - m + 2 * k);
        res.coeff[res.num_terms] = sign * std::exp(log_pref - log_denom);
        res.num_terms++;
    }
    return res;
}

static inline double eval_wigner_d_half(
    const WignerDCoeffs& coeffs, double sin_half, double cos_half
) {
    if (coeffs.num_terms == 0) return 0.0;

    double sum = 0.0;
    for (int i = 0; i < coeffs.num_terms; ++i) {
        double term = coeffs.coeff[i];
        if (coeffs.p_ct[i] > 0) term *= fast_pow(cos_half, coeffs.p_ct[i]);
        if (coeffs.p_st[i] > 0) term *= fast_pow(sin_half, coeffs.p_st[i]);
        sum += term;
    }
    return sum;
}

static inline double eval_wigner_d(const WignerDCoeffs& coeffs, double theta) {
    return eval_wigner_d_half(coeffs, std::sin(theta * 0.5), std::cos(theta * 0.5));
}

static bool sht_profile_enabled() {
    const char* env = std::getenv("TORCHSKY_SHT_PROFILE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

#if defined(__linux__)
static int linux_affinity_cpu_count() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return 0;
    }
    return CPU_COUNT(&set);
}

static int linux_cgroup_cpu_quota() {
    // ponytail: Skaha --cpu is often CFS quota, not cpuset. Affinity then
    // still reports host cores. Upgrade: read the job request if a runtime
    // API exposes it.
    {
        std::ifstream in("/sys/fs/cgroup/cpu.max");
        std::string quota;
        std::string period;
        if (in && (in >> quota >> period) && quota != "max") {
            const long q = std::strtol(quota.c_str(), nullptr, 10);
            const long p = std::strtol(period.c_str(), nullptr, 10);
            if (q > 0 && p > 0) {
                return std::max(1, static_cast<int>((q + p - 1) / p));
            }
        }
    }
    const char* quota_paths[] = {
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us",
    };
    const char* period_paths[] = {
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us",
    };
    for (int i = 0; i < 2; ++i) {
        std::ifstream qf(quota_paths[i]);
        std::ifstream pf(period_paths[i]);
        long q = 0;
        long p = 0;
        if (qf && pf && (qf >> q) && (pf >> p) && q > 0 && p > 0) {
            return std::max(1, static_cast<int>((q + p - 1) / p));
        }
    }
    return 0;
}
#endif

class SpinMPool {
 public:
  static SpinMPool& instance() {
    static SpinMPool pool;
    return pool;
  }

  int nworkers() const { return nworkers_; }

  template <typename Fn>
  void parallel_m(int64_t mmax, Fn&& fn) {
    const int64_t nm = mmax + 1;
    if (nworkers_ <= 1 || nm <= 1) {
      fn(0, 1);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      job_ = [&fn](int64_t a, int64_t b) { fn(a, b); };
      remaining_.store(nworkers_, std::memory_order_relaxed);
      epoch_ += 1;
    }
    cv_start_.notify_all();
    std::unique_lock<std::mutex> lock(mu_);
    cv_done_.wait(lock, [&] { return remaining_.load(std::memory_order_acquire) == 0; });
    job_ = nullptr;
  }

 private:
  static int choose_nworkers() {
    if (const char* env = std::getenv("TORCHSKY_SPIN_THREADS"); env != nullptr && env[0] != '\0') {
      return std::max(1, std::atoi(env));
    }
    int n = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
#if defined(__linux__)
    const int aff = linux_affinity_cpu_count();
    if (aff > 0) {
      n = std::min(n, aff);
    }
    const int quota = linux_cgroup_cpu_quota();
    if (quota > 0) {
      n = std::min(n, quota);
    }
    const int aten = at::get_num_threads();
    if (aten > 0) {
      n = std::min(n, aten);
    }
#endif
    return std::max(1, n);
  }

  SpinMPool() {
    nworkers_ = choose_nworkers();
    remaining_.store(0, std::memory_order_relaxed);
    workers_.reserve(static_cast<size_t>(nworkers_));
    for (int i = 0; i < nworkers_; ++i) {
      workers_.emplace_back([this, i] { worker(i); });
    }
    if (sht_profile_enabled()) {
      std::fprintf(
          stderr,
          "TORCHSKY_SHT_PROFILE SpinMPool workers=%d aten_threads=%d hw=%u\n",
          nworkers_,
          at::get_num_threads(),
          std::thread::hardware_concurrency()
      );
    }
  }

  ~SpinMPool() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_start_.notify_all();
    for (auto& t : workers_) {
      t.join();
    }
  }

  SpinMPool(const SpinMPool&) = delete;
  SpinMPool& operator=(const SpinMPool&) = delete;

  void worker(int id) {
    int seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> lock(mu_);
      cv_start_.wait(lock, [&] { return stop_ || epoch_ != seen; });
      if (stop_) {
        return;
      }
      seen = epoch_;
      const auto job = job_;
      const int n = nworkers_;
      lock.unlock();
      if (job) {
        job(id, n);
      }
      if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> done(mu_);
        cv_done_.notify_one();
      }
    }
  }

  std::mutex mu_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  std::vector<std::thread> workers_;
  std::function<void(int64_t, int64_t)> job_;
  int nworkers_ = 1;
  int epoch_ = 0;
  std::atomic<int> remaining_{0};
  bool stop_ = false;
};

template <typename Fn>
static void parallel_m_ranges(int64_t mmax, Fn&& fn) {
  SpinMPool::instance().parallel_m(mmax, std::forward<Fn>(fn));
}

static inline void spin_active_pair_span(
    const uint8_t* __restrict__ active,
    int n_pairs,
    int& pair0,
    int& n_run
) {
    int i0 = 0;
    while (i0 < n_pairs && active[i0] == 0) {
        ++i0;
    }
    int i1 = n_pairs;
    while (i1 > i0 && active[i1 - 1] == 0) {
        --i1;
    }
    pair0 = i0;
    n_run = i1 - i0;
}

static inline void spin_integrate_dot_rings(
    int n_pairs,
    double parity,
    const double* __restrict__ d_p,
    const double* __restrict__ d_m,
    const double* __restrict__ Np_r,
    const double* __restrict__ Np_i,
    const double* __restrict__ Nm_r,
    const double* __restrict__ Nm_i,
    const double* __restrict__ Sp_r,
    const double* __restrict__ Sp_i,
    const double* __restrict__ Sm_r,
    const double* __restrict__ Sm_i,
    double& r_p,
    double& i_p,
    double& r_m,
    double& i_m
) {
    int i = 0;
#if defined(__aarch64__)
    float64x2_t acc_rp = vdupq_n_f64(0.0);
    float64x2_t acc_ip = vdupq_n_f64(0.0);
    float64x2_t acc_rm = vdupq_n_f64(0.0);
    float64x2_t acc_im = vdupq_n_f64(0.0);
    const float64x2_t vpar = vdupq_n_f64(parity);
    auto acc_pair = [&](int off) {
        const float64x2_t vp = vld1q_f64(d_p + off);
        const float64x2_t vm = vld1q_f64(d_m + off);
        const float64x2_t vms = vmulq_f64(vm, vpar);
        const float64x2_t vps = vmulq_f64(vp, vpar);
        acc_rp = vfmaq_f64(acc_rp, vld1q_f64(Np_r + off), vp);
        acc_rp = vfmaq_f64(acc_rp, vld1q_f64(Sp_r + off), vms);
        acc_ip = vfmaq_f64(acc_ip, vld1q_f64(Np_i + off), vp);
        acc_ip = vfmaq_f64(acc_ip, vld1q_f64(Sp_i + off), vms);
        acc_rm = vfmaq_f64(acc_rm, vld1q_f64(Nm_r + off), vm);
        acc_rm = vfmaq_f64(acc_rm, vld1q_f64(Sm_r + off), vps);
        acc_im = vfmaq_f64(acc_im, vld1q_f64(Nm_i + off), vm);
        acc_im = vfmaq_f64(acc_im, vld1q_f64(Sm_i + off), vps);
    };
    for (; i + 4 <= n_pairs; i += 4) {
        acc_pair(i);
        acc_pair(i + 2);
    }
    if (i + 2 <= n_pairs) {
        acc_pair(i);
        i += 2;
    }
    r_p += vaddvq_f64(acc_rp);
    i_p += vaddvq_f64(acc_ip);
    r_m += vaddvq_f64(acc_rm);
    i_m += vaddvq_f64(acc_im);
#endif
    for (; i < n_pairs; ++i) {
        const double vp = d_p[i];
        const double vm = d_m[i];
        const double vms = vm * parity;
        const double vps = vp * parity;
        r_p += Np_r[i] * vp + Sp_r[i] * vms;
        i_p += Np_i[i] * vp + Sp_i[i] * vms;
        r_m += Nm_r[i] * vm + Sm_r[i] * vps;
        i_m += Nm_i[i] * vm + Sm_i[i] * vps;
    }
}

static inline void spin_integrate_recurrence_l_rings(
    int n_pairs,
    double c1,
    double c2,
    double c3,
    double parity,
    const double* __restrict__ ct,
    const double* __restrict__ dp1,
    const double* __restrict__ dp2,
    const double* __restrict__ dm1,
    const double* __restrict__ dm2,
    double* __restrict__ dpc,
    double* __restrict__ dmc,
    const double* __restrict__ Np_r,
    const double* __restrict__ Np_i,
    const double* __restrict__ Nm_r,
    const double* __restrict__ Nm_i,
    const double* __restrict__ Sp_r,
    const double* __restrict__ Sp_i,
    const double* __restrict__ Sm_r,
    const double* __restrict__ Sm_i,
    double& r_p,
    double& i_p,
    double& r_m,
    double& i_m
) {
    int i = 0;
#if defined(__aarch64__)
    float64x2_t acc_rp0 = vdupq_n_f64(0.0);
    float64x2_t acc_ip0 = vdupq_n_f64(0.0);
    float64x2_t acc_rm0 = vdupq_n_f64(0.0);
    float64x2_t acc_im0 = vdupq_n_f64(0.0);
    float64x2_t acc_rp1 = vdupq_n_f64(0.0);
    float64x2_t acc_ip1 = vdupq_n_f64(0.0);
    float64x2_t acc_rm1 = vdupq_n_f64(0.0);
    float64x2_t acc_im1 = vdupq_n_f64(0.0);
    const float64x2_t vc1 = vdupq_n_f64(c1);
    const float64x2_t vc2 = vdupq_n_f64(c2);
    const float64x2_t vc3 = vdupq_n_f64(c3);
    const float64x2_t vpar = vdupq_n_f64(parity);
    auto acc_pair = [&](int off, float64x2_t& arp, float64x2_t& aip, float64x2_t& arm, float64x2_t& aim) {
        const float64x2_t vct = vld1q_f64(ct + off);
        const float64x2_t vcterm = vmulq_f64(vc1, vct);
        const float64x2_t vp = vfmaq_f64(
            vmulq_f64(vaddq_f64(vcterm, vc2), vld1q_f64(dp1 + off)),
            vc3,
            vld1q_f64(dp2 + off)
        );
        const float64x2_t vm = vfmaq_f64(
            vmulq_f64(vsubq_f64(vcterm, vc2), vld1q_f64(dm1 + off)),
            vc3,
            vld1q_f64(dm2 + off)
        );
        vst1q_f64(dpc + off, vp);
        vst1q_f64(dmc + off, vm);
        const float64x2_t vms = vmulq_f64(vm, vpar);
        const float64x2_t vps = vmulq_f64(vp, vpar);
        arp = vfmaq_f64(arp, vld1q_f64(Np_r + off), vp);
        arp = vfmaq_f64(arp, vld1q_f64(Sp_r + off), vms);
        aip = vfmaq_f64(aip, vld1q_f64(Np_i + off), vp);
        aip = vfmaq_f64(aip, vld1q_f64(Sp_i + off), vms);
        arm = vfmaq_f64(arm, vld1q_f64(Nm_r + off), vm);
        arm = vfmaq_f64(arm, vld1q_f64(Sm_r + off), vps);
        aim = vfmaq_f64(aim, vld1q_f64(Nm_i + off), vm);
        aim = vfmaq_f64(aim, vld1q_f64(Sm_i + off), vps);
    };
    for (; i + 4 <= n_pairs; i += 4) {
        acc_pair(i, acc_rp0, acc_ip0, acc_rm0, acc_im0);
        acc_pair(i + 2, acc_rp1, acc_ip1, acc_rm1, acc_im1);
    }
    if (i + 2 <= n_pairs) {
        acc_pair(i, acc_rp0, acc_ip0, acc_rm0, acc_im0);
        i += 2;
    }
    r_p += vaddvq_f64(vaddq_f64(acc_rp0, acc_rp1));
    i_p += vaddvq_f64(vaddq_f64(acc_ip0, acc_ip1));
    r_m += vaddvq_f64(vaddq_f64(acc_rm0, acc_rm1));
    i_m += vaddvq_f64(vaddq_f64(acc_im0, acc_im1));
#endif
    for (; i < n_pairs; ++i) {
        const double vpN = (c1 * ct[i] + c2) * dp1[i] + c3 * dp2[i];
        const double vmN = (c1 * ct[i] - c2) * dm1[i] + c3 * dm2[i];
        dpc[i] = vpN;
        dmc[i] = vmN;
        const double vms = vmN * parity;
        const double vps = vpN * parity;
        r_p += Np_r[i] * vpN + Sp_r[i] * vms;
        i_p += Np_i[i] * vpN + Sp_i[i] * vms;
        r_m += Nm_r[i] * vmN + Sm_r[i] * vps;
        i_m += Nm_i[i] * vmN + Sm_i[i] * vps;
    }
}

static inline void spin_integrate_recurrence_ll_rings(
    int n_pairs,
    double c1A,
    double c2A,
    double c3A,
    double c1B,
    double c2B,
    double c3B,
    double parityA,
    const double* __restrict__ ct,
    double* __restrict__ dp1,
    double* __restrict__ dp2,
    double* __restrict__ dm1,
    double* __restrict__ dm2,
    const double* __restrict__ Np_r,
    const double* __restrict__ Np_i,
    const double* __restrict__ Nm_r,
    const double* __restrict__ Nm_i,
    const double* __restrict__ Sp_r,
    const double* __restrict__ Sp_i,
    const double* __restrict__ Sm_r,
    const double* __restrict__ Sm_i,
    double& r_pA,
    double& i_pA,
    double& r_mA,
    double& i_mA,
    double& r_pB,
    double& i_pB,
    double& r_mB,
    double& i_mB
) {
    int i = 0;
    const double parityB = -parityA;
#if defined(__aarch64__)
    float64x2_t acc_rpA = vdupq_n_f64(0.0);
    float64x2_t acc_ipA = vdupq_n_f64(0.0);
    float64x2_t acc_rmA = vdupq_n_f64(0.0);
    float64x2_t acc_imA = vdupq_n_f64(0.0);
    float64x2_t acc_rpB = vdupq_n_f64(0.0);
    float64x2_t acc_ipB = vdupq_n_f64(0.0);
    float64x2_t acc_rmB = vdupq_n_f64(0.0);
    float64x2_t acc_imB = vdupq_n_f64(0.0);
    const float64x2_t vc1A = vdupq_n_f64(c1A);
    const float64x2_t vc2A = vdupq_n_f64(c2A);
    const float64x2_t vc3A = vdupq_n_f64(c3A);
    const float64x2_t vc1B = vdupq_n_f64(c1B);
    const float64x2_t vc2B = vdupq_n_f64(c2B);
    const float64x2_t vc3B = vdupq_n_f64(c3B);
    const float64x2_t vparA = vdupq_n_f64(parityA);
    const float64x2_t vparB = vdupq_n_f64(parityB);
    for (; i + 2 <= n_pairs; i += 2) {
        const float64x2_t vct = vld1q_f64(ct + i);
        const float64x2_t vdp1 = vld1q_f64(dp1 + i);
        const float64x2_t vdp2 = vld1q_f64(dp2 + i);
        const float64x2_t vdm1 = vld1q_f64(dm1 + i);
        const float64x2_t vdm2 = vld1q_f64(dm2 + i);
        const float64x2_t vctermA = vmulq_f64(vc1A, vct);
        const float64x2_t vpA = vfmaq_f64(vmulq_f64(vaddq_f64(vctermA, vc2A), vdp1), vc3A, vdp2);
        const float64x2_t vmA = vfmaq_f64(vmulq_f64(vsubq_f64(vctermA, vc2A), vdm1), vc3A, vdm2);
        const float64x2_t vmsA = vmulq_f64(vmA, vparA);
        const float64x2_t vpsA = vmulq_f64(vpA, vparA);
        acc_rpA = vfmaq_f64(acc_rpA, vld1q_f64(Np_r + i), vpA);
        acc_rpA = vfmaq_f64(acc_rpA, vld1q_f64(Sp_r + i), vmsA);
        acc_ipA = vfmaq_f64(acc_ipA, vld1q_f64(Np_i + i), vpA);
        acc_ipA = vfmaq_f64(acc_ipA, vld1q_f64(Sp_i + i), vmsA);
        acc_rmA = vfmaq_f64(acc_rmA, vld1q_f64(Nm_r + i), vmA);
        acc_rmA = vfmaq_f64(acc_rmA, vld1q_f64(Sm_r + i), vpsA);
        acc_imA = vfmaq_f64(acc_imA, vld1q_f64(Nm_i + i), vmA);
        acc_imA = vfmaq_f64(acc_imA, vld1q_f64(Sm_i + i), vpsA);
        const float64x2_t vctermB = vmulq_f64(vc1B, vct);
        const float64x2_t vpB = vfmaq_f64(vmulq_f64(vaddq_f64(vctermB, vc2B), vpA), vc3B, vdp1);
        const float64x2_t vmB = vfmaq_f64(vmulq_f64(vsubq_f64(vctermB, vc2B), vmA), vc3B, vdm1);
        const float64x2_t vmsB = vmulq_f64(vmB, vparB);
        const float64x2_t vpsB = vmulq_f64(vpB, vparB);
        acc_rpB = vfmaq_f64(acc_rpB, vld1q_f64(Np_r + i), vpB);
        acc_rpB = vfmaq_f64(acc_rpB, vld1q_f64(Sp_r + i), vmsB);
        acc_ipB = vfmaq_f64(acc_ipB, vld1q_f64(Np_i + i), vpB);
        acc_ipB = vfmaq_f64(acc_ipB, vld1q_f64(Sp_i + i), vmsB);
        acc_rmB = vfmaq_f64(acc_rmB, vld1q_f64(Nm_r + i), vmB);
        acc_rmB = vfmaq_f64(acc_rmB, vld1q_f64(Sm_r + i), vpsB);
        acc_imB = vfmaq_f64(acc_imB, vld1q_f64(Nm_i + i), vmB);
        acc_imB = vfmaq_f64(acc_imB, vld1q_f64(Sm_i + i), vpsB);
        vst1q_f64(dp2 + i, vpA);
        vst1q_f64(dp1 + i, vpB);
        vst1q_f64(dm2 + i, vmA);
        vst1q_f64(dm1 + i, vmB);
    }
    r_pA += vaddvq_f64(acc_rpA);
    i_pA += vaddvq_f64(acc_ipA);
    r_mA += vaddvq_f64(acc_rmA);
    i_mA += vaddvq_f64(acc_imA);
    r_pB += vaddvq_f64(acc_rpB);
    i_pB += vaddvq_f64(acc_ipB);
    r_mB += vaddvq_f64(acc_rmB);
    i_mB += vaddvq_f64(acc_imB);
#endif
    for (; i < n_pairs; ++i) {
        const double vpA = (c1A * ct[i] + c2A) * dp1[i] + c3A * dp2[i];
        const double vmA = (c1A * ct[i] - c2A) * dm1[i] + c3A * dm2[i];
        const double vmsA = vmA * parityA;
        const double vpsA = vpA * parityA;
        r_pA += Np_r[i] * vpA + Sp_r[i] * vmsA;
        i_pA += Np_i[i] * vpA + Sp_i[i] * vmsA;
        r_mA += Nm_r[i] * vmA + Sm_r[i] * vpsA;
        i_mA += Nm_i[i] * vmA + Sm_i[i] * vpsA;
        const double vpB = (c1B * ct[i] + c2B) * vpA + c3B * dp1[i];
        const double vmB = (c1B * ct[i] - c2B) * vmA + c3B * dm1[i];
        const double vmsB = vmB * parityB;
        const double vpsB = vpB * parityB;
        r_pB += Np_r[i] * vpB + Sp_r[i] * vmsB;
        i_pB += Np_i[i] * vpB + Sp_i[i] * vmsB;
        r_mB += Nm_r[i] * vmB + Sm_r[i] * vpsB;
        i_mB += Nm_i[i] * vmB + Sm_i[i] * vpsB;
        dp2[i] = vpA;
        dp1[i] = vpB;
        dm2[i] = vmA;
        dm1[i] = vmB;
    }
}

static inline void spin_interpolate_dot_rings(
    int n_pairs,
    double r0,
    double i0,
    double r1,
    double i1,
    double parity,
    const double* __restrict__ d_p,
    const double* __restrict__ d_m,
    double* __restrict__ rNp,
    double* __restrict__ iNp,
    double* __restrict__ rNm,
    double* __restrict__ iNm,
    double* __restrict__ rSp,
    double* __restrict__ iSp,
    double* __restrict__ rSm,
    double* __restrict__ iSm
) {
    int i = 0;
#if defined(__aarch64__)
    const float64x2_t vr0 = vdupq_n_f64(r0);
    const float64x2_t vi0 = vdupq_n_f64(i0);
    const float64x2_t vr1 = vdupq_n_f64(r1);
    const float64x2_t vi1 = vdupq_n_f64(i1);
    const float64x2_t vpar = vdupq_n_f64(parity);
    auto acc_pair = [&](int off) {
        const float64x2_t vp = vld1q_f64(d_p + off);
        const float64x2_t vm = vld1q_f64(d_m + off);
        const float64x2_t vms = vmulq_f64(vm, vpar);
        const float64x2_t vps = vmulq_f64(vp, vpar);
        vst1q_f64(rNp + off, vfmaq_f64(vld1q_f64(rNp + off), vr0, vp));
        vst1q_f64(iNp + off, vfmaq_f64(vld1q_f64(iNp + off), vi0, vp));
        vst1q_f64(rNm + off, vfmaq_f64(vld1q_f64(rNm + off), vr1, vm));
        vst1q_f64(iNm + off, vfmaq_f64(vld1q_f64(iNm + off), vi1, vm));
        vst1q_f64(rSp + off, vfmaq_f64(vld1q_f64(rSp + off), vr0, vms));
        vst1q_f64(iSp + off, vfmaq_f64(vld1q_f64(iSp + off), vi0, vms));
        vst1q_f64(rSm + off, vfmaq_f64(vld1q_f64(rSm + off), vr1, vps));
        vst1q_f64(iSm + off, vfmaq_f64(vld1q_f64(iSm + off), vi1, vps));
    };
    for (; i + 4 <= n_pairs; i += 4) {
        acc_pair(i);
        acc_pair(i + 2);
    }
    if (i + 2 <= n_pairs) {
        acc_pair(i);
        i += 2;
    }
#endif
    for (; i < n_pairs; ++i) {
        rNp[i] += r0 * d_p[i];
        iNp[i] += i0 * d_p[i];
        rNm[i] += r1 * d_m[i];
        iNm[i] += i1 * d_m[i];
        rSp[i] += r0 * d_m[i] * parity;
        iSp[i] += i0 * d_m[i] * parity;
        rSm[i] += r1 * d_p[i] * parity;
        iSm[i] += i1 * d_p[i] * parity;
    }
}














static std::vector<double> precomputed_l_factors;
static std::mutex l_factors_mutex;

void ensure_l_factors(int64_t lmax) {
    std::lock_guard<std::mutex> lock(l_factors_mutex);
    if (precomputed_l_factors.size() <= (size_t)lmax) {
        precomputed_l_factors.resize(lmax + 1);
        for (int64_t l = 0; l <= lmax; ++l) {
            precomputed_l_factors[l] = std::sqrt((2.0 * l + 1.0) / (4.0 * M_PI));
        }
    }
}

static torch::Tensor ishioka_spin_lowered_interpolate_cpu(
    const c10::complex<double>* alm0_ptr,
    const c10::complex<double>* alm1_ptr,
    const torch::Tensor& theta,
    int64_t lmax,
    int64_t mmax,
    int64_t spin_abs,
    const c10::complex<double>* phase_ptr,
    c10::complex<double>* eq_plus,
    c10::complex<double>* eq_minus,
    int64_t eq_start,
    int64_t n_eq,
    int64_t nph_eq
);

static torch::Tensor ishioka_spin_lowered_integrate_cpu(
    const torch::Tensor& s_plus,
    const torch::Tensor& s_minus,
    const torch::Tensor& theta,
    const torch::Tensor& weight,
    int64_t lmax,
    int64_t mmax,
    int64_t spin_abs
);


torch::Tensor healpix_spin_integrate_recurrence_cpu(
    const torch::Tensor& s_plus,
    const torch::Tensor& s_minus,
    const torch::Tensor& theta,
    const torch::Tensor& weight,
    int64_t lmax,
    int64_t mmax,
    int64_t spin
) {
    auto sp_f = s_plus.contiguous(); auto sm_f = s_minus.contiguous();
    auto th_ptr = theta.data_ptr<double>();
    auto w_ptr = weight.data_ptr<double>();
    const bool w_scalar = weight.numel() == 1;
    const c10::complex<double>* __restrict__ sp_ptr = sp_f.data_ptr<c10::complex<double>>();
    const c10::complex<double>* __restrict__ sm_ptr = sm_f.data_ptr<c10::complex<double>>();
    const int64_t spin_abs = std::abs(spin);
    if ((spin_abs == 1 || spin_abs == 2) && spin_ishioka_enabled()) {
        return ishioka_spin_lowered_integrate_cpu(
            sp_f, sm_f, theta, weight, lmax, mmax, spin_abs
        );
    }

    int64_t nrings = theta.size(0);
    int64_t n_eff = (nrings + 1) / 2;
    int64_t n_pairs = nrings / 2;
    bool has_equator = (nrings % 2 != 0);
    ensure_l_factors(lmax);

    auto options = torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU);
    const int64_t nalm = alm_size(lmax, mmax);
    torch::Tensor alms = torch::zeros({2, nalm}, options);
    auto cp_out = alms.select(0, 0).data_ptr<c10::complex<double>>();
    auto cm_out = alms.select(0, 1).data_ptr<c10::complex<double>>();

    std::vector<int64_t> m_offsets(mmax + 1);
    int64_t current_lut_off = 0;
    for (int64_t m = 0; m <= mmax; ++m) {
        m_offsets[m] = current_lut_off;
        int64_t l0 = std::max((int64_t)std::abs(spin), m);
        if (lmax >= l0 + 2) {
            current_lut_off += (lmax - (l0 + 2) + 1);
        }
    }
    const auto& lut = g_recurrence_cache.get_coeffs(lmax, mmax, spin, m_offsets);

    std::vector<double> cos_theta_all(n_eff), sin_half_all(n_eff), cos_half_all(n_eff);
    for (int i = 0; i < n_eff; ++i) {
        const double th = th_ptr[i];
        cos_theta_all[static_cast<size_t>(i)] = std::cos(th);
        sin_half_all[static_cast<size_t>(i)] = std::sin(th * 0.5);
        cos_half_all[static_cast<size_t>(i)] = std::cos(th * 0.5);
    }
    std::vector<int64_t> mlim_north(n_eff, lmax);
    if (spin_mlim_prune_enabled()) {
        const int64_t spin_abs = std::abs(spin);
        for (int64_t i = 0; i < n_eff; ++i) {
            const double cth = cos_theta_all[static_cast<size_t>(i)];
            const double sth = std::sqrt(std::max(0.0, 1.0 - cth * cth));
            mlim_north[static_cast<size_t>(i)] = sharp_mlim_estimate(lmax, spin_abs, sth, cth);
        }
    }

    parallel_m_ranges(mmax, [&](int64_t m0, int64_t stride) {
        // Allocate working buffers ONCE per thread task, rather than per inner loop.
        std::vector<double> v_dp2(n_eff), v_dp1(n_eff), v_dpc(n_eff);
        std::vector<double> v_dm2(n_eff), v_dm1(n_eff), v_dmc(n_eff);
        std::vector<uint8_t> v_active(n_eff);
        double * __restrict__ dp2 = v_dp2.data();
        double * __restrict__ dp1 = v_dp1.data();
        double * __restrict__ dpc = v_dpc.data();
        double * __restrict__ dm2 = v_dm2.data();
        double * __restrict__ dm1 = v_dm1.data();
        double * __restrict__ dmc = v_dmc.data();
        uint8_t * __restrict__ active = v_active.data();
        const double * __restrict__ ct_ptr = cos_theta_all.data();

        std::vector<double> vNp_r(n_eff), vNp_i(n_eff), vNm_r(n_eff), vNm_i(n_eff);
        std::vector<double> vSp_r(n_eff), vSp_i(n_eff), vSm_r(n_eff), vSm_i(n_eff);
        double * __restrict__ Np_r = vNp_r.data(); double * __restrict__ Np_i = vNp_i.data();
        double * __restrict__ Nm_r = vNm_r.data(); double * __restrict__ Nm_i = vNm_i.data();
        double * __restrict__ Sp_r = vSp_r.data(); double * __restrict__ Sp_i = vSp_i.data();
        double * __restrict__ Sm_r = vSm_r.data(); double * __restrict__ Sm_i = vSm_i.data();
        const double * __restrict__ sh_ptr = sin_half_all.data();
        const double * __restrict__ ch_ptr = cos_half_all.data();

        for (int64_t m = m0; m <= mmax; m += stride) {
            int64_t l0 = std::max((int64_t)std::abs(spin), m);
            int64_t idx_base = m * (lmax + 1) - (m * (m - 1)) / 2;
            const double pre_phase = ((m % 2 != 0) ? -1.0 : 1.0) * ((std::abs(spin) % 2 != 0) ? -1.0 : 1.0);
            const bool spin_active = (spin != 0);
            const auto* lut_ptr = &lut[m_offsets[m]];
            bool any_active = false;
            const int64_t m_off = m * nrings;

            for (int i=0; i < n_eff; ++i) {
                const bool on = (m <= mlim_north[static_cast<size_t>(i)]);
                active[i] = static_cast<uint8_t>(on ? 1 : 0);
                any_active = any_active || on;
                if (!on) {
                    Np_r[i] = Np_i[i] = Nm_r[i] = Nm_i[i] = 0.0;
                    Sp_r[i] = Sp_i[i] = Sm_r[i] = Sm_i[i] = 0.0;
                    continue;
                }
                const double wN = w_scalar ? w_ptr[0] : w_ptr[i];
                const auto np = sp_ptr[m_off + i];
                const auto nm = sm_ptr[m_off + i];
                Np_r[i] = np.real() * wN; Np_i[i] = np.imag() * wN;
                Nm_r[i] = nm.real() * wN; Nm_i[i] = nm.imag() * wN;
                if (i < n_pairs) {
                    const int64_t idx_S = nrings - 1 - i;
                    const double wS = w_scalar ? w_ptr[0] : w_ptr[idx_S];
                    const auto spv = sp_ptr[m_off + idx_S];
                    const auto smv = sm_ptr[m_off + idx_S];
                    Sp_r[i] = spv.real() * wS; Sp_i[i] = spv.imag() * wS;
                    Sm_r[i] = smv.real() * wS; Sm_i[i] = smv.imag() * wS;
                }
            }
            if (!any_active) {
                continue;
            }
            int pair0 = 0;
            int n_run = 0;
            spin_active_pair_span(active, static_cast<int>(n_pairs), pair0, n_run);

            WignerDCoeffs coeffs_p1 = get_wigner_d_coeffs(l0, m, -spin);
            WignerDCoeffs coeffs_m1; if (spin_active) coeffs_m1 = get_wigner_d_coeffs(l0, m, spin);

            for (int i=0; i < n_eff; ++i) {
                if (!active[i]) {
                    dp1[i] = 0.0;
                    dm1[i] = 0.0;
                    dp2[i] = 0.0;
                    dm2[i] = 0.0;
                    continue;
                }
                dp1[i] = eval_wigner_d_half(coeffs_p1, sh_ptr[i], ch_ptr[i]);
                if(spin_active) dm1[i] = eval_wigner_d_half(coeffs_m1, sh_ptr[i], ch_ptr[i]);
                else dm1[i] = dp1[i];
                dp2[i] = 0.0;
                dm2[i] = 0.0;
            }

            auto accumulate_l_internal = [&](int64_t l, const double* __restrict__ d_p, const double* __restrict__ d_m) {
                double r_p = 0, i_p = 0, r_m = 0, i_m = 0;
                const double parity = ((l + m) % 2 == 0) ? 1.0 : -1.0;
                spin_integrate_dot_rings(
                    n_run,
                    parity,
                    d_p + pair0,
                    d_m + pair0,
                    Np_r + pair0,
                    Np_i + pair0,
                    Nm_r + pair0,
                    Nm_i + pair0,
                    Sp_r + pair0,
                    Sp_i + pair0,
                    Sm_r + pair0,
                    Sm_i + pair0,
                    r_p,
                    i_p,
                    r_m,
                    i_m
                );
                if (has_equator && active[n_pairs]) {
                    int i = n_pairs;
                    r_p += Np_r[i]*d_p[i]; i_p += Np_i[i]*d_p[i];
                    r_m += Nm_r[i]*d_m[i]; i_m += Nm_i[i]*d_m[i];
                }
                const double sc = pre_phase * precomputed_l_factors[l];
                cp_out[idx_base+(l-m)] = c10::complex<double>(r_p*sc, i_p*sc);
                cm_out[idx_base+(l-m)] = c10::complex<double>(r_m*sc, i_m*sc);
            };

            accumulate_l_internal(l0, dp1, dm1);

            if (l0 < lmax) {
                WignerDCoeffs coeffs_pc = get_wigner_d_coeffs(l0+1, m, -spin);
                WignerDCoeffs coeffs_mc; if (spin_active) coeffs_mc = get_wigner_d_coeffs(l0+1, m, spin);

                for (int i=0; i < n_eff; ++i) {
                    if (!active[i]) {
                        dpc[i] = 0.0;
                        dmc[i] = 0.0;
                        continue;
                    }
                    dpc[i] = eval_wigner_d_half(coeffs_pc, sh_ptr[i], ch_ptr[i]);
                    if(spin_active) dmc[i] = eval_wigner_d_half(coeffs_mc, sh_ptr[i], ch_ptr[i]);
                    else dmc[i] = dpc[i];
                }
                accumulate_l_internal(l0+1, dpc, dmc);
                { auto t = dp2; dp2 = dp1; dp1 = dpc; dpc = t; }
                { auto t = dm2; dm2 = dm1; dm1 = dmc; dmc = t; }

                int64_t l = l0 + 2;
                for (; l + 1 <= lmax; l += 2) {
                    const auto& cA = lut_ptr[l - (l0 + 2)];
                    const auto& cB = lut_ptr[l + 1 - (l0 + 2)];
                    const double parityA = ((l + m) % 2 == 0) ? 1.0 : -1.0;
                    double r_pA = 0, i_pA = 0, r_mA = 0, i_mA = 0;
                    double r_pB = 0, i_pB = 0, r_mB = 0, i_mB = 0;
                    spin_integrate_recurrence_ll_rings(
                        n_run,
                        cA.c1, cA.c2_p, cA.c3,
                        cB.c1, cB.c2_p, cB.c3,
                        parityA,
                        ct_ptr + pair0,
                        dp1 + pair0,
                        dp2 + pair0,
                        dm1 + pair0,
                        dm2 + pair0,
                        Np_r + pair0,
                        Np_i + pair0,
                        Nm_r + pair0,
                        Nm_i + pair0,
                        Sp_r + pair0,
                        Sp_i + pair0,
                        Sm_r + pair0,
                        Sm_i + pair0,
                        r_pA, i_pA, r_mA, i_mA,
                        r_pB, i_pB, r_mB, i_mB
                    );
                    if (has_equator && active[n_pairs]) {
                        int eq = static_cast<int>(n_pairs);
                        const double vpA = (cA.c1 * ct_ptr[eq] + cA.c2_p) * dp1[eq] + cA.c3 * dp2[eq];
                        const double vmA = (cA.c1 * ct_ptr[eq] - cA.c2_p) * dm1[eq] + cA.c3 * dm2[eq];
                        r_pA += Np_r[eq] * vpA; i_pA += Np_i[eq] * vpA;
                        r_mA += Nm_r[eq] * vmA; i_mA += Nm_i[eq] * vmA;
                        const double vpB = (cB.c1 * ct_ptr[eq] + cB.c2_p) * vpA + cB.c3 * dp1[eq];
                        const double vmB = (cB.c1 * ct_ptr[eq] - cB.c2_p) * vmA + cB.c3 * dm1[eq];
                        r_pB += Np_r[eq] * vpB; i_pB += Np_i[eq] * vpB;
                        r_mB += Nm_r[eq] * vmB; i_mB += Nm_i[eq] * vmB;
                        dp2[eq] = vpA; dp1[eq] = vpB;
                        dm2[eq] = vmA; dm1[eq] = vmB;
                    }
                    const double scA = pre_phase * precomputed_l_factors[static_cast<size_t>(l)];
                    const double scB = pre_phase * precomputed_l_factors[static_cast<size_t>(l + 1)];
                    cp_out[idx_base + (l - m)] = c10::complex<double>(r_pA * scA, i_pA * scA);
                    cm_out[idx_base + (l - m)] = c10::complex<double>(r_mA * scA, i_mA * scA);
                    cp_out[idx_base + (l + 1 - m)] = c10::complex<double>(r_pB * scB, i_pB * scB);
                    cm_out[idx_base + (l + 1 - m)] = c10::complex<double>(r_mB * scB, i_mB * scB);
                }
                if (l <= lmax) {
                    const auto& c = lut_ptr[l - (l0 + 2)];
                    const double c1 = c.c1, c2 = c.c2_p, c3 = c.c3;
                    const double parity = ((l+m)%2 == 0) ? 1.0 : -1.0;
                    double r_p = 0, i_p = 0, r_m = 0, i_m = 0;
                    spin_integrate_recurrence_l_rings(
                        n_run,
                        c1,
                        c2,
                        c3,
                        parity,
                        ct_ptr + pair0,
                        dp1 + pair0,
                        dp2 + pair0,
                        dm1 + pair0,
                        dm2 + pair0,
                        dpc + pair0,
                        dmc + pair0,
                        Np_r + pair0,
                        Np_i + pair0,
                        Nm_r + pair0,
                        Nm_i + pair0,
                        Sp_r + pair0,
                        Sp_i + pair0,
                        Sm_r + pair0,
                        Sm_i + pair0,
                        r_p,
                        i_p,
                        r_m,
                        i_m
                    );
                    if (has_equator && active[n_pairs]) {
                        int eq = static_cast<int>(n_pairs);
                        const double vpN = (c1*ct_ptr[eq] + c2)*dp1[eq] + c3*dp2[eq];
                        const double vmN = (c1*ct_ptr[eq] - c2)*dm1[eq] + c3*dm2[eq];
                        dpc[eq] = vpN; dmc[eq] = vmN;
                        r_p += Np_r[eq]*vpN; i_p += Np_i[eq]*vpN;
                        r_m += Nm_r[eq]*vmN; i_m += Nm_i[eq]*vmN;
                    }
                    const double sc = pre_phase * precomputed_l_factors[static_cast<size_t>(l)];
                    cp_out[idx_base+(l-m)] = c10::complex<double>(r_p*sc, i_p*sc);
                    cm_out[idx_base+(l-m)] = c10::complex<double>(r_m*sc, i_m*sc);
                }
            }
        }
    });

    return alms;
}

static inline void spin_interpolate_l_scan_one(
    int64_t l0,
    int64_t lmax,
    int64_t m,
    const RecurrenceCoeffs* lut_ptr,
    const c10::complex<double>* alm0_ptr,
    const c10::complex<double>* alm1_ptr,
    int64_t idx_base,
    double pre_phase,
    bool spin_active,
    double ct,
    double dp1,
    double dp2,
    double dm1,
    double dm2,
    double& rNp,
    double& iNp,
    double& rNm,
    double& iNm,
    double& rSp,
    double& iSp,
    double& rSm,
    double& iSm
) {
    double par = ((l0 + 2 + m) % 2 == 0) ? 1.0 : -1.0;
    int64_t l = l0 + 2;
    for (; l + 1 <= lmax; l += 2) {
        const auto& cA = lut_ptr[l - (l0 + 2)];
        const auto& cB = lut_ptr[l + 1 - (l0 + 2)];
        const auto a0A = alm0_ptr[idx_base + (l - m)];
        const auto a1A = spin_active ? alm1_ptr[idx_base + (l - m)] : a0A;
        const auto a0B = alm0_ptr[idx_base + (l + 1 - m)];
        const auto a1B = spin_active ? alm1_ptr[idx_base + (l + 1 - m)] : a0B;
        const double scA = pre_phase * precomputed_l_factors[static_cast<size_t>(l)];
        const double scB = pre_phase * precomputed_l_factors[static_cast<size_t>(l + 1)];
        const double r0A = a0A.real() * scA, i0A = a0A.imag() * scA;
        const double r1A = a1A.real() * scA, i1A = a1A.imag() * scA;
        const double r0B = a0B.real() * scB, i0B = a0B.imag() * scB;
        const double r1B = a1B.real() * scB, i1B = a1B.imag() * scB;
        const double vpA = (cA.c1 * ct + cA.c2_p) * dp1 + cA.c3 * dp2;
        const double vmA = (cA.c1 * ct - cA.c2_p) * dm1 + cA.c3 * dm2;
        rNp += r0A * vpA; iNp += i0A * vpA;
        rNm += r1A * vmA; iNm += i1A * vmA;
        rSp += r0A * vmA * par; iSp += i0A * vmA * par;
        rSm += r1A * vpA * par; iSm += i1A * vpA * par;
        const double parB = -par;
        const double vpB = (cB.c1 * ct + cB.c2_p) * vpA + cB.c3 * dp1;
        const double vmB = (cB.c1 * ct - cB.c2_p) * vmA + cB.c3 * dm1;
        rNp += r0B * vpB; iNp += i0B * vpB;
        rNm += r1B * vmB; iNm += i1B * vmB;
        rSp += r0B * vmB * parB; iSp += i0B * vmB * parB;
        rSm += r1B * vpB * parB; iSm += i1B * vpB * parB;
        dp2 = vpA; dp1 = vpB;
        dm2 = vmA; dm1 = vmB;
    }
    if (l <= lmax) {
        const auto& c = lut_ptr[l - (l0 + 2)];
        const auto a0 = alm0_ptr[idx_base + (l - m)];
        const auto a1 = spin_active ? alm1_ptr[idx_base + (l - m)] : a0;
        const double sc = pre_phase * precomputed_l_factors[static_cast<size_t>(l)];
        const double r0 = a0.real() * sc, i0 = a0.imag() * sc;
        const double r1 = a1.real() * sc, i1 = a1.imag() * sc;
        const double vp = (c.c1 * ct + c.c2_p) * dp1 + c.c3 * dp2;
        const double vm = (c.c1 * ct - c.c2_p) * dm1 + c.c3 * dm2;
        rNp += r0 * vp; iNp += i0 * vp;
        rNm += r1 * vm; iNm += i1 * vm;
        rSp += r0 * vm * par; iSp += i0 * vm * par;
        rSm += r1 * vp * par; iSm += i1 * vp * par;
    }
}

#if defined(__aarch64__)
static inline void spin_interpolate_l_scan_four(
    int64_t l0,
    int64_t lmax,
    int64_t m,
    const RecurrenceCoeffs* lut_ptr,
    const c10::complex<double>* alm0_ptr,
    const c10::complex<double>* alm1_ptr,
    int64_t idx_base,
    double pre_phase,
    bool spin_active,
    const double* __restrict__ ct,
    const double* __restrict__ dp1_in,
    const double* __restrict__ dp2_in,
    const double* __restrict__ dm1_in,
    const double* __restrict__ dm2_in,
    double* __restrict__ rNp,
    double* __restrict__ iNp,
    double* __restrict__ rNm,
    double* __restrict__ iNm,
    double* __restrict__ rSp,
    double* __restrict__ iSp,
    double* __restrict__ rSm,
    double* __restrict__ iSm
) {
    float64x2_t vctA = vld1q_f64(ct);
    float64x2_t vctB = vld1q_f64(ct + 2);
    float64x2_t vdp1A = vld1q_f64(dp1_in);
    float64x2_t vdp1B = vld1q_f64(dp1_in + 2);
    float64x2_t vdp2A = vld1q_f64(dp2_in);
    float64x2_t vdp2B = vld1q_f64(dp2_in + 2);
    float64x2_t vdm1A = vld1q_f64(dm1_in);
    float64x2_t vdm1B = vld1q_f64(dm1_in + 2);
    float64x2_t vdm2A = vld1q_f64(dm2_in);
    float64x2_t vdm2B = vld1q_f64(dm2_in + 2);
    float64x2_t vrNpA = vld1q_f64(rNp);
    float64x2_t vrNpB = vld1q_f64(rNp + 2);
    float64x2_t viNpA = vld1q_f64(iNp);
    float64x2_t viNpB = vld1q_f64(iNp + 2);
    float64x2_t vrNmA = vld1q_f64(rNm);
    float64x2_t vrNmB = vld1q_f64(rNm + 2);
    float64x2_t viNmA = vld1q_f64(iNm);
    float64x2_t viNmB = vld1q_f64(iNm + 2);
    float64x2_t vrSpA = vld1q_f64(rSp);
    float64x2_t vrSpB = vld1q_f64(rSp + 2);
    float64x2_t viSpA = vld1q_f64(iSp);
    float64x2_t viSpB = vld1q_f64(iSp + 2);
    float64x2_t vrSmA = vld1q_f64(rSm);
    float64x2_t vrSmB = vld1q_f64(rSm + 2);
    float64x2_t viSmA = vld1q_f64(iSm);
    float64x2_t viSmB = vld1q_f64(iSm + 2);
    float64x2_t vpar = vdupq_n_f64(((l0 + 2 + m) % 2 == 0) ? 1.0 : -1.0);
    int64_t l = l0 + 2;
    for (; l + 1 <= lmax; l += 2) {
        const auto& cA = lut_ptr[l - (l0 + 2)];
        const auto& cB = lut_ptr[l + 1 - (l0 + 2)];
        const auto a0A = alm0_ptr[idx_base + (l - m)];
        const auto a1A = spin_active ? alm1_ptr[idx_base + (l - m)] : a0A;
        const auto a0B = alm0_ptr[idx_base + (l + 1 - m)];
        const auto a1B = spin_active ? alm1_ptr[idx_base + (l + 1 - m)] : a0B;
        const double scA = pre_phase * precomputed_l_factors[static_cast<size_t>(l)];
        const double scB = pre_phase * precomputed_l_factors[static_cast<size_t>(l + 1)];
        const float64x2_t vr0A = vdupq_n_f64(a0A.real() * scA);
        const float64x2_t vi0A = vdupq_n_f64(a0A.imag() * scA);
        const float64x2_t vr1A = vdupq_n_f64(a1A.real() * scA);
        const float64x2_t vi1A = vdupq_n_f64(a1A.imag() * scA);
        const float64x2_t vr0B = vdupq_n_f64(a0B.real() * scB);
        const float64x2_t vi0B = vdupq_n_f64(a0B.imag() * scB);
        const float64x2_t vr1B = vdupq_n_f64(a1B.real() * scB);
        const float64x2_t vi1B = vdupq_n_f64(a1B.imag() * scB);
        const float64x2_t vc1A = vdupq_n_f64(cA.c1);
        const float64x2_t vc2A = vdupq_n_f64(cA.c2_p);
        const float64x2_t vc3A = vdupq_n_f64(cA.c3);
        const float64x2_t vc1B = vdupq_n_f64(cB.c1);
        const float64x2_t vc2B = vdupq_n_f64(cB.c2_p);
        const float64x2_t vc3B = vdupq_n_f64(cB.c3);
        const float64x2_t vctermA0 = vmulq_f64(vc1A, vctA);
        const float64x2_t vctermA1 = vmulq_f64(vc1A, vctB);
        const float64x2_t vpA0 = vfmaq_f64(vmulq_f64(vaddq_f64(vctermA0, vc2A), vdp1A), vc3A, vdp2A);
        const float64x2_t vpA1 = vfmaq_f64(vmulq_f64(vaddq_f64(vctermA1, vc2A), vdp1B), vc3A, vdp2B);
        const float64x2_t vmA0 = vfmaq_f64(vmulq_f64(vsubq_f64(vctermA0, vc2A), vdm1A), vc3A, vdm2A);
        const float64x2_t vmA1 = vfmaq_f64(vmulq_f64(vsubq_f64(vctermA1, vc2A), vdm1B), vc3A, vdm2B);
        vrNpA = vfmaq_f64(vrNpA, vr0A, vpA0);
        vrNpB = vfmaq_f64(vrNpB, vr0A, vpA1);
        viNpA = vfmaq_f64(viNpA, vi0A, vpA0);
        viNpB = vfmaq_f64(viNpB, vi0A, vpA1);
        vrNmA = vfmaq_f64(vrNmA, vr1A, vmA0);
        vrNmB = vfmaq_f64(vrNmB, vr1A, vmA1);
        viNmA = vfmaq_f64(viNmA, vi1A, vmA0);
        viNmB = vfmaq_f64(viNmB, vi1A, vmA1);
        vrSpA = vfmaq_f64(vrSpA, vr0A, vmulq_f64(vmA0, vpar));
        vrSpB = vfmaq_f64(vrSpB, vr0A, vmulq_f64(vmA1, vpar));
        viSpA = vfmaq_f64(viSpA, vi0A, vmulq_f64(vmA0, vpar));
        viSpB = vfmaq_f64(viSpB, vi0A, vmulq_f64(vmA1, vpar));
        vrSmA = vfmaq_f64(vrSmA, vr1A, vmulq_f64(vpA0, vpar));
        vrSmB = vfmaq_f64(vrSmB, vr1A, vmulq_f64(vpA1, vpar));
        viSmA = vfmaq_f64(viSmA, vi1A, vmulq_f64(vpA0, vpar));
        viSmB = vfmaq_f64(viSmB, vi1A, vmulq_f64(vpA1, vpar));
        const float64x2_t vparB = vnegq_f64(vpar);
        const float64x2_t vctermB0 = vmulq_f64(vc1B, vctA);
        const float64x2_t vctermB1 = vmulq_f64(vc1B, vctB);
        const float64x2_t vpB0 = vfmaq_f64(vmulq_f64(vaddq_f64(vctermB0, vc2B), vpA0), vc3B, vdp1A);
        const float64x2_t vpB1 = vfmaq_f64(vmulq_f64(vaddq_f64(vctermB1, vc2B), vpA1), vc3B, vdp1B);
        const float64x2_t vmB0 = vfmaq_f64(vmulq_f64(vsubq_f64(vctermB0, vc2B), vmA0), vc3B, vdm1A);
        const float64x2_t vmB1 = vfmaq_f64(vmulq_f64(vsubq_f64(vctermB1, vc2B), vmA1), vc3B, vdm1B);
        vrNpA = vfmaq_f64(vrNpA, vr0B, vpB0);
        vrNpB = vfmaq_f64(vrNpB, vr0B, vpB1);
        viNpA = vfmaq_f64(viNpA, vi0B, vpB0);
        viNpB = vfmaq_f64(viNpB, vi0B, vpB1);
        vrNmA = vfmaq_f64(vrNmA, vr1B, vmB0);
        vrNmB = vfmaq_f64(vrNmB, vr1B, vmB1);
        viNmA = vfmaq_f64(viNmA, vi1B, vmB0);
        viNmB = vfmaq_f64(viNmB, vi1B, vmB1);
        vrSpA = vfmaq_f64(vrSpA, vr0B, vmulq_f64(vmB0, vparB));
        vrSpB = vfmaq_f64(vrSpB, vr0B, vmulq_f64(vmB1, vparB));
        viSpA = vfmaq_f64(viSpA, vi0B, vmulq_f64(vmB0, vparB));
        viSpB = vfmaq_f64(viSpB, vi0B, vmulq_f64(vmB1, vparB));
        vrSmA = vfmaq_f64(vrSmA, vr1B, vmulq_f64(vpB0, vparB));
        vrSmB = vfmaq_f64(vrSmB, vr1B, vmulq_f64(vpB1, vparB));
        viSmA = vfmaq_f64(viSmA, vi1B, vmulq_f64(vpB0, vparB));
        viSmB = vfmaq_f64(viSmB, vi1B, vmulq_f64(vpB1, vparB));
        vdp2A = vpA0; vdp1A = vpB0;
        vdp2B = vpA1; vdp1B = vpB1;
        vdm2A = vmA0; vdm1A = vmB0;
        vdm2B = vmA1; vdm1B = vmB1;
    }
    if (l <= lmax) {
        const auto& c = lut_ptr[l - (l0 + 2)];
        const auto a0 = alm0_ptr[idx_base + (l - m)];
        const auto a1 = spin_active ? alm1_ptr[idx_base + (l - m)] : a0;
        const double sc = pre_phase * precomputed_l_factors[static_cast<size_t>(l)];
        const float64x2_t vr0 = vdupq_n_f64(a0.real() * sc);
        const float64x2_t vi0 = vdupq_n_f64(a0.imag() * sc);
        const float64x2_t vr1 = vdupq_n_f64(a1.real() * sc);
        const float64x2_t vi1 = vdupq_n_f64(a1.imag() * sc);
        const float64x2_t vc1 = vdupq_n_f64(c.c1);
        const float64x2_t vc2 = vdupq_n_f64(c.c2_p);
        const float64x2_t vc3 = vdupq_n_f64(c.c3);
        const float64x2_t vctermA = vmulq_f64(vc1, vctA);
        const float64x2_t vctermB = vmulq_f64(vc1, vctB);
        const float64x2_t vpA = vfmaq_f64(vmulq_f64(vaddq_f64(vctermA, vc2), vdp1A), vc3, vdp2A);
        const float64x2_t vpB = vfmaq_f64(vmulq_f64(vaddq_f64(vctermB, vc2), vdp1B), vc3, vdp2B);
        const float64x2_t vmA = vfmaq_f64(vmulq_f64(vsubq_f64(vctermA, vc2), vdm1A), vc3, vdm2A);
        const float64x2_t vmB = vfmaq_f64(vmulq_f64(vsubq_f64(vctermB, vc2), vdm1B), vc3, vdm2B);
        vrNpA = vfmaq_f64(vrNpA, vr0, vpA);
        vrNpB = vfmaq_f64(vrNpB, vr0, vpB);
        viNpA = vfmaq_f64(viNpA, vi0, vpA);
        viNpB = vfmaq_f64(viNpB, vi0, vpB);
        vrNmA = vfmaq_f64(vrNmA, vr1, vmA);
        vrNmB = vfmaq_f64(vrNmB, vr1, vmB);
        viNmA = vfmaq_f64(viNmA, vi1, vmA);
        viNmB = vfmaq_f64(viNmB, vi1, vmB);
        vrSpA = vfmaq_f64(vrSpA, vr0, vmulq_f64(vmA, vpar));
        vrSpB = vfmaq_f64(vrSpB, vr0, vmulq_f64(vmB, vpar));
        viSpA = vfmaq_f64(viSpA, vi0, vmulq_f64(vmA, vpar));
        viSpB = vfmaq_f64(viSpB, vi0, vmulq_f64(vmB, vpar));
        vrSmA = vfmaq_f64(vrSmA, vr1, vmulq_f64(vpA, vpar));
        vrSmB = vfmaq_f64(vrSmB, vr1, vmulq_f64(vpB, vpar));
        viSmA = vfmaq_f64(viSmA, vi1, vmulq_f64(vpA, vpar));
        viSmB = vfmaq_f64(viSmB, vi1, vmulq_f64(vpB, vpar));
    }
    vst1q_f64(rNp, vrNpA); vst1q_f64(rNp + 2, vrNpB);
    vst1q_f64(iNp, viNpA); vst1q_f64(iNp + 2, viNpB);
    vst1q_f64(rNm, vrNmA); vst1q_f64(rNm + 2, vrNmB);
    vst1q_f64(iNm, viNmA); vst1q_f64(iNm + 2, viNmB);
    vst1q_f64(rSp, vrSpA); vst1q_f64(rSp + 2, vrSpB);
    vst1q_f64(iSp, viSpA); vst1q_f64(iSp + 2, viSpB);
    vst1q_f64(rSm, vrSmA); vst1q_f64(rSm + 2, vrSmB);
    vst1q_f64(iSm, viSmA); vst1q_f64(iSm + 2, viSmB);
}

static inline void spin_interpolate_l_scan_two(
    int64_t l0,
    int64_t lmax,
    int64_t m,
    const RecurrenceCoeffs* lut_ptr,
    const c10::complex<double>* alm0_ptr,
    const c10::complex<double>* alm1_ptr,
    int64_t idx_base,
    double pre_phase,
    bool spin_active,
    const double* __restrict__ ct,
    const double* __restrict__ dp1_in,
    const double* __restrict__ dp2_in,
    const double* __restrict__ dm1_in,
    const double* __restrict__ dm2_in,
    double* __restrict__ rNp,
    double* __restrict__ iNp,
    double* __restrict__ rNm,
    double* __restrict__ iNm,
    double* __restrict__ rSp,
    double* __restrict__ iSp,
    double* __restrict__ rSm,
    double* __restrict__ iSm
) {
    float64x2_t vct = vld1q_f64(ct);
    float64x2_t vdp1 = vld1q_f64(dp1_in);
    float64x2_t vdp2 = vld1q_f64(dp2_in);
    float64x2_t vdm1 = vld1q_f64(dm1_in);
    float64x2_t vdm2 = vld1q_f64(dm2_in);
    float64x2_t vrNp = vld1q_f64(rNp);
    float64x2_t viNp = vld1q_f64(iNp);
    float64x2_t vrNm = vld1q_f64(rNm);
    float64x2_t viNm = vld1q_f64(iNm);
    float64x2_t vrSp = vld1q_f64(rSp);
    float64x2_t viSp = vld1q_f64(iSp);
    float64x2_t vrSm = vld1q_f64(rSm);
    float64x2_t viSm = vld1q_f64(iSm);
    float64x2_t vpar = vdupq_n_f64(((l0 + 2 + m) % 2 == 0) ? 1.0 : -1.0);
    int64_t l = l0 + 2;
    for (; l + 1 <= lmax; l += 2) {
        const auto& cA = lut_ptr[l - (l0 + 2)];
        const auto& cB = lut_ptr[l + 1 - (l0 + 2)];
        const auto a0A = alm0_ptr[idx_base + (l - m)];
        const auto a1A = spin_active ? alm1_ptr[idx_base + (l - m)] : a0A;
        const auto a0B = alm0_ptr[idx_base + (l + 1 - m)];
        const auto a1B = spin_active ? alm1_ptr[idx_base + (l + 1 - m)] : a0B;
        const double scA = pre_phase * precomputed_l_factors[static_cast<size_t>(l)];
        const double scB = pre_phase * precomputed_l_factors[static_cast<size_t>(l + 1)];
        const float64x2_t vr0A = vdupq_n_f64(a0A.real() * scA);
        const float64x2_t vi0A = vdupq_n_f64(a0A.imag() * scA);
        const float64x2_t vr1A = vdupq_n_f64(a1A.real() * scA);
        const float64x2_t vi1A = vdupq_n_f64(a1A.imag() * scA);
        const float64x2_t vr0B = vdupq_n_f64(a0B.real() * scB);
        const float64x2_t vi0B = vdupq_n_f64(a0B.imag() * scB);
        const float64x2_t vr1B = vdupq_n_f64(a1B.real() * scB);
        const float64x2_t vi1B = vdupq_n_f64(a1B.imag() * scB);
        const float64x2_t vc1A = vdupq_n_f64(cA.c1);
        const float64x2_t vc2A = vdupq_n_f64(cA.c2_p);
        const float64x2_t vc3A = vdupq_n_f64(cA.c3);
        const float64x2_t vctermA = vmulq_f64(vc1A, vct);
        const float64x2_t vpA = vfmaq_f64(vmulq_f64(vaddq_f64(vctermA, vc2A), vdp1), vc3A, vdp2);
        const float64x2_t vmA = vfmaq_f64(vmulq_f64(vsubq_f64(vctermA, vc2A), vdm1), vc3A, vdm2);
        vrNp = vfmaq_f64(vrNp, vr0A, vpA);
        viNp = vfmaq_f64(viNp, vi0A, vpA);
        vrNm = vfmaq_f64(vrNm, vr1A, vmA);
        viNm = vfmaq_f64(viNm, vi1A, vmA);
        vrSp = vfmaq_f64(vrSp, vr0A, vmulq_f64(vmA, vpar));
        viSp = vfmaq_f64(viSp, vi0A, vmulq_f64(vmA, vpar));
        vrSm = vfmaq_f64(vrSm, vr1A, vmulq_f64(vpA, vpar));
        viSm = vfmaq_f64(viSm, vi1A, vmulq_f64(vpA, vpar));
        const float64x2_t vparB = vnegq_f64(vpar);
        const float64x2_t vc1B = vdupq_n_f64(cB.c1);
        const float64x2_t vc2B = vdupq_n_f64(cB.c2_p);
        const float64x2_t vc3B = vdupq_n_f64(cB.c3);
        const float64x2_t vctermB = vmulq_f64(vc1B, vct);
        const float64x2_t vpB = vfmaq_f64(vmulq_f64(vaddq_f64(vctermB, vc2B), vpA), vc3B, vdp1);
        const float64x2_t vmB = vfmaq_f64(vmulq_f64(vsubq_f64(vctermB, vc2B), vmA), vc3B, vdm1);
        vrNp = vfmaq_f64(vrNp, vr0B, vpB);
        viNp = vfmaq_f64(viNp, vi0B, vpB);
        vrNm = vfmaq_f64(vrNm, vr1B, vmB);
        viNm = vfmaq_f64(viNm, vi1B, vmB);
        vrSp = vfmaq_f64(vrSp, vr0B, vmulq_f64(vmB, vparB));
        viSp = vfmaq_f64(viSp, vi0B, vmulq_f64(vmB, vparB));
        vrSm = vfmaq_f64(vrSm, vr1B, vmulq_f64(vpB, vparB));
        viSm = vfmaq_f64(viSm, vi1B, vmulq_f64(vpB, vparB));
        vdp2 = vpA;
        vdp1 = vpB;
        vdm2 = vmA;
        vdm1 = vmB;
    }
    if (l <= lmax) {
        const auto& c = lut_ptr[l - (l0 + 2)];
        const auto a0 = alm0_ptr[idx_base + (l - m)];
        const auto a1 = spin_active ? alm1_ptr[idx_base + (l - m)] : a0;
        const double sc = pre_phase * precomputed_l_factors[static_cast<size_t>(l)];
        const float64x2_t vr0 = vdupq_n_f64(a0.real() * sc);
        const float64x2_t vi0 = vdupq_n_f64(a0.imag() * sc);
        const float64x2_t vr1 = vdupq_n_f64(a1.real() * sc);
        const float64x2_t vi1 = vdupq_n_f64(a1.imag() * sc);
        const float64x2_t vc1 = vdupq_n_f64(c.c1);
        const float64x2_t vc2 = vdupq_n_f64(c.c2_p);
        const float64x2_t vc3 = vdupq_n_f64(c.c3);
        const float64x2_t vcterm = vmulq_f64(vc1, vct);
        const float64x2_t vp = vfmaq_f64(vmulq_f64(vaddq_f64(vcterm, vc2), vdp1), vc3, vdp2);
        const float64x2_t vm = vfmaq_f64(vmulq_f64(vsubq_f64(vcterm, vc2), vdm1), vc3, vdm2);
        vrNp = vfmaq_f64(vrNp, vr0, vp);
        viNp = vfmaq_f64(viNp, vi0, vp);
        vrNm = vfmaq_f64(vrNm, vr1, vm);
        viNm = vfmaq_f64(viNm, vi1, vm);
        vrSp = vfmaq_f64(vrSp, vr0, vmulq_f64(vm, vpar));
        viSp = vfmaq_f64(viSp, vi0, vmulq_f64(vm, vpar));
        vrSm = vfmaq_f64(vrSm, vr1, vmulq_f64(vp, vpar));
        viSm = vfmaq_f64(viSm, vi1, vmulq_f64(vp, vpar));
    }
    vst1q_f64(rNp, vrNp);
    vst1q_f64(iNp, viNp);
    vst1q_f64(rNm, vrNm);
    vst1q_f64(iNm, viNm);
    vst1q_f64(rSp, vrSp);
    vst1q_f64(iSp, viSp);
    vst1q_f64(rSm, vrSm);
    vst1q_f64(iSm, viSm);
}
#endif


torch::Tensor healpix_spin_interpolate_recurrence_cpu(
    const torch::Tensor& alms,
    const torch::Tensor& theta,
    int64_t lmax,
    int64_t mmax,
    int64_t spin,
    const c10::complex<double>* phase_ptr = nullptr,
    const c10::complex<double>* alm0_ptr_in = nullptr,
    const c10::complex<double>* alm1_ptr_in = nullptr,
    c10::complex<double>* eq_plus = nullptr,
    c10::complex<double>* eq_minus = nullptr,
    int64_t eq_start = -1,
    int64_t n_eq = 0,
    int64_t nph_eq = 0
) {
    torch::Tensor alms_keep;
    const c10::complex<double>* alm0_ptr = alm0_ptr_in;
    const c10::complex<double>* alm1_ptr = alm1_ptr_in;
    if (alm0_ptr == nullptr) {
        alms_keep = alms.contiguous();
        alm0_ptr = alms_keep.data_ptr<c10::complex<double>>();
        alm1_ptr = alm0_ptr + alms_keep.size(1);
    } else {
        (void)alms;
    }
    const int64_t spin_abs = std::abs(spin);
    if ((spin_abs == 1 || spin_abs == 2) && spin_ishioka_enabled()) {
        return ishioka_spin_lowered_interpolate_cpu(
            alm0_ptr,
            alm1_ptr,
            theta,
            lmax,
            mmax,
            spin_abs,
            phase_ptr,
            eq_plus,
            eq_minus,
            eq_start,
            n_eq,
            nph_eq
        );
    }
    int64_t nrings = theta.size(0);
    int64_t n_eff = (nrings + 1) / 2;
    int64_t n_pairs = nrings / 2;
    bool has_equator = (nrings % 2 != 0);
    const int64_t eq_end = eq_start + n_eq;
    const bool write_eq =
        (eq_plus != nullptr && eq_minus != nullptr && n_eq > 0 && nph_eq > mmax && eq_start >= 0);
    ensure_l_factors(lmax);

    auto options = torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU);
    torch::Tensor s_modes = torch::zeros({2, mmax + 1, nrings}, options);
    auto* s0_ptr = s_modes.data_ptr<c10::complex<double>>();
    auto* s1_ptr = s0_ptr + (mmax + 1) * nrings;
    auto th_ptr = theta.data_ptr<double>();

    std::vector<int64_t> m_offsets(mmax + 1);
    int64_t current_lut_off = 0;
    for (int64_t m = 0; m <= mmax; ++m) {
        m_offsets[m] = current_lut_off;
        int64_t l0 = std::max(std::abs(spin), m);
        if (lmax >= l0 + 2) {
            current_lut_off += (lmax - (l0 + 2) + 1);
        }
    }
    const auto& lut = g_recurrence_cache.get_coeffs(lmax, mmax, spin, m_offsets);

    std::vector<double> cos_theta_all(n_eff), sin_half_all(n_eff), cos_half_all(n_eff);
    for (int i = 0; i < n_eff; ++i) {
        const double th = th_ptr[i];
        cos_theta_all[static_cast<size_t>(i)] = std::cos(th);
        sin_half_all[static_cast<size_t>(i)] = std::sin(th * 0.5);
        cos_half_all[static_cast<size_t>(i)] = std::cos(th * 0.5);
    }
    std::vector<int64_t> mlim_north(n_eff, lmax);
    if (spin_mlim_prune_enabled()) {
        const int64_t mlim_spin = std::abs(spin);
        for (int64_t i = 0; i < n_eff; ++i) {
            const double cth = cos_theta_all[static_cast<size_t>(i)];
            const double sth = std::sqrt(std::max(0.0, 1.0 - cth * cth));
            mlim_north[static_cast<size_t>(i)] = sharp_mlim_estimate(lmax, mlim_spin, sth, cth);
        }
    }

    parallel_m_ranges(mmax, [&](int64_t m0, int64_t stride) {
        // Allocate working buffers ONCE per thread task, rather than per inner loop.
        std::vector<double> v_dp2(n_eff), v_dp1(n_eff), v_dpc(n_eff);
        std::vector<double> v_dm2(n_eff), v_dm1(n_eff), v_dmc(n_eff);
        std::vector<double> v_rNp(n_eff), v_iNp(n_eff), v_rNm(n_eff), v_iNm(n_eff);
        std::vector<double> v_rSp(n_eff), v_iSp(n_eff), v_rSm(n_eff), v_iSm(n_eff);
        std::vector<uint8_t> v_active(n_eff);
        double * __restrict__ dp2 = v_dp2.data();
        double * __restrict__ dp1 = v_dp1.data();
        double * __restrict__ dpc = v_dpc.data();
        double * __restrict__ dm2 = v_dm2.data();
        double * __restrict__ dm1 = v_dm1.data();
        double * __restrict__ dmc = v_dmc.data();
        double * __restrict__ rNp = v_rNp.data(); double * __restrict__ iNp = v_iNp.data();
        double * __restrict__ rNm = v_rNm.data(); double * __restrict__ iNm = v_iNm.data();
        double * __restrict__ rSp = v_rSp.data(); double * __restrict__ iSp = v_iSp.data();
        double * __restrict__ rSm = v_rSm.data(); double * __restrict__ iSm = v_iSm.data();
        uint8_t * __restrict__ active = v_active.data();
        const double * __restrict__ ct_ptr = cos_theta_all.data();
        const double * __restrict__ sh_ptr = sin_half_all.data();
        const double * __restrict__ ch_ptr = cos_half_all.data();

        for (int64_t m = m0; m <= mmax; m += stride) {
            int64_t l0 = std::max(std::abs(spin), m);
            int64_t idx_base = m * (lmax + 1) - (m * (m - 1)) / 2;
            const double pre_phase = ((m % 2 != 0) ? -1.0 : 1.0)
                * ((std::abs(spin) % 2 != 0) ? -1.0 : 1.0);
            const bool distinct_d = (spin != 0);
            const auto* lut_ptr = &lut[m_offsets[m]];
            bool any_active = false;

            for(int i=0; i<n_eff; ++i) {
                const bool on = (m <= mlim_north[static_cast<size_t>(i)]);
                active[i] = static_cast<uint8_t>(on ? 1 : 0);
                any_active = any_active || on;
                rNp[i]=0; iNp[i]=0; rNm[i]=0; iNm[i]=0;
                rSp[i]=0; iSp[i]=0; rSm[i]=0; iSm[i]=0;
            }
            if (!any_active) {
                continue;
            }
            int pair0 = 0;
            int n_run = 0;
            spin_active_pair_span(active, static_cast<int>(n_pairs), pair0, n_run);

            WignerDCoeffs coeffs_p1 = get_wigner_d_coeffs(l0, m, -spin);
            WignerDCoeffs coeffs_m1;
            if (distinct_d) {
                coeffs_m1 = get_wigner_d_coeffs(l0, m, spin);
            }

            for(int i=0; i<n_eff; ++i) {
                if (!active[i]) {
                    dp1[i] = 0.0;
                    dm1[i] = 0.0;
                    dp2[i] = 0.0;
                    dm2[i] = 0.0;
                    continue;
                }
                dp1[i] = eval_wigner_d_half(coeffs_p1, sh_ptr[i], ch_ptr[i]);
                if (distinct_d) {
                    dm1[i] = eval_wigner_d_half(coeffs_m1, sh_ptr[i], ch_ptr[i]);
                } else {
                    dm1[i] = dp1[i];
                }
                dp2[i] = 0.0;
                dm2[i] = 0.0;
            }


            auto acc_spin = [&](int64_t l, const double* __restrict__ d_p, const double* __restrict__ d_m) {
                const c10::complex<double> a0 = alm0_ptr[idx_base+(l-m)];
                const c10::complex<double> a1 = alm1_ptr[idx_base+(l-m)];
                const double sc = pre_phase * precomputed_l_factors[l];
                const double r0 = a0.real()*sc, i0 = a0.imag()*sc, r1 = a1.real()*sc, i1 = a1.imag()*sc;
                const double parity = ((l+m)%2 == 0) ? 1.0 : -1.0;
                spin_interpolate_dot_rings(
                    n_run,
                    r0,
                    i0,
                    r1,
                    i1,
                    parity,
                    d_p + pair0,
                    d_m + pair0,
                    rNp + pair0,
                    iNp + pair0,
                    rNm + pair0,
                    iNm + pair0,
                    rSp + pair0,
                    iSp + pair0,
                    rSm + pair0,
                    iSm + pair0
                );
                if (has_equator && active[n_pairs]) {
                    int i = n_pairs;
                    rNp[i] += r0*d_p[i]; iNp[i] += i0*d_p[i];
                    rNm[i] += r1*d_m[i]; iNm[i] += i1*d_m[i];
                }
            };

            acc_spin(l0, dp1, dm1);
            if (l0 < lmax) {
                WignerDCoeffs coeffs_pc = get_wigner_d_coeffs(l0+1, m, -spin);
                WignerDCoeffs coeffs_mc;
                if (distinct_d) {
                    coeffs_mc = get_wigner_d_coeffs(l0+1, m, spin);
                }

                for(int i=0; i<n_eff; ++i) {
                    if (!active[i]) {
                        dpc[i] = 0.0;
                        dmc[i] = 0.0;
                        continue;
                    }
                    dpc[i] = eval_wigner_d_half(coeffs_pc, sh_ptr[i], ch_ptr[i]);
                    if (distinct_d) {
                        dmc[i] = eval_wigner_d_half(coeffs_mc, sh_ptr[i], ch_ptr[i]);
                    } else {
                        dmc[i] = dpc[i];
                    }
                }
                acc_spin(l0+1, dpc, dmc);
                { auto t = dp2; dp2 = dp1; dp1 = dpc; dpc = t; }
                { auto t = dm2; dm2 = dm1; dm1 = dmc; dmc = t; }

                int i = 0;
#if defined(__aarch64__)
                for (; i + 4 <= n_run; i += 4) {
                    spin_interpolate_l_scan_four(
                        l0,
                        lmax,
                        m,
                        lut_ptr,
                        alm0_ptr,
                        alm1_ptr,
                        idx_base,
                        pre_phase,
                        true,
                        ct_ptr + pair0 + i,
                        dp1 + pair0 + i,
                        dp2 + pair0 + i,
                        dm1 + pair0 + i,
                        dm2 + pair0 + i,
                        rNp + pair0 + i,
                        iNp + pair0 + i,
                        rNm + pair0 + i,
                        iNm + pair0 + i,
                        rSp + pair0 + i,
                        iSp + pair0 + i,
                        rSm + pair0 + i,
                        iSm + pair0 + i
                    );
                }
                for (; i + 2 <= n_run; i += 2) {
                    spin_interpolate_l_scan_two(
                        l0,
                        lmax,
                        m,
                        lut_ptr,
                        alm0_ptr,
                        alm1_ptr,
                        idx_base,
                        pre_phase,
                        true,
                        ct_ptr + pair0 + i,
                        dp1 + pair0 + i,
                        dp2 + pair0 + i,
                        dm1 + pair0 + i,
                        dm2 + pair0 + i,
                        rNp + pair0 + i,
                        iNp + pair0 + i,
                        rNm + pair0 + i,
                        iNm + pair0 + i,
                        rSp + pair0 + i,
                        iSp + pair0 + i,
                        rSm + pair0 + i,
                        iSm + pair0 + i
                    );
                }
#endif
                for (; i < n_run; ++i) {
                    const int j = pair0 + i;
                    spin_interpolate_l_scan_one(
                        l0,
                        lmax,
                        m,
                        lut_ptr,
                        alm0_ptr,
                        alm1_ptr,
                        idx_base,
                        pre_phase,
                        true,
                        ct_ptr[j],
                        dp1[j],
                        dp2[j],
                        dm1[j],
                        dm2[j],
                        rNp[j],
                        iNp[j],
                        rNm[j],
                        iNm[j],
                        rSp[j],
                        iSp[j],
                        rSm[j],
                        iSm[j]
                    );
                }
                if (has_equator && active[n_pairs]) {
                    int eq = static_cast<int>(n_pairs);
                    spin_interpolate_l_scan_one(
                        l0,
                        lmax,
                        m,
                        lut_ptr,
                        alm0_ptr,
                        alm1_ptr,
                        idx_base,
                        pre_phase,
                        true,
                        ct_ptr[eq],
                        dp1[eq],
                        dp2[eq],
                        dm1[eq],
                        dm2[eq],
                        rNp[eq],
                        iNp[eq],
                        rNm[eq],
                        iNm[eq],
                        rSp[eq],
                        iSp[eq],
                        rSm[eq],
                        iSm[eq]
                    );
                }
            }

            auto store_pair = [&](int64_t idx, int /*ring_i*/, double rP, double iP, double rM, double iM) {
                auto zp = c10::complex<double>(rP, iP);
                auto zm = c10::complex<double>(rM, iM);
                if (phase_ptr != nullptr) {
                    const auto ph = phase_ptr[m * nrings + idx];
                    zp = zp * ph;
                    zm = zm * ph;
                }
                if (write_eq && idx >= eq_start && idx < eq_end && m < nph_eq) {
                    const int64_t off = (idx - eq_start) * nph_eq + m;
                    eq_plus[off] = zp;
                    eq_minus[off] = zm;
                    if (m == 0) {
                        s0_ptr[idx] = zp;
                        s1_ptr[idx] = zm;
                    }
                } else {
                    s0_ptr[m * nrings + idx] = zp;
                    s1_ptr[m * nrings + idx] = zm;
                }
            };
            for(int i=0; i<n_eff; ++i) {
                if (!active[i]) {
                    continue;
                }
                store_pair(i, i, rNp[i], iNp[i], rNm[i], iNm[i]);
                if (i < n_pairs) {
                    store_pair(nrings - 1 - i, i, rSp[i], iSp[i], rSm[i], iSm[i]);
                }
            }
        }
    });

    return s_modes;
}

// ---------------------------------------------------------------------------
// Dedicated scalar (spin=0) recurrence kernels
// ---------------------------------------------------------------------------
// These avoid the 2x data duplication and 2x FLOP waste of routing spin=0
// through the general spin kernel. For spin=0, the Wigner-d recurrence
// degenerates to the standard Legendre Plm recurrence:
//   P_l^m(cos θ) = c1 * cos θ * P_{l-1}^m - c3 * P_{l-2}^m
// with c2_p = 0 (since shift = m*(-spin)/(l*(l-1)) = 0).
//
// Bootstrap: P_m^m(cos θ) = (-1)^m * mfac[m] * sin^m(θ)
//   where mfac[m] = 1/sqrt(4π) * prod_{k=1}^{m} sqrt((2k+1)/(2k))
// ---------------------------------------------------------------------------

static std::vector<double> precomputed_mfac;
static std::mutex mfac_mutex;

static void ensure_mfac(int64_t mmax) {
    std::lock_guard<std::mutex> lock(mfac_mutex);
    if (precomputed_mfac.size() <= (size_t)mmax) {
        precomputed_mfac.resize(mmax + 1);
        precomputed_mfac[0] = 1.0 / std::sqrt(4.0 * M_PI);
        for (int64_t i = 1; i <= mmax; ++i) {
            precomputed_mfac[i] = precomputed_mfac[i - 1] * std::sqrt((2.0 * i + 1.0) / (2.0 * i));
        }
    }
}

static inline double ishioka_eps(int64_t n, int64_t m) {
    if (n <= m) {
        return 0.0;
    }
    const double nn = static_cast<double>(n) * static_cast<double>(n);
    const double mm = static_cast<double>(m) * static_cast<double>(m);
    return std::sqrt((nn - mm) / (4.0 * nn - 1.0));
}

// Ishioka 2018 (JMSJ, CC BY 4.0) skip-2 coefficients for one m.
struct IshiokaMCoeffs {
    std::vector<double> eps;
    std::vector<double> alpha;
    std::vector<double> a;
    std::vector<double> b;
    int64_t kmax = 0;
    int64_t kmax_odd = -1;
};

class IshiokaCache {
    std::mutex mutex_;
    int64_t last_lmax_ = -1;
    int64_t last_mmax_ = -1;
    std::vector<IshiokaMCoeffs> by_m_;

public:
    const std::vector<IshiokaMCoeffs>& get(int64_t lmax, int64_t mmax) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (lmax == last_lmax_ && mmax == last_mmax_) {
            return by_m_;
        }
        by_m_.assign(static_cast<size_t>(mmax + 1), IshiokaMCoeffs{});
        for (int64_t m = 0; m <= mmax; ++m) {
            IshiokaMCoeffs& c = by_m_[static_cast<size_t>(m)];
            c.kmax = (lmax - m) / 2;
            c.kmax_odd = (lmax >= m + 1) ? (lmax - m - 1) / 2 : -1;
            const int64_t nmax = lmax + 2;
            c.eps.assign(static_cast<size_t>(nmax + 1), 0.0);
            for (int64_t n = m + 1; n <= nmax; ++n) {
                c.eps[static_cast<size_t>(n)] = ishioka_eps(n, m);
            }
            c.alpha.assign(static_cast<size_t>(c.kmax + 1), 0.0);
            c.a.assign(static_cast<size_t>(std::max(c.kmax, int64_t{0})), 0.0);
            c.b.assign(c.a.size(), 0.0);
            if (c.kmax < 0) {
                continue;
            }
            const double e1 = c.eps[static_cast<size_t>(m + 1)];
            c.alpha[0] = (e1 == 0.0) ? 0.0 : 1.0 / e1;
            if (c.kmax >= 1) {
                c.alpha[1] = e1 / (c.eps[static_cast<size_t>(m + 2)] * c.eps[static_cast<size_t>(m + 3)]);
                for (int64_t k = 1; k < c.kmax; ++k) {
                    const double den =
                        c.alpha[static_cast<size_t>(k)] *
                        c.eps[static_cast<size_t>(m + 2 * k + 3)] *
                        c.eps[static_cast<size_t>(m + 2 * k + 2)];
                    c.alpha[static_cast<size_t>(k + 1)] = ((k & 1) ? -1.0 : 1.0) / den;
                }
            }
            for (int64_t k = 0; k < c.kmax; ++k) {
                const double ak = ((k & 1) ? -1.0 : 1.0) *
                    c.alpha[static_cast<size_t>(k)] * c.alpha[static_cast<size_t>(k)];
                c.a[static_cast<size_t>(k)] = ak;
                const double e_a = c.eps[static_cast<size_t>(m + 2 * k + 2)];
                const double e_b = c.eps[static_cast<size_t>(m + 2 * k + 1)];
                c.b[static_cast<size_t>(k)] = -ak * (e_a * e_a + e_b * e_b);
            }
        }
        last_lmax_ = lmax;
        last_mmax_ = mmax;
        return by_m_;
    }
};

static IshiokaCache g_ishioka_cache;

struct RecoupleSrc {
    double cm2, cm1, c0, cp1, cp2;
};

class RecoupleCache {
    std::mutex mutex_;
    int64_t last_lmax_ = -1;
    int64_t last_mmax_ = -1;
    int64_t last_spin_ = -1;
    std::vector<int64_t> m_off_;
    std::vector<int64_t> lmin_;
    std::vector<RecoupleSrc> src_;

public:
    void get(
        int64_t lmax,
        int64_t mmax,
        int64_t spin,
        const RecoupleSrc*& src_base,
        const int64_t*& m_off,
        const int64_t*& lmin
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (lmax != last_lmax_ || mmax != last_mmax_ || spin != last_spin_) {
            m_off_.assign(static_cast<size_t>(mmax + 1), 0);
            lmin_.assign(static_cast<size_t>(mmax + 1), 0);
            src_.clear();
            for (int64_t m = 0; m <= mmax; ++m) {
                m_off_[static_cast<size_t>(m)] = static_cast<int64_t>(src_.size());
                const int64_t lmin_m = std::max(m, spin);
                lmin_[static_cast<size_t>(m)] = lmin_m;
                for (int64_t l = lmin_m; l <= lmax; ++l) {
                    RecoupleSrc c{0.0, 0.0, 0.0, 0.0, 0.0};
                    if (spin == 1) {
                        const double den = std::sqrt(static_cast<double>(l) * static_cast<double>(l + 1));
                        c.cm1 = -(static_cast<double>(l + 1)) * ishioka_eps(l, m) / den;
                        c.c0 = -static_cast<double>(m) / den;
                        c.cp1 = static_cast<double>(l) * ishioka_eps(l + 1, m) / den;
                    } else {
                        const double Nl = std::sqrt(
                            static_cast<double>(l - 1) * static_cast<double>(l) *
                            static_cast<double>(l + 1) * static_cast<double>(l + 2)
                        );
                        c.cm2 = static_cast<double>(l + 1) * static_cast<double>(l + 2) *
                            ishioka_eps(l, m) * ishioka_eps(l - 1, m) / Nl;
                        c.cm1 = 2.0 * static_cast<double>(m) * static_cast<double>(l + 2) *
                            ishioka_eps(l, m) / Nl;
                        c.c0 = 2.0 * static_cast<double>(l - 1) * static_cast<double>(l + 2) *
                            (3.0 * static_cast<double>(m) * static_cast<double>(m) -
                             static_cast<double>(l) * static_cast<double>(l + 1)) /
                            (Nl * static_cast<double>(2 * l - 1) * static_cast<double>(2 * l + 3));
                        c.cp1 = -2.0 * static_cast<double>(m) * static_cast<double>(l - 1) *
                            ishioka_eps(l + 1, m) / Nl;
                        c.cp2 = static_cast<double>(l) * static_cast<double>(l - 1) *
                            ishioka_eps(l + 1, m) * ishioka_eps(l + 2, m) / Nl;
                    }
                    src_.push_back(c);
                }
            }
            last_lmax_ = lmax;
            last_mmax_ = mmax;
            last_spin_ = spin;
        }
        src_base = src_.data();
        m_off = m_off_.data();
        lmin = lmin_.data();
    }
};

static RecoupleCache g_recouple_cache;

static void recouple_spin_to_scalar_m(
    const c10::complex<double>* plus_m,
    const c10::complex<double>* minus_m,
    int64_t lmax,
    int64_t m,
    int64_t spin,
    const RecoupleSrc* src_base,
    const int64_t* m_off,
    const int64_t* lmin_arr,
    c10::complex<double>* s0,
    c10::complex<double>* s1
) {
    const int64_t lmax_s = lmax + spin;
    const int64_t n_s = lmax_s - m + 1;
    std::fill(s0, s0 + n_s, c10::complex<double>(0.0, 0.0));
    std::fill(s1, s1 + n_s, c10::complex<double>(0.0, 0.0));
    const int64_t lmin = lmin_arr[m];
    const RecoupleSrc* cs = src_base + m_off[m];
    for (int64_t l = lmin; l <= lmax; ++l) {
        const RecoupleSrc& c = cs[l - lmin];
        const c10::complex<double> p = plus_m[l - m];
        const c10::complex<double> q = minus_m[l - m];
        if (c.cm2 != 0.0 && l - 2 >= m) {
            s0[l - 2 - m] += p * c.cm2;
            s1[l - 2 - m] += q * c.cm2;
        }
        if (c.cm1 != 0.0 && l - 1 >= m) {
            s0[l - 1 - m] += p * c.cm1;
            s1[l - 1 - m] += q * (-c.cm1);
        }
        s0[l - m] += p * c.c0;
        s1[l - m] += q * c.c0;
        if (c.cp1 != 0.0 && l + 1 <= lmax_s) {
            s0[l + 1 - m] += p * c.cp1;
            s1[l + 1 - m] += q * (-c.cp1);
        }
        if (c.cp2 != 0.0 && l + 2 <= lmax_s) {
            s0[l + 2 - m] += p * c.cp2;
            s1[l + 2 - m] += q * c.cp2;
        }
    }
}

static void recouple_scalar_to_spin_m(
    const c10::complex<double>* s0,
    const c10::complex<double>* s1,
    int64_t lmax,
    int64_t m,
    int64_t spin,
    const RecoupleSrc* src_base,
    const int64_t* m_off,
    const int64_t* lmin_arr,
    c10::complex<double>* plus_m,
    c10::complex<double>* minus_m
) {
    const int64_t lmax_s = lmax + spin;
    const int64_t n_spin = lmax - m + 1;
    std::fill(plus_m, plus_m + n_spin, c10::complex<double>(0.0, 0.0));
    std::fill(minus_m, minus_m + n_spin, c10::complex<double>(0.0, 0.0));
    const int64_t lmin = lmin_arr[m];
    const RecoupleSrc* cs = src_base + m_off[m];
    for (int64_t l = lmin; l <= lmax; ++l) {
        const RecoupleSrc& c = cs[l - lmin];
        c10::complex<double> p(0.0, 0.0);
        c10::complex<double> q(0.0, 0.0);
        if (c.cm2 != 0.0 && l - 2 >= m) {
            p += s0[l - 2 - m] * c.cm2;
            q += s1[l - 2 - m] * c.cm2;
        }
        if (c.cm1 != 0.0 && l - 1 >= m) {
            p += s0[l - 1 - m] * c.cm1;
            q += s1[l - 1 - m] * (-c.cm1);
        }
        p += s0[l - m] * c.c0;
        q += s1[l - m] * c.c0;
        if (c.cp1 != 0.0 && l + 1 <= lmax_s) {
            p += s0[l + 1 - m] * c.cp1;
            q += s1[l + 1 - m] * (-c.cp1);
        }
        if (c.cp2 != 0.0 && l + 2 <= lmax_s) {
            p += s0[l + 2 - m] * c.cp2;
            q += s1[l + 2 - m] * c.cp2;
        }
        plus_m[l - m] = p;
        minus_m[l - m] = q;
    }
}

template <bool DoOdd, bool Step>
static inline void ishioka_k_step_rings_impl(
    int n,
    double we0r,
    double we0i,
    double we1r,
    double we1i,
    double wo0r,
    double wo0i,
    double wo1r,
    double wo1i,
    double ak,
    double bk,
    const double* __restrict__ mu2,
    double* __restrict__ p,
    double* __restrict__ pold,
    double* __restrict__ s0r,
    double* __restrict__ s0i,
    double* __restrict__ s1r,
    double* __restrict__ s1i,
    double* __restrict__ a0r,
    double* __restrict__ a0i,
    double* __restrict__ a1r,
    double* __restrict__ a1i
) {
    int i = 0;
#if defined(__aarch64__)
    const float64x2_t vwe0r = vdupq_n_f64(we0r);
    const float64x2_t vwe0i = vdupq_n_f64(we0i);
    const float64x2_t vwe1r = vdupq_n_f64(we1r);
    const float64x2_t vwe1i = vdupq_n_f64(we1i);
    const float64x2_t vwo0r = vdupq_n_f64(wo0r);
    const float64x2_t vwo0i = vdupq_n_f64(wo0i);
    const float64x2_t vwo1r = vdupq_n_f64(wo1r);
    const float64x2_t vwo1i = vdupq_n_f64(wo1i);
    const float64x2_t vak = vdupq_n_f64(ak);
    const float64x2_t vbk = vdupq_n_f64(bk);
    for (; i + 4 <= n; i += 4) {
        for (int off : {i, i + 2}) {
            const float64x2_t vp = vld1q_f64(p + off);
            vst1q_f64(s0r + off, vfmaq_f64(vld1q_f64(s0r + off), vwe0r, vp));
            vst1q_f64(s0i + off, vfmaq_f64(vld1q_f64(s0i + off), vwe0i, vp));
            vst1q_f64(s1r + off, vfmaq_f64(vld1q_f64(s1r + off), vwe1r, vp));
            vst1q_f64(s1i + off, vfmaq_f64(vld1q_f64(s1i + off), vwe1i, vp));
            if constexpr (DoOdd) {
                vst1q_f64(a0r + off, vfmaq_f64(vld1q_f64(a0r + off), vwo0r, vp));
                vst1q_f64(a0i + off, vfmaq_f64(vld1q_f64(a0i + off), vwo0i, vp));
                vst1q_f64(a1r + off, vfmaq_f64(vld1q_f64(a1r + off), vwo1r, vp));
                vst1q_f64(a1i + off, vfmaq_f64(vld1q_f64(a1i + off), vwo1i, vp));
            }
            if constexpr (Step) {
                const float64x2_t pnew = vfmaq_f64(
                    vld1q_f64(pold + off),
                    vfmaq_f64(vbk, vak, vld1q_f64(mu2 + off)),
                    vp
                );
                vst1q_f64(pold + off, vp);
                vst1q_f64(p + off, pnew);
            }
        }
    }
#elif defined(__AVX2__) && defined(__FMA__)
    const __m256d vwe0r = _mm256_set1_pd(we0r);
    const __m256d vwe0i = _mm256_set1_pd(we0i);
    const __m256d vwe1r = _mm256_set1_pd(we1r);
    const __m256d vwe1i = _mm256_set1_pd(we1i);
    const __m256d vwo0r = _mm256_set1_pd(wo0r);
    const __m256d vwo0i = _mm256_set1_pd(wo0i);
    const __m256d vwo1r = _mm256_set1_pd(wo1r);
    const __m256d vwo1i = _mm256_set1_pd(wo1i);
    const __m256d vak = _mm256_set1_pd(ak);
    const __m256d vbk = _mm256_set1_pd(bk);
    for (; i + 4 <= n; i += 4) {
        const __m256d vp = _mm256_loadu_pd(p + i);
        _mm256_storeu_pd(s0r + i, _mm256_fmadd_pd(vwe0r, vp, _mm256_loadu_pd(s0r + i)));
        _mm256_storeu_pd(s0i + i, _mm256_fmadd_pd(vwe0i, vp, _mm256_loadu_pd(s0i + i)));
        _mm256_storeu_pd(s1r + i, _mm256_fmadd_pd(vwe1r, vp, _mm256_loadu_pd(s1r + i)));
        _mm256_storeu_pd(s1i + i, _mm256_fmadd_pd(vwe1i, vp, _mm256_loadu_pd(s1i + i)));
        if constexpr (DoOdd) {
            _mm256_storeu_pd(a0r + i, _mm256_fmadd_pd(vwo0r, vp, _mm256_loadu_pd(a0r + i)));
            _mm256_storeu_pd(a0i + i, _mm256_fmadd_pd(vwo0i, vp, _mm256_loadu_pd(a0i + i)));
            _mm256_storeu_pd(a1r + i, _mm256_fmadd_pd(vwo1r, vp, _mm256_loadu_pd(a1r + i)));
            _mm256_storeu_pd(a1i + i, _mm256_fmadd_pd(vwo1i, vp, _mm256_loadu_pd(a1i + i)));
        }
        if constexpr (Step) {
            const __m256d pnew = _mm256_fmadd_pd(
                _mm256_fmadd_pd(vak, _mm256_loadu_pd(mu2 + i), vbk),
                vp,
                _mm256_loadu_pd(pold + i)
            );
            _mm256_storeu_pd(pold + i, vp);
            _mm256_storeu_pd(p + i, pnew);
        }
    }
#endif
    for (; i < n; ++i) {
        const double pj = p[i];
        s0r[i] += we0r * pj;
        s0i[i] += we0i * pj;
        s1r[i] += we1r * pj;
        s1i[i] += we1i * pj;
        if constexpr (DoOdd) {
            a0r[i] += wo0r * pj;
            a0i[i] += wo0i * pj;
            a1r[i] += wo1r * pj;
            a1i[i] += wo1i * pj;
        }
        if constexpr (Step) {
            const double pnew = (ak * mu2[i] + bk) * pj + pold[i];
            pold[i] = pj;
            p[i] = pnew;
        }
    }
}

static inline void ishioka_k_step_rings(
    int n,
    bool do_odd,
    bool step,
    double we0r,
    double we0i,
    double we1r,
    double we1i,
    double wo0r,
    double wo0i,
    double wo1r,
    double wo1i,
    double ak,
    double bk,
    const double* __restrict__ mu2,
    double* __restrict__ p,
    double* __restrict__ pold,
    double* __restrict__ s0r,
    double* __restrict__ s0i,
    double* __restrict__ s1r,
    double* __restrict__ s1i,
    double* __restrict__ a0r,
    double* __restrict__ a0i,
    double* __restrict__ a1r,
    double* __restrict__ a1i
) {
    if (do_odd) {
        if (step) {
            ishioka_k_step_rings_impl<true, true>(
                n, we0r, we0i, we1r, we1i, wo0r, wo0i, wo1r, wo1i, ak, bk,
                mu2, p, pold, s0r, s0i, s1r, s1i, a0r, a0i, a1r, a1i
            );
        } else {
            ishioka_k_step_rings_impl<true, false>(
                n, we0r, we0i, we1r, we1i, wo0r, wo0i, wo1r, wo1i, ak, bk,
                mu2, p, pold, s0r, s0i, s1r, s1i, a0r, a0i, a1r, a1i
            );
        }
    } else if (step) {
        ishioka_k_step_rings_impl<false, true>(
            n, we0r, we0i, we1r, we1i, wo0r, wo0i, wo1r, wo1i, ak, bk,
            mu2, p, pold, s0r, s0i, s1r, s1i, a0r, a0i, a1r, a1i
        );
    } else {
        ishioka_k_step_rings_impl<false, false>(
            n, we0r, we0i, we1r, we1i, wo0r, wo0i, wo1r, wo1i, ak, bk,
            mu2, p, pold, s0r, s0i, s1r, s1i, a0r, a0i, a1r, a1i
        );
    }
}

template <bool DoOdd, bool Step>
static inline void ishioka_k_reduce_rings_impl(
    int n,
    double ak,
    double bk,
    const double* __restrict__ mu2,
    double* __restrict__ p,
    double* __restrict__ pold,
    const double* __restrict__ s0r,
    const double* __restrict__ s0i,
    const double* __restrict__ s1r,
    const double* __restrict__ s1i,
    const double* __restrict__ a0r,
    const double* __restrict__ a0i,
    const double* __restrict__ a1r,
    const double* __restrict__ a1i,
    double& we0r,
    double& we0i,
    double& we1r,
    double& we1i,
    double& wo0r,
    double& wo0i,
    double& wo1r,
    double& wo1i
) {
    int i = 0;
#if defined(__aarch64__)
    float64x2_t acc_we0r = vdupq_n_f64(0.0);
    float64x2_t acc_we0i = vdupq_n_f64(0.0);
    float64x2_t acc_we1r = vdupq_n_f64(0.0);
    float64x2_t acc_we1i = vdupq_n_f64(0.0);
    float64x2_t acc_wo0r = vdupq_n_f64(0.0);
    float64x2_t acc_wo0i = vdupq_n_f64(0.0);
    float64x2_t acc_wo1r = vdupq_n_f64(0.0);
    float64x2_t acc_wo1i = vdupq_n_f64(0.0);
    const float64x2_t vak = vdupq_n_f64(ak);
    const float64x2_t vbk = vdupq_n_f64(bk);
    for (; i + 4 <= n; i += 4) {
        for (int off : {i, i + 2}) {
            const float64x2_t vp = vld1q_f64(p + off);
            acc_we0r = vfmaq_f64(acc_we0r, vld1q_f64(s0r + off), vp);
            acc_we0i = vfmaq_f64(acc_we0i, vld1q_f64(s0i + off), vp);
            acc_we1r = vfmaq_f64(acc_we1r, vld1q_f64(s1r + off), vp);
            acc_we1i = vfmaq_f64(acc_we1i, vld1q_f64(s1i + off), vp);
            if constexpr (DoOdd) {
                acc_wo0r = vfmaq_f64(acc_wo0r, vld1q_f64(a0r + off), vp);
                acc_wo0i = vfmaq_f64(acc_wo0i, vld1q_f64(a0i + off), vp);
                acc_wo1r = vfmaq_f64(acc_wo1r, vld1q_f64(a1r + off), vp);
                acc_wo1i = vfmaq_f64(acc_wo1i, vld1q_f64(a1i + off), vp);
            }
            if constexpr (Step) {
                const float64x2_t pnew = vfmaq_f64(
                    vld1q_f64(pold + off),
                    vfmaq_f64(vbk, vak, vld1q_f64(mu2 + off)),
                    vp
                );
                vst1q_f64(pold + off, vp);
                vst1q_f64(p + off, pnew);
            }
        }
    }
    we0r += vaddvq_f64(acc_we0r);
    we0i += vaddvq_f64(acc_we0i);
    we1r += vaddvq_f64(acc_we1r);
    we1i += vaddvq_f64(acc_we1i);
    if constexpr (DoOdd) {
        wo0r += vaddvq_f64(acc_wo0r);
        wo0i += vaddvq_f64(acc_wo0i);
        wo1r += vaddvq_f64(acc_wo1r);
        wo1i += vaddvq_f64(acc_wo1i);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    const auto hsum4 = [](__m256d v) -> double {
        const __m128d lo = _mm256_castpd256_pd128(v);
        const __m128d hi = _mm256_extractf128_pd(v, 1);
        const __m128d s = _mm_add_pd(lo, hi);
        const __m128d s2 = _mm_add_pd(s, _mm_shuffle_pd(s, s, 1));
        return _mm_cvtsd_f64(s2);
    };
    __m256d acc_we0r = _mm256_setzero_pd();
    __m256d acc_we0i = _mm256_setzero_pd();
    __m256d acc_we1r = _mm256_setzero_pd();
    __m256d acc_we1i = _mm256_setzero_pd();
    __m256d acc_wo0r = _mm256_setzero_pd();
    __m256d acc_wo0i = _mm256_setzero_pd();
    __m256d acc_wo1r = _mm256_setzero_pd();
    __m256d acc_wo1i = _mm256_setzero_pd();
    const __m256d vak = _mm256_set1_pd(ak);
    const __m256d vbk = _mm256_set1_pd(bk);
    for (; i + 4 <= n; i += 4) {
        const __m256d vp = _mm256_loadu_pd(p + i);
        acc_we0r = _mm256_fmadd_pd(_mm256_loadu_pd(s0r + i), vp, acc_we0r);
        acc_we0i = _mm256_fmadd_pd(_mm256_loadu_pd(s0i + i), vp, acc_we0i);
        acc_we1r = _mm256_fmadd_pd(_mm256_loadu_pd(s1r + i), vp, acc_we1r);
        acc_we1i = _mm256_fmadd_pd(_mm256_loadu_pd(s1i + i), vp, acc_we1i);
        if constexpr (DoOdd) {
            acc_wo0r = _mm256_fmadd_pd(_mm256_loadu_pd(a0r + i), vp, acc_wo0r);
            acc_wo0i = _mm256_fmadd_pd(_mm256_loadu_pd(a0i + i), vp, acc_wo0i);
            acc_wo1r = _mm256_fmadd_pd(_mm256_loadu_pd(a1r + i), vp, acc_wo1r);
            acc_wo1i = _mm256_fmadd_pd(_mm256_loadu_pd(a1i + i), vp, acc_wo1i);
        }
        if constexpr (Step) {
            const __m256d pnew = _mm256_fmadd_pd(
                _mm256_fmadd_pd(vak, _mm256_loadu_pd(mu2 + i), vbk),
                vp,
                _mm256_loadu_pd(pold + i)
            );
            _mm256_storeu_pd(pold + i, vp);
            _mm256_storeu_pd(p + i, pnew);
        }
    }
    we0r += hsum4(acc_we0r);
    we0i += hsum4(acc_we0i);
    we1r += hsum4(acc_we1r);
    we1i += hsum4(acc_we1i);
    if constexpr (DoOdd) {
        wo0r += hsum4(acc_wo0r);
        wo0i += hsum4(acc_wo0i);
        wo1r += hsum4(acc_wo1r);
        wo1i += hsum4(acc_wo1i);
    }
#endif
    for (; i < n; ++i) {
        const double pj = p[i];
        we0r += s0r[i] * pj;
        we0i += s0i[i] * pj;
        we1r += s1r[i] * pj;
        we1i += s1i[i] * pj;
        if constexpr (DoOdd) {
            wo0r += a0r[i] * pj;
            wo0i += a0i[i] * pj;
            wo1r += a1r[i] * pj;
            wo1i += a1i[i] * pj;
        }
        if constexpr (Step) {
            const double pnew = (ak * mu2[i] + bk) * pj + pold[i];
            pold[i] = pj;
            p[i] = pnew;
        }
    }
}

static inline void ishioka_k_reduce_rings(
    int n,
    bool do_odd,
    bool step,
    double ak,
    double bk,
    const double* __restrict__ mu2,
    double* __restrict__ p,
    double* __restrict__ pold,
    const double* __restrict__ s0r,
    const double* __restrict__ s0i,
    const double* __restrict__ s1r,
    const double* __restrict__ s1i,
    const double* __restrict__ a0r,
    const double* __restrict__ a0i,
    const double* __restrict__ a1r,
    const double* __restrict__ a1i,
    double& we0r,
    double& we0i,
    double& we1r,
    double& we1i,
    double& wo0r,
    double& wo0i,
    double& wo1r,
    double& wo1i
) {
    if (do_odd) {
        if (step) {
            ishioka_k_reduce_rings_impl<true, true>(
                n, ak, bk, mu2, p, pold, s0r, s0i, s1r, s1i, a0r, a0i, a1r, a1i,
                we0r, we0i, we1r, we1i, wo0r, wo0i, wo1r, wo1i
            );
        } else {
            ishioka_k_reduce_rings_impl<true, false>(
                n, ak, bk, mu2, p, pold, s0r, s0i, s1r, s1i, a0r, a0i, a1r, a1i,
                we0r, we0i, we1r, we1i, wo0r, wo0i, wo1r, wo1i
            );
        }
    } else if (step) {
        ishioka_k_reduce_rings_impl<false, true>(
            n, ak, bk, mu2, p, pold, s0r, s0i, s1r, s1i, a0r, a0i, a1r, a1i,
            we0r, we0i, we1r, we1i, wo0r, wo0i, wo1r, wo1i
        );
    } else {
        ishioka_k_reduce_rings_impl<false, false>(
            n, ak, bk, mu2, p, pold, s0r, s0i, s1r, s1i, a0r, a0i, a1r, a1i,
            we0r, we0i, we1r, we1i, wo0r, wo0i, wo1r, wo1i
        );
    }
}

static torch::Tensor ishioka_spin_lowered_interpolate_cpu(
    const c10::complex<double>* alm0_ptr,
    const c10::complex<double>* alm1_ptr,
    const torch::Tensor& theta,
    int64_t lmax,
    int64_t mmax,
    int64_t spin_abs,
    const c10::complex<double>* phase_ptr,
    c10::complex<double>* eq_plus,
    c10::complex<double>* eq_minus,
    int64_t eq_start,
    int64_t n_eq,
    int64_t nph_eq
) {
    const int64_t lmax_s = lmax + spin_abs;
    const int64_t nrings = theta.size(0);
    const int64_t n_eff = (nrings + 1) / 2;
    const int64_t n_pairs = nrings / 2;
    const bool has_equator = (nrings % 2 != 0);
    const int64_t eq_end = eq_start + n_eq;
    const bool write_eq =
        (eq_plus != nullptr && eq_minus != nullptr && n_eq > 0 && nph_eq > mmax && eq_start >= 0);
    ensure_mfac(mmax);
    const auto& packs = g_ishioka_cache.get(lmax_s, mmax);
    const RecoupleSrc* rec_src = nullptr;
    const int64_t* rec_off = nullptr;
    const int64_t* rec_lmin = nullptr;
    g_recouple_cache.get(lmax, mmax, spin_abs, rec_src, rec_off, rec_lmin);

    auto options = torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU);
    torch::Tensor s_modes = torch::zeros({2, mmax + 1, nrings}, options);
    auto* s0_ptr = s_modes.data_ptr<c10::complex<double>>();
    auto* s1_ptr = s0_ptr + (mmax + 1) * nrings;
    auto th_ptr = theta.data_ptr<double>();

    std::vector<double> cos_theta_all(n_eff), sin_theta_all(n_eff), inv_sin_s(n_eff), mu2_all(n_eff);
    for (int i = 0; i < n_eff; ++i) {
        const double th = th_ptr[i];
        const double ct = std::cos(th);
        const double st = std::sin(th);
        cos_theta_all[static_cast<size_t>(i)] = ct;
        sin_theta_all[static_cast<size_t>(i)] = st;
        mu2_all[static_cast<size_t>(i)] = ct * ct;
        inv_sin_s[static_cast<size_t>(i)] = 1.0 / fast_pow(st, static_cast<int>(spin_abs));
    }
    std::vector<int64_t> mlim_north(n_eff, lmax_s);
    if (spin_mlim_prune_enabled()) {
        for (int64_t i = 0; i < n_eff; ++i) {
            const double cth = cos_theta_all[static_cast<size_t>(i)];
            const double sth = sin_theta_all[static_cast<size_t>(i)];
            mlim_north[static_cast<size_t>(i)] = sharp_mlim_estimate(lmax_s, 0, sth, cth);
        }
    }

    parallel_m_ranges(mmax, [&](int64_t m0, int64_t stride) {
        if (m0 > mmax) {
            return;
        }
        const int64_t n_s_max = lmax_s - m0 + 1;
        std::vector<c10::complex<double>> v_s0(static_cast<size_t>(n_s_max));
        std::vector<c10::complex<double>> v_s1(static_cast<size_t>(n_s_max));
        std::vector<double> v_p(n_eff), v_pold(n_eff);
        std::vector<double> v_s0r(n_eff), v_s0i(n_eff), v_a0r(n_eff), v_a0i(n_eff);
        std::vector<double> v_s1r(n_eff), v_s1i(n_eff), v_a1r(n_eff), v_a1i(n_eff);
        std::vector<uint8_t> v_active(n_eff);
        double* __restrict__ p = v_p.data();
        double* __restrict__ pold = v_pold.data();
        double* __restrict__ s0r = v_s0r.data();
        double* __restrict__ s0i = v_s0i.data();
        double* __restrict__ a0r = v_a0r.data();
        double* __restrict__ a0i = v_a0i.data();
        double* __restrict__ s1r = v_s1r.data();
        double* __restrict__ s1i = v_s1i.data();
        double* __restrict__ a1r = v_a1r.data();
        double* __restrict__ a1i = v_a1i.data();
        uint8_t* __restrict__ active = v_active.data();
        const double* __restrict__ ct_ptr = cos_theta_all.data();
        const double* __restrict__ st_ptr = sin_theta_all.data();
        const double* __restrict__ invs_ptr = inv_sin_s.data();
        const double* __restrict__ mu2_ptr = mu2_all.data();

        for (int64_t m = m0; m <= mmax; m += stride) {
            const IshiokaMCoeffs& pc = packs[static_cast<size_t>(m)];
            recouple_spin_to_scalar_m(
                alm0_ptr + alm_packed_index(m, m, lmax),
                alm1_ptr + alm_packed_index(m, m, lmax),
                lmax,
                m,
                spin_abs,
                rec_src,
                rec_off,
                rec_lmin,
                v_s0.data(),
                v_s1.data()
            );
            const c10::complex<double>* __restrict__ sc0 = v_s0.data();
            const c10::complex<double>* __restrict__ sc1 = v_s1.data();
            const double pre_phase = (m % 2 != 0) ? -1.0 : 1.0;
            const double seed = precomputed_mfac[static_cast<size_t>(m)];

            for (int i = 0; i < n_eff; ++i) {
                const bool on = (m <= mlim_north[static_cast<size_t>(i)]);
                active[i] = static_cast<uint8_t>(on);
                p[i] = on ? seed * fast_pow(st_ptr[i], static_cast<int>(m)) : 0.0;
                pold[i] = 0.0;
                s0r[i] = 0.0;
                s0i[i] = 0.0;
                a0r[i] = 0.0;
                a0i[i] = 0.0;
                s1r[i] = 0.0;
                s1i[i] = 0.0;
                a1r[i] = 0.0;
                a1i[i] = 0.0;
            }
            int pair0 = 0;
            int n_run = 0;
            spin_active_pair_span(active, static_cast<int>(n_pairs), pair0, n_run);

            auto s_at = [&](const c10::complex<double>* sc, int64_t n) -> c10::complex<double> {
                if (n < m || n > lmax_s) {
                    return c10::complex<double>(0.0, 0.0);
                }
                return sc[n - m];
            };

            for (int64_t k = 0; k <= pc.kmax; ++k) {
                const int64_t n_even = m + 2 * k;
                const double e1 = pc.eps[static_cast<size_t>(n_even + 1)];
                c10::complex<double> t0 = s_at(sc0, n_even) * e1;
                c10::complex<double> t1 = s_at(sc1, n_even) * e1;
                if (k < pc.kmax) {
                    const double e2 = pc.eps[static_cast<size_t>(n_even + 2)];
                    t0 += s_at(sc0, n_even + 2) * e2;
                    t1 += s_at(sc1, n_even + 2) * e2;
                }
                const double al = pc.alpha[static_cast<size_t>(k)];
                const c10::complex<double> we0 = t0 * al;
                const c10::complex<double> we1 = t1 * al;
                const bool do_odd = (k <= pc.kmax_odd);
                c10::complex<double> wo0(0.0, 0.0);
                c10::complex<double> wo1(0.0, 0.0);
                if (do_odd) {
                    wo0 = s_at(sc0, m + 2 * k + 1) * al;
                    wo1 = s_at(sc1, m + 2 * k + 1) * al;
                }
                const double we0r = we0.real(), we0i = we0.imag();
                const double we1r = we1.real(), we1i = we1.imag();
                const double wo0r = wo0.real(), wo0i = wo0.imag();
                const double wo1r = wo1.real(), wo1i = wo1.imag();
                const bool step = (k < pc.kmax);
                const double ak = step ? pc.a[static_cast<size_t>(k)] : 0.0;
                const double bk = step ? pc.b[static_cast<size_t>(k)] : 0.0;
                ishioka_k_step_rings(
                    n_run,
                    do_odd,
                    step,
                    we0r,
                    we0i,
                    we1r,
                    we1i,
                    wo0r,
                    wo0i,
                    wo1r,
                    wo1i,
                    ak,
                    bk,
                    mu2_ptr + pair0,
                    p + pair0,
                    pold + pair0,
                    s0r + pair0,
                    s0i + pair0,
                    s1r + pair0,
                    s1i + pair0,
                    a0r + pair0,
                    a0i + pair0,
                    a1r + pair0,
                    a1i + pair0
                );
                if (has_equator && active[n_pairs]) {
                    const int j = static_cast<int>(n_pairs);
                    const double pj = p[j];
                    s0r[j] += we0r * pj;
                    s0i[j] += we0i * pj;
                    s1r[j] += we1r * pj;
                    s1i[j] += we1i * pj;
                    if (do_odd) {
                        a0r[j] += wo0r * pj;
                        a0i[j] += wo0i * pj;
                        a1r[j] += wo1r * pj;
                        a1i[j] += wo1i * pj;
                    }
                    if (step) {
                        const double pnew = (ak * (ct_ptr[j] * ct_ptr[j]) + bk) * pj + pold[j];
                        pold[j] = pj;
                        p[j] = pnew;
                    }
                }
            }

            auto store_pair = [&](int64_t idx, double rP, double iP, double rM, double iM) {
                auto zp = c10::complex<double>(rP, iP);
                auto zm = c10::complex<double>(rM, iM);
                if (phase_ptr != nullptr) {
                    const auto ph = phase_ptr[m * nrings + idx];
                    zp = zp * ph;
                    zm = zm * ph;
                }
                if (write_eq && idx >= eq_start && idx < eq_end && m < nph_eq) {
                    const int64_t off = (idx - eq_start) * nph_eq + m;
                    eq_plus[off] = zp;
                    eq_minus[off] = zm;
                    if (m == 0) {
                        s0_ptr[idx] = zp;
                        s1_ptr[idx] = zm;
                    }
                } else {
                    s0_ptr[m * nrings + idx] = zp;
                    s1_ptr[m * nrings + idx] = zm;
                }
            };
            for (int i = 0; i < n_eff; ++i) {
                if (!active[i]) {
                    continue;
                }
                const double mu = ct_ptr[i];
                const double sc = pre_phase * invs_ptr[i];
                const double rP = sc * (s0r[i] + mu * a0r[i]);
                const double iP = sc * (s0i[i] + mu * a0i[i]);
                const double rM = sc * (s1r[i] + mu * a1r[i]);
                const double iM = sc * (s1i[i] + mu * a1i[i]);
                store_pair(i, rP, iP, rM, iM);
                if (i < n_pairs) {
                    const double rPs = sc * (s0r[i] - mu * a0r[i]);
                    const double iPs = sc * (s0i[i] - mu * a0i[i]);
                    const double rMs = sc * (s1r[i] - mu * a1r[i]);
                    const double iMs = sc * (s1i[i] - mu * a1i[i]);
                    store_pair(nrings - 1 - i, rPs, iPs, rMs, iMs);
                }
            }
        }
    });

    return s_modes;
}

static torch::Tensor ishioka_spin_lowered_integrate_cpu(
    const torch::Tensor& s_plus,
    const torch::Tensor& s_minus,
    const torch::Tensor& theta,
    const torch::Tensor& weight,
    int64_t lmax,
    int64_t mmax,
    int64_t spin_abs
) {
    const c10::complex<double>* __restrict__ sp_ptr = s_plus.data_ptr<c10::complex<double>>();
    const c10::complex<double>* __restrict__ sm_ptr = s_minus.data_ptr<c10::complex<double>>();
    auto th_ptr = theta.data_ptr<double>();
    auto w_ptr = weight.data_ptr<double>();
    const bool w_scalar = weight.numel() == 1;
    const int64_t lmax_s = lmax + spin_abs;
    const int64_t nrings = theta.size(0);
    const int64_t n_eff = (nrings + 1) / 2;
    const int64_t n_pairs = nrings / 2;
    const bool has_equator = (nrings % 2 != 0);
    ensure_mfac(mmax);
    const auto& packs = g_ishioka_cache.get(lmax_s, mmax);
    const RecoupleSrc* rec_src = nullptr;
    const int64_t* rec_off = nullptr;
    const int64_t* rec_lmin = nullptr;
    g_recouple_cache.get(lmax, mmax, spin_abs, rec_src, rec_off, rec_lmin);

    auto options = torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU);
    torch::Tensor alms = torch::zeros({2, alm_size(lmax, mmax)}, options);
    auto* cp_out = alms.select(0, 0).data_ptr<c10::complex<double>>();
    auto* cm_out = alms.select(0, 1).data_ptr<c10::complex<double>>();

    std::vector<double> cos_theta_all(n_eff), sin_theta_all(n_eff), inv_sin_s(n_eff), mu2_all(n_eff);
    for (int i = 0; i < n_eff; ++i) {
        const double th = th_ptr[i];
        const double ct = std::cos(th);
        const double st = std::sin(th);
        cos_theta_all[static_cast<size_t>(i)] = ct;
        sin_theta_all[static_cast<size_t>(i)] = st;
        mu2_all[static_cast<size_t>(i)] = ct * ct;
        inv_sin_s[static_cast<size_t>(i)] = 1.0 / fast_pow(st, static_cast<int>(spin_abs));
    }
    std::vector<int64_t> mlim_north(n_eff, lmax_s);
    if (spin_mlim_prune_enabled()) {
        for (int64_t i = 0; i < n_eff; ++i) {
            const double cth = cos_theta_all[static_cast<size_t>(i)];
            const double sth = sin_theta_all[static_cast<size_t>(i)];
            mlim_north[static_cast<size_t>(i)] = sharp_mlim_estimate(lmax_s, 0, sth, cth);
        }
    }

    parallel_m_ranges(mmax, [&](int64_t m0, int64_t stride) {
        if (m0 > mmax) {
            return;
        }
        const int64_t n_s_max = lmax_s - m0 + 1;
        std::vector<c10::complex<double>> v_s0(static_cast<size_t>(n_s_max));
        std::vector<c10::complex<double>> v_s1(static_cast<size_t>(n_s_max));
        std::vector<double> v_p(n_eff), v_pold(n_eff);
        std::vector<double> v_s0r(n_eff), v_s0i(n_eff), v_a0r(n_eff), v_a0i(n_eff);
        std::vector<double> v_s1r(n_eff), v_s1i(n_eff), v_a1r(n_eff), v_a1i(n_eff);
        std::vector<uint8_t> v_active(n_eff);
        double* __restrict__ p = v_p.data();
        double* __restrict__ pold = v_pold.data();
        double* __restrict__ s0r = v_s0r.data();
        double* __restrict__ s0i = v_s0i.data();
        double* __restrict__ a0r = v_a0r.data();
        double* __restrict__ a0i = v_a0i.data();
        double* __restrict__ s1r = v_s1r.data();
        double* __restrict__ s1i = v_s1i.data();
        double* __restrict__ a1r = v_a1r.data();
        double* __restrict__ a1i = v_a1i.data();
        uint8_t* __restrict__ active = v_active.data();
        const double* __restrict__ ct_ptr = cos_theta_all.data();
        const double* __restrict__ st_ptr = sin_theta_all.data();
        const double* __restrict__ invs_ptr = inv_sin_s.data();
        const double* __restrict__ mu2_ptr = mu2_all.data();

        for (int64_t m = m0; m <= mmax; m += stride) {
            const IshiokaMCoeffs& pc = packs[static_cast<size_t>(m)];
            const int64_t n_s = lmax_s - m + 1;
            std::fill(v_s0.begin(), v_s0.begin() + n_s, c10::complex<double>(0.0, 0.0));
            std::fill(v_s1.begin(), v_s1.begin() + n_s, c10::complex<double>(0.0, 0.0));
            c10::complex<double>* __restrict__ sc0 = v_s0.data();
            c10::complex<double>* __restrict__ sc1 = v_s1.data();
            const double pre_phase = (m % 2 != 0) ? -1.0 : 1.0;
            const double seed = precomputed_mfac[static_cast<size_t>(m)];
            const int64_t m_off = m * nrings;
            bool any_active = false;

            for (int i = 0; i < n_eff; ++i) {
                const bool on = (m <= mlim_north[static_cast<size_t>(i)]);
                active[i] = static_cast<uint8_t>(on);
                any_active = any_active || on;
                p[i] = on ? seed * fast_pow(st_ptr[i], static_cast<int>(m)) : 0.0;
                pold[i] = 0.0;
                if (!on) {
                    s0r[i] = s0i[i] = a0r[i] = a0i[i] = 0.0;
                    s1r[i] = s1i[i] = a1r[i] = a1i[i] = 0.0;
                    continue;
                }
                const double wN = w_scalar ? w_ptr[0] : w_ptr[i];
                const double sc = pre_phase * invs_ptr[i];
                const auto np = sp_ptr[m_off + i];
                const auto nm = sm_ptr[m_off + i];
                double npr = np.real() * wN;
                double npi = np.imag() * wN;
                double nmr = nm.real() * wN;
                double nmi = nm.imag() * wN;
                double spr = 0.0, spi = 0.0, smr = 0.0, smi = 0.0;
                if (i < n_pairs) {
                    const int64_t idx_S = nrings - 1 - i;
                    const double wS = w_scalar ? w_ptr[0] : w_ptr[idx_S];
                    const auto spv = sp_ptr[m_off + idx_S];
                    const auto smv = sm_ptr[m_off + idx_S];
                    spr = spv.real() * wS;
                    spi = spv.imag() * wS;
                    smr = smv.real() * wS;
                    smi = smv.imag() * wS;
                }
                const double mu = ct_ptr[i];
                s0r[i] = sc * (npr + spr);
                s0i[i] = sc * (npi + spi);
                s1r[i] = sc * (nmr + smr);
                s1i[i] = sc * (nmi + smi);
                a0r[i] = sc * mu * (npr - spr);
                a0i[i] = sc * mu * (npi - spi);
                a1r[i] = sc * mu * (nmr - smr);
                a1i[i] = sc * mu * (nmi - smi);
            }
            if (!any_active) {
                continue;
            }
            int pair0 = 0;
            int n_run = 0;
            spin_active_pair_span(active, static_cast<int>(n_pairs), pair0, n_run);

            for (int64_t k = 0; k <= pc.kmax; ++k) {
                const int64_t n_even = m + 2 * k;
                const double e1 = pc.eps[static_cast<size_t>(n_even + 1)];
                const bool do_odd = (k <= pc.kmax_odd);
                const bool step = (k < pc.kmax);
                const double ak = step ? pc.a[static_cast<size_t>(k)] : 0.0;
                const double bk = step ? pc.b[static_cast<size_t>(k)] : 0.0;
                double we0r = 0.0, we0i = 0.0, we1r = 0.0, we1i = 0.0;
                double wo0r = 0.0, wo0i = 0.0, wo1r = 0.0, wo1i = 0.0;
                ishioka_k_reduce_rings(
                    n_run,
                    do_odd,
                    step,
                    ak,
                    bk,
                    mu2_ptr + pair0,
                    p + pair0,
                    pold + pair0,
                    s0r + pair0,
                    s0i + pair0,
                    s1r + pair0,
                    s1i + pair0,
                    a0r + pair0,
                    a0i + pair0,
                    a1r + pair0,
                    a1i + pair0,
                    we0r,
                    we0i,
                    we1r,
                    we1i,
                    wo0r,
                    wo0i,
                    wo1r,
                    wo1i
                );
                if (has_equator && active[n_pairs]) {
                    const int j = static_cast<int>(n_pairs);
                    const double pj = p[j];
                    we0r += s0r[j] * pj;
                    we0i += s0i[j] * pj;
                    we1r += s1r[j] * pj;
                    we1i += s1i[j] * pj;
                    if (do_odd) {
                        wo0r += a0r[j] * pj;
                        wo0i += a0i[j] * pj;
                        wo1r += a1r[j] * pj;
                        wo1i += a1i[j] * pj;
                    }
                    if (step) {
                        const double pnew = (ak * mu2_ptr[j] + bk) * pj + pold[j];
                        pold[j] = pj;
                        p[j] = pnew;
                    }
                }
                const double al = pc.alpha[static_cast<size_t>(k)];
                const c10::complex<double> t0(we0r * al, we0i * al);
                const c10::complex<double> t1(we1r * al, we1i * al);
                sc0[n_even - m] += t0 * e1;
                sc1[n_even - m] += t1 * e1;
                if (k < pc.kmax) {
                    const double e2 = pc.eps[static_cast<size_t>(n_even + 2)];
                    sc0[n_even + 2 - m] += t0 * e2;
                    sc1[n_even + 2 - m] += t1 * e2;
                }
                if (do_odd) {
                    const int64_t n_odd = m + 2 * k + 1;
                    sc0[n_odd - m] += c10::complex<double>(wo0r * al, wo0i * al);
                    sc1[n_odd - m] += c10::complex<double>(wo1r * al, wo1i * al);
                }
            }

            recouple_scalar_to_spin_m(
                sc0,
                sc1,
                lmax,
                m,
                spin_abs,
                rec_src,
                rec_off,
                rec_lmin,
                cp_out + alm_packed_index(m, m, lmax),
                cm_out + alm_packed_index(m, m, lmax)
            );
        }
    });

    return alms;
}

// Scalar recurrence coefficients for spin=0.
// For spin=0: c2_p = 0, so we only need {c1, c3}.
struct ScalarRecurrenceCoeffs {
    double c1, c3;
};

class ScalarRecurrenceCache {
    std::mutex mutex_;
    int64_t last_lmax_ = -1;
    int64_t last_mmax_ = -1;
    std::vector<ScalarRecurrenceCoeffs> coeffs_;

public:
    const std::vector<ScalarRecurrenceCoeffs>& get_coeffs(int64_t lmax, int64_t mmax, const std::vector<int64_t>& offsets) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (lmax == last_lmax_ && mmax == last_mmax_) {
            return coeffs_;
        }

        int64_t total_size = 0;
        for (int64_t m = 0; m <= mmax; ++m) {
            int64_t l0 = m;  // spin=0 => l0 = max(0, m) = m
            if (lmax >= l0 + 2) {
                total_size += (lmax - (l0 + 2) + 1);
            }
        }

        coeffs_.assign(total_size, {0, 0});
        int64_t current_off = 0;
        for (int64_t m = 0; m <= mmax; ++m) {
            int64_t l0 = m;
            for (int64_t l = l0 + 2; l <= lmax; ++l) {
                double lsq_msq = (double)(l * l - m * m);
                double kl = std::sqrt(lsq_msq);  // spin=0 => l*l - spin*spin = l*l
                // But the full formula is sqrt((l²-m²)(l²-s²)) = sqrt((l²-m²)*l²) = l*sqrt(l²-m²)
                // Wait - let me re-derive. For spin=0:
                // kl = sqrt((l²-m²)(l²-0²)) = sqrt((l²-m²)*l²) = l * sqrt(l²-m²)
                // preFactor = l / kl = l / (l * sqrt(l²-m²)) = 1 / sqrt(l²-m²)
                // c1 = preFactor * (2l-1) = (2l-1) / sqrt(l²-m²)
                // c3_num = sqrt(((l-1)²-m²)((l-1)²-0)) = (l-1) * sqrt((l-1)²-m²)
                // c3 = -preFactor * c3_num / (l-1) = -sqrt((l-1)²-m²) / sqrt(l²-m²)
                double inv_kl = 1.0 / (l * std::sqrt(lsq_msq));
                double c1 = inv_kl * l * (2.0 * l - 1.0);
                double lm1sq_msq = (double)((l - 1) * (l - 1) - m * m);
                double c3 = -inv_kl * l * ((l - 1.0) * std::sqrt(lm1sq_msq) / (l - 1));
                coeffs_[current_off++] = {c1, c3};
            }
        }
        last_lmax_ = lmax;
        last_mmax_ = mmax;
        return coeffs_;
    }
};

static ScalarRecurrenceCache g_scalar_recurrence_cache;


torch::Tensor healpix_scalar_integrate_recurrence_cpu(
    const torch::Tensor& s_modes,  // (mmax+1, nrings) complex128
    const torch::Tensor& theta,    // (nrings,) float64
    const torch::Tensor& weight,   // (nrings,) float64
    int64_t lmax,
    int64_t mmax
) {
    auto sm_f = s_modes.contiguous();
    auto sm_a = sm_f.accessor<c10::complex<double>, 2>();
    auto th_ptr = theta.data_ptr<double>();
    auto w_ptr = weight.data_ptr<double>();

    int64_t nrings = theta.size(0);
    int64_t n_eff = (nrings + 1) / 2;
    int64_t n_pairs = nrings / 2;
    bool has_equator = (nrings % 2 != 0);
    ensure_l_factors(lmax);
    ensure_mfac(mmax);

    auto options = torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU);
    torch::Tensor alms = torch::zeros({alm_size(lmax, mmax)}, options);
    auto alm_out = alms.data_ptr<c10::complex<double>>();

    // Build scalar recurrence LUT
    std::vector<int64_t> m_offsets(mmax + 1);
    int64_t current_lut_off = 0;
    for (int64_t m = 0; m <= mmax; ++m) {
        m_offsets[m] = current_lut_off;
        int64_t l0 = m;  // spin=0
        if (lmax >= l0 + 2) {
            current_lut_off += (lmax - (l0 + 2) + 1);
        }
    }
    const auto& lut = g_scalar_recurrence_cache.get_coeffs(lmax, mmax, m_offsets);

    std::vector<double> cos_theta_all(n_eff);
    std::vector<double> sin_theta_all(n_eff);
    for (int i = 0; i < n_eff; ++i) {
        cos_theta_all[i] = std::cos(th_ptr[i]);
        sin_theta_all[i] = std::sin(th_ptr[i]);
    }

    parallel_m_ranges(mmax, [&](int64_t m0, int64_t stride) {
        std::vector<double> v_lam2(n_eff), v_lam1(n_eff), v_lamc(n_eff);
        double * __restrict__ lam2 = v_lam2.data();
        double * __restrict__ lam1 = v_lam1.data();
        double * __restrict__ lamc = v_lamc.data();

        // North/South data buffers
        std::vector<double> vNr(n_eff), vNi(n_eff), vSr(n_eff), vSi(n_eff);
        std::vector<double> v_weightsN(n_eff), v_weightsS(n_eff);
        double * __restrict__ Nr = vNr.data(); double * __restrict__ Ni = vNi.data();
        double * __restrict__ Sr = vSr.data(); double * __restrict__ Si = vSi.data();
        double * __restrict__ weightsN = v_weightsN.data();
        double * __restrict__ weightsS = v_weightsS.data();
        const double * __restrict__ ct_ptr = cos_theta_all.data();
        const double * __restrict__ st_ptr = sin_theta_all.data();

        for (int64_t m = m0; m <= mmax; m += stride) {
            int64_t l0 = m;
            int64_t idx_base = m * (lmax + 1) - (m * (m - 1)) / 2;
            const double pre_phase = (m % 2 != 0) ? -1.0 : 1.0;
            const auto* lut_ptr = &lut[m_offsets[m]];

            // Load Fourier modes and weights for north/south rings
            for (int i = 0; i < n_eff; ++i) {
                weightsN[i] = w_ptr[i];
                Nr[i] = sm_a[m][i].real();
                Ni[i] = sm_a[m][i].imag();
                if (i < n_pairs) {
                    int64_t idx_S = nrings - 1 - i;
                    weightsS[i] = w_ptr[idx_S];
                    Sr[i] = sm_a[m][idx_S].real();
                    Si[i] = sm_a[m][idx_S].imag();
                }
            }

            // Seed using eval_wigner_d (matches spin kernel convention)
            WignerDCoeffs coeffs_c1 = get_wigner_d_coeffs(l0, m, 0);
            for (int i = 0; i < n_eff; ++i) {
                lam1[i] = eval_wigner_d(coeffs_c1, th_ptr[i]);
                lam2[i] = 0.0;
            }

            auto accumulate_l = [&](int64_t l, const double* __restrict__ lam) {
                double r_alm = 0, i_alm = 0;
                const double parity = ((l + m) % 2 == 0) ? 1.0 : -1.0;
                #pragma clang loop vectorize(enable)
                for (int i = 0; i < n_pairs; ++i) {
                    double d_val = lam[i];
                    r_alm += (Nr[i] * weightsN[i] * d_val + Sr[i] * weightsS[i] * d_val * parity);
                    i_alm += (Ni[i] * weightsN[i] * d_val + Si[i] * weightsS[i] * d_val * parity);
                }
                if (has_equator) {
                    int i = n_pairs;
                    r_alm += Nr[i] * weightsN[i] * lam[i];
                    i_alm += Ni[i] * weightsN[i] * lam[i];
                }
                const double sc = pre_phase * precomputed_l_factors[l];
                alm_out[idx_base + (l - m)] = c10::complex<double>(r_alm * sc, i_alm * sc);
            };

            accumulate_l(l0, lam1);

            if (l0 < lmax) {
                WignerDCoeffs coeffs_cc = get_wigner_d_coeffs(l0 + 1, m, 0);
                for (int i = 0; i < n_eff; ++i) {
                    lamc[i] = eval_wigner_d(coeffs_cc, th_ptr[i]);
                }
                accumulate_l(l0 + 1, lamc);
                { auto t = lam2; lam2 = lam1; lam1 = lamc; lamc = t; }

                // Main recurrence loop l = l0+2 ... lmax
                for (int64_t l = l0 + 2; l <= lmax; ++l) {
                    const auto& c = lut_ptr[l - (l0 + 2)];
                    const double c1 = c.c1, c3 = c.c3;
                    const double parity = ((l + m) % 2 == 0) ? 1.0 : -1.0;
                    double r_alm = 0, i_alm = 0;

                    #pragma clang loop vectorize(enable)
                    for (int i = 0; i < n_pairs; ++i) {
                        const double d_val = c1 * ct_ptr[i] * lam1[i] + c3 * lam2[i];
                        lamc[i] = d_val;
                        r_alm += (Nr[i] * weightsN[i] * d_val + Sr[i] * weightsS[i] * d_val * parity);
                        i_alm += (Ni[i] * weightsN[i] * d_val + Si[i] * weightsS[i] * d_val * parity);
                    }
                    if (has_equator) {
                        int i = n_pairs;
                        const double d_val = c1 * ct_ptr[i] * lam1[i] + c3 * lam2[i];
                        lamc[i] = d_val;
                        r_alm += Nr[i] * weightsN[i] * d_val;
                        i_alm += Ni[i] * weightsN[i] * d_val;
                    }
                    const double sc = pre_phase * precomputed_l_factors[l];
                    alm_out[idx_base + (l - m)] = c10::complex<double>(r_alm * sc, i_alm * sc);
                    { auto t = lam2; lam2 = lam1; lam1 = lamc; lamc = t; }
                }
            }
        }
    });

    return alms;
}

torch::Tensor healpix_scalar_interpolate_recurrence_cpu(
    const torch::Tensor& alm,    // (nalm,) complex128
    const torch::Tensor& theta,  // (nrings,) float64
    int64_t lmax,
    int64_t mmax
) {
    auto alm_f = alm.contiguous();
    auto alm_a = alm_f.data_ptr<c10::complex<double>>();
    auto th_ptr = theta.data_ptr<double>();

    int64_t nrings = theta.size(0);
    int64_t n_eff = (nrings + 1) / 2;
    int64_t n_pairs = nrings / 2;
    bool has_equator = (nrings % 2 != 0);
    ensure_l_factors(lmax);
    ensure_mfac(mmax);

    auto options = torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU);
    torch::Tensor s_modes = torch::zeros({mmax + 1, nrings}, options);
    auto sm_a = s_modes.accessor<c10::complex<double>, 2>();

    // Build scalar recurrence LUT
    std::vector<int64_t> m_offsets(mmax + 1);
    int64_t current_lut_off = 0;
    for (int64_t m = 0; m <= mmax; ++m) {
        m_offsets[m] = current_lut_off;
        int64_t l0 = m;
        if (lmax >= l0 + 2) {
            current_lut_off += (lmax - (l0 + 2) + 1);
        }
    }
    const auto& lut = g_scalar_recurrence_cache.get_coeffs(lmax, mmax, m_offsets);

    std::vector<double> cos_theta_all(n_eff);
    std::vector<double> sin_theta_all(n_eff);
    for (int i = 0; i < n_eff; ++i) {
        cos_theta_all[i] = std::cos(th_ptr[i]);
        sin_theta_all[i] = std::sin(th_ptr[i]);
    }

    parallel_m_ranges(mmax, [&](int64_t m0, int64_t stride) {
        std::vector<double> v_lam2(n_eff), v_lam1(n_eff), v_lamc(n_eff);
        double * __restrict__ lam2 = v_lam2.data();
        double * __restrict__ lam1 = v_lam1.data();
        double * __restrict__ lamc = v_lamc.data();

        std::vector<double> v_rN(n_eff), v_iN(n_eff), v_rS(n_eff), v_iS(n_eff);
        double * __restrict__ rN = v_rN.data(); double * __restrict__ iN = v_iN.data();
        double * __restrict__ rS = v_rS.data(); double * __restrict__ iS = v_iS.data();
        const double * __restrict__ ct_ptr = cos_theta_all.data();
        const double * __restrict__ st_ptr = sin_theta_all.data();

        for (int64_t m = m0; m <= mmax; m += stride) {
            int64_t l0 = m;
            int64_t idx_base = m * (lmax + 1) - (m * (m - 1)) / 2;
            const double pre_phase = (m % 2 != 0) ? -1.0 : 1.0;
            const auto* lut_ptr = &lut[m_offsets[m]];

            // Zero accumulators
            for (int i = 0; i < n_eff; ++i) {
                rN[i] = 0; iN[i] = 0;
                rS[i] = 0; iS[i] = 0;
            }

            // Seed using eval_wigner_d (matches spin kernel convention)
            WignerDCoeffs coeffs_c1 = get_wigner_d_coeffs(l0, m, 0);
            for (int i = 0; i < n_eff; ++i) {
                lam1[i] = eval_wigner_d(coeffs_c1, th_ptr[i]);
                lam2[i] = 0.0;
            }

            auto update_block = [&](int64_t l, const double* __restrict__ lam) {
                const c10::complex<double> a = alm_a[idx_base + (l - m)];
                const double sc = pre_phase * precomputed_l_factors[l];
                const double r0 = a.real() * sc, i0 = a.imag() * sc;
                const double parity = ((l + m) % 2 == 0) ? 1.0 : -1.0;

                #pragma clang loop vectorize(enable)
                for (int i = 0; i < n_pairs; ++i) {
                    rN[i] += r0 * lam[i];
                    iN[i] += i0 * lam[i];
                    rS[i] += r0 * lam[i] * parity;
                    iS[i] += i0 * lam[i] * parity;
                }
                if (has_equator) {
                    int i = n_pairs;
                    rN[i] += r0 * lam[i];
                    iN[i] += i0 * lam[i];
                }
            };

            update_block(l0, lam1);

            if (l0 < lmax) {
                WignerDCoeffs coeffs_cc = get_wigner_d_coeffs(l0 + 1, m, 0);
                for (int i = 0; i < n_eff; ++i) {
                    lamc[i] = eval_wigner_d(coeffs_cc, th_ptr[i]);
                }
                update_block(l0 + 1, lamc);
                { auto t = lam2; lam2 = lam1; lam1 = lamc; lamc = t; }

                for (int64_t l = l0 + 2; l <= lmax; ++l) {
                    const auto& c = lut_ptr[l - (l0 + 2)];
                    const c10::complex<double> a = alm_a[idx_base + (l - m)];
                    const double sc = pre_phase * precomputed_l_factors[l];
                    const double r0 = a.real() * sc, i0 = a.imag() * sc;
                    const double parity = ((l + m) % 2 == 0) ? 1.0 : -1.0;
                    const double c1 = c.c1, c3 = c.c3;

                    #pragma clang loop vectorize(enable)
                    for (int i = 0; i < n_pairs; ++i) {
                        const double d_val = c1 * ct_ptr[i] * lam1[i] + c3 * lam2[i];
                        lamc[i] = d_val;
                        rN[i] += r0 * d_val;
                        iN[i] += i0 * d_val;
                        rS[i] += r0 * d_val * parity;
                        iS[i] += i0 * d_val * parity;
                    }
                    if (has_equator) {
                        int i = n_pairs;
                        const double d_val = c1 * ct_ptr[i] * lam1[i] + c3 * lam2[i];
                        lamc[i] = d_val;
                        rN[i] += r0 * d_val;
                        iN[i] += i0 * d_val;
                    }
                    { auto t = lam2; lam2 = lam1; lam1 = lamc; lamc = t; }
                }
            }

            for (int i = 0; i < n_eff; ++i) {
                sm_a[m][i] = c10::complex<double>(rN[i], iN[i]);
                if (i < n_pairs) {
                    int64_t idx_S = nrings - 1 - i;
                    sm_a[m][idx_S] = c10::complex<double>(rS[i], iS[i]);
                }
            }
        }
    });

    return s_modes;
}

torch::Tensor healpix_query_disc_cpu(

    int64_t nside,
    const torch::Tensor& vec_xyz,
    double cos_lim,
    bool nest
) {
    if (!is_valid_nside(nside)) {
        throw std::runtime_error("nside must be a positive power of two");
    }
    if (!vec_xyz.device().is_cpu() || vec_xyz.scalar_type() != torch::kDouble) {
        throw std::runtime_error("healpix_query_disc_cpu expects float64 CPU vec");
    }
    auto vec = vec_xyz.contiguous().reshape({-1});
    if (vec.numel() != 3) {
        throw std::runtime_error("vec must have length 3");
    }

    double vx = vec.data_ptr<double>()[0];
    double vy = vec.data_ptr<double>()[1];
    double vz = vec.data_ptr<double>()[2];
    const double vnorm = std::sqrt(vx * vx + vy * vy + vz * vz);
    if (!(vnorm > 0.0)) {
        throw std::runtime_error("vec norm must be positive");
    }
    vx /= vnorm;
    vy /= vnorm;
    vz /= vnorm;

    const int64_t npix = 12 * nside * nside;
    if (cos_lim <= -1.0) {
        return torch::arange(npix, torch::TensorOptions().dtype(torch::kInt64));
    }
    if (cos_lim > 1.0) {
        return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64));
    }

    const double vxy = std::sqrt(std::max(0.0, vx * vx + vy * vy));
    const double phi0 = wrap_rad_2pi(std::atan2(vy, vx));
    const double eps = 1.0e-14;
    struct Range {
        int64_t start;
        int64_t len;
    };
    std::vector<Range> ranges;
    ranges.reserve(static_cast<size_t>(std::max<int64_t>(64, nside * 4)));

    auto append_range = [&](int64_t startpix, int64_t ringpix, int64_t k0, int64_t k1) {
        if (k1 < 0 || k0 >= ringpix || k0 > k1) {
            return;
        }
        if (k0 < 0) {
            k0 = 0;
        }
        if (k1 >= ringpix) {
            k1 = ringpix - 1;
        }
        if (k0 > k1) {
            return;
        }
        const int64_t start = startpix + k0;
        const int64_t len = k1 - k0 + 1;
        if (!ranges.empty()) {
            auto& prev = ranges.back();
            const int64_t prev_end = prev.start + prev.len;
            if (start <= prev_end) {
                const int64_t new_end = std::max(prev_end, start + len);
                prev.len = new_end - prev.start;
                return;
            }
        }
        ranges.push_back(Range{start, len});
    };

    for (int64_t ring = 1; ring < 4 * nside; ++ring) {
        int64_t startpix = 0;
        int64_t ringpix = 0;
        double theta = 0.0;
        bool shifted = false;
        get_ring_info2_scalar(nside, ring, startpix, ringpix, theta, shifted);

        double z = std::cos(theta);
        if (z > 1.0) {
            z = 1.0;
        } else if (z < -1.0) {
            z = -1.0;
        }
        const double rxy = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double dot_center = z * vz;
        const double span = rxy * vxy;

        if (span <= eps) {
            if (dot_center >= cos_lim) {
                append_range(startpix, ringpix, 0, ringpix - 1);
            }
            continue;
        }
        if (dot_center + span < cos_lim) {
            continue;
        }
        if (dot_center - span >= cos_lim) {
            append_range(startpix, ringpix, 0, ringpix - 1);
            continue;
        }

        double c = (cos_lim - dot_center) / span;
        if (c <= -1.0) {
            append_range(startpix, ringpix, 0, ringpix - 1);
            continue;
        }
        if (c > 1.0) {
            continue;
        }
        if (c > 1.0 - 1.0e-15) {
            c = 1.0;
        } else if (c < -1.0 + 1.0e-15) {
            c = -1.0;
        }
        const double alpha = std::acos(c);
        if (alpha >= (PI - 1.0e-15)) {
            append_range(startpix, ringpix, 0, ringpix - 1);
            continue;
        }

        const double dphi = TWO_PI / static_cast<double>(ringpix);
        const double phase = shifted ? 0.5 : 0.0;
        const double lo = wrap_rad_2pi(phi0 - alpha);
        const double hi = wrap_rad_2pi(phi0 + alpha);

        auto interval_to_k = [&](double a, double b, int64_t& k0, int64_t& k1) {
            k0 = static_cast<int64_t>(std::ceil(a / dphi - phase - eps));
            k1 = static_cast<int64_t>(std::floor(b / dphi - phase + eps));
        };

        if (lo <= hi) {
            int64_t k0 = 0;
            int64_t k1 = -1;
            interval_to_k(lo, hi, k0, k1);
            append_range(startpix, ringpix, k0, k1);
        } else {
            // Wrap-around interval: emit low segment first to keep ascending ring order.
            int64_t k0a = 0;
            int64_t k1a = -1;
            interval_to_k(0.0, hi, k0a, k1a);
            append_range(startpix, ringpix, k0a, k1a);

            int64_t k0b = 0;
            int64_t k1b = -1;
            interval_to_k(lo, TWO_PI - eps, k0b, k1b);
            append_range(startpix, ringpix, k0b, k1b);
        }
    }

    if (ranges.empty()) {
        return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64));
    }

    int64_t total = 0;
    for (const auto& r : ranges) {
        total += r.len;
    }
    auto ring_out = torch::empty({total}, torch::TensorOptions().dtype(torch::kInt64));
    auto* ptr = ring_out.data_ptr<int64_t>();
    int64_t off = 0;
    for (const auto& r : ranges) {
        std::iota(ptr + off, ptr + off + r.len, r.start);
        off += r.len;
    }
    if (!nest) {
        return ring_out;
    }
    const int64_t npface = nside * nside;
    const int64_t ncap = 2 * nside * (nside - 1);
    const int64_t nl2 = 2 * nside;
    const int64_t nside_shift = ilog2_pow2(nside);
    const int64_t four_nside_shift = nside_shift + 2;
    const int64_t four_nside_mask = (4 * nside) - 1;

    std::vector<int64_t> nest_pix(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i) {
        const int64_t p = ptr[i];
        int64_t ix;
        int64_t iy;
        int64_t face_num;
        ring_to_xyf(
            nside,
            ncap,
            npix,
            nl2,
            nside_shift,
            four_nside_shift,
            four_nside_mask,
            p,
            ix,
            iy,
            face_num
        );
        nest_pix[static_cast<size_t>(i)] = xyf_to_nest(npface, ix, iy, face_num);
    }
    std::sort(nest_pix.begin(), nest_pix.end());
    return torch::from_blob(
        nest_pix.data(),
        {total},
        torch::TensorOptions().dtype(torch::kInt64)
    ).clone();
}

torch::Tensor healpix_spin_map2alm_ring_recurrence_cpu(
    const torch::Tensor& maps,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    const torch::Tensor& theta,
    const torch::Tensor& phase0_neg,
    int64_t lmax,
    int64_t mmax,
    int64_t spin,
    double pix_w
) {
    auto phase = phase0_neg.contiguous();
    auto theta_c = theta.contiguous();
    torch::Tensor s_stacked;
    if (maps.dim() == 2 && maps.scalar_type() == torch::kDouble) {
        s_stacked = healpix_ring_fourier_modes_spin_conj_cpu(
            maps, starts, lengths, mmax, phase.data_ptr<c10::complex<double>>(), true
        );
    } else {
        s_stacked = healpix_ring_fourier_modes_spin_conj_cpu(
            maps, starts, lengths, mmax, phase.data_ptr<c10::complex<double>>()
        );
    }
    auto s_plus = s_stacked.select(0, 0);
    auto s_minus = s_stacked.select(0, 1);
    auto weight = torch::empty({1}, torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU));
    weight.fill_(pix_w);
    return healpix_spin_integrate_recurrence_cpu(
        s_plus,
        s_minus,
        theta_c,
        weight,
        lmax,
        mmax,
        spin
    );
}

torch::Tensor healpix_spin_alm2map_ring_recurrence_cpu(
    const torch::Tensor& alm_plus,
    const torch::Tensor& starts,
    const torch::Tensor& lengths,
    const torch::Tensor& theta,
    const torch::Tensor& phase0_pos,
    int64_t lmax,
    int64_t mmax,
    int64_t spin,
    const torch::Tensor& alm_minus = torch::Tensor()
) {
    auto phase = phase0_pos.contiguous();
    const c10::complex<double>* a0 = nullptr;
    const c10::complex<double>* a1 = nullptr;
    torch::Tensor keep0;
    torch::Tensor keep1;
    torch::Tensor stacked;
    if (alm_minus.defined()) {
        keep0 = alm_plus.contiguous();
        keep1 = alm_minus.contiguous();
        a0 = keep0.data_ptr<c10::complex<double>>();
        a1 = keep1.data_ptr<c10::complex<double>>();
    } else {
        stacked = alm_plus.contiguous();
        a0 = stacked.data_ptr<c10::complex<double>>();
        a1 = a0 + stacked.size(1);
    }

    auto starts_c = starts.contiguous();
    auto lengths_c = lengths.contiguous();
    const auto* starts_ptr = starts_c.data_ptr<int64_t>();
    const auto* lengths_ptr = lengths_c.data_ptr<int64_t>();
    const int64_t nrings = starts_c.numel();
    int64_t eq_start = -1;
    int64_t n_eq = 0;
    int64_t nph_eq = 0;
    int64_t ring = 0;
    while (ring < nrings) {
        const int64_t run_start = ring;
        const int64_t nph = lengths_ptr[ring];
        int64_t pix_cursor = starts_ptr[ring] + nph;
        ++ring;
        while (ring < nrings && lengths_ptr[ring] == nph && starts_ptr[ring] == pix_cursor) {
            pix_cursor += nph;
            ++ring;
        }
        const int64_t count = ring - run_start;
        if (count > 1) {
            eq_start = run_start;
            n_eq = count;
            nph_eq = nph;
            break;
        }
    }

    torch::Tensor eq_spec;
    c10::complex<double>* eq_plus = nullptr;
    c10::complex<double>* eq_minus = nullptr;
    if (n_eq > 0 && nph_eq > mmax) {
        eq_spec = torch::zeros(
            {2, n_eq, nph_eq},
            torch::TensorOptions().dtype(torch::kComplexDouble).device(torch::kCPU)
        );
        eq_plus = eq_spec.select(0, 0).data_ptr<c10::complex<double>>();
        eq_minus = eq_spec.select(0, 1).data_ptr<c10::complex<double>>();
    }

    const torch::Tensor& alms_ref = alm_minus.defined() ? keep0 : stacked;
    auto theta_c = theta.contiguous();
    const bool profile = sht_profile_enabled();
    const auto t0 = std::chrono::steady_clock::now();
    auto s_modes = healpix_spin_interpolate_recurrence_cpu(
        alms_ref,
        theta_c,
        lmax,
        mmax,
        spin,
        phase.data_ptr<c10::complex<double>>(),
        a0,
        a1,
        eq_plus,
        eq_minus,
        eq_start,
        n_eq,
        nph_eq
    );
    const auto t1 = std::chrono::steady_clock::now();
    torch::Tensor out;
    if (eq_spec.defined() && eq_spec.numel() > 0) {
        out = healpix_spin_synthesize_qu_cpu(
            s_modes, starts_c, lengths_c, eq_spec, eq_start, n_eq, nph_eq
        );
    } else {
        auto p_pm_pos = healpix_ring_fourier_synthesis_cpu(s_modes, starts_c, lengths_c);
        out = spin_qu_from_ifft_cpu(
            p_pm_pos, s_modes.select(0, 0), s_modes.select(0, 1), starts_c, lengths_c
        );
    }
    if (profile) {
        const auto t2 = std::chrono::steady_clock::now();
        const double interp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double synth_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::fprintf(
            stderr,
            "TORCHSKY_SHT_PROFILE alm2map_spin interpolate=%.3f ms synthesize=%.3f ms "
            "spin_threads=%d aten_threads=%d\n",
            interp_ms,
            synth_ms,
            SpinMPool::instance().nworkers(),
            at::get_num_threads()
        );
    }
    return out;
}

template <typename T>
torch::Tensor sparse_get_values_blocks_impl(
    const torch::Tensor& query,
    const torch::Tensor& block_index,
    const torch::Tensor& block_values,
    const torch::Tensor& fill
) {
    const int64_t n = query.numel();
    const auto bv_sizes = block_values.sizes();
    if (bv_sizes.size() < 2) {
        throw std::runtime_error("block_values must be at least 2D");
    }
    const int64_t ratio2 = bv_sizes[bv_sizes.size() - 1];
    const int64_t n_blocks = bv_sizes[bv_sizes.size() - 2];
    if (ratio2 <= 0 || (ratio2 & (ratio2 - 1)) != 0) {
        throw std::runtime_error("block_values last dim must be a power of two");
    }
    int64_t lead_n = 1;
    std::vector<int64_t> out_sizes;
    out_sizes.reserve(bv_sizes.size() - 1);
    for (int64_t i = 0; i < static_cast<int64_t>(bv_sizes.size()) - 2; ++i) {
        lead_n *= bv_sizes[i];
        out_sizes.push_back(bv_sizes[i]);
    }
    out_sizes.push_back(n);
    auto out = torch::empty(
        out_sizes,
        torch::TensorOptions()
            .dtype(block_values.dtype())
            .device(block_values.device())
    );
    const auto* pix = query.data_ptr<int64_t>();
    const auto* index = block_index.data_ptr<int64_t>();
    const int64_t npix_cov = block_index.numel();
    const T fill_v = *fill.data_ptr<T>();
    const T* vals = block_values.data_ptr<T>();
    T* dst = out.data_ptr<T>();
    const int64_t shift = static_cast<int64_t>(
        __builtin_ctzll(static_cast<unsigned long long>(ratio2))
    );
    const int64_t mask = ratio2 - 1;
    const uint64_t npix = static_cast<uint64_t>(npix_cov) * static_cast<uint64_t>(ratio2);
    const auto body = [&](int64_t begin, int64_t end) {
        if (lead_n == 1) {
            for (int64_t i = begin; i < end; ++i) {
                const uint64_t p = static_cast<uint64_t>(pix[i]);
                if (p >= npix) {
                    dst[i] = fill_v;
                    continue;
                }
                const int64_t bid = index[static_cast<int64_t>(p >> shift)];
                if (static_cast<uint64_t>(bid) >= static_cast<uint64_t>(n_blocks)) {
                    dst[i] = fill_v;
                    continue;
                }
                dst[i] = vals[bid * ratio2 + (static_cast<int64_t>(p) & mask)];
            }
            return;
        }
        for (int64_t i = begin; i < end; ++i) {
            const uint64_t p = static_cast<uint64_t>(pix[i]);
            if (p >= npix || n_blocks <= 0) {
                for (int64_t c = 0; c < lead_n; ++c) {
                    dst[c * n + i] = fill_v;
                }
                continue;
            }
            const int64_t bid = index[static_cast<int64_t>(p >> shift)];
            if (static_cast<uint64_t>(bid) >= static_cast<uint64_t>(n_blocks)) {
                for (int64_t c = 0; c < lead_n; ++c) {
                    dst[c * n + i] = fill_v;
                }
                continue;
            }
            const int64_t off = static_cast<int64_t>(p) & mask;
            for (int64_t c = 0; c < lead_n; ++c) {
                dst[c * n + i] = vals[(c * n_blocks + bid) * ratio2 + off];
            }
        }
    };
    if (n >= PARALLEL_MIN_ELEMS) {
        at::parallel_for(0, n, 4096, body);
    } else {
        body(0, n);
    }
    return out;
}

torch::Tensor sparse_get_values_blocks_cpu(
    const torch::Tensor& query,
    const torch::Tensor& block_index,
    const torch::Tensor& block_values,
    const torch::Tensor& fill
) {
    if (!query.defined() || !block_index.defined() || !block_values.defined()
        || !fill.defined()) {
        throw std::runtime_error("sparse_get_values_blocks_cpu expects defined tensors");
    }
    if (query.device().type() != torch::kCPU
        || block_index.device().type() != torch::kCPU
        || block_values.device().type() != torch::kCPU
        || fill.device().type() != torch::kCPU) {
        throw std::runtime_error("sparse_get_values_blocks_cpu expects CPU tensors");
    }
    if (query.scalar_type() != torch::kLong || block_index.scalar_type() != torch::kLong) {
        throw std::runtime_error("query and block_index must be int64");
    }
    if (fill.numel() != 1) {
        throw std::runtime_error("fill must be a scalar");
    }
    if (fill.scalar_type() != block_values.scalar_type()) {
        throw std::runtime_error("fill dtype must match block_values");
    }
    auto q = query.contiguous();
    auto idx = block_index.contiguous();
    auto bv = block_values.contiguous();
    auto f = fill.contiguous();
    switch (bv.scalar_type()) {
        case torch::kFloat64:
            return sparse_get_values_blocks_impl<double>(q, idx, bv, f);
        case torch::kFloat32:
            return sparse_get_values_blocks_impl<float>(q, idx, bv, f);
        case torch::kInt64:
            return sparse_get_values_blocks_impl<int64_t>(q, idx, bv, f);
        case torch::kInt32:
            return sparse_get_values_blocks_impl<int32_t>(q, idx, bv, f);
        case torch::kBool:
            return sparse_get_values_blocks_impl<bool>(q, idx, bv, f);
        default:
            throw std::runtime_error("unsupported block_values dtype");
    }
}

}  // namespace torchsky
