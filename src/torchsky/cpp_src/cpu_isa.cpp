#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#if defined(_WIN32) && defined(_M_ARM64)
#include <windows.h>
#endif

#if defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#include "torchsky/ishioka_kernels.h"

namespace torchsky {
namespace {

std::once_flag dispatch_once;
IshiokaStepKernel step_kernel = ishioka_k_step_scalar;
IshiokaReduceKernel reduce_kernel = ishioka_k_reduce_scalar;
const char* isa_name = "scalar";

bool avx2_fma_supported() {
#if defined(TORCHSKY_HAVE_AVX2_KERNEL) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#elif defined(TORCHSKY_HAVE_AVX2_KERNEL) && defined(_MSC_VER)
    int registers[4] = {};
    __cpuid(registers, 1);
    const bool osxsave = (registers[2] & (1 << 27)) != 0;
    const bool fma = (registers[2] & (1 << 12)) != 0;
    if (!osxsave || !fma || (_xgetbv(0) & 0x6) != 0x6) {
        return false;
    }
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
#else
    return false;
#endif
}

bool neon_supported() {
#if defined(TORCHSKY_HAVE_NEON_KERNEL) && defined(__APPLE__)
    return true;
#elif defined(TORCHSKY_HAVE_NEON_KERNEL) && defined(__linux__) && defined(__aarch64__)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0;
#elif defined(TORCHSKY_HAVE_NEON_KERNEL) && defined(_WIN32) && defined(_M_ARM64)
    return IsProcessorFeaturePresent(PF_ARM_V8_INSTRUCTIONS_AVAILABLE) != 0;
#else
    return false;
#endif
}

void select_dispatch() {
    const char* value = std::getenv("TORCHSKY_CPU_ISA");
    const std::string requested = value == nullptr ? "auto" : value;
    if (requested != "auto" && requested != "native" && requested != "scalar"
        && requested != "avx2" && requested != "neon") {
        throw std::runtime_error(
            "TORCHSKY_CPU_ISA must be one of auto, native, scalar, avx2, or neon"
        );
    }
    if (requested == "scalar") {
        return;
    }
#if defined(TORCHSKY_HAVE_AVX2_KERNEL)
    if ((requested == "auto" || requested == "native" || requested == "avx2")
        && avx2_fma_supported()) {
        step_kernel = ishioka_k_step_avx2;
        reduce_kernel = ishioka_k_reduce_avx2;
        isa_name = "avx2+fma";
        return;
    }
#endif
#if defined(TORCHSKY_HAVE_NEON_KERNEL)
    if ((requested == "auto" || requested == "native" || requested == "neon")
        && neon_supported()) {
        step_kernel = ishioka_k_step_neon;
        reduce_kernel = ishioka_k_reduce_neon;
        isa_name = "neon";
        return;
    }
#endif
    if (requested == "avx2" || requested == "neon") {
        throw std::runtime_error(
            "TORCHSKY_CPU_ISA=" + requested + " is not supported by this build and CPU/OS"
        );
    }
}

void ensure_dispatch() {
    std::call_once(dispatch_once, select_dispatch);
}

}

void ishioka_k_step_rings(
    int n, bool do_odd, bool step, double we0r, double we0i, double we1r,
    double we1i, double wo0r, double wo0i, double wo1r, double wo1i,
    double ak, double bk, const double* mu2, double* p, double* pold,
    double* s0r, double* s0i, double* s1r, double* s1i, double* a0r,
    double* a0i, double* a1r, double* a1i
) {
    ensure_dispatch();
    step_kernel(
        n, do_odd, step, we0r, we0i, we1r, we1i, wo0r, wo0i, wo1r,
        wo1i, ak, bk, mu2, p, pold, s0r, s0i, s1r, s1i, a0r, a0i, a1r, a1i
    );
}

void ishioka_k_reduce_rings(
    int n, bool do_odd, bool step, double ak, double bk, const double* mu2,
    double* p, double* pold, const double* s0r, const double* s0i,
    const double* s1r, const double* s1i, const double* a0r,
    const double* a0i, const double* a1r, const double* a1i, double& we0r,
    double& we0i, double& we1r, double& we1i, double& wo0r, double& wo0i,
    double& wo1r, double& wo1i
) {
    ensure_dispatch();
    reduce_kernel(
        n, do_odd, step, ak, bk, mu2, p, pold, s0r, s0i, s1r, s1i, a0r,
        a0i, a1r, a1i, &we0r, &we0i, &we1r, &we1i, &wo0r, &wo0i, &wo1r,
        &wo1i
    );
}

const char* selected_cpu_isa() {
    ensure_dispatch();
    return isa_name;
}

}
