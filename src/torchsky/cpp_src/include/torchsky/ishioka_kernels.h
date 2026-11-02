#pragma once

namespace torchsky {

using IshiokaStepKernel = void (*)(
    int,
    bool,
    bool,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    const double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*
);

using IshiokaReduceKernel = void (*)(
    int,
    bool,
    bool,
    double,
    double,
    const double*,
    double*,
    double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*
);

void ishioka_k_step_scalar(
    int, bool, bool, double, double, double, double, double, double, double,
    double, double, double, const double*, double*, double*, double*, double*,
    double*, double*, double*, double*, double*, double*
);
void ishioka_k_reduce_scalar(
    int, bool, bool, double, double, const double*, double*, double*,
    const double*, const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, double*, double*, double*,
    double*, double*, double*, double*, double*
);

#if defined(TORCHSKY_HAVE_AVX2_KERNEL)
void ishioka_k_step_avx2(
    int, bool, bool, double, double, double, double, double, double, double,
    double, double, double, const double*, double*, double*, double*, double*,
    double*, double*, double*, double*, double*, double*
);
void ishioka_k_reduce_avx2(
    int, bool, bool, double, double, const double*, double*, double*,
    const double*, const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, double*, double*, double*,
    double*, double*, double*, double*, double*
);
#endif

#if defined(TORCHSKY_HAVE_NEON_KERNEL)
void ishioka_k_step_neon(
    int, bool, bool, double, double, double, double, double, double, double,
    double, double, double, const double*, double*, double*, double*, double*,
    double*, double*, double*, double*, double*, double*
);
void ishioka_k_reduce_neon(
    int, bool, bool, double, double, const double*, double*, double*,
    const double*, const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, double*, double*, double*,
    double*, double*, double*, double*, double*
);
#endif

void ishioka_k_step_rings(
    int, bool, bool, double, double, double, double, double, double, double,
    double, double, double, const double*, double*, double*, double*, double*,
    double*, double*, double*, double*, double*, double*
);
void ishioka_k_reduce_rings(
    int, bool, bool, double, double, const double*, double*, double*,
    const double*, const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, double&, double&, double&,
    double&, double&, double&, double&, double&
);
const char* selected_cpu_isa();

}
