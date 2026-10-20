#include "torchsky/ishioka_kernels.h"

namespace torchsky {

void ishioka_k_step_scalar(
    int n, bool do_odd, bool step, double we0r, double we0i, double we1r,
    double we1i, double wo0r, double wo0i, double wo1r, double wo1i,
    double ak, double bk, const double* mu2, double* p, double* pold,
    double* s0r, double* s0i, double* s1r, double* s1i, double* a0r,
    double* a0i, double* a1r, double* a1i
) {
    for (int i = 0; i < n; ++i) {
        const double pj = p[i];
        s0r[i] += we0r * pj;
        s0i[i] += we0i * pj;
        s1r[i] += we1r * pj;
        s1i[i] += we1i * pj;
        if (do_odd) {
            a0r[i] += wo0r * pj;
            a0i[i] += wo0i * pj;
            a1r[i] += wo1r * pj;
            a1i[i] += wo1i * pj;
        }
        if (step) {
            const double pnew = (ak * mu2[i] + bk) * pj + pold[i];
            pold[i] = pj;
            p[i] = pnew;
        }
    }
}

void ishioka_k_reduce_scalar(
    int n, bool do_odd, bool step, double ak, double bk, const double* mu2,
    double* p, double* pold, const double* s0r, const double* s0i,
    const double* s1r, const double* s1i, const double* a0r,
    const double* a0i, const double* a1r, const double* a1i, double* we0r,
    double* we0i, double* we1r, double* we1i, double* wo0r, double* wo0i,
    double* wo1r, double* wo1i
) {
    for (int i = 0; i < n; ++i) {
        const double pj = p[i];
        *we0r += s0r[i] * pj;
        *we0i += s0i[i] * pj;
        *we1r += s1r[i] * pj;
        *we1i += s1i[i] * pj;
        if (do_odd) {
            *wo0r += a0r[i] * pj;
            *wo0i += a0i[i] * pj;
            *wo1r += a1r[i] * pj;
            *wo1i += a1i[i] * pj;
        }
        if (step) {
            const double pnew = (ak * mu2[i] + bk) * pj + pold[i];
            pold[i] = pj;
            p[i] = pnew;
        }
    }
}

}
