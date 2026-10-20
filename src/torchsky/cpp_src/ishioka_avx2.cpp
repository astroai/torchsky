#include <immintrin.h>

#include "torchsky/ishioka_kernels.h"

namespace torchsky {

void ishioka_k_step_avx2(
    int n, bool do_odd, bool step, double we0r, double we0i, double we1r,
    double we1i, double wo0r, double wo0i, double wo1r, double wo1i,
    double ak, double bk, const double* mu2, double* p, double* pold,
    double* s0r, double* s0i, double* s1r, double* s1i, double* a0r,
    double* a0i, double* a1r, double* a1i
) {
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
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d vp = _mm256_loadu_pd(p + i);
        _mm256_storeu_pd(s0r + i, _mm256_fmadd_pd(vwe0r, vp, _mm256_loadu_pd(s0r + i)));
        _mm256_storeu_pd(s0i + i, _mm256_fmadd_pd(vwe0i, vp, _mm256_loadu_pd(s0i + i)));
        _mm256_storeu_pd(s1r + i, _mm256_fmadd_pd(vwe1r, vp, _mm256_loadu_pd(s1r + i)));
        _mm256_storeu_pd(s1i + i, _mm256_fmadd_pd(vwe1i, vp, _mm256_loadu_pd(s1i + i)));
        if (do_odd) {
            _mm256_storeu_pd(a0r + i, _mm256_fmadd_pd(vwo0r, vp, _mm256_loadu_pd(a0r + i)));
            _mm256_storeu_pd(a0i + i, _mm256_fmadd_pd(vwo0i, vp, _mm256_loadu_pd(a0i + i)));
            _mm256_storeu_pd(a1r + i, _mm256_fmadd_pd(vwo1r, vp, _mm256_loadu_pd(a1r + i)));
            _mm256_storeu_pd(a1i + i, _mm256_fmadd_pd(vwo1i, vp, _mm256_loadu_pd(a1i + i)));
        }
        if (step) {
            const __m256d pnew = _mm256_fmadd_pd(
                _mm256_fmadd_pd(vak, _mm256_loadu_pd(mu2 + i), vbk),
                vp,
                _mm256_loadu_pd(pold + i)
            );
            _mm256_storeu_pd(pold + i, vp);
            _mm256_storeu_pd(p + i, pnew);
        }
    }
    ishioka_k_step_scalar(
        n - i, do_odd, step, we0r, we0i, we1r, we1i, wo0r, wo0i, wo1r,
        wo1i, ak, bk, mu2 + i, p + i, pold + i, s0r + i, s0i + i,
        s1r + i, s1i + i, a0r + i, a0i + i, a1r + i, a1i + i
    );
}

static double hsum4(__m256d value) {
    const __m128d lo = _mm256_castpd256_pd128(value);
    const __m128d hi = _mm256_extractf128_pd(value, 1);
    const __m128d sum = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_add_pd(sum, _mm_shuffle_pd(sum, sum, 1)));
}

void ishioka_k_reduce_avx2(
    int n, bool do_odd, bool step, double ak, double bk, const double* mu2,
    double* p, double* pold, const double* s0r, const double* s0i,
    const double* s1r, const double* s1i, const double* a0r,
    const double* a0i, const double* a1r, const double* a1i, double* we0r,
    double* we0i, double* we1r, double* we1i, double* wo0r, double* wo0i,
    double* wo1r, double* wo1i
) {
    __m256d e0r = _mm256_setzero_pd();
    __m256d e0i = _mm256_setzero_pd();
    __m256d e1r = _mm256_setzero_pd();
    __m256d e1i = _mm256_setzero_pd();
    __m256d o0r = _mm256_setzero_pd();
    __m256d o0i = _mm256_setzero_pd();
    __m256d o1r = _mm256_setzero_pd();
    __m256d o1i = _mm256_setzero_pd();
    const __m256d vak = _mm256_set1_pd(ak);
    const __m256d vbk = _mm256_set1_pd(bk);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d vp = _mm256_loadu_pd(p + i);
        e0r = _mm256_fmadd_pd(_mm256_loadu_pd(s0r + i), vp, e0r);
        e0i = _mm256_fmadd_pd(_mm256_loadu_pd(s0i + i), vp, e0i);
        e1r = _mm256_fmadd_pd(_mm256_loadu_pd(s1r + i), vp, e1r);
        e1i = _mm256_fmadd_pd(_mm256_loadu_pd(s1i + i), vp, e1i);
        if (do_odd) {
            o0r = _mm256_fmadd_pd(_mm256_loadu_pd(a0r + i), vp, o0r);
            o0i = _mm256_fmadd_pd(_mm256_loadu_pd(a0i + i), vp, o0i);
            o1r = _mm256_fmadd_pd(_mm256_loadu_pd(a1r + i), vp, o1r);
            o1i = _mm256_fmadd_pd(_mm256_loadu_pd(a1i + i), vp, o1i);
        }
        if (step) {
            const __m256d pnew = _mm256_fmadd_pd(
                _mm256_fmadd_pd(vak, _mm256_loadu_pd(mu2 + i), vbk),
                vp,
                _mm256_loadu_pd(pold + i)
            );
            _mm256_storeu_pd(pold + i, vp);
            _mm256_storeu_pd(p + i, pnew);
        }
    }
    *we0r += hsum4(e0r);
    *we0i += hsum4(e0i);
    *we1r += hsum4(e1r);
    *we1i += hsum4(e1i);
    if (do_odd) {
        *wo0r += hsum4(o0r);
        *wo0i += hsum4(o0i);
        *wo1r += hsum4(o1r);
        *wo1i += hsum4(o1i);
    }
    ishioka_k_reduce_scalar(
        n - i, do_odd, step, ak, bk, mu2 + i, p + i, pold + i, s0r + i,
        s0i + i, s1r + i, s1i + i, a0r + i, a0i + i, a1r + i,
        a1i + i, we0r, we0i, we1r, we1i, wo0r, wo0i, wo1r, wo1i
    );
}

}
