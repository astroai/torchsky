#include <arm_neon.h>

#include "torchsky/ishioka_kernels.h"

namespace torchsky {

void ishioka_k_step_neon(
    int n, bool do_odd, bool step, double we0r, double we0i, double we1r,
    double we1i, double wo0r, double wo0i, double wo1r, double wo1i,
    double ak, double bk, const double* mu2, double* p, double* pold,
    double* s0r, double* s0i, double* s1r, double* s1i, double* a0r,
    double* a0i, double* a1r, double* a1i
) {
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
    int i = 0;
    for (; i + 2 <= n; i += 2) {
        const float64x2_t vp = vld1q_f64(p + i);
        vst1q_f64(s0r + i, vfmaq_f64(vld1q_f64(s0r + i), vwe0r, vp));
        vst1q_f64(s0i + i, vfmaq_f64(vld1q_f64(s0i + i), vwe0i, vp));
        vst1q_f64(s1r + i, vfmaq_f64(vld1q_f64(s1r + i), vwe1r, vp));
        vst1q_f64(s1i + i, vfmaq_f64(vld1q_f64(s1i + i), vwe1i, vp));
        if (do_odd) {
            vst1q_f64(a0r + i, vfmaq_f64(vld1q_f64(a0r + i), vwo0r, vp));
            vst1q_f64(a0i + i, vfmaq_f64(vld1q_f64(a0i + i), vwo0i, vp));
            vst1q_f64(a1r + i, vfmaq_f64(vld1q_f64(a1r + i), vwo1r, vp));
            vst1q_f64(a1i + i, vfmaq_f64(vld1q_f64(a1i + i), vwo1i, vp));
        }
        if (step) {
            const float64x2_t pnew = vfmaq_f64(
                vld1q_f64(pold + i),
                vfmaq_f64(vbk, vak, vld1q_f64(mu2 + i)),
                vp
            );
            vst1q_f64(pold + i, vp);
            vst1q_f64(p + i, pnew);
        }
    }
    ishioka_k_step_scalar(
        n - i, do_odd, step, we0r, we0i, we1r, we1i, wo0r, wo0i, wo1r,
        wo1i, ak, bk, mu2 + i, p + i, pold + i, s0r + i, s0i + i,
        s1r + i, s1i + i, a0r + i, a0i + i, a1r + i, a1i + i
    );
}

void ishioka_k_reduce_neon(
    int n, bool do_odd, bool step, double ak, double bk, const double* mu2,
    double* p, double* pold, const double* s0r, const double* s0i,
    const double* s1r, const double* s1i, const double* a0r,
    const double* a0i, const double* a1r, const double* a1i, double* we0r,
    double* we0i, double* we1r, double* we1i, double* wo0r, double* wo0i,
    double* wo1r, double* wo1i
) {
    float64x2_t e0r = vdupq_n_f64(0.0);
    float64x2_t e0i = vdupq_n_f64(0.0);
    float64x2_t e1r = vdupq_n_f64(0.0);
    float64x2_t e1i = vdupq_n_f64(0.0);
    float64x2_t o0r = vdupq_n_f64(0.0);
    float64x2_t o0i = vdupq_n_f64(0.0);
    float64x2_t o1r = vdupq_n_f64(0.0);
    float64x2_t o1i = vdupq_n_f64(0.0);
    const float64x2_t vak = vdupq_n_f64(ak);
    const float64x2_t vbk = vdupq_n_f64(bk);
    int i = 0;
    for (; i + 2 <= n; i += 2) {
        const float64x2_t vp = vld1q_f64(p + i);
        e0r = vfmaq_f64(e0r, vld1q_f64(s0r + i), vp);
        e0i = vfmaq_f64(e0i, vld1q_f64(s0i + i), vp);
        e1r = vfmaq_f64(e1r, vld1q_f64(s1r + i), vp);
        e1i = vfmaq_f64(e1i, vld1q_f64(s1i + i), vp);
        if (do_odd) {
            o0r = vfmaq_f64(o0r, vld1q_f64(a0r + i), vp);
            o0i = vfmaq_f64(o0i, vld1q_f64(a0i + i), vp);
            o1r = vfmaq_f64(o1r, vld1q_f64(a1r + i), vp);
            o1i = vfmaq_f64(o1i, vld1q_f64(a1i + i), vp);
        }
        if (step) {
            const float64x2_t pnew = vfmaq_f64(
                vld1q_f64(pold + i),
                vfmaq_f64(vbk, vak, vld1q_f64(mu2 + i)),
                vp
            );
            vst1q_f64(pold + i, vp);
            vst1q_f64(p + i, pnew);
        }
    }
    *we0r += vaddvq_f64(e0r);
    *we0i += vaddvq_f64(e0i);
    *we1r += vaddvq_f64(e1r);
    *we1i += vaddvq_f64(e1i);
    if (do_odd) {
        *wo0r += vaddvq_f64(o0r);
        *wo0i += vaddvq_f64(o0i);
        *wo1r += vaddvq_f64(o1r);
        *wo1i += vaddvq_f64(o1i);
    }
    ishioka_k_reduce_scalar(
        n - i, do_odd, step, ak, bk, mu2 + i, p + i, pold + i, s0r + i,
        s0i + i, s1r + i, s1i + i, a0r + i, a0i + i, a1r + i,
        a1i + i, we0r, we0i, we1r, we1i, wo0r, wo0i, wo1r, wo1i
    );
}

}
