// Scalar (spin-0) HEALPix SHT Legendre kernels with exponent scaling.
//
// Included from healpix.cpp inside namespace torchsky (after SpinMPool /
// parallel_m_ranges are defined). Both kernels evaluate
//   d_l = d^l_{m,0}(theta)       (Wigner small-d; Y_lm = sqrt((2l+1)/4pi) d_l e^{im phi})
// with the three-term recurrence in l, vectorised over blocks of rings.
//
// Underflow: the sectoral seed D_m sin^m(theta) leaves the float64 range for
// m*ln(1/sin) > ~745. Each (m, ring) carries an integer exponent in units of
// 2^256 (as libsharp/ducc): the seed is formed as (mantissa, binary exponent)
// by repeated squaring, the recurrence runs on the mantissa with a branch-free
// rescale while the ring is still "scaled", and the ring only accumulates once
// its true value exceeds 2^-128 (contributions below that are < 1e-38 of an
// O(1) value and are dropped). Rings in the classically forbidden region
// m > lmax*sin(theta) + margin are skipped; the margin is the WKB bound
// exp(-(2*sqrt(2)/3) * delta^1.5 / sqrt(lmax*sin)) < e^-90.
//
// Work per m is split in ring blocks (kBlock rings, register resident state);
// north/south rings are folded into even/odd-parity sums so each (l, ring)
// costs one recurrence step plus two fused multiply-adds.

namespace sht_scalar {

constexpr int VL = 2;  // doubles per vector (16-byte NEON / SSE2 registers)
typedef double v4d __attribute__((vector_size(16)));
typedef long long v4l __attribute__((vector_size(16)));

inline v4d splat_mask() {
    const long long m = 0x7fffffffffffffffLL;
    return (v4d)v4l{m, m};
}

inline v4d vsel(v4l mask, v4d a, v4d b) {
    return (v4d)(((v4l)a & mask) | ((v4l)b & ~mask));
}

constexpr double kUp = 0x1p256;
constexpr double kDn = 0x1p-256;
constexpr double kBig = 0x1p128;
#ifndef TS_KINT
#define TS_KINT 8
#endif
#ifndef TS_KSYN
#define TS_KSYN 4
#endif
constexpr int kInt = TS_KINT;        // vectors per block in the integrate kernel
constexpr int kSyn = TS_KSYN;        // vectors per block in the synthesis kernel
constexpr int kMaxLanes = VL * (TS_KINT > TS_KSYN ? TS_KINT : TS_KSYN);

inline v4d splat(double x) {
    return v4d{x, x};
}

inline double hsum(v4d v) {
    return v[0] + v[1];
}

inline void renorm(double& x, int& e) {
    int k = 0;
    x = std::frexp(x, &k);
    e += k;
}

// base^n as x * 2^e with x in [0.5, 1).
inline void scaled_pow(double base, int64_t n, double& x, int& e) {
    x = 1.0;
    e = 0;
    renorm(x, e);
    double b = base;
    int be = 0;
    renorm(b, be);
    while (n > 0) {
        if (n & 1) {
            x *= b;
            e += be;
            renorm(x, e);
        }
        n >>= 1;
        if (n > 0) {
            b *= b;
            be *= 2;
            renorm(b, be);
        }
    }
}

// D_m = sqrt((2m)!) / (2^m m!)  (so that d^m_{m0} = D_m sin^m(theta)).
inline std::vector<double> sectoral_coeffs(int64_t mmax) {
    std::vector<double> d(static_cast<size_t>(mmax + 1));
    long double v = 1.0L;
    d[0] = 1.0;
    for (int64_t m = 1; m <= mmax; ++m) {
        v *= std::sqrt((2.0L * m - 1.0L) / (2.0L * m));
        d[static_cast<size_t>(m)] = static_cast<double>(v);
    }
    return d;
}

// Largest m for which a ring at colatitude (sin_theta) still carries weight.
inline int64_t forbidden_mlim(int64_t lmax, double sin_theta) {
    const double ls = static_cast<double>(lmax) * sin_theta;
    const double margin = std::max(50.0, 24.0 * std::cbrt(ls));
    const double lim = ls + margin;
    if (lim >= static_cast<double>(lmax)) {
        return lmax;
    }
    return static_cast<int64_t>(lim) + 1;
}

struct RingGeometry {
    int64_t nrings = 0;
    int64_t n_eff = 0;
    int64_t n_pairs = 0;
    bool has_equator = false;
    std::vector<double> ct, st;
    std::vector<int64_t> mlim;
    explicit RingGeometry(const double* theta, int64_t nr, int64_t lmax) {
        nrings = nr;
        n_eff = nr / 2 + nr % 2;
        n_pairs = nr / 2;
        has_equator = (nr % 2) != 0;
        ct.resize(static_cast<size_t>(n_eff));
        st.resize(static_cast<size_t>(n_eff));
        mlim.resize(static_cast<size_t>(n_eff));
        for (int64_t i = 0; i < n_eff; ++i) {
            ct[static_cast<size_t>(i)] = std::cos(theta[i]);
            st[static_cast<size_t>(i)] = std::sin(theta[i]);
            mlim[static_cast<size_t>(i)] = forbidden_mlim(lmax, st[static_cast<size_t>(i)]);
        }
    }
    // First north ring (by index) that is still active at this m; mlim grows
    // with the ring index, so active rings form the suffix [first, n_eff).
    int64_t first_active(int64_t m) const {
        int64_t lo = 0;
        int64_t hi = n_eff;
        while (lo < hi) {
            const int64_t mid = (lo + hi) / 2;
            if (mlim[static_cast<size_t>(mid)] >= m) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }
};

// Per-thread running seed sin(theta)^m as (x, e); advanced by `stride` in m.
struct SeedState {
    std::vector<double> x, sx;
    std::vector<int> e, se;
    int64_t m_cur = 0;
    int64_t stride = 1;
    void init(const RingGeometry& g, int64_t m0, int64_t stride_) {
        const size_t n = static_cast<size_t>(g.n_eff);
        x.assign(n, 1.0);
        sx.assign(n, 1.0);
        e.assign(n, 0);
        se.assign(n, 0);
        m_cur = m0;
        stride = stride_;
        for (size_t i = 0; i < n; ++i) {
            scaled_pow(g.st[i], m0, x[i], e[i]);
            scaled_pow(g.st[i], stride_, sx[i], se[i]);
        }
    }
    void advance_to(int64_t m) {
        while (m_cur < m) {
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] *= sx[i];
                e[i] += se[i];
                renorm(x[i], e[i]);
            }
            m_cur += stride;
        }
    }
};

// 2^n for n in [-1022, 1023] without a libm call.
inline double pow2i(int n) {
    const uint64_t bits = static_cast<uint64_t>(n + 1023) << 52;
    double r;
    std::memcpy(&r, &bits, sizeof(r));
    return r;
}

// Seed d_m = D_m sin^m as mantissa * 2^(256*ex), ex <= 0, mantissa in [2^-128, 2^128).
// px in [0.5, 1) and dm in (0, 1], so the product only needs its exponent shifted.
inline void ring_seed(double dm, double px, int pe, double& mant, double& ex) {
    const double x = dm * px;
    const int k = pe + 128;
    int q = (k >= 0) ? 0 : -((-k + 255) / 256);  // min(0, floor(k / 256))
    mant = x * pow2i(pe - 256 * q);
    ex = static_cast<double>(q);
}

// Per-m recurrence coefficients: d_l = c1[l-m] cos(theta) d_{l-1} + c3[l-m] d_{l-2}, l >= m+1.
inline void fill_coeffs(int64_t m, int64_t lmax, double* c1, double* c3) {
    double prev_den = 0.0;
    for (int64_t l = m + 1; l <= lmax; ++l) {
        const double den = std::sqrt(static_cast<double>(l * l - m * m));
        const double inv = 1.0 / den;
        c1[l - m] = (2.0 * static_cast<double>(l) - 1.0) * inv;
        c3[l - m] = -prev_den * inv;
        prev_den = den;
    }
}

// One block of up to kMaxLanes rings; NV vectors of 4 lanes.
template <int NV>
struct Lanes {
    alignas(16) double ct[NV * VL];
    alignas(16) double cur[NV * VL];
    alignas(16) double prev[NV * VL];
    alignas(16) double ex[NV * VL];  // < 0 while the ring is still scaled
    alignas(16) double live[NV * VL];  // 1 once the ring is normal
    bool any_scaled = false;

    void load(const RingGeometry& g, const SeedState& s, const double* dm_ptr, int64_t i0, int64_t nb) {
        any_scaled = false;
        for (int j = 0; j < NV * VL; ++j) {
            prev[j] = 0.0;
            if (j < nb) {
                const size_t i = static_cast<size_t>(i0 + j);
                ct[j] = g.ct[i];
                ring_seed(*dm_ptr, s.x[i], s.e[i], cur[j], ex[j]);
                any_scaled = any_scaled || ex[j] < 0.0;
            } else {
                ct[j] = 0.0;
                cur[j] = 0.0;
                ex[j] = 0.0;
            }
            live[j] = ex[j] < 0.0 ? 0.0 : 1.0;
        }
    }

    // Phase A: advance while some ring is still scaled. Groups of kGroup steps
    // run without overflow checks (growth per step <= c1 + |c3| < 2^9 for
    // m < 2^17, so a group stays far below the 2^1023 limit after crossing
    // 2^128); rings that are still scaled are rescaled at group boundaries.
    // `acc(l, cur)` is invoked per step only while some ring is live.
    template <typename Acc>
    inline int64_t run_scaled(int64_t l, int64_t lmax, int64_t m, const double* c1, const double* c3, Acc&& acc) {
        constexpr int kGroup = 4;
        v4d* vct = reinterpret_cast<v4d*>(ct);
        v4d* vcur = reinterpret_cast<v4d*>(cur);
        v4d* vprev = reinterpret_cast<v4d*>(prev);
        v4d* vex = reinterpret_cast<v4d*>(ex);
        const v4d big = splat(kBig), dn = splat(kDn), one = splat(1.0), zero = splat(0.0);
        v4d lct[NV], lcur[NV], lprev[NV];
        for (int k = 0; k < NV; ++k) {
            lct[k] = vct[k];
            lcur[k] = vcur[k];
            lprev[k] = vprev[k];
        }
        bool has_live = false;
        for (int j = 0; j < NV * VL; ++j) {
            has_live = has_live || live[j] != 0.0;
        }
        while (any_scaled && l < lmax) {
            const int64_t nstep = std::min<int64_t>(kGroup, lmax - l);
            for (int64_t s = 0; s < nstep; ++s) {
                ++l;
                const v4d C1 = splat(c1[l - m]), C3 = splat(c3[l - m]);
                for (int k = 0; k < NV; ++k) {
                    const v4d d = C1 * (lct[k] * lcur[k]) + C3 * lprev[k];
                    lprev[k] = lcur[k];
                    lcur[k] = d;
                }
                if (has_live) {
                    acc(l, lcur);
                }
            }
            bool changed = false;
            for (int k = 0; k < NV; ++k) {
                const v4d ad = (v4d)((v4l)lcur[k] & (v4l)splat_mask());
                const v4l hit = (ad > big) & (vex[k] < zero);
                const v4d f = vsel(hit, dn, one);
                lcur[k] *= f;
                lprev[k] *= f;
                vex[k] += vsel(hit, one, zero);
                changed = changed || (hit[0] | hit[1]) != 0;
            }
            if (changed) {
                any_scaled = false;
                for (int j = 0; j < NV * VL; ++j) {
                    any_scaled = any_scaled || ex[j] < 0.0;
                    live[j] = ex[j] < 0.0 ? 0.0 : 1.0;
                }
                has_live = false;
                for (int j = 0; j < NV * VL; ++j) {
                    has_live = has_live || live[j] != 0.0;
                }
            }
        }
        for (int k = 0; k < NV; ++k) {
            vcur[k] = lcur[k];
            vprev[k] = lprev[k];
        }
        return l;
    }
};

// alm[l] += scale * sum_i W_i d_l(i) for one ring block: map2alm direction.
inline void integrate_block(
    const RingGeometry& g,
    const SeedState& seeds,
    const double* dm_ptr,
    int64_t m,
    int64_t lmax,
    int64_t i0,
    int64_t nb,
    const double* c1,
    const double* c3,
    const double* wpr,  // even-parity weights, padded, indexed by ring
    const double* wpi,
    const double* wqr,  // odd-parity weights
    const double* wqi,
    const double* lscale,  // per-l scale pre_phase * sqrt((2l+1)/4pi), indexed by l
    c10::complex<double>* alm_m  // alm_out + idx_base - m  (index by l)
) {
    constexpr int NV = kInt;
    Lanes<NV> ln;
    ln.load(g, seeds, dm_ptr, i0, nb);
    alignas(16) double pr[NV * VL], pi_[NV * VL], qr[NV * VL], qi[NV * VL];
    for (int j = 0; j < NV * VL; ++j) {
        const bool on = j < nb;
        pr[j] = on ? wpr[i0 + j] : 0.0;
        pi_[j] = on ? wpi[i0 + j] : 0.0;
        qr[j] = on ? wqr[i0 + j] : 0.0;
        qi[j] = on ? wqi[i0 + j] : 0.0;
    }
    int64_t l = m;
    // l == m: d = seed (even parity).
    if (!ln.any_scaled) {
        double ar = 0.0, ai = 0.0;
        for (int j = 0; j < NV * VL; ++j) {
            ar += pr[j] * ln.cur[j];
            ai += pi_[j] * ln.cur[j];
        }
        reinterpret_cast<double*>(alm_m)[2 * l] += ar * lscale[l];
        reinterpret_cast<double*>(alm_m)[2 * l + 1] += ai * lscale[l];
    } else {
        double ar = 0.0, ai = 0.0;
        for (int j = 0; j < NV * VL; ++j) {
            const double live = ln.live[j];
            ar += pr[j] * ln.cur[j] * live;
            ai += pi_[j] * ln.cur[j] * live;
        }
        reinterpret_cast<double*>(alm_m)[2 * l] += ar * lscale[l];
        reinterpret_cast<double*>(alm_m)[2 * l + 1] += ai * lscale[l];
    }
    // Phase A: some rings still scaled.
    const v4d* vlive = reinterpret_cast<const v4d*>(ln.live);
    l = ln.run_scaled(l, lmax, m, c1, c3, [&](int64_t ll, const v4d* cur) {
        const v4d* wr = reinterpret_cast<const v4d*>(((ll - m) & 1) ? qr : pr);
        const v4d* wi = reinterpret_cast<const v4d*>(((ll - m) & 1) ? qi : pi_);
        v4d ar = splat(0.0), ai = splat(0.0);
        for (int k = 0; k < NV; ++k) {
            ar += wr[k] * cur[k] * vlive[k];
            ai += wi[k] * cur[k] * vlive[k];
        }
        reinterpret_cast<double*>(alm_m)[2 * ll] += hsum(ar) * lscale[ll];
        reinterpret_cast<double*>(alm_m)[2 * ll + 1] += hsum(ai) * lscale[ll];
    });
    if (l >= lmax) {
        return;
    }
    // Phase B: all rings normal; register-resident state.
    v4d ct[NV], cur[NV], prev[NV];
    for (int k = 0; k < NV; ++k) {
        for (int t = 0; t < VL; ++t) {
            ct[k][t] = ln.ct[VL * k + t];
            cur[k][t] = ln.cur[VL * k + t];
            prev[k][t] = ln.prev[VL * k + t];
        }
    }
    const v4d* Pr = reinterpret_cast<const v4d*>(pr);
    const v4d* Pi = reinterpret_cast<const v4d*>(pi_);
    const v4d* Qr = reinterpret_cast<const v4d*>(qr);
    const v4d* Qi = reinterpret_cast<const v4d*>(qi);
    double* out = reinterpret_cast<double*>(alm_m);
    for (++l; l <= lmax; ++l) {
        const v4d C1 = splat(c1[l - m]);
        const v4d C3 = splat(c3[l - m]);
        const bool odd = ((l - m) & 1) != 0;
        const v4d* Wr = odd ? Qr : Pr;
        const v4d* Wi = odd ? Qi : Pi;
        v4d ar = splat(0.0), ai = splat(0.0);
        for (int k = 0; k < NV; ++k) {
            const v4d d = C1 * (ct[k] * cur[k]) + C3 * prev[k];
            prev[k] = cur[k];
            cur[k] = d;
            ar += Wr[k] * d;
            ai += Wi[k] * d;
        }
        out[2 * l] += hsum(ar) * lscale[l];
        out[2 * l + 1] += hsum(ai) * lscale[l];
    }
}

// f_N[i], f_S[i] += alm contributions for one ring block: alm2map direction.
inline void synth_block(
    const RingGeometry& g,
    const SeedState& seeds,
    const double* dm_ptr,
    int64_t m,
    int64_t lmax,
    int64_t i0,
    int64_t nb,
    const double* c1,
    const double* c3,
    const double* alm_re_im,  // interleaved alm * lscale, indexed by l (already offset)
    double* er,
    double* ei,
    double* orr,
    double* oi  // each kSyn*4 long: even/odd parity accumulators
) {
    constexpr int NV = kSyn;
    Lanes<NV> ln;
    ln.load(g, seeds, dm_ptr, i0, nb);
    for (int j = 0; j < NV * VL; ++j) {
        er[j] = ei[j] = orr[j] = oi[j] = 0.0;
    }
    int64_t l = m;
    {
        const double ar = alm_re_im[2 * l], ai = alm_re_im[2 * l + 1];
        for (int j = 0; j < NV * VL; ++j) {
            const double live = ln.live[j];
            er[j] += ar * ln.cur[j] * live;
            ei[j] += ai * ln.cur[j] * live;
        }
    }
    const v4d* vlive = reinterpret_cast<const v4d*>(ln.live);
    l = ln.run_scaled(l, lmax, m, c1, c3, [&](int64_t ll, const v4d* cur) {
        const v4d AR = splat(alm_re_im[2 * ll]), AI = splat(alm_re_im[2 * ll + 1]);
        const bool odd = ((ll - m) & 1) != 0;
        v4d* xr = reinterpret_cast<v4d*>(odd ? orr : er);
        v4d* xi = reinterpret_cast<v4d*>(odd ? oi : ei);
        for (int k = 0; k < NV; ++k) {
            xr[k] += AR * cur[k] * vlive[k];
            xi[k] += AI * cur[k] * vlive[k];
        }
    });
    if (l >= lmax) {
        return;
    }
    v4d ct[NV], cur[NV], prev[NV], ER[NV], EI[NV], OR[NV], OI[NV];
    for (int k = 0; k < NV; ++k) {
        for (int t = 0; t < VL; ++t) {
            ct[k][t] = ln.ct[VL * k + t];
            cur[k][t] = ln.cur[VL * k + t];
            prev[k][t] = ln.prev[VL * k + t];
            ER[k][t] = er[VL * k + t];
            EI[k][t] = ei[VL * k + t];
            OR[k][t] = orr[VL * k + t];
            OI[k][t] = oi[VL * k + t];
        }
    }
    for (++l; l <= lmax; ++l) {
        const v4d C1 = splat(c1[l - m]);
        const v4d C3 = splat(c3[l - m]);
        const v4d AR = splat(alm_re_im[2 * l]);
        const v4d AI = splat(alm_re_im[2 * l + 1]);
        if (((l - m) & 1) == 0) {
            for (int k = 0; k < NV; ++k) {
                const v4d d = C1 * ct[k] * cur[k] + C3 * prev[k];
                prev[k] = cur[k];
                cur[k] = d;
                ER[k] += AR * d;
                EI[k] += AI * d;
            }
        } else {
            for (int k = 0; k < NV; ++k) {
                const v4d d = C1 * ct[k] * cur[k] + C3 * prev[k];
                prev[k] = cur[k];
                cur[k] = d;
                OR[k] += AR * d;
                OI[k] += AI * d;
            }
        }
    }
    for (int k = 0; k < NV; ++k) {
        for (int t = 0; t < VL; ++t) {
            er[VL * k + t] = ER[k][t];
            ei[VL * k + t] = EI[k][t];
            orr[VL * k + t] = OR[k][t];
            oi[VL * k + t] = OI[k][t];
        }
    }
}

// s_modes (mmax+1, nrings) -> alm (nalm); weights scalar or per ring.
inline void integrate(
    const c10::complex<double>* s_modes,
    const double* theta,
    const double* weight,
    bool w_scalar,
    int64_t nrings,
    int64_t lmax,
    int64_t mmax,
    c10::complex<double>* alm_out
) {
    const RingGeometry g(theta, nrings, lmax);
    const std::vector<double> dcoef = sectoral_coeffs(mmax);
    const auto lfac = get_l_factors(lmax);
    parallel_m_ranges(mmax, [&](int64_t m0, int64_t stride) {
        if (m0 > mmax) {
            return;
        }
        SeedState seeds;
        seeds.init(g, m0, stride);
        const size_t pad = static_cast<size_t>(g.n_eff + kMaxLanes);
        std::vector<double> wpr(pad, 0.0), wpi(pad, 0.0), wqr(pad, 0.0), wqi(pad, 0.0);
        std::vector<double> c1(static_cast<size_t>(lmax + 2)), c3(static_cast<size_t>(lmax + 2));
        std::vector<double> lscale(static_cast<size_t>(lmax + 1));
        for (int64_t m = m0; m <= mmax; m += stride) {
            seeds.advance_to(m);
            const int64_t first = g.first_active(m);
            if (first >= g.n_eff) {
                continue;
            }
            const c10::complex<double>* sm = s_modes + m * nrings;
            for (int64_t i = first; i < g.n_eff; ++i) {
                const double wn = w_scalar ? weight[0] : weight[i];
                const double nr = sm[i].real() * wn, ni = sm[i].imag() * wn;
                if (i < g.n_pairs) {
                    const int64_t is = nrings - 1 - i;
                    const double ws = w_scalar ? weight[0] : weight[is];
                    const double sr = sm[is].real() * ws, si = sm[is].imag() * ws;
                    wpr[i] = nr + sr;
                    wpi[i] = ni + si;
                    wqr[i] = nr - sr;
                    wqi[i] = ni - si;
                } else {
                    wpr[i] = wqr[i] = nr;
                    wpi[i] = wqi[i] = ni;
                }
            }
            fill_coeffs(m, lmax, c1.data(), c3.data());
            const double pre_phase = (m % 2 != 0) ? -1.0 : 1.0;
            for (int64_t l = m; l <= lmax; ++l) {
                lscale[static_cast<size_t>(l)] = pre_phase * (*lfac)[static_cast<size_t>(l)];
            }
            c10::complex<double>* alm_m = alm_out + (m * (lmax + 1) - (m * (m - 1)) / 2) - m;
            for (int64_t i0 = first; i0 < g.n_eff; i0 += VL * kInt) {
                const int64_t nb = std::min<int64_t>(VL * kInt, g.n_eff - i0);
                integrate_block(
                    g, seeds, &dcoef[static_cast<size_t>(m)], m, lmax, i0, nb, c1.data(), c3.data(),
                    wpr.data(), wpi.data(), wqr.data(), wqi.data(), lscale.data(), alm_m
                );
            }
        }
    });
}

// alm (nalm) -> s_modes (mmax+1, nrings) complex.
inline void synthesize(
    const c10::complex<double>* alm,
    const double* theta,
    int64_t nrings,
    int64_t lmax,
    int64_t mmax,
    c10::complex<double>* s_modes
) {
    const RingGeometry g(theta, nrings, lmax);
    const std::vector<double> dcoef = sectoral_coeffs(mmax);
    const auto lfac = get_l_factors(lmax);
    parallel_m_ranges(mmax, [&](int64_t m0, int64_t stride) {
        if (m0 > mmax) {
            return;
        }
        SeedState seeds;
        seeds.init(g, m0, stride);
        std::vector<double> c1(static_cast<size_t>(lmax + 2)), c3(static_cast<size_t>(lmax + 2));
        std::vector<double> alm_s(static_cast<size_t>(2 * (lmax + 1)), 0.0);
        alignas(16) double er[kSyn * 4], ei[kSyn * 4], orr[kSyn * 4], oi[kSyn * 4];
        for (int64_t m = m0; m <= mmax; m += stride) {
            seeds.advance_to(m);
            const int64_t first = g.first_active(m);
            if (first >= g.n_eff) {
                continue;
            }
            fill_coeffs(m, lmax, c1.data(), c3.data());
            const double pre_phase = (m % 2 != 0) ? -1.0 : 1.0;
            const c10::complex<double>* am = alm + (m * (lmax + 1) - (m * (m - 1)) / 2) - m;
            for (int64_t l = m; l <= lmax; ++l) {
                const double sc = pre_phase * (*lfac)[static_cast<size_t>(l)];
                alm_s[static_cast<size_t>(2 * l)] = am[l].real() * sc;
                alm_s[static_cast<size_t>(2 * l + 1)] = am[l].imag() * sc;
            }
            c10::complex<double>* out = s_modes + m * nrings;
            for (int64_t i0 = first; i0 < g.n_eff; i0 += VL * kSyn) {
                const int64_t nb = std::min<int64_t>(VL * kSyn, g.n_eff - i0);
                synth_block(
                    g, seeds, &dcoef[static_cast<size_t>(m)], m, lmax, i0, nb, c1.data(), c3.data(),
                    alm_s.data(), er, ei, orr, oi
                );
                for (int64_t j = 0; j < nb; ++j) {
                    const int64_t i = i0 + j;
                    out[i] = c10::complex<double>(er[j] + orr[j], ei[j] + oi[j]);
                    if (i < g.n_pairs) {
                        out[nrings - 1 - i] = c10::complex<double>(er[j] - orr[j], ei[j] - oi[j]);
                    }
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Exponent-scaled seeding for the Ishioka skip-2 spin kernels.
// p_k (k = 0, 1, ...) obeys p_{k+1} = (a_k mu^2 + b_k) p_k + p_{k-1}; the seed
// p_0 = coeff * sin^m underflows for large m. Rings whose seed is below 2^-128
// are advanced here with the same recurrence on a rescaled mantissa until their
// true value is representable; the state at that k is injected into the SIMD
// k-loop (p and pold stay 0, so they contribute nothing before that).
// ---------------------------------------------------------------------------

struct SpinSeedEvent {
    int64_t k;
    int i;
    double p;
    double pold;
};

// Spin-aware forbidden-region cut: sharp_mlim_estimate with the WKB margin
// max(50, 24 (lmax sin)^(1/3)) in place of libsharp's max(100, 0.01 lmax).
inline int64_t spin_mlim_estimate(int64_t lmax, int64_t spin_abs, double sth, double cth) {
    const double ofs = std::max(50.0, 24.0 * std::cbrt(static_cast<double>(lmax) * sth));
    const double b = -2.0 * static_cast<double>(spin_abs) * std::abs(cth);
    const double t1 = static_cast<double>(lmax) * sth + ofs;
    const double c = static_cast<double>(spin_abs * spin_abs) - t1 * t1;
    const double discr = b * b - 4.0 * c;
    if (discr <= 0.0) {
        return lmax;
    }
    double res = (-b + std::sqrt(discr)) * 0.5;
    res = std::min(std::max(res, 0.0), static_cast<double>(lmax));
    return static_cast<int64_t>(res + 0.5);
}

inline void spin_seed_rings(
    const RingGeometry& g,
    const SeedState& seeds,
    double coeff,
    const IshiokaMCoeffs& pc,
    const double* mu2,
    const uint8_t* active,
    double* p,
    double* pold,
    std::vector<SpinSeedEvent>& events
) {
    events.clear();
    for (int64_t i = 0; i < g.n_eff; ++i) {
        if (!active[i]) {
            continue;
        }
        double mant;
        double ex;
        ring_seed(coeff, seeds.x[static_cast<size_t>(i)], seeds.e[static_cast<size_t>(i)], mant, ex);
        if (ex == 0.0) {
            p[i] = mant;
            continue;
        }
        double pv = mant;
        double po = 0.0;
        int64_t k = 0;
        bool found = false;
        while (k < pc.kmax) {
            const double pn = (pc.a[static_cast<size_t>(k)] * mu2[i] + pc.b[static_cast<size_t>(k)]) * pv + po;
            po = pv;
            pv = pn;
            ++k;
            if (std::fabs(pv) > kBig) {
                pv *= kDn;
                po *= kDn;
                ex += 1.0;
                if (ex >= 0.0) {
                    found = true;
                    break;
                }
            }
        }
        if (found) {
            events.push_back(SpinSeedEvent{k, static_cast<int>(i), pv, po});
        }
    }
    std::sort(events.begin(), events.end(), [](const SpinSeedEvent& a, const SpinSeedEvent& b) {
        return a.k < b.k;
    });
}

}  // namespace sht_scalar
