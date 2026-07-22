# Minimal tensor-native ICRS phase-space propagation

Status: review basis. The completed implementation ported the ERFA/SOFA
changing-light-time and relativistic path in the same iteration, rather than
stopping at the geometric first step proposed below.

## Recommendation

Add one new ICRS-only primitive instead of widening the existing angular-only
`propagate_proper_motion`:

```python
@dataclass(frozen=True)
class ICRSPhaseSpace:
    ra_deg: Tensor
    dec_deg: Tensor
    distance_pc: Tensor
    pm_ra_cosdec_mas_yr: Tensor
    pm_dec_mas_yr: Tensor
    radial_velocity_km_s: Tensor


def propagate_icrs_space_motion(
    ra_deg: Any,
    dec_deg: Any,
    pm_ra_cosdec_mas_yr: Any,
    pm_dec_mas_yr: Any,
    radial_velocity_km_s: Any,
    *,
    distance_pc: Any | None = None,
    parallax_mas: Any | None = None,
    source_epoch_jyear: Any,
    target_epoch_jyear: Any,
) -> ICRSPhaseSpace: ...
```

Exactly one of `distance_pc` and `parallax_mas` must be supplied. Requiring the
radial velocity, even when the caller deliberately passes zero, keeps the 6D
scientific assumption visible. The existing `propagate_proper_motion` remains
the distance-free API and keeps its current two-tensor return contract.

All inputs broadcast. All result fields have the broadcast shape, remain on the
input device, and are floating tensors. Longitude is wrapped to `[0, 360)`.
The implementation should use float64 on CPU/CUDA, matching the current frame
code. MPS does not support float64 and should fail clearly until an explicit,
accuracy-tested float32 contract exists.

Return the full six-coordinate state. Distance, radial velocity, and both
proper-motion components change even under constant Cartesian velocity because
the line of sight and tangent basis change. Returning only RA/Dec would discard
the state needed for a second propagation and for covariance/Jacobian work.
Parallax is deliberately not a seventh, redundant output:
`parallax_mas = 1000 / distance_pc`.

## Units and exact conversion constants

Let `alpha`, `delta` be RA and Dec in radians, `rho` the distance in parsecs,
`mu_a` the conventional `pm_ra_cosdec`, and `mu_d` the Dec proper motion.

Use:

```text
RAD_PER_MAS       = pi / (180 * 3600 * 1000)
SECONDS_PER_JYEAR = 365.25 * 86400 = 31_557_600
AU_KM             = 149_597_870.7                 (IAU exact)
PC_KM             = AU_KM * 648_000 / pi
KM_S_TO_PC_YR     = SECONDS_PER_JYEAR / PC_KM
distance_pc       = 1000 / parallax_mas
```

Equivalently, the familiar transverse-velocity constant is
`4.7404704635 km/s` per `(arcsec/year * pc)`.

Positive radial velocity means receding. Epoch arguments are numeric Julian
years, so `dt = target_epoch_jyear - source_epoch_jyear`; one Julian year is
exactly 365.25 days. No leap-year or calendar conversion belongs in this
primitive. Astropy parity fixtures should use
`Time(..., format="jyear", scale="tdb")`, for which a 100-year numeric interval
is exactly 100 Julian years. Converting TCB endpoints to TDB changes that
interval by about 1.55e-6 years per century and should not be mixed into a
numeric-epoch kernel.

## Exact geometric constant-velocity core

Define the local orthonormal basis:

```text
r  = (cos(delta) cos(alpha), cos(delta) sin(alpha), sin(delta))
ea = (-sin(alpha), cos(alpha), 0)
ed = (-sin(delta) cos(alpha), -sin(delta) sin(alpha), cos(delta))
```

Here `ea` is why the API accepts `pm_ra_cosdec`: no division by
`cos(delta)` is needed, including near a pole.

Convert the rates and form barycentric Cartesian position and velocity:

```text
mu_a_rad_yr = pm_ra_cosdec_mas_yr * RAD_PER_MAS
mu_d_rad_yr = pm_dec_mas_yr       * RAD_PER_MAS
rho_dot     = radial_velocity_km_s * KM_S_TO_PC_YR

p = rho * r
v = rho_dot * r + rho * (mu_a_rad_yr * ea + mu_d_rad_yr * ed)
p2 = p + dt * v
rho2 = norm(p2)
r2 = p2 / rho2
```

Recover RA/Dec with `atan2(y, x)` and `atan2(z, hypot(x, y))`; the latter is
better behaved at the poles than `asin(z)`. At the propagated direction, form
the new `ea2`, `ed2` basis and recover all observable rates:

```text
rv2_km_s = dot(v, r2) / KM_S_TO_PC_YR
mu_a2_mas_yr = dot(v, ea2) / rho2 / RAD_PER_MAS
mu_d2_mas_yr = dot(v, ed2) / rho2 / RAD_PER_MAS
```

This is exact for inertial Euclidean straight-line motion. It automatically
includes classical perspective acceleration and supports negative `dt`.
It is vectorized, O(N), differentiable, and requires no native extension.

## Astropy `SkyCoord.apply_space_motion` semantics

Astropy 8.0 does more than the geometric equations above:

1. It requires exactly one of `new_obstime` or `dt` and requires a velocity
   differential.
2. It converts the coordinate to ICRS and to a spherical differential without
   the `cos(dec)` factor before calling ERFA.
3. It invokes `erfa.pmsafe`, which implements catalog-observable space motion
   through `starpv -> linear PV update -> pvstar`.
4. ERFA assumes constant inertial velocity but applies changing-light-time and
   special-relativistic corrections following Stumpff (1985). Proper motion is
   per Julian year, parallax is arcseconds, and RV is km/s, positive receding.
5. It transforms the propagated ICRS result back into the original frame and
   restores the input differential type. The proposed primitive is explicitly
   ICRS-only and therefore does not reproduce that frame-wrapper behavior.
6. Missing radial velocity is treated as zero. Missing distance invokes
   `pmsafe` with zero parallax; ERFA substitutes an artificial large distance
   and can return artificial distance/RV values. Astropy then hides the
   substituted distance. Torchsky already has a clearer distance-free angular
   primitive, so the new 6D API should not copy this convention.
7. ERFA warns and overrides very small/non-positive parallax, can zero an
   excessive velocity, and can warn if its relativistic iteration does not
   converge. The torch-native API should fail explicitly on invalid physical
   6D inputs rather than silently alter a catalog row.

The geometric core is not exact `apply_space_motion` parity for the highest
precision use cases. Local float64 comparisons gave:

| case | interval | largest RA/Dec difference vs Astropy |
|---|---:|---:|
| 10 pc, 100/50 mas yr-1, +20 km/s | 100 yr | about 0.1 microarcsec |
| Barnard-like, 1.83 pc, 10.4 arcsec yr-1, -110 km/s | 25 yr | about 69 microarcsec |
| deliberately extreme, 1 pc, 11 arcsec yr-1, +1000 km/s | 100 yr | about 0.20 arcsec |

Therefore there are two honest implementation choices:

- A first release may document **geometric ICRS constant-velocity** semantics
  and test Astropy only within physically justified tolerances. It must not be
  advertised as exact `SkyCoord.apply_space_motion` parity.
- For Torchsky/USPM high-precision catalog semantics, port the small
  ERFA/SOFA `starpv` and `pvstar` light-time/relativistic transforms into torch
  around the same linear PV update. That should be the eventual default before
  claiming Astropy parity. Astropy/ERFA remains a test-only oracle.

The recommended path is to land and validate the geometric kernel first, then
add the Stumpff correction before wiring it into precision USPM inference. Do
not add a public `model=` switch until both semantics have real callers.

## Domain and numerical edge cases

- Require finite RA, Dec, proper motions, RV, epochs, and distance/parallax.
- Require `-90 <= dec <= 90`, `distance_pc > 0`, or `parallax_mas > 0`.
  Negative noisy parallaxes require a distance posterior; they are not physical
  6D positions. Zero/missing parallax belongs on the angular-only path.
- Reject states with total derived speed `>= c` rather than reproducing ERFA's
  silent velocity override. A lower warning threshold is not needed in the
  geometric kernel; high-precision catalog use should select the relativistic
  implementation.
- Reject/mark undefined any propagated state for which `norm(p2)` is numerically
  zero (trajectory through the solar-system barycenter).
- Exact poles are coordinate singularities: the supplied RA defines the local
  tangent basis. The Cartesian propagation is finite, but RA itself and its
  derivative are conventional there.
- RA wrap creates an unavoidable derivative discontinuity at 0/360 degrees.
  Covariance code should operate on Cartesian or tangent-plane residuals.
- Support `dt == 0` and negative intervals. Add an identity regression because
  round-tripping through spherical coordinates can expose signed-zero/wrap
  behavior.
- Do not clamp a nonphysical distance or speed to make a row appear valid.
  For survey ingestion, filter or mask invalid rows before this strict kernel.
- Use `atan2`, vector norms, and float64; do not divide by `cos(dec)`.

## Minimal test gate

One focused test module is enough for the first implementation:

1. Astropy/ERFA parity on randomized, physically realistic positive-distance
   6D states, including returned distance, RV, and both PM components.
2. A Barnard-like regression that prevents loose generic tolerances from hiding
   the light-time/relativistic gap.
3. Zero and negative `dt`, RA wrap, near-pole, zero-PM, zero-RV, scalar/tensor
   broadcasting, and invalid distance/parallax/speed cases.
4. `torch.autograd.gradcheck` for a non-wrapped, non-polar state.
5. CPU/CUDA parity when CUDA is available. Mark MPS unsupported until its
   float32 error envelope is measured.

For the geometric first step, compare its six outputs to the equations above
at tight float64 tolerance and keep the Astropy discrepancy test explicit. For
the ERFA-compatible step, use TDB Julian-year fixtures and tolerances scaled in
microarcseconds, relative distance, mas/year, and km/s rather than one mixed
absolute tolerance.

Covariance propagation is intentionally not part of this primitive. The full
six-output differentiable map exposes the Jacobian needed for a later
`J C J.T` covariance primitive without embedding uncertainty policy here.
