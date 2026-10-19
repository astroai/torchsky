# FOSSIL science cases and open-discovery directions

Primordial \(\mu\) distortions, dark-sector physics, and underexplored analysis
directions for a TorchSky contribution.

**Status:** working document
**Date:** July 2026

> Converted from the user-authored Word note `fossil_science.docx`. Wording is
> the original July 2026 text; Markdown structure, tables, math, and reference
> links were cleaned up for the repository.

## Purpose

> A technically grounded starting point for a possible contribution combining
> CMB spectral-distortion science, open-world discovery analysis, full-sky
> modelling, signal processing, and TorchSky development.

> **Core recommendation.** Use the standard primordial \(\mu\) distortion as
> the validation and precision target; use the dark sector as the discovery
> extension. The distinctive contribution is a discovery-preserving analysis
> that learns foreground/noise structure from data but keeps unexplained,
> cross-view-stable spectral structure visible.

**Scope.** This note describes two science cases that match an ML, large-data,
statistics and signal-processing profile. It also identifies three
underexplored fundamental-physics directions. "Underexplored" means that
adjacent theory or component-separation work exists, but I did not find a
mature end-to-end programme combining open-world residual discovery, full-sky
spatial information, and raw FTS redundancies. It is not a claim that no
related idea has ever appeared.

## 1. Mission context

**FOSSIL** (FTS fOr CMB Spectral diStortIon expLoration) is an ESA M8 proposal
for a full-sky absolute spectrometric survey. Public June 2026 material
describes a differential Fourier-transform spectrometer comparing the sky with
an internal blackbody reference, with approximately 130 bands over 50-2000 GHz,
about 15 GHz spectral resolution and roughly \(1.6^\circ\) angular resolution.
The Phase-2 proposal was submitted in April 2026; selection milestones in
public slides remain prospective.

| Public baseline item | Description |
| --- | --- |
| Measurement | Differential FTS: sky intensity relative to an internal blackbody reference |
| Coverage | Full sky from L2; public baseline 50-2000 GHz, goal toward 30 GHz |
| Spectral sampling | Approximately 15 GHz resolution, about 130 bands |
| Angular resolution | Approximately \(1.6^\circ\) equivalent Gaussian beam |
| Science emphasis | Primordial perturbations/inflation, dark sector, \(y\) and relativistic-\(y\), foreground/CIB/line science |
| Public \(\mu\) forecast | \(\sigma(\mu) \approx 1.5\times10^{-8}\) from the monopole; about \(5\times10^{-9}\) when spatial information is used |

The central analysis challenge is that astrophysical foreground emission is
much brighter than the target distortions and has spatially varying spectral
dependence. Current FOSSIL/BISOU work publicly emphasizes parametric sky and
instrument simulations, foreground complexity, instrument optimisation and an
end-to-end interferogram pipeline. A data-driven, open-discovery analysis would
be complementary rather than competitive.

## 2. Science case A: primordial \(\mu\) distortion and small-scale primordial power

> **Why this should be the first case.** The target is low dimensional and
> theoretically clean enough to validate signal preservation, while the
> foreground and instrumental nuisance remain high dimensional. It is the best
> environment for proving that a data-driven method does not absorb the science
> signal.

### 2.1 Physical content

Acoustic perturbations on scales far smaller than those resolved by ordinary
CMB anisotropy maps are damped in the early photon-baryon plasma. The
dissipated perturbation energy heats the radiation field. During the \(\mu\)
era, Compton scattering efficiently redistributes photon energies but
photon-production processes cannot fully restore a blackbody, leaving an
approximately chemical-potential distortion.

- The standard slow-roll/near-scale-invariant prediction is of order
  \(\mu \approx 2\times10^{-8}\).
- The signal probes primordial power over roughly \(k \approx 50\) to
  \(10^{4}\,\mathrm{Mpc}^{-1}\), depending on the observable and window
  convention.
- This accesses physical scales unavailable to ordinary CMB temperature and
  polarization anisotropies.
- Departures can indicate features or running in the primordial spectrum,
  isocurvature, enhanced small-scale power, primordial-black-hole scenarios, or
  non-standard damping physics.

### 2.2 Fundamental-physics questions

- Does the primordial spectrum remain close to a smooth nearly scale-invariant
  form over many additional decades in scale?
- Are there features or enhanced/suppressed power associated with inflationary
  dynamics?
- Can small-scale power discriminate primordial explanations of compact objects
  from late-time dark-matter effects?
- Do \(\mu\) anisotropies and their correlations reveal primordial
  non-Gaussianity or statistical anisotropy?

### 2.3 Contribution matched to your profile

1. Build a full-sky, spatial-spectral nuisance model using TorchSky fields,
   HEALPix and multi-survey context.
2. Estimate the \(\mu\) target with cross-fitting so that nuisance learning and
   target evaluation occur on separated sky/detector/time blocks.
3. Measure target attenuation with positive/negative injections at the map and
   interferogram levels.
4. Exploit spatial information without requiring a fixed foreground
   distribution: learn predictable sky structure and retain an unexplained
   residual map.
5. Cross-correlate a recovered \(\mu\) map and open residual modes with CMB
   temperature/polarization and large-scale-structure tracers.

### 2.4 A realistic first paper

**Working title:** "Discovery-preserving recovery of primordial CMB spectral
distortions with cross-fitted spatial-spectral nuisance models."

- Baseline: parametric or moment-expanded foreground model.
- Alternatives: PCA/NMF, learned spectral basis, continuous field and hybrid
  protected-target model.
- Data: FOSSIL-like simulations plus real multi-frequency sky backgrounds with
  injected \(\mu\) signals.
- Primary metrics: \(\mu\) attenuation, bias, split stability,
  foreground-prediction error, null failures and unexplained residual power.
- Deliverable: a mission-neutral benchmark and TorchSky primitives, not only
  one estimator.

## 3. Science case B: dark-sector and exotic energy/photon injection

> **Why this is the discovery extension.** Spectral distortions probe energy
> and photon exchange over redshifts and particle masses that can be difficult
> to access elsewhere. The space of possible spectra is broader than a single
> \(\mu\) template, making closed-model analysis more likely to miss novelty.

### 3.1 Physical content

Dark matter decay, annihilation, scattering with baryons or photons,
dark-photon conversion, extra radiation and other hidden-sector processes can
alter the CMB spectrum. The final shape depends on when energy or photons are
injected, their spectrum, the thermalization regime and possible changes to
expansion or interaction rates.

- Recent examples include light-dark-matter energy injection, neutrino-linked
  extra radiation, and resonant ordinary/dark-photon conversion in atomic dark
  sectors.
- Different processes can generate similar leading \(\mu\)/\(y\) amplitudes but
  differ in residual spectral structure or in correlations with anisotropy
  observables.
- FOSSIL public science material explicitly lists dark-matter decay,
  annihilation, scattering and photon-dark-photon conversion among primary
  goals.

### 3.2 Fundamental-physics questions

- Does the dark sector exchange energy or photon number with the Standard Model
  during the spectral-distortion era?
- Are there light mediators or ultra-small kinetic mixings inaccessible to
  collider and direct-detection experiments?
- Can spectral shapes distinguish energy injection, photon conversion and
  modified thermalization?
- Are there unexplained spectral components inconsistent with known
  astrophysical and instrumental modes but not represented by existing particle
  templates?

### 3.3 Contribution matched to your profile

6. Treat theory templates as optional probes, not as an exhaustive partition of
   the data.
7. Learn foreground and correlated-noise structure from repeated sky, detector,
   scan and reference views.
8. Build a discovery residual after removing only components that are
   demonstrably predictable and stable under held-out evaluation.
9. Search the residual with a hierarchy: known template bank, broad residual
   spectral basis, anomaly/stability tests, then physical interpretation.
10. Use FOSSIL/BISOU simulations to determine sensitivity, but test whether
    conclusions survive simulator and nuisance-model changes.

### 3.4 A realistic second paper

**Working title:** "Open-world searches for exotic CMB spectral distortions
with learned nuisance structure and cross-view stability tests."

- Validate on several published dark-sector spectra without training the
  nuisance model to those spectra.
- Include unknown-shape injections withheld from model development.
- Report detection efficiency as a function of spectral morphology and overlap
  with nuisance modes, not only particle parameters.
- Require consistency across detector, scan, time, sky/reference and
  map/interferogram analyses.

## 4. Three underexplored fundamental-physics directions

### 4.1 A full-sky "open residual atlas" for spectral distortions

**Idea.** Produce a public or collaboration-internal set of unexplained
spectral components after explicit known targets and held-out-predictable
nuisance have been separated. Each component carries full-sky morphology,
spectrum, detector/scan/time stability, response to calibration alternatives,
and correlation with external tracers.

**What is adjacent.** Existing work includes parametric foreground forecasts,
moment expansions, blind/semi-blind component separation, and physics-derived
residual-distortion eigenmodes. These methods are essential, but they normally
compress the residual within a chosen foreground or thermalization basis.

**What is new in the proposed programme.** The unexplained residual is a
first-class product rather than a fit failure. Components are ranked by
cross-view reproducibility and celestial consistency, not solely by likelihood
gain or agreement with a predefined signal bank.

**Fundamental reach.** Unknown photon injection, unanticipated dark-sector
conversion, non-standard radiative processes, or other departures whose spectra
were not included in the proposal model.

### 4.2 Joint monopole-anisotropy residual tomography

**Idea.** Generalize the usual monopole \(\mu\)/\(y\)/residual analysis to a
set of spatial-spectral modes. Infer which residual modes are isotropic, trace
Galactic morphology, correlate with CMB T/E, or correlate with CIB, LIM and
large-scale-structure tracers.

**What is adjacent.** Residual eigenmodes compress energy-release histories in
the mean spectrum, while \(\mu\)-anisotropy correlations have been developed as
probes of primordial non-Gaussianity. FOSSIL forecasts also show that spatial
information can materially improve \(\mu\) sensitivity.

**What is new in the proposed programme.** Use a joint data-derived
spatial-spectral residual basis, then interpret stable modes through
cross-correlations. This could separate a primordial isotropic/large-scale
component from late-time or Galactic structure without assuming that every
residual follows a known emission law.

**Fundamental reach.** Small-scale primordial non-Gaussianity, statistical
anisotropy, spatially modulated energy release, and exotic sectors coupled to
structure or radiation backgrounds.

### 4.3 Falsifying the standard thermalization manifold

**Idea.** Define the broad observable manifold generated by arbitrary energy
and photon-number injection histories evolved through standard Compton
scattering, double Compton emission and bremsstrahlung. Rather than select a
particle model, test whether a robust celestial residual lies inside that
standard-thermalization manifold.

**What is adjacent.** CosmoTherm calculations, energy-release principal
components and recent general parametrizations describe broad families of
distorted spectra. They are usually used to forecast or fit specified histories
and models.

**What is new in the proposed programme.** Use the standard manifold as a
falsifiable boundary, not a complete prior. A component robustly orthogonal to
it would motivate new photon kinetics, missing interactions, calibration
re-examination, or genuinely non-standard thermalization.

**Fundamental reach.** Tests of early-Universe photon kinetics and detailed
thermalization that are more open than a catalogue of decay, annihilation and
conversion models.

> **Best ordering.** Start with Direction 1 using standard \(\mu\) injections.
> Add Direction 2 when full-sky maps and cross-correlation machinery are
> stable. Direction 3 is the most fundamental and should be developed with a
> thermalization theorist after the measurement-space pipeline is trusted.

## 5. Recommended contribution programme

| Phase | Deliverable |
| --- | --- |
| Phase 1 - trustworthy substrate | Full-sky spectral data model, spectral/FTS operators, detector/time/scan splits, injections, open residual products and exact-adjoint benchmarks. |
| Phase 2 - standard \(\mu\) validation | Cross-fitted nuisance/noise learning; demonstrate recovery of \(\mu \approx 2\times10^{-8}\) under foreground and instrument misspecification; quantify attenuation and spatial-information gain. |
| Phase 3 - dark-sector challenge set | Withhold several exotic spectra during method design, then evaluate recovery and anomaly sensitivity across shapes, amplitudes and nuisance overlap. |
| Phase 4 - science white-paper contribution | Frame open discovery as a complementary mission capability: targeted high-precision measurements plus an explicit programme to preserve, validate and publish unexplained spectral structure. |

## 6. Candidate white-paper language

> **Proposed paragraph.** A FOSSIL analysis programme should combine targeted
> inference for predicted distortions with an open-discovery branch. Known
> target spectra and reliable measurement transformations remain explicit,
> while foregrounds, correlated noise and residual systematics may be
> represented by flexible data-driven models. Scientific validation should
> include held-out prediction, signal injections, independent detector/scan/time
> splits and explicit identifiability diagnostics. Stable unexplained spectral
> structure should remain a science product rather than being forced into a
> closed catalogue of nuisance components.

## References

1. B. Maffei, "Measuring spectral distortions with FOSSIL and BISOU,"
   Challenges at the CMB Frontier, June-July 2026.
   [PDF](https://indico.in2p3.fr/event/39689/contributions/172737/attachments/102686/159782/CMB-FRONTIER_Maffei.pdf).
2. Institut d'Astrophysique Spatiale, "BISOU" project page.
   [Link](https://www.ias.universite-paris-saclay.fr/en/content/bisou).
3. J. Chluba et al., "New horizons in cosmology with spectral distortions of
   the cosmic microwave background," *Experimental Astronomy* 51, 1515-1554
   (2021),
   [doi:10.1007/s10686-021-09729-5](https://doi.org/10.1007/s10686-021-09729-5).
4. J. Chluba and R. A. Sunyaev, "The evolution of CMB spectral distortions in
   the early Universe," *MNRAS* 419, 1294-1314 (2012),
   [doi:10.1111/j.1365-2966.2011.19786.x](https://doi.org/10.1111/j.1365-2966.2011.19786.x).
5. J. Chluba and D. Jeong, "Teasing bits of information out of the CMB energy
   spectrum," *MNRAS* 438, 2065-2082 (2014),
   [doi:10.1093/mnras/stt2327](https://doi.org/10.1093/mnras/stt2327).
6. M. H. Abitbol, J. Chluba, J. C. Hill and B. R. Johnson, "Prospects for
   measuring CMB spectral distortions in the presence of foregrounds,"
   *MNRAS* 471, 1126-1140 (2017),
   [doi:10.1093/mnras/stx1653](https://doi.org/10.1093/mnras/stx1653).
7. F. Bianchini and G. Fabbian, "CMB spectral distortions revisited: a new take
   on \(\mu\) distortions and primordial non-Gaussianities from FIRAS data,"
   *Phys. Rev. D* 106, 063527 (2022),
   [doi:10.1103/PhysRevD.106.063527](https://doi.org/10.1103/PhysRevD.106.063527).
8. J. Chluba, E. Dimastrogiovanni, M. A. Amin and M. Kamionkowski, "Evolution
   of CMB spectral distortion anisotropies and tests of primordial
   non-Gaussianity," *MNRAS* 466, 2390-2401 (2017),
   [doi:10.1093/mnras/stw3230](https://doi.org/10.1093/mnras/stw3230).
9. M. Lucca et al., "The synergy between CMB spectral distortions and
   anisotropies," *JCAP* 02, 026 (2020),
   [doi:10.1088/1475-7516/2020/02/026](https://doi.org/10.1088/1475-7516/2020/02/026).
10. X. Coulon, B. Maffei and N. Aghanim, "Towards measurements of CMB spectral
    distortions," *EPJ Web of Conferences* 293, 00012 (2024),
    [doi:10.1051/epjconf/202429300012](https://doi.org/10.1051/epjconf/202429300012).
11. D. K. Adams, J. Barron, B. Cyr and X. Zhang, "CMB spectral distortions from
    resonant conversions in atomic dark sectors," accepted in *Phys. Rev. D*
    (2026),
    [doi:10.1103/gpbc-6kr6](https://doi.org/10.1103/gpbc-6kr6).
12. S.-P. Li and J. Chluba, "Neutrinogenic CMB spectral distortions,"
    *Phys. Rev. D* 113, 043504 (2026),
    [doi:10.1103/gl24-7pqt](https://doi.org/10.1103/gl24-7pqt).
13. H. Liu, W. Qin, G. W. Ridgway and T. R. Slatyer, "Exotic energy injection
    in the early universe II: CMB spectral distortions and constraints on light
    dark matter," [arXiv:2303.07370](https://arxiv.org/abs/2303.07370).
14. G. Barenboim, J. Froustey, C. Pitrou and H. Sanchis, "Parametrizations of
    spectral distortions and applications in cosmology," *Phys. Rev. D* 113,
    063566 (2026),
    [doi:10.1103/dq6j-fzz6](https://doi.org/10.1103/dq6j-fzz6).
15. CNES/IAS doctoral project description, "Preparing for CMB spectral
    distortion measurement with space missions" (2026).
    [Link](https://recrutement.cnes.fr/en/annonce/4148993-26-212-preparing-for-cmb-spectral-distortion-measurement-with-space-missions-91405-orsay).

## Related Torchsky documents

- [FOSSIL/BISOU mission-neutral track](fossil.md): executable implementation
  backlog.
- [FOSSIL primitives plan](fossil_primitives_plan.md): observation-operator
  roadmap.
- [Source unmixing](source_unmixing_problem_formulation.md): shared
  mathematical formulation.
