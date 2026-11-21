"""Configuration of :func:`torchsky.map.exposure_fit.fit_sky`.

:class:`SkyFitConfig` holds the settings a user chooses. Every other setting that the moved fitting code still reads is kept in the private
mapping ``_INTERNAL_SETTINGS``, whose values equal the defaults of the research prototype's command-line parser exactly (a test pins every
value), so that the numbers a fit gives do not change. ``_namespace`` merges a configuration with that mapping into the attribute object
the moved code reads (the role the parser's namespace had).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping

# Internal settings of the moved code, outside the public configuration. Defaults = the prototype's parser defaults. Several only matter for
# several sky tiles or for features that are not part of this move (listed because the moved code still reads them). Do not change a value
# without remaking the bit-identity check against the prototype (compare_port.py); `_fit_exposures(..., internal=...)` overrides single
# entries for tests and for that check.
_INTERNAL_SETTINGS: Mapping[str, Any] = MappingProxyType(
    {
        # --- sky and spectrum
        "sky_margin": 2.0,  # sky domain = data box + 2 x margin [arcsec]; used for the sky domain, see fit_sky
        "data_margin": 1.0,  # data region = box + 2 x margin [arcsec]; used for the sky domain, see fit_sky
        "spec_k_rule": "fixed",  # 'fixed': K = n_spectral whatever the spectral domain; 'span': K grows with the log-wavelength span
        "spectral_domain": "union",  # 'union': reference domain 0.70-5.30 um widened to cover every filter; 'fixed': the reference domain
        "item_levels": 3,  # coarse-to-fine seeding lattices of the items
        "item_chunk": 0,  # elements (items x patch nodes x K) per rasterisation chunk; 0 = the module default (4 million)
        "item_lr_pos": 0.05,  # Adam step of item positions [sky pitches]
        "item_lr_width": 0.03,  # Adam step of item log widths
        "item_lr_angle": 0.05,  # Adam step of item angles [rad]
        "item_lr_coef": 0.05,  # Adam step of item log spectral coefficients
        # --- kernel and rendering
        "render_pitch": 0.0,  # render pitch [arcsec]; 0 = per channel (largest sky_pitch/k at or below pixel / render_os_pix)
        "render_os_pix": 6.0,  # oversampling of the detector pixel by the render grid
        "level_factor": 2,  # pitch ratio between nested kernel level grids
        "level_nh": 24,  # half-size (nodes) of each nested kernel level grid
        "level_overlap": 0.5,  # level l hands over to l+1 between a*R_l and R_l
        "region_px": 256,  # detector lattice (pixels) of the region nodes at which the kernel is evaluated
        "n_lambda": 5,  # wavelength quadrature nodes per filter (Q)
        "k_hidden": 64,  # kernel network width
        "k_layers": 3,  # kernel network depth
        "dh": 16,  # dimension of the kernel code
        "enc_r0": 0.0,  # core radius of the log-radius encoding (0 = half the render pitch)
        "render_jitter": 1,  # random sub-pitch phase of the level-0 kernel grid in training (1 = on)
        "enc_nrad": 8,  # radial Fourier frequencies of the kernel encoding
        "enc_mmax": 12,  # largest angular harmonic of the kernel encoding
        "filter_input": "curve",  # 'curve' (filter curve vector) or 'token' (learned filter embedding)
        "k_exp": "learned",  # per-exposure kernel code: 'on' (scale fixed), 'off', or 'learned' (EM)
        "warm_kernel_steps": 400,  # steps of the kernel warm start
        "render_frame": "detector",  # frame of the fine level grids: 'detector' or 'sky'
        "ckpt_kernel": 0,  # activation checkpointing of the kernel field (GPU memory; 1 = on)
        # --- variability scales
        "tau_mode": "learned",  # scales of the per-exposure effects: 'learned' (EM) or 'fixed'
        "tau0_code": 0.3,  # starting scale of the per-exposure kernel code
        "tau0_g": 0.02,  # starting scale of the per-exposure gain
        "tau_start": 0.35,  # step fraction at which the EM update of the scales starts
        "val_every": 4,  # cadence (steps) of the EM evaluations
        "tau_accum": 4,  # EM evaluations accumulated per update of the scales
        "em_damp": 0.5,  # exponent of the damped EM step
        # --- gauges, registration and detector terms
        "amplitude": "structured",  # 'structured' (A = alpha g), 'unit' or 'free'
        "dilation_stop": 0.0,  # end the fit when the common-blur dilation exceeds this (0 = off)
        "blur_floor": 1.0,  # blur gauge: core second moment floor in units of sky_pitch^2 / 3 (0 = off)
        "kernel_centroid_gauge": 0,  # 1 = hold the windowed first moment of every kernel class's mean kernel at zero (exact projection; off until validated on real data)
        "pedestal_gauge": 1,  # per filter, training background constants sum to zero (1 = on)
        "coarse_reg": "off",  # 'off' or 'xcorr' (coarse registration by cross-correlation)
        "coarse_reg_radius": 1.0,  # search radius [arcsec] of the coarse registration
        "coarse_reg_instruments": "",  # instruments to register: comma list of names or indices, 'all', or '' (all but the reference)
        "coarse_reg_min_peak": 0.05,  # refuse an exposure whose correlation peak is below this
        "coarse_reg_min_ratio": 1.5,  # refuse an exposure whose peak is less than this times the second peak
        "coarse_reg_block": 0.6,  # background block [arcsec] of the high-pass
        "coarse_reg_smooth": 0.1,  # Gaussian width [arcsec] smoothing the high-passed images
        "affine_prior": 0.01,  # scale of the prior on the per-exposure affine matrix
        "nuisance_steps": "exposure",  # 'exposure': Adam step of background and stripes scaled by each exposure's own median sigma; 'mean': by the mean sigma
        "stripes": 1,  # row / column stripes for instruments that have them (1 = on)
        "stripe_l2": 0.001,  # L2 penalty of the stripes
        "floor0": 0.01,  # starting fractional noise floor
        # --- loss and optimiser
        "loss": "huber",  # 'huber' or 'gauss'
        "huber_delta": 3.0,  # Huber transition [sigma]
        "inner_sky": 2,  # sky-only inner steps per joint step
        "lr_op": 0.001,  # Adam step of the operator
        "lr_bkg": 0.1,  # Adam step of the backgrounds (x mean sigma)
        "lr_affine": 0.1,  # Adam step of the raw affine terms
        "lr_codes": 0.02,  # Adam step of the per-exposure codes
        "clip": 10.0,  # gradient-norm clip
        "prior_pixels": 0.0,  # random-effect priors are divided by this pixel count (0 = mean valid pixels per training exposure)
        "is_frac": 0.5,  # fraction of patches drawn around bright local maxima
        "is_snr": 20.0,  # signal-to-noise above which a local maximum counts as bright
        "is_refresh": 200,  # steps between refreshes of the bright-patch strata
        # --- held-out prediction
        "held_steps": 150,  # steps of the held-out nuisance fit
        "held_xfit": 1,  # cross-fitted held-out prediction (1 = on)
        "held_step_rows": 1,  # held-out steps move only the exposures in the batch (1 = on)
        "held_reg_steps": 100,  # held-out translation + background pre-fit steps
        "reg_prepass_iters": 2,  # registration pre-pass iterations
        "reg_prepass_steps": 150,  # steps per registration pre-pass iteration
        "pred_margin": 0.3,  # prediction margin [arcsec] inside the sky edge
        "pred_batch": 8,  # patches per prediction batch
        "pix_default": 0.031,  # pixel scale [arcsec] used when an exposure has no Jacobian
        # --- several tiles (not used by fit_sky in part 1, read by the moved tile code)
        "tile_weight_power": 0.5,  # several tiles: visit probability proportional to (training pixels)^power
        "visit_steps": 10,  # several tiles: optimisation steps per tile visit
        "gpu_resident_gb": 4.0,  # several tiles: pixel data kept on the device [GB]
        "sky_swap": 0,  # several tiles: inactive skies kept on the host (1 = on)
        "held_rounds": 4,  # several tiles: rounds over the tiles in the held-out fit
        "held_ref_cuts": 24.0,  # several tiles: reference number of held-out cuts per tile
        # --- logging
        "log_every": 25,  # steps between log lines
    }
)


@dataclass(frozen=True)
class SkyFitConfig:
    """Settings of the item-sky fit. Lengths are in arcseconds, wavelengths in micrometres.

    The defaults are those of the research prototype at real-data scale (60,000 items, a 30 minute budget); tests and quick looks pass small
    values. The configuration is validated when it is built and cannot be changed afterwards (use :func:`dataclasses.replace`).

    Attributes:
        sky_pitch: node spacing of the sky grid on which the items are rasterised [arcsec]; positive.
        item_count: number of Gaussian items that make up the sky; at least 1.
        item_width_floor: smallest item width [arcsec]; 0 means 0.4 x ``sky_pitch``.
        item_width_max: largest item width [arcsec]; 0 means 16 x ``sky_pitch``. Must exceed the floor when both are given.
        spectral_k: number of spectral coefficients per item (radial-basis bumps, uniform in log wavelength); at least 2.
        kernel_extent: outer radius of the response kernel [arcsec]; positive.
        kernel_norm_radius: radius [arcsec] inside which the kernel has unit integral; 0 means the whole support; a value beyond ``kernel_extent`` also means the whole support.
        background_order: polynomial order of the per-exposure background (Legendre terms in the detector coordinates). The moved code
            fits a quadratic (six terms); only 2 is accepted.
        noise_scale_max: the learned noise scale of every detector and filter is kept within [1/value, value]; 0 means no bound, otherwise
            at least 1.
        noise_floor_max: upper bound of the learned fractional noise floor (a fraction of the model, in (0, 1]); 0 means no bound.
        reference_instrument: index (in the sorted instrument names of the observations) of the instrument whose training exposures fix the
            absolute flux scale; -1 means all training exposures; None is accepted for one instrument only.
        steps: maximum number of optimisation steps; the fit stops at ``steps`` or after ``budget_minutes``, whichever comes first.
        budget_minutes: wall-clock budget of the fit [minutes]; positive.
        patches_per_step: image patches per optimisation step; at least 1.
        patch_size: side of an image patch [arcsec]; positive.
        seed: seed of every random stream of the fit; it also seeds torch's global generator at the start of the fit.
        device: ``"auto"`` (CUDA when available, otherwise CPU), ``"cpu"``, or a torch device string such as ``"cuda:0"``.
    """

    sky_pitch: float = 0.04
    item_count: int = 60000
    item_width_floor: float = 0.0
    item_width_max: float = 2.56
    spectral_k: int = 12
    kernel_extent: float = 10.0
    kernel_norm_radius: float = 2.0
    background_order: int = 2
    noise_scale_max: float = 2.0
    noise_floor_max: float = 0.2
    reference_instrument: int | None = None
    steps: int = 1000000
    budget_minutes: float = 30.0
    patches_per_step: int = 5
    patch_size: float = 2.5
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        def finite(
            name: str, lo: float | None = None, *, strict: bool = False
        ) -> float:
            v = getattr(self, name)
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
            ):
                raise ValueError(
                    f"SkyFitConfig.{name} must be a finite number, got {v!r}"
                )
            if lo is not None and (v <= lo if strict else v < lo):
                raise ValueError(
                    f"SkyFitConfig.{name} must be {'>' if strict else '>='} {lo}, got {v!r}"
                )
            return float(v)

        def integer(name: str, lo: int) -> int:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < lo:
                raise ValueError(
                    f"SkyFitConfig.{name} must be an integer >= {lo}, got {v!r}"
                )
            return v

        finite("sky_pitch", 0.0, strict=True)
        integer("item_count", 1)
        floor, wmax = finite("item_width_floor", 0.0), finite("item_width_max", 0.0)
        if floor > 0 and wmax > 0 and not floor < wmax:
            raise ValueError(
                f"SkyFitConfig.item_width_floor ({floor}) must be below item_width_max ({wmax})"
            )
        integer("spectral_k", 2)
        finite("kernel_extent", 0.0, strict=True)
        finite("kernel_norm_radius", 0.0)
        if self.background_order != 2 or isinstance(self.background_order, bool):
            raise ValueError(
                "SkyFitConfig.background_order: only 2 (six Legendre terms) is implemented"
            )
        scale = finite("noise_scale_max", 0.0)
        if scale != 0 and scale < 1:
            raise ValueError(
                f"SkyFitConfig.noise_scale_max must be 0 (no bound) or >= 1, got {scale}"
            )
        nf = finite("noise_floor_max", 0.0)
        if nf > 1:
            raise ValueError(
                f"SkyFitConfig.noise_floor_max must be in [0, 1], got {nf}"
            )
        ref = self.reference_instrument
        if ref is not None and (
            isinstance(ref, bool) or not isinstance(ref, int) or ref < -1
        ):
            raise ValueError(
                f"SkyFitConfig.reference_instrument must be None, -1 or an instrument index >= 0, got {ref!r}"
            )
        integer("steps", 1)
        finite("budget_minutes", 0.0, strict=True)
        integer("patches_per_step", 1)
        finite("patch_size", 0.0, strict=True)
        integer("seed", 0)
        if not isinstance(self.device, str):
            raise ValueError(
                f"SkyFitConfig.device must be a string, got {self.device!r}"
            )
        if self.device != "auto":
            import torch

            try:
                torch.device(self.device)
            except (RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"SkyFitConfig.device {self.device!r} is not a torch device"
                ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (JSON-serialisable)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SkyFitConfig":
        """Inverse of :meth:`to_dict`; an unknown key raises ``TypeError``."""
        return cls(**dict(d))


# public field -> the attribute name the moved code reads
_PUBLIC_TO_ATTR = {
    "sky_pitch": "sky_pitch",
    "item_count": "item_count",
    "item_width_floor": "item_floor",
    "item_width_max": "item_max_width",
    "spectral_k": "spec_k",
    "kernel_extent": "kernel_extent",
    "kernel_norm_radius": "kernel_norm_radius",
    "noise_scale_max": "noise_scale_max",
    "noise_floor_max": "noise_floor_max",
    "reference_instrument": "gauge_ref_inst",
    "steps": "steps",
    "budget_minutes": "budget_min",
    "patches_per_step": "batch_patches",
    "patch_size": "patch_arcsec",
    "seed": "seed",
    "device": "device",
}


def _namespace(
    config: SkyFitConfig, internal: Mapping[str, Any] | None = None
) -> SimpleNamespace:
    """The attribute object the moved fitting code reads (the role of the prototype's argparse namespace).

    A fresh, mutable object on every call: the moved code writes to it (``spec_k`` after the spectral setup, ``gauge_ref_inst`` when it is None
    with one instrument, ``prior_pixels`` when it is 0), so one object must never be shared between two fits. ``internal`` overrides entries of
    ``_INTERNAL_SETTINGS``; an unknown name raises ``KeyError``.
    """
    ns = {attr: getattr(config, pub) for pub, attr in _PUBLIC_TO_ATTR.items()}
    ns.update(_INTERNAL_SETTINGS)
    for k, v in (internal or {}).items():
        if k not in _INTERNAL_SETTINGS:
            raise KeyError(f"unknown internal setting {k!r}")
        ns[k] = v
    return SimpleNamespace(**ns)
