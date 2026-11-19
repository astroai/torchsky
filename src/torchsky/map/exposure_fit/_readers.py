"""Read-outs of the fitted response (no modelling): the kernel stamp, the throughput curve. Used by ``SkyFit.response``."""

from __future__ import annotations

from typing import NamedTuple

import torch

from ._render import kernel_level_values, smooth_down


class KernelStamp(NamedTuple):
    """A kernel sampled on a regular grid.

    Attributes:
        stamp: ``[n, n]`` float32 tensor, the kernel density per square arcsec; ``n`` is odd and the centre pixel is the kernel centre. Row index is
            along the detector rows (Y) and column index along the detector columns (X) of the detector-aligned frame the kernel is defined in.
        pitch_arcsec: the grid spacing [arcsec].
    """

    stamp: torch.Tensor
    pitch_arcsec: float


def _pair(fit, instrument: str, band: str):
    op, st = fit.operator, fit._structure
    insts, names = list(st["instrument_names"]), list(op.fset.names)
    if instrument not in insts:
        raise ValueError(f"unknown instrument {instrument!r}; the fit has {insts}")
    if band not in names:
        raise ValueError(f"unknown band {band!r}; the fit has {names}")
    ins, f = insts.index(instrument), names.index(band)
    rows = [
        i
        for i in range(op.E)
        if int(op.meta["inst"][i]) == ins and int(op.meta["filt"][i]) == f
    ]
    if not rows:
        raise ValueError(
            f"the fit has no exposure of band {band!r} with instrument {instrument!r}"
        )
    return op, ins, f, rows


def kernel_stamp(
    fit, instrument, band, exposure=None, *, pitch_arcsec=None, half_size_arcsec=None
) -> KernelStamp:
    """The fitted kernel (point-spread function) of an instrument and band; see ``_Response.kernel``."""
    op, ins, f, rows = _pair(fit, instrument, band)
    row = None
    if exposure is not None:
        names = fit.exposure_names
        if exposure not in names:
            raise ValueError(f"unknown exposure {exposure!r}")
        row = names.index(exposure)
        if row not in rows:
            raise ValueError(
                f"exposure {exposure!r} is not an exposure of {instrument!r} in band {band!r}"
            )
    lv = op.levels_ch[int(op.meta["chan"][rows[0]])]
    half = half_size_arcsec
    if half is None:
        half = min(lv.norm_r, lv.r_ext) if lv.norm_r > 0 else lv.r_ext
    pitch = lv.p_r if pitch_arcsec is None else pitch_arcsec
    if not (pitch > 0 and half > 0):
        raise ValueError("pitch_arcsec and half_size_arcsec must be positive")
    nh = max(1, int(round(half / pitch)))
    Q = op.fset.Q
    dev = op.dev
    with torch.no_grad():
        P = {k: v.detach() for k, v in op.prepare().items()}
        fi = torch.as_tensor(f, device=dev)
        h = op.codes.hbar_at(
            op.piv[fi].expand(Q),
            fi.expand(Q),
            torch.as_tensor(ins, device=dev).expand(Q),
        )
        if row is not None and op.tau("k_exp") is not None:
            h = h + op.tau("k_exp") * op.codes.z[row][None]
        eye = torch.eye(2, device=dev)
        dil = P.get("dil")
        _, logZ = kernel_level_values(
            op.net, lv, eye, h, amp_free=(op.amp == "free"), dil=dil
        )
        g = (torch.arange(-nh, nh + 1, device=dev, dtype=torch.float32)) * pitch
        yy, xx = torch.meshgrid(g, g, indexing="ij")
        u = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)
        # kernel-centroid gauge: the finished kernel is moved by its class's offset
        if getattr(op, "kc_on", False):
            u = u - op.kc_off[op.kc_pairs.index((int(f), int(ins)))]
        v = u if dil is None else u / dil
        lg = op.net(op.net.enc(v), h)  # (Q, n*n)
        taper = smooth_down(u.norm(dim=-1), lv.taper * lv.r_ext, lv.r_ext)
        k = taper[None] * torch.exp(lg - logZ[:, None])
        w = op.wq[f]
        stamp = ((w[:, None] * k).sum(0) / w.sum()).view(2 * nh + 1, 2 * nh + 1)
    return KernelStamp(stamp.float().cpu(), float(pitch))


def throughput_curve(fit, instrument, band):
    """Wavelengths [micrometre] and throughput of an instrument and band; see ``_Response.throughput``."""
    op, ins, f, _ = _pair(fit, instrument, band)
    Q = op.fset.Q
    with torch.no_grad():
        P = {k: v.detach() for k, v in op.prepare().items()}
        fi = torch.as_tensor(f, device=op.dev)
        lam = op.lam[f]
        if op.amp == "structured":
            gam = (
                op.gbar_at(
                    lam, fi.expand(Q), torch.as_tensor(ins, device=op.dev).expand(Q)
                )
                - P["m"][f]
            )
        else:
            gam = torch.zeros(Q, device=op.dev)
        T = op.T_ref[f, f * Q : (f + 1) * Q]
        return lam.float().cpu(), (T * torch.exp(gam)).float().cpu()
