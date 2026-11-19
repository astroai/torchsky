"""In-memory exposure container, patches and pixel access (moved from the prototype's data.py).

This record type is private. The prototype's file loader (astropy FITS reading, protocol splits) is not part of this module: core code does not
open files. `torchsky.map.exposure_fit.load_exposures` reads files through torchfits into `FieldObservation`s, and `_observations.py`
converts those into `Exposure` records.

Every exposure keeps a rectangular CUT of its detector: all pixels whose header-WCS position falls in the data region (scoring box +
--data-margin).  Header-WCS positions X, Y [arcsec] in the tangent plane at the box centre (sky.py convention) are precomputed per pixel;
the learned geometry acts on top of them.  Exposures are never resampled or pre-aligned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ...instrument import instrument_description as _description

# telescope diameter [m] by instrument name (from torchsky.instrument); a name that is not a known instrument gets 6.5 m in `build_meta`
INST_DIAM = {n: _description(n).diameter_m for n in ("NIRCAM", "WFC3_IR", "ACS_WFC")}


@dataclass
class Exposure:
    name: str
    filt: int
    inst: int
    chan: int
    det: int
    det_shape: tuple
    cut0: tuple  # (y0, x0) of the cut in detector pixels
    d: torch.Tensor  # (ny, nx)
    sig: torch.Tensor  # pipeline sigma
    valid: torch.Tensor  # bool
    X: torch.Tensor  # header-WCS tangent-plane positions [arcsec]
    Y: torch.Tensor
    R: np.ndarray  # 2x2 rotation (parity) of the mean Jacobian
    split: str = "train"
    group: str = ""
    shift0: tuple = (0.0, 0.0)
    stripes: bool = False
    bkg0: float = 0.0
    eid: int = -1  # GLOBAL exposure id: the row of every per-exposure parameter (shared by all cuts of the exposure)
    patches: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    tile: int = 0  # scoring box (sky patch) this cut belongs to
    c0: tuple = (
        0.0,
        0.0,
    )  # tangent-plane offset of the tile centre from the field reference [arcsec] (X west, Y north; 0 for one tile)
    cid: int = -1  # index of this cut in the list of cuts (== eid with one tile)
    host: dict = None  # multi-tile: CPU copies of the pixel tensors; the attributes point to them unless the tile is resident
    # Added in the move (not in the prototype): the filter curve and the names behind the integer indices, so that a list of exposures is
    # self-contained for fit_sky. None of them enters the arithmetic; the names only label logs and reports.
    bandpass: object = None  # _bandpass.Filter of this exposure's filter (every exposure with the same `filt` must carry the same curve)
    inst_name: str = ""
    det_name: str = ""
    chan_name: str = ""

    DATA = ("d", "sig", "valid", "X", "Y")

    @property
    def shape(self):
        return tuple((self.d if self.d is not None else self.host["d"]).shape)

    def nbytes(self):
        src = (
            self.host
            if self.host is not None
            else {k: getattr(self, k) for k in self.DATA}
        )
        return int(
            sum(v.numel() * v.element_size() for v in src.values() if v is not None)
        )

    def hostv(self, k):
        """The tensor k without touching the device (host copy when swapping is on)."""
        return self.host[k] if self.host is not None else getattr(self, k)

    def to_dev(self, dev):
        """Make the pixel tensors resident on dev (a fresh copy; the host copy is kept). No-op without host storage."""
        if self.host is None:
            return
        for k in self.DATA:
            v = self.host.get(k)
            setattr(
                self, k, None if v is None else v.to(dev, non_blocking=True, copy=True)
            )

    def release(self):
        """Drop the device copies (the attributes point back to the host tensors)."""
        if self.host is None:
            return
        for k in self.DATA:
            setattr(self, k, self.host.get(k))

    def pix(self, sl, only_valid=False):
        y0, y1, x0, x1 = sl
        dev = self.d.device
        rows = torch.arange(y0, y1, device=dev) + self.cut0[0]
        cols = torch.arange(x0, x1, device=dev) + self.cut0[1]
        RR, CC = torch.meshgrid(rows, cols, indexing="ij")
        ny, nx = self.det_shape
        out = {
            "X": self.X[y0:y1, x0:x1].reshape(-1),
            "Y": self.Y[y0:y1, x0:x1].reshape(-1),
            "row": RR.reshape(-1),
            "col": CC.reshape(-1),
            "xi_px": torch.stack([CC.reshape(-1).float(), RR.reshape(-1).float()], -1),
            "d": self.d[y0:y1, x0:x1].reshape(-1),
            "sig": self.sig[y0:y1, x0:x1].reshape(-1),
            "valid": self.valid[y0:y1, x0:x1].reshape(-1),
        }
        out["xi"] = torch.stack(
            [
                out["xi_px"][:, 0] / (nx - 1) * 2 - 1,
                out["xi_px"][:, 1] / (ny - 1) * 2 - 1,
            ],
            -1,
        )
        out["bounds"] = (
            x0 + self.cut0[1],
            x1 + self.cut0[1],
            y0 + self.cut0[0],
            y1 + self.cut0[0],
        )
        out["c0"] = self.c0
        if only_valid:
            v = out["valid"]
            out = {
                k: (x[v] if torch.is_tensor(x) and x.shape[0] == v.shape[0] else x)
                for k, x in out.items()
            }
        return out


def make_patches(e: Exposure, patch_px, min_valid=0.15):
    ny, nx = e.shape
    out = []
    for y0 in range(0, ny, patch_px):
        for x0 in range(0, nx, patch_px):
            sl = (y0, min(ny, y0 + patch_px), x0, min(nx, x0 + patch_px))
            v = e.valid[sl[0] : sl[1], sl[2] : sl[3]]
            if float(v.float().mean()) >= min_valid and int(v.sum()) >= 16:
                out.append(sl)
    e.patches = out
    return out


def rotation_of(J):
    """Polar factor of a 2x2 Jacobian (rotation, with the parity flip when det J < 0)."""
    U, _, Vt = np.linalg.svd(np.asarray(J, np.float64))
    return U @ Vt


def mean_jacobian(X, Y):
    """Least-squares linear map d(X,Y)/d(col,row) of a cut (numpy arrays)."""
    ny, nx = X.shape
    rr, cc = np.mgrid[0:ny, 0:nx]
    s = (slice(None, None, max(1, ny // 64)), slice(None, None, max(1, nx // 64)))
    A = np.stack(
        [np.ones(rr[s].size), cc[s].ravel() - cc.mean(), rr[s].ravel() - rr.mean()], 1
    )
    coef = np.linalg.lstsq(A, np.stack([X[s].ravel(), Y[s].ravel()], 1), rcond=None)[0]
    return coef[1:].T  # J[i, j] = d(X_i)/d(xi_j), xi = (col, row)


def affine_refs(cuts_by_eid):
    """Per exposure: centroid of its valid pixels over all its cuts in the FIELD frame (header X, Y + tile offset c0) [arcsec] (F32)."""
    out = []
    for cs in cuts_by_eid:
        sx, sy, n = 0.0, 0.0, 0
        for c in cs:
            v = c.hostv("valid")
            k = int(v.sum())
            sx += float(c.hostv("X")[v].sum()) + k * c.c0[0]
            sy += float(c.hostv("Y")[v].sum()) + k * c.c0[1]
            n += k
        out.append([sx / max(n, 1), sy / max(n, 1)])
    return out


def build_meta(exps, dets, chans, insts, cuts_by_eid=None):
    """Per GLOBAL exposure metadata from one representative cut per exposure (exps[i].eid == i); `cuts_by_eid` (optional) lists every cut of
    each exposure (detector rows / columns used, for the stripe gauge)."""
    cb = cuts_by_eid if cuts_by_eid is not None else [[e] for e in exps]
    # telescope diameter per exposure, by the same rule as the kernel warm start (cli.filt_info_of), so that the blur gauge (model.py) and
    # the warm start use one value (audit 17 Nov 2026: the blur gauge read a key that was never set and always used 6.5 m)
    diam = [
        float(
            e.extra.get(
                "diameter_m",
                INST_DIAM.get(insts[e.inst] if e.inst < len(insts) else "", 6.5),
            )
        )
        for e in exps
    ]
    return {
        "diameter_m_e": diam,
        "filt": [e.filt for e in exps],
        "inst": [e.inst for e in exps],
        "chan": [e.chan for e in exps],
        "det": [e.det for e in exps],
        "train": [e.split == "train" for e in exps],
        "R": np.stack([e.R for e in exps]),
        "sigbar": [float(e.sig[e.valid].median()) for e in exps],
        "det_shape": [list(e.det_shape) for e in exps],
        "shift0": [list(e.shift0) for e in exps],
        "stripes": [e.stripes for e in exps],
        "bkg0": [e.bkg0 for e in exps],
        "n_det": len(dets),
        "n_inst": len(insts),
        "n_chan": len(chans),
        "det_names": list(dets),
        "cut": [
            (int(e.cut0[0]), int(e.cut0[1]), int(e.shape[0]), int(e.shape[1]))
            for e in exps
        ],
        "cuts": [
            [
                (int(c.cut0[0]), int(c.cut0[1]), int(c.shape[0]), int(c.shape[1]))
                for c in cs
            ]
            for cs in cb
        ],
        "chan_pix": [
            float(
                np.median(
                    [
                        abs(np.linalg.det(np.asarray(e.extra["J"]))) ** 0.5
                        for e in exps
                        if e.chan == c
                    ]
                )
            )
            for c in range(len(chans))
        ],
    }


# ----------------------------------------------------------------------------- sky warm start
def untrained_filters(exps, nf):
    """Sorted filter indices (of nf) with no training exposure (split 'train'); [] when every filter has training data."""
    have = {e.filt for e in exps if e.split == "train"}
    return [f for f in range(nf) if f not in have]


def splat_coadd(exps, nf, N, pitch, dev, extra_shift=None):
    """Inverse-variance bilinear splat of (d - bkg0) of the given exposures at their header positions (+ shift0) onto the sky nodes,
    per filter: (nf, N, N) and the weight maps. Used only for the warm start of the sky and the bright-patch strata."""
    num = torch.zeros(nf, N * N, device=dev)
    den = torch.zeros(nf, N * N, device=dev)
    x0 = -(N - 1) / 2 * pitch
    for e in exps:
        v = e.valid
        es = (0.0, 0.0) if extra_shift is None else extra_shift.get(e.eid, (0.0, 0.0))
        X = (e.X[v] + e.shift0[0] + es[0] - x0) / pitch
        Y = (e.Y[v] + e.shift0[1] + es[1] - x0) / pitch
        w = 1.0 / e.sig[v] ** 2
        val = (e.d[v] - e.bkg0) * w
        ix, iy = X.floor().long(), Y.floor().long()
        fx, fy = X - ix, Y - iy
        for dy in (0, 1):
            for dx in (0, 1):
                jx, jy = ix + dx, iy + dy
                ww = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
                ok = (jx >= 0) & (jx < N) & (jy >= 0) & (jy < N)
                idx = (jy * N + jx)[ok]
                num[e.filt].index_add_(0, idx, (val * ww)[ok])
                den[e.filt].index_add_(0, idx, (w * ww)[ok])
    img = torch.where(den > 0, num / den.clamp(min=1e-30), torch.zeros_like(num))
    return img.view(nf, N, N), den.view(nf, N, N)
