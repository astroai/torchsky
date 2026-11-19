"""One golden step made inside the test at small size: the joint-step loss and its gradient on a fixed batch, in float64.

The loss is the data term plus the prior of the batch's exposure effects (as in the moved fit loop). Checks: the value is reproducible bit for bit, every
parameter the loss reaches gets a finite gradient, and the autograd gradient of a few scalar parameters matches central finite differences. Nothing is
stored in the repository; the bit-for-bit comparison with the research prototype is made outside it (compare_port.py).
"""

import contextlib

import numpy as np
import torch

from _exposure_fit_common import TINY_INTERNAL, tiny_config, tiny_field
from torchsky.map.exposure_fit import _fit as F
from torchsky.map.exposure_fit import _items
from torchsky.map.exposure_fit._config import _namespace
from torchsky.map.exposure_fit._exposures import make_patches
from torchsky.map.exposure_fit._fit import make_sky
from torchsky.map.exposure_fit._response import FieldOperator
from torchsky.map.exposure_fit._skyfit import _filters_of, _meta_of
from torchsky.map.exposure_fit._spectral import spectral_setup


@contextlib.contextmanager
def float64_mode():
    old = (torch.get_default_dtype(), torch.Tensor.float, _items.ITEM_CHUNK)
    torch.set_default_dtype(torch.float64)
    torch.Tensor.float = lambda self: self.to(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(old[0])
        torch.Tensor.float = old[1]
        _items.ITEM_CHUNK = old[2]


def _problem():
    torch.manual_seed(0)
    exps = tiny_field()
    exps = [e for e in exps if e.inst_name == "A"]
    for e in exps:
        for k in ("d", "sig", "X", "Y"):
            setattr(e, k, getattr(e, k).to(torch.float64))
        e.extra = dict(e.extra, clean=None)
    cfg = _namespace(tiny_config(1), TINY_INTERNAL)
    cfg.batch_patches = 6
    fset = _filters_of(exps)
    meta, _ = _meta_of(exps)
    spectral_setup(fset, cfg, log=lambda s: None)
    sky, _ = make_sky(exps, fset, cfg, 1.6, "cpu")
    model = FieldOperator(cfg, fset, meta, sky, torch.device("cpu"))
    model.double()
    for lv in model.levels_ch:  # the renderer's level grids are built as float32
        lv.u = [u.double() for u in lv.u]
        lv.w = [w.double() for w in lv.w]
    g = torch.Generator().manual_seed(1000)
    with torch.no_grad():  # parameters that start at exactly zero get a small seeded value: every gradient is generic
        for n, p in model.named_parameters():
            if not (n.startswith("sky.") or n.startswith("tau.")) and bool(
                (p == 0).all()
            ):
                p.add_(0.05 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    F.set_prior_pixels(cfg, exps, log=lambda s: None)
    for e in exps:
        make_patches(e, 12)
    tr = [i for i, e in enumerate(exps) if e.split == "train"]
    pool = F.PatchPool(exps, tr, cfg)
    pool.refresh(model)
    items = pool.draw(cfg.batch_patches, np.random.default_rng(0))
    return model, exps, cfg, pool, items


def _loss(model, exps, cfg, pool, items):
    P = model.prepare()
    data, _, _ = F.batch_loss(model, exps, items, P, cfg, pool.mean_valid, jitter=False)
    return data + model.priors(F.batch_eids(exps, items, model.dev)) / cfg.prior_pixels


def test_golden_step():
    with float64_mode():
        model, exps, cfg, pool, items = _problem()
        l1 = _loss(model, exps, cfg, pool, items)
        l2 = _loss(model, exps, cfg, pool, items)
        assert torch.isfinite(l1) and torch.equal(l1, l2)
        params = dict(model.named_parameters())
        grads = dict(
            zip(
                params,
                torch.autograd.grad(l1, list(params.values()), allow_unused=True),
            )
        )
        reached = {n: g for n, g in grads.items() if g is not None}
        assert len(reached) > 20 and all(
            torch.isfinite(g).all() for g in reached.values()
        )
        e0 = int(exps[items[0][0]].eid)
        picks = [
            ("u", (e0,)),
            ("bkg", (e0, 1)),
            ("hs", (int(model.e_det[e0]), int(model.e_filt[e0]))),
            ("floor_raw", (0,)),
        ]
        checked = 0
        for n, idx in picks:
            if grads.get(n) is None:
                continue
            p, eps = params[n], 1e-6
            with torch.no_grad():
                p[idx] += eps
            lp = float(_loss(model, exps, cfg, pool, items))
            with torch.no_grad():
                p[idx] -= 2 * eps
            lm = float(_loss(model, exps, cfg, pool, items))
            with torch.no_grad():
                p[idx] += eps
            fd, ad = (lp - lm) / (2 * eps), float(grads[n][idx])
            assert abs(fd - ad) < 1e-5 * max(1.0, abs(fd)), (n, fd, ad)
            checked += 1
        assert checked >= 3
