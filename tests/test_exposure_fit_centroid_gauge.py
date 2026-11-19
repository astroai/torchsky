"""Kernel-centroid gauge (internal setting ``kernel_centroid_gauge``, off by default): fast checks on the tiny synthetic field.

The convention holds the windowed first moment of every kernel class's mean kernel at zero by an exact projection after each optimiser step;
the shift goes into the translations (and the training mean of those changes into the sky), so predictions do not change. The same code
lives in the research prototype; the two are kept bit-identical (``compare_port.py --kernel-centroid-gauge 1``).
"""

import pytest
import torch

from _exposure_fit_common import TINY_INTERNAL, tiny_config, tiny_field
from torchsky.map.exposure_fit import SkyFit
from torchsky.map.exposure_fit._config import _INTERNAL_SETTINGS
from torchsky.map.exposure_fit._skyfit import _fit_exposures, _prepare


def _fit(gauge, steps=6):
    torch.set_num_threads(1)
    exps = tiny_field()
    sf = _fit_exposures(
        exps,
        tiny_config(steps),
        internal={**TINY_INTERNAL, "kernel_centroid_gauge": gauge},
        sky_half_extent=1.8,
    )
    return exps, sf


@pytest.fixture(scope="module")
def on_fit():
    return _fit(1)


@torch.no_grad()
def _predict(op, exps):
    P = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in op.prepare().items()}
    out = []
    for e in exps:
        for sl in e.patches:
            out.append(op.render_batch([(e.eid, e.pix(sl))], P)[0]["m"].flatten())
    return torch.cat(out)


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def test_off_by_default_and_nothing_is_made():
    assert _INTERNAL_SETTINGS["kernel_centroid_gauge"] == 0
    _, sf = _fit(0, steps=2)
    assert not sf.operator.kc_on and not hasattr(sf.operator, "kc_off")
    assert "kernel_centroid_gauge" not in sf.report
    assert not any(k.startswith("kc_") for k in sf.operator.state_dict())


def test_fit_reports_the_gauge_and_holds_the_moment(on_fit):
    _, sf = on_fit
    kr = sf.report["kernel_centroid_gauge"]
    assert kr["projections"] == 6 and len(kr["classes"]) == 3
    assert kr["held_moment_after_last_projection_mas"] < 1e-4
    op = sf.operator
    dl = op.dilation()
    dil = None if dl is None else dl.detach()
    with torch.no_grad():
        for c in range(len(op.kc_pairs)):
            assert float(op.kernel_centroid(c, op.kc_off[c], dil).norm()) < 1e-7
    # the kernel stamp is the rendered (moved) kernel: its windowed centroid is near zero too
    st = sf.response.kernel(sf.instruments[0], "A1")
    n = st.stamp.shape[0]
    g = (torch.arange(n) - n // 2) * st.pitch_arcsec
    yy, xx = torch.meshgrid(g, g, indexing="ij")
    w = st.stamp * torch.exp(-0.5 * (xx**2 + yy**2) / op.kc_sigma[0] ** 2)
    cx, cy = float((w * xx).sum() / w.sum()), float((w * yy).sum() / w.sum())
    assert (
        abs(cx) < 2e-3 and abs(cy) < 2e-3
    )  # the stamp's own quadrature (pitch 0.04 arcsec)


def test_projection_preserves_predictions(on_fit):
    exps, sf = on_fit
    op = sf.operator
    exps = _prepare(exps, op.dev)
    keep = {k: v.clone() for k, v in op.state_dict().items()}
    try:
        R, cls, tr = op.e_R, op.kc_class, op.ref_eids
        C = len(op.kc_pairs)
        with torch.no_grad():
            dk = torch.stack(
                [torch.tensor([5e-5, -3e-5]) * (-1) ** c for c in range(C)]
            )
            op.kc_off += dk
            op.affine.data[:, 4:6] -= (
                torch.einsum("eij,ej->ei", R, dk[cls]) / op.aff_scale[4:6]
            )
            pre = _predict(op, exps)
            t_mean = (op.affine[tr, 4:6] * op.aff_scale[4:6]).mean(0).clone()
            delta, _ = op.project_kernel_centroid()
            assert float(delta.abs().max()) > 1e-5
            assert _rel(_predict(op, exps), pre) < 1e-5
            t_after = (op.affine[tr, 4:6] * op.aff_scale[4:6]).mean(0)
            assert float((t_after - t_mean).abs().max()) < 1e-9
            op.kc_off[0] += torch.tensor(
                [1e-3, 0.0]
            )  # a 1 mas kernel shift alone is visible
            assert _rel(_predict(op, exps), pre) > 1e-4
    finally:
        op.load_state_dict(keep)
        op.sky._mut += 1
        op.sky._cache = None


def test_save_load_keeps_the_offsets(on_fit, tmp_path):
    _, sf = on_fit
    sf.save(tmp_path / "fit")
    lf = SkyFit.load(tmp_path / "fit")
    assert lf.operator.kc_on
    assert torch.equal(lf.operator.kc_off, sf.operator.kc_off)
