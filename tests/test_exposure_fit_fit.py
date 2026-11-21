"""fit_sky on a tiny synthetic field: shapes, finite values, falling loss, save/load, prediction, placeholders, private path."""

import math
from types import MappingProxyType

import pytest
import torch

import torchsky.map.exposure_fit._config as cfgmod
import torchsky.map.exposure_fit._skyfit as skymod
from _exposure_fit_common import (
    TINY_INTERNAL,
    cached_fit,
    tiny_config,
    tiny_field,
    tiny_fit,
    tiny_observations,
)
from torchsky.map.exposure_fit import (
    ExposurePrediction,
    ReferenceCatalogue,
    SkyFit,
    fit_sky,
    predict_exposure,
)


@pytest.fixture(scope="module")
def public_fit(tmp_path_factory):
    """One run of the public fit_sky with the small private settings (patched in, because they are not public)."""
    torch.set_num_threads(1)
    mp = pytest.MonkeyPatch()
    small = MappingProxyType({**cfgmod._INTERNAL_SETTINGS, **TINY_INTERNAL})
    mp.setattr(cfgmod, "_INTERNAL_SETTINGS", small)
    mp.setattr(skymod, "_INTERNAL_SETTINGS", small)
    obs = tiny_observations()
    lines = []
    try:
        sf = fit_sky(obs, tiny_config(40), log=lines.append)
    finally:
        mp.undo()
    return obs, sf, lines


def test_public_fit_shapes_finite_and_loss_falls(public_fit):
    obs, sf, lines = public_fit
    assert isinstance(sf, SkyFit) and lines
    assert sf.sky.n_items == 300 and sf.config.item_count == 300
    log = sf.report["loss_log"]
    assert len(log) >= 6
    assert all(math.isfinite(r[2]) for r in log)
    assert sum(r[2] for r in log[-2:]) < sum(r[2] for r in log[:2])
    terms = sf.exposure_terms
    assert set(terms) == {o.header["TSNAME"] for o in obs}
    for t in terms.values():
        assert (
            len(t["affine"]) == 6
            and len(t["background"]) == 6
            and all(math.isfinite(v) for v in t["affine"] + t["background"])
        )
    for name, p in sf.state_dict().items():
        assert torch.isfinite(p).all(), name
    nt = sf.response.noise_terms()
    assert set(nt) == {"floor", "scale"} and all(v > 0 for v in nt["scale"].values())
    assert sf.response.kernel_network is not None


def test_public_inputs_not_modified_and_cards_required(public_fit):
    obs, _, _ = public_fit
    assert obs[0].header["TSSPLIT"] in ("train", "held") and "TSNAME" in obs[0].header
    bad = tiny_observations()[:3]
    del bad[0].header["TSBOXRA"]
    with pytest.raises(ValueError, match="TSBOXRA"):
        fit_sky(bad, tiny_config(1))
    with pytest.raises(ValueError):
        fit_sky([], tiny_config(1))
    nocurve = tiny_observations()[:3]
    nocurve[0].transmission_wavelength = None
    with pytest.raises(ValueError, match="filter curve"):
        fit_sky(nocurve, tiny_config(1))


def test_placeholders_raise():
    obs = tiny_observations()[:2]
    with pytest.raises(NotImplementedError):
        fit_sky(obs, tiny_config(1), catalogue=object())
    _, sf = cached_fit()
    with pytest.raises(NotImplementedError):
        fit_sky(obs, tiny_config(1), init=sf)
    with pytest.raises(NotImplementedError):
        ReferenceCatalogue()


def test_predict_with_and_without_cross_fit(public_fit):
    obs, sf, _ = public_fit
    held = [o for o in obs if o.header["TSSPLIT"] == "held"]
    assert held
    chi = {}
    for o in held:
        for xf in (True, False):
            p = predict_exposure(sf, o, cross_fit=xf)
            assert isinstance(p, ExposurePrediction) and p._fields == (
                "model",
                "sigma",
                "terms",
                "chi2",
            )
            assert p.model.shape == o.image.shape == p.sigma.shape
            assert (
                torch.isfinite(p.model).any()
                and torch.isfinite(p.sigma).all()
                and (p.sigma > 0).all()
            )
            assert p.chi2["n"] > 0 and math.isfinite(p.chi2["chi2"])
            chi[(o.header["TSNAME"], xf)] = p.chi2["chi2"]
    assert any(
        chi[(n, True)] != chi[(n, False)] for n, _ in chi
    )  # the two modes are different predictions
    train = next(o for o in obs if o.header["TSSPLIT"] == "train")
    with pytest.raises(ValueError, match="training exposure"):
        predict_exposure(sf, train)
    stranger = tiny_observations()[0]
    stranger.header["TSNAME"] = "not_in_fit"
    stranger.band_name = "no_such_band"
    with pytest.raises(ValueError, match="not in the fit"):
        predict_exposure(sf, stranger)


def test_save_load_round_trip_gives_identical_predictions(public_fit, tmp_path):
    obs, sf, _ = public_fit
    sf.save(tmp_path / "fit")
    assert {p.name for p in (tmp_path / "fit").iterdir()} == {
        "manifest.json",
        "state.pt",
    }
    lf = SkyFit.load(tmp_path / "fit")
    a, b = sf.state_dict(), lf.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert (
        lf.config == sf.config
        and lf.exposure_names == sf.exposure_names
        and lf.exposure_terms == sf.exposure_terms
    )
    for o in (o for o in obs if o.header["TSSPLIT"] == "held"):
        for xf in (True, False):
            p1, p2 = (
                predict_exposure(sf, o, cross_fit=xf),
                predict_exposure(lf, o, cross_fit=xf),
            )
            assert (
                torch.equal(p1.model, p2.model)
                and torch.equal(p1.sigma, p2.sigma)
                and p1.chi2 == p2.chi2
                and p1.terms == p2.terms
            )
    with pytest.raises(ValueError):
        (tmp_path / "bad").mkdir()
        (tmp_path / "bad" / "manifest.json").write_text('{"format": "other"}')
        SkyFit.load(tmp_path / "bad")


def test_private_fit_is_deterministic_and_leaves_inputs_alone():
    torch.set_num_threads(1)
    exps = tiny_field()
    before = [(e.eid, e.shift0, len(e.patches)) for e in exps]
    a, b = tiny_fit(exps, 6), tiny_fit(exps, 6)
    assert before == [(e.eid, e.shift0, len(e.patches)) for e in exps]
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert a.report["noise_floor_final"] == b.report["noise_floor_final"]


def test_private_fit_rejects_bad_records():
    exps = tiny_field()
    exps[0].eid = 5
    with pytest.raises(ValueError, match="eid"):
        skymod._fit_exposures(exps, tiny_config(1))
    exps = tiny_field()
    for e in exps:
        e.split = "held"
    with pytest.raises(ValueError, match="training"):
        skymod._fit_exposures(exps, tiny_config(1))
