"""FieldObservation -> private record conversion: positions, values, indices, and the cards it requires."""

import numpy as np
import pytest
import torch

from _exposure_fit_common import tiny_field, to_observation
from torchsky.map.exposure_fit._observations import observations_to_exposures


def test_round_trip_positions_values_and_indices():
    exps = tiny_field()
    obs = [to_observation(e) for e in exps]
    recs, box_half = observations_to_exposures(obs, n_lambda=3)
    assert box_half == 0.8 and len(recs) == len(exps)
    for e, r in zip(exps, recs):
        assert (
            r.name == e.name
            and r.split == e.split
            and r.det_shape == e.det_shape
            and r.cut0 == (0, 0)
        )
        assert (
            torch.equal(r.d, e.d)
            and torch.equal(r.valid, e.valid)
            and torch.equal(r.sig, e.sig)
        )
        assert (
            float((r.X - e.X).abs().max()) < 1e-5
            and float((r.Y - e.Y).abs().max()) < 1e-5
        )  # arcsec; float32 round-off of the tangent plane
        assert np.allclose(r.R, e.R, atol=1e-6)
        assert r.bandpass.name == e.bandpass.name
    # instruments sorted by name (NIRCAM before WFC3_IR); filters by pivot wavelength with two instruments
    assert [r.inst for r in recs if r.bandpass.name.startswith("A")][0] == 0
    assert [r.inst for r in recs if r.bandpass.name.startswith("B")][0] == 1
    piv = {r.filt: r.bandpass.pivot for r in recs}
    assert [piv[i] for i in sorted(piv)] == sorted(piv.values())


def test_observation_inputs_are_not_modified():
    o = to_observation(tiny_field()[0])
    h, img = dict(o.header), o.image.clone()
    observations_to_exposures([o], n_lambda=3)
    assert o.header == h and torch.equal(o.image, img)


def test_errors():
    o = to_observation(tiny_field()[0])
    p = to_observation(tiny_field()[0])
    with pytest.raises(ValueError, match="share the name"):
        observations_to_exposures([o, p], n_lambda=3)
    bad = to_observation(tiny_field()[0])
    bad.spectral_unit = "angstrom"
    with pytest.raises(ValueError, match="spectral_unit"):
        observations_to_exposures([bad], n_lambda=3)
    bad = to_observation(tiny_field()[0])
    bad.mask = None
    with pytest.raises(ValueError, match="err and mask"):
        observations_to_exposures([bad], n_lambda=3)
    bad = to_observation(tiny_field()[0])
    bad.mask = torch.zeros_like(bad.mask)
    with pytest.raises(ValueError, match="no valid pixel"):
        observations_to_exposures([bad], n_lambda=3)
