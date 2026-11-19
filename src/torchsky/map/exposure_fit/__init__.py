"""Item-sky fit to individual exposures (moved from a verified research prototype; the numerics are unchanged).

    from torchsky.map.exposure_fit import SkyFitConfig, fit_sky, SkyFit, predict_exposure, ExposurePrediction, load_exposures

``load_exposures`` reads calibrated exposure files (the only function here that opens files) into ``FieldObservation`` objects; ``fit_sky``
fits an item sky and an instrument response to them; ``predict_exposure`` scores a held-out exposure. ``SkyProduct``, ``ReferenceCatalogue`` and
``SkyFit.product`` are placeholders for later pieces. See ``docs/models/exposure_fit.md``.
"""

from ._load import load_exposures
from ._config import SkyFitConfig
from ._placeholders import ReferenceCatalogue, SkyProduct
from ._predict import ExposurePrediction, predict_exposure
from ._skyfit import SkyFit, fit_sky

__all__ = [
    "ExposurePrediction",
    "ReferenceCatalogue",
    "SkyFit",
    "SkyFitConfig",
    "SkyProduct",
    "fit_sky",
    "load_exposures",
    "predict_exposure",
]
