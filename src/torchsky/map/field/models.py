"""Neural field-model classes for :mod:`torchsky.map.field`.

The implementation is grouped by representation family in sibling modules; the
class names are re-exported here so the historical import path stays stable.
"""

from __future__ import annotations

from .models_foam import (
    SpectralPowerFoamField,
    FoamInitializedINRField,
    DiffuseFoamHybridField,
)
from .models_component import (
    FieldComponentDecomposition,
)
from .models_implicit import (
    ExplicitSpectralGridField,
    SineImplicitField,
    _GaborLayer,
    WireImplicitField,
    GaussianBasisField,
    FourierMLPField,
    SpectralBasisField,
    BandpassConditionedField,
    LearnedSpectralBasisField,
)
from .models_encoder import (
    SkyEncoderBandpassField,
    ConditionedSpectralField,
    HashGridConditionedField,
    SkyEncoderMLPField,
    StrainerSkyEncoderMLPField,
    SkyEncoderResMLPField,
    HashGridMLPField,
    HashGridResMLPField,
    HashGridSpectralField,
)
from .models_skylambda import (
    SkyLambdaPlanesField,
    SkyLambdaKPlanesField,
    SkyLambdaFeatureField,
    OperatorConstrainedFeatureField,
    SkyLambdaAtlasField,
    SkyLambdaFeatureAtlasField,
)

__all__ = [
    "SpectralPowerFoamField",
    "FoamInitializedINRField",
    "DiffuseFoamHybridField",
    "FieldComponentDecomposition",
    "ExplicitSpectralGridField",
    "SineImplicitField",
    "_GaborLayer",
    "WireImplicitField",
    "GaussianBasisField",
    "FourierMLPField",
    "SpectralBasisField",
    "BandpassConditionedField",
    "LearnedSpectralBasisField",
    "SkyEncoderBandpassField",
    "ConditionedSpectralField",
    "HashGridConditionedField",
    "SkyEncoderMLPField",
    "StrainerSkyEncoderMLPField",
    "SkyEncoderResMLPField",
    "HashGridMLPField",
    "HashGridResMLPField",
    "HashGridSpectralField",
    "SkyLambdaPlanesField",
    "SkyLambdaKPlanesField",
    "SkyLambdaFeatureField",
    "OperatorConstrainedFeatureField",
    "SkyLambdaAtlasField",
    "SkyLambdaFeatureAtlasField",
]
