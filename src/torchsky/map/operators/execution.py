"""Executable operator helpers shared by image-plane mapmaking paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor

from .background import SpatialSkyBackgroundOp
from .bundle import ObservationOperatorBundle, image_observation_operator_bundle
from .geometry import (
    Conservative