"""Names of later pieces of the move: importable and documented, but not implemented. Every use raises ``NotImplementedError``."""

from __future__ import annotations


class SkyProduct:
    """Sky image and standard deviation on a stated grid and beam, with its header and valid mask (``SkyFit.product`` returns one).

    Not implemented yet: the uncertainty and product step is a later piece of the move. Constructing one raises ``NotImplementedError``.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SkyProduct is not implemented yet (a later piece of the move)"
        )


class ReferenceCatalogue:
    """Reference sources (positions, epochs, optional proper motions, fluxes, point-source flags, uncertainties) that can tie the frame.

    Not implemented yet: ``fit_sky(catalogue=...)`` is a later piece of the move. Constructing one raises ``NotImplementedError``.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ReferenceCatalogue is not implemented yet (a later piece of the move)"
        )
