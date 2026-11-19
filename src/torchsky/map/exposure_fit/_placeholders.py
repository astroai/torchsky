"""Names of later pieces of the move: importable and documented, but not implemented. Every use raises ``NotImplementedError``."""

from __future__ import annotations


class ReferenceCatalogue:
    """Reference sources (positions, epochs, optional proper motions, fluxes, point-source flags, uncertainties) that can tie the frame.

    Not implemented yet: ``fit_sky(catalogue=...)`` is a later piece of the move. Constructing one raises ``NotImplementedError``.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ReferenceCatalogue is not implemented yet (a later piece of the move)"
        )
