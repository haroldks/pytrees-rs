"""Exceptions raised by the pytrees estimators."""


class TreeNotFoundError(Exception):
    """The search found no tree, so the estimator cannot predict."""


class SearchFailedError(Exception):
    """The native search failed."""
