"""Exceptions raised by the pytrees estimators."""


class TreeNotFoundError(Exception):
    """The search found no tree, so the estimator cannot predict."""


class SearchFailedError(Exception):
    """The native search failed."""


class ThresholdSkippingWarning(UserWarning):
    """Threshold skipping was forced on for a loss it may not suit.

    Skipping assumes what misclassification guarantees: moving ``n`` rows
    into a leaf never lowers its cost and raises it by at most ``n``. Raised when ``threshold_skipping=True`` is used with an
    ``error_function``, and again after the fit if the search saw the loss
    break that assumption.
    """
