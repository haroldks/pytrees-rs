"""Rules that bound or relax the DL8.5 search.

Pass them to ``DL85Classifier`` or ``DL85Cluster``::

    >>> from pytrees import DL85Classifier
    >>> from pytrees.rules import GainRule, PurityRule
    >>> clf = DL85Classifier(max_depth=4, gain=GainRule(min_gain=0.01),
    ...                      purity=PurityRule(min_purity=0.9))

They are plain frozen dataclasses, so an estimator holding one can be
cloned, pickled and compared; the search reads their fields when ``fit``
starts.

Several rules grow a budget each time the search restarts. ``step_strategy``
says how: ``"monotonic"`` adds ``base`` each time, ``"exponential"``
multiplies by ``base``, and ``"luby"`` follows the Luby sequence scaled by
``base``.
"""

from dataclasses import dataclass
from typing import Optional

__all__ = ["DiscrepancyRule", "GainRule", "PurityRule", "RestartRule", "TopKRule"]


@dataclass(frozen=True)
class DiscrepancyRule:
    """Limited discrepancy search.

    Taking the feature of rank ``i`` at a node costs ``i`` discrepancies. A
    pass explores only the trees whose total cost is within the budget, which
    starts at ``initial_value`` and grows on each restart up to ``limit``.
    This puts the effort on the trees the heuristic prefers first
    (LDS-DL8.5).

    Parameters
    ----------
    initial_value : int, default=0
        Discrepancies allowed on the first pass.
    limit : int or None, default=None
        Largest budget; ``None`` lets it grow until the search is complete.
    step_strategy : {"monotonic", "exponential", "luby"}, default="monotonic"
    base : int, default=1
    """

    initial_value: int = 0
    limit: Optional[int] = None
    step_strategy: str = "monotonic"
    base: int = 1


@dataclass(frozen=True)
class GainRule:
    """Explore only the paths that stay close to the heuristic's choices.

    At each node, choosing a feature other than the best-ranked one loses the
    difference between their heuristic scores. A pass skips the paths whose
    accumulated loss exceeds a gap, and the gap widens on each restart.

    Parameters
    ----------
    min_gain : float, default=0.0
        Gap allowed on the first pass.
    epsilon : float, default=1e-4
        Step by which the gap widens on each restart.
    limit : float, default=6.0
        Largest gap; usually the maximum depth.
    step_strategy : {"monotonic", "exponential", "luby"}, default="monotonic"
    base : int, default=1
    """

    min_gain: float = 0.0
    epsilon: float = 1e-4
    limit: float = 6.0
    step_strategy: str = "monotonic"
    base: int = 1


@dataclass(frozen=True)
class PurityRule:
    """Do not split nodes that are already pure enough, on the current pass.

    The threshold rises on each restart until every node may be split.

    Parameters
    ----------
    min_purity : float, default=0.0
        Share of correctly classified rows from which a node is not split.
    epsilon : float, default=1e-4
        Amount the threshold rises by on each restart.
    """

    min_purity: float = 0.0
    epsilon: float = 1e-4


@dataclass(frozen=True)
class TopKRule:
    """Explore only the best-ranked features at each node (Top-k search).

    A pass with budget ``k`` tries the ``k + 1`` best features of each node,
    and ``k`` grows on each restart.

    Parameters
    ----------
    initial_value : int, default=0
        ``k`` on the first pass.
    limit : int or None, default=None
        Largest ``k``; ``None`` lets it grow to every feature.
    step_strategy : {"monotonic", "exponential", "luby"}, default="monotonic"
    base : int, default=1
    """

    initial_value: int = 0
    limit: Optional[int] = None
    step_strategy: str = "monotonic"
    base: int = 1


@dataclass(frozen=True)
class RestartRule:
    """Restart the search every ``limit`` seconds with relaxed rules.

    Parameters
    ----------
    limit : float, default=1.0
        Seconds per pass.
    """

    limit: float = 1.0
