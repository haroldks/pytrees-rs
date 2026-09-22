"""Builds the native DL8.5 search from an estimator's parameters.

Shared by DL85Classifier and DL85Cluster, which differ only in what they
fix: the clustering search always hands row indices to its error function.
"""

from pytrees._native.dtrees import RawDL85


def dl85_search(estimator, *, fast_d2, error_function_input, error_function):
    """A ``RawDL85`` configured from ``estimator``'s parameters."""
    rules = (
        estimator.discrepancy,
        estimator.gain,
        estimator.topk,
        estimator.restart,
        estimator.purity,
    )
    # The similarity bound and dynamic branching assume a single complete
    # pass, so they are turned off when a rule makes the search run in passes.
    bounded = any(rule is not None for rule in rules)
    # The similarity bound also assumes that each row adds at most 1 to the
    # error, which only the built-in misclassification error guarantees.
    similarity_lb = estimator.similarity_lb and not bounded and error_function is None
    return RawDL85(
        min_sup=estimator.min_sup,
        max_depth=estimator.max_depth,
        max_error=estimator.max_error,
        time_limit=estimator.max_time,
        always_sort=estimator.always_sort,
        heuristic=estimator.heuristic,
        fast_d2=fast_d2,
        similarity_lb=similarity_lb,
        dynamic_branching=estimator.dynamic_branching and not bounded,
        error_function_input=error_function_input,
        discrepancy=estimator.discrepancy,
        gain=estimator.gain,
        topk=estimator.topk,
        restart=estimator.restart,
        purity=estimator.purity,
        error_function=error_function,
    )
