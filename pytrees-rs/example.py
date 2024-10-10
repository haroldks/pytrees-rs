import graphviz
import numpy as np

from pytrees import (
    DL85Classifier,
    RestartDL85Classifier,
    ExposedDataFormat,
    ExposedSpecialization,
    ExposedLowerBoundStrategy,
    ExposedBranchingStrategy,
    ExposedSearchHeuristic,
)

dataset = np.genfromtxt("../test_data/kr-vs-kp.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]

clf = RestartDL85Classifier(
    1,
    4,
    use_partial_fit=True,
    specialization=ExposedSpecialization.None_,
    lower_bound=ExposedLowerBoundStrategy.None_,
    branching_type=ExposedBranchingStrategy.None_,
    heuristic=ExposedSearchHeuristic.None_,
)

clf.partial_fit(X, y, 600)
print(clf.statistics)
graphviz.Source(clf.export_to_graphviz_dot()).view()


runtime = 10

while not clf.is_optimal:
    clf.partial_fit(X, y, runtime=runtime)
    x = clf.statistics
    del x["constraints"]
    del x["duration"]
    x["runtime"] = clf.results.duration
    x["optimal"] = clf.is_optimal
    print(x)


# print(clf.score(X, y)
#
#
# graphviz.Source(clf.export_to_graphviz_dot()).view()
