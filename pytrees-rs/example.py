import graphviz
import numpy as np
from pytreesrs.enums import ExposedDiscrepancyStrategy

from pytrees import (
    DL85Classifier,
    LDSDL85Classifier,
    RestartDL85Classifier,
    ExposedDataFormat,
    ExposedSpecialization,
    ExposedLowerBoundStrategy,
    ExposedBranchingStrategy,
    ExposedSearchHeuristic,
)

dataset = np.genfromtxt("../test_data/anneal.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]

clf = LDSDL85Classifier(
    1,
    4,
    budget=39,
    specialization=ExposedSpecialization.None_,
    lower_bound=ExposedLowerBoundStrategy.None_,
    branching_type=ExposedBranchingStrategy.None_,
    heuristic=ExposedSearchHeuristic.None_,
    discrepancy_strategy=ExposedDiscrepancyStrategy.Exponential,
)

clf.fit(X, y)
print(clf.statistics)
graphviz.Source(clf.export_to_graphviz_dot()).view()


# print(clf.score(X, y)
#
#
# graphviz.Source(clf.export_to_graphviz_dot()).view()
