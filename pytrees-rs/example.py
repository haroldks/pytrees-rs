# import graphviz
# import numpy as np
# from pytrees import DL85Classifier, LGDTCLassifier, ExposedSearchStrategy
#
# dataset = np.genfromtxt("../test_data/anneal.txt", delimiter=" ")
# X, y = dataset[:, 1:], dataset[:, 0]
#
#
# clf = LGDTCLassifier(1, 19, search_strategy=ExposedSearchStrategy.LGDTErrorMinimizer)
#
#
# clf.fit(X, y)
# print(clf.accuracy_)
# print(clf.tree_error_)
# print(clf.statistics)
#
# print(clf.score(X, y))
# #
# #
# graphviz.Source(clf.export_to_graphviz_dot()).view()
import graphviz


def error(tids):
    print(tids)
    return 1.0, 1.0


#
import numpy as np
from pytrees import LGDTClassifier, DL85Classifier
from pytrees.common import (
    ExposedSearchStrategy,
    ExposedDiscrepancyRule,
    ExposedStepStrategy,
    ExposedLowerBoundPolicy,
    ExposedBranchingPolicy,
)

dataset = np.genfromtxt("../test_data/anneal.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]

discrepancy = ExposedDiscrepancyRule(
    step_strategy=ExposedStepStrategy.Exponential, base=2
)

classifier = DL85Classifier(min_sup=1, max_depth=4, discrepancy=discrepancy)
classifier.fit(X, y)

print(classifier.statistics)
graphviz.Source(classifier.export_to_graphviz_dot()).view()
print(help(ExposedDiscrepancyRule))
