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


def error(tids):
    print(tids)
    return 1.0, 1.0


import numpy as np
from pytreesrs.enums import ExposedNodeDataType
from pytreesrs.odt import PyDL85

dataset = np.genfromtxt("../test_data/anneal.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]

classifier = PyDL85(data_type=ExposedNodeDataType.ClassSupportso, error_function=error)
classifier.fit(X, y)
stats = classifier.stats

print(stats.error)


# print(X.dtype)
# X = X.astype(int, copy=False)
# print(X.dtype)
# rapid(X)
