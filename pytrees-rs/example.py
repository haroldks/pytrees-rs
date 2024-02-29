import graphviz
import numpy as np
from pytrees import DL85Classifier, ExposedDataFormat

dataset = np.genfromtxt("../test_data/vote.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]


clf = DL85Classifier(1, 2)


clf.fit(X, y)
print(clf.accuracy_)
print(clf.tree_error_)
print(clf.statistics)

print(clf.score(X, y))


graphviz.Source(clf.export_to_graphviz_dot()).view()
