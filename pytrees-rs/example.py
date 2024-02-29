import numpy as np
from pytrees import DL85Cluster, ExposedDataFormat

dataset = np.genfromtxt("../test_data/anneal.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]


def error(tids, y):
    classes, supports = np.unique(y.take(list(tids)), return_counts=True)
    maxindex = np.argmax(supports)
    return sum(supports) - supports[maxindex], classes[maxindex]


clf = DL85Cluster(1, 2)

clf.fit(X)

print(clf.results.statistics)
