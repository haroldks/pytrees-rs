import numpy as np
import time
import sys
import graphviz
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from pytrees import (
    DL85Classifier,
    ExposedSpecialization,
    ExposedDataFormat,
    ExposedBranchingStrategy,
    ExposedLowerBoundStrategy,
)

inside_func = 0
count = [0]


def evaluate_runtime(model, X, y, cv=5):
    pass


# return the error and the majority class
def error(sup_iter):
    global inside_func
    global count
    count[0] += 1
    start = time.time()
    supports = list(sup_iter)
    maxindex = np.argmax(supports)
    inside_func += time.time() - start
    return sum(supports) - supports[maxindex], maxindex


def xerror(sup_iter):
    global count
    count += 1
    supports = list(sup_iter)
    maxindex = np.argmax(supports)
    return sum(supports) - supports[maxindex], maxindex


cc = [0]
# return the error and the majority class
def mserror(tids):
    global inside_func
    global count
    count[0] += 1
    start = time.time()
    classes, supports = np.unique(y.take(tids), return_counts=True)
    maxindex = np.argmax(supports)
    inside_func += time.time() - start
    return sum(supports) - supports[maxindex], int(classes[maxindex])


# def error(tids):
#     # tids = list(tids)
#     global count
#     count += 1
#     if len(np.unique(y[tids])) < 2:
#         return 0.0, y[0]
#     linear_model = LogisticRegression()
#     linear_model.fit(X[tidscsv, :], y[tids])
#     y_pred = linear_model.predict(X[tids, :])
#     error_val = (y_pred != y[tids]).sum()
#     unique_elements, counts = np.unique(y_pred, return_counts=True)
#     max_count_index = np.argmax(counts)
#     leaf_value = unique_elements[max_count_index]
#     return error_val, leaf_value


dataset = np.genfromtxt("../test_data/paper.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]

clf = DL85Classifier(
    min_sup=1,
    max_depth=3,
    error_function=error,
    # data_format=ExposedDataFormat.Tids,
    branching_type=ExposedBranchingStrategy.None_,
    lower_bound=ExposedLowerBoundStrategy.None_,
    specialization=ExposedSpecialization.None_,
)

clf.fit(X, y)
print(clf.results.statistics)
print(f"Total inside the function = {inside_func} secs")
print(f"Went in python {count} times")
print(f"Average time spent by one call : {inside_func * 1000/ count[0]}")
# print((x != y).sum())
# graphviz.Source(clf.export_to_graphviz_dot()).view()
