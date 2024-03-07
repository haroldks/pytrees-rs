# Pytrees-rs: Dynamic Programming-based Decision Tree Learning Library

Pytrees-rs is a library that encompasses decision tree learning algorithms implemented using dynamic programming.

The currently available algorithms include:


### DL8.5: Optimal Decision Tree Learning Algorithm
DL8.5 is an algorithm designed for learning optimal decision trees,
incorporating default misclassification error and a user-provided objective function.
The method uses a branch-and-bound approach with caching to minimize the search space exploration and a
speed-up the exploration. For further insights, refer to the relevant [paper](https://dial.uclouvain.be/pr/boreal/fr/object/boreal%3A223390/datastream/PDF_01/view),
and find the native [implementation](https://github.com/aia-uclouvain/pydl8.5).
Both implementations incorporate concepts introduced by [Murtree](https://www.jmlr.org/papers/volume23/20-520/20-520.pdf), another optimal decision learning algorithm.


### LDS-DL8.5 :Anytime Optimal Decision Tree Learning Algorithm
An anytime decision tree learning algorithm, building upon the foundation of DL8.5.
This algorithm addresses dynamic programming challenges, preventing stagnation in a particular part of the search space.
It ensures minimal tree quality within time constraints by leveraging Limited Discrepancy Search and a strategically designed restart strategy.
For in-depth details, refer to the comprehensive information available in the associated [paper](https://dial.uclouvain.be/pr/boreal/fr/object/boreal%3A266060/datastream/PDF_01/view).


### LGDT : Less Greedy Decision Tree Learning Algorithm

LGDT employs a reversible sparse bitset data structure, also found in DL8.5 and LDS-DL8.5.
Additionally, it incorporates Murtree's specialized depth-2 algorithm to provide decision trees with less
myopic evaluation. Unlike purely greedy algorithms such as CART and C4.5,
LGDT assesses error or information gain over two levels to determine the local best node value.
For a more in-depth understanding, refer to the details provided in the accompanying [paper](https://www.info.ucl.ac.be/~pschaus/assets/publi/ida2024_efficient_lookehead_decision_trees.pdf).


For more information on how to use Pytrees-rs, check the [documentation]().
