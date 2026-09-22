# Publications

The algorithms in pytrees-rs were introduced in the following papers. If you
use them in your work, please cite the relevant one.

**Time Constrained DL8.5 Using Limited Discrepancy Search.**
H. Kiossou, P. Schaus, S. Nijssen and V. R. Houndji.
ECML PKDD 2022, LNCS 13717, pp. 443-459.
[doi:10.1007/978-3-031-26419-1_27](https://doi.org/10.1007/978-3-031-26419-1_27)

Limited discrepancy search for DL8.5 (LDS-DL8.5), so that the search returns
good trees under a time limit. In pytrees: `DL85Classifier` with a
[`DiscrepancyRule`](estimators/rules.md#discrepancyrule).

**Efficient Lookahead Decision Trees.**
H. Kiossou, P. Schaus, S. Nijssen and G. Aglin.
IDA 2024, pp. 133-144.
[doi:10.1007/978-3-031-58553-1_11](https://doi.org/10.1007/978-3-031-58553-1_11)

LGDT, a top-down learner that chooses each test with an efficient depth-2
lookahead. In pytrees: [`LGDTClassifier`](estimators/lgdt.md).

**A Generic Complete Anytime Beam Search for Optimal Decision Tree.**
H. Kiossou and P. Schaus.
IDA 2026.
[doi:10.1007/978-3-032-23833-7_8](https://doi.org/10.1007/978-3-032-23833-7_8),
[arXiv:2508.06064](https://arxiv.org/abs/2508.06064)

CA-DL8.5, a framework that generalises LDS-DL8.5 and Top-k-DL8.5: rules
restrict each pass of the search and are relaxed at each restart. In pytrees:
the [search rules](estimators/rules.md) of `DL85Classifier`.

**Anytime Optimal Decision Tree Learning with Continuous Features.**
H. Kiossou, P. Schaus and S. Nijssen.
ECML PKDD 2026.
[arXiv:2601.14765](https://arxiv.org/abs/2601.14765)

An anytime version of ConTree based on limited discrepancy search. In
pytrees: [`ConTreeClassifier`](estimators/contree.md) with `use_lds=True` or
`fit_anytime`.

## Related work

- G. Aglin, S. Nijssen and P. Schaus. *Learning Optimal Decision Trees Using
  Caching Branch-and-Bound Search.* AAAI 2020. DL8.5; the original
  implementation is [pydl8.5](https://github.com/aia-uclouvain/pydl8.5).
- E. Demirović, A. Lukina, E. Hebrard, J. Chan, J. Bailey, C. Leckie,
  K. Ramamohanarao and P. J. Stuckey. *MurTree: Optimal Decision Trees via
  Dynamic Programming and Search.* JMLR 23, 2022. The depth-2 solver used by
  DL8.5 and LGDT.
- C. E. Briţa, J. G. M. van der Linden and E. Demirović. *Optimal
  Classification Trees for Continuous Feature Data Using Dynamic Programming
  with Branch-and-Bound.* AAAI 2025. ConTree; the original implementation is
  [ConSol-Lab/contree](https://github.com/ConSol-Lab/contree).
