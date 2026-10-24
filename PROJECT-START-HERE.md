# Start Here — Novum Labs Decision Tree AI

## Stage 1 acceptance checklist
- [ ] Entropy and Gini functions handle edge cases and are covered by unit tests.
- [ ] Information gain / variance reduction split criteria are documented.
- [ ] Classification tree supports `fit` and `predict`.
- [ ] Regression tree supports `fit` and `predict`.
- [ ] Maximum depth, minimum samples split/leaf, and stopping conditions work.
- [ ] Tests pass and results are compared with scikit-learn.
- [ ] Stage 1 report includes formulas, design decisions, benchmark results and limitations.

## Stage 2 acceptance checklist
- [ ] Pruning experiments demonstrate the trade-off between complexity and generalization.
- [ ] Missing and categorical data handling is specified and tested.
- [ ] Class imbalance support is evaluated.
- [ ] Decision path and readable rule output are available.
- [ ] Tree visualization and feature importance are documented.
- [ ] API and frontend are integrated only after the core engine is stable.
- [ ] CI, packaging, Docker instructions and reproducible examples are checked.

## Weekly routine
1. Read the relevant mathematical notes.
2. Implement one small feature.
3. Add tests before expanding the feature.
4. Compare against a trusted baseline.
5. Update `docs/experiments.md` and submit a concise weekly report.
