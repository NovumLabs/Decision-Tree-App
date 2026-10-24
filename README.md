# Novum Labs — Novum Labs Advanced Decision Tree AI

Two-stage development workspace for the Novum Labs repository:
https://github.com/NovumLabs/Decision-Tree-App

This workspace is designed to help Shreya (B.Tech fresher) progress from a mathematically grounded Decision Tree implementation to a tested, explainable, deployable AI application.

## Stage 1 — Core algorithm and mathematical foundations (Weeks 1–6)
1. Python, NumPy, pandas, testing and Git workflow review.
2. Entropy, Gini impurity, information gain, variance reduction and split criteria.
3. Implement classification and regression trees from scratch.
4. Add stopping conditions, prediction, tree inspection and reproducible tests.
5. Compare results against scikit-learn baselines; record accuracy, F1, MAE/RMSE and runtime.
6. Deliver a Stage 1 review with mathematical notes, examples, test results and known limitations.

## Stage 2 — Advanced features and application (Weeks 7–14)
1. Pre-pruning and post-pruning; evaluate overfitting and generalization.
2. Handle missing values, categorical features, class weights and imbalanced data.
3. Explore bagging and Random Forest as extensions (keep the single-tree engine independently testable).
4. Add explainability: decision paths, human-readable rules, feature importance and tree visualization.
5. Build a FastAPI backend and a React + TypeScript interface if the existing repository architecture supports them.
6. Add packaging, Docker, CI, documentation, examples and a release checklist.

## Suggested first steps
1. Review the existing GitHub repository and preserve working code before moving files.
2. Create a branch: `feature/stage-1-core-engine`.
3. Set up the environment: `python -m venv .venv` and install `requirements-dev.txt`.
4. Run tests: `pytest`.
5. Implement and test impurity criteria before implementing tree growth.

## Project layout
- `docs/`: mathematics, algorithm design, roadmap, experiments and API notes.
- `src/decision_tree_ai/`: Python package for the engine.
- `tests/`: unit, regression and integration tests.
- `examples/`: classification and regression examples.
- `app/backend/`: API starter.
- `app/frontend/`: frontend planning notes.
- `deployment/`: Docker starter.

## Important status note
This is a **starter workspace**, not a finished Decision Tree engine and not a verified copy of the live GitHub repository. The repository URL could not be fully inspected in this run. Compare this structure with the current repository before merging; adapt rather than overwrite existing work.

Maintained for **Novum Labs Pvt. Limited**.
