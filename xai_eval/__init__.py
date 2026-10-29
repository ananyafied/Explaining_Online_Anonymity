"""Shared code for step 2 (explainability). See docs/step2_plan.md.

Modules
- data:         loads the synthetic data exactly as step 1 did, plus the answer key (truth/)
- step1_models: the neural-network classes, copied from notebooks/01_fit_models.ipynb
- loaders:      reloads the step-1 models and rebuilds their scoring functions
- metrics:      correctness / faithfulness / agreement / stability helpers
- io:           saving results and scorecards
"""
# xgboost and lightgbm must be imported before torch so they share one OpenMP runtime on macOS
# (same workaround as in the step-1 notebook).
try:  # noqa: SIM105
    import xgboost  # noqa: F401
    import lightgbm  # noqa: F401
except Exception:  # pragma: no cover - they are only needed by the tree models
    pass

SEED = 0
