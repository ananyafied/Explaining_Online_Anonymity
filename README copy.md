# Explaining_Anonymity
Comparing the interpretability of black-box models used for anonymous fake account detection, bot and troll behavior analysis in social media spaces. Final project for Dr Brinnae Bent's class on Explainable AI (XAI) at Duke University.

## Status
1. **Synthetic data with known ground truth** (done): `anonsynth/`, documented in [docs/synthetic_data.md](docs/synthetic_data.md)
2. Model fitting (done, branch `step1-model-fitting`): `notebooks/01_fit_models.ipynb`, results in `outputs/step1_fit/`
3. Explainability evaluation (branch `step2-explainability`): plan in [docs/step2_plan.md](docs/step2_plan.md), notebooks `notebooks/02` to `07`, shared code in `xai_eval/`, results written to `outputs/step2_explain/`

## Quick start
```bash
pip install -r requirements-dev.txt
python -m anonsynth --n-accounts 5000 --seed 0 --out data/synthetic   # writes data/synthetic/
python -m pytest                                                       # checks the generator
```

## Step 2: running the explainability notebooks
```bash
pip install -r requirements-xai.txt
python -m anonsynth --n-accounts 5000 --seed 0 --out data/synthetic   # if you have not generated the data yet
# then run, one cell at a time and in this order:
#   notebooks/02_explain_tabular.ipynb    notebooks/03_explain_sequences.ipynb
#   notebooks/04_explain_text.ipynb       notebooks/05_explain_graph.ipynb
#   notebooks/06_explain_multimodal.ipynb notebooks/07_scorecards.ipynb
```
Each notebook has a `QUICK = True` switch for a few-minute smoke test (its numbers are not meaningful). The step-1 models in `outputs/step1_fit/models/` are required.
