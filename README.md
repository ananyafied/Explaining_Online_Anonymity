# Explaining_Online_Anonymity
Comparing the interpretability of black-box models used for anonymous fake account detection, bot and troll behavior analysis in social media spaces. Final project for Dr Brinnae Bent's class on Explainable AI (XAI) at Duke University.

This is v2 of the Explaining_Anonymity repository. v1 is under review and cannot be touched, so all work continues here. Everything from v1 is on `main`.

## Status
1. **Synthetic data with known ground truth** (done): `anonsynth/`, documented in [docs/synthetic_data.md](docs/synthetic_data.md)
2. **Model fitting** (done): `notebooks/01_fit_models.ipynb`, extra packages in `requirements-models.txt`, results and saved models in `outputs/step1_fit/`
3. **Explainability evaluation**: plan in [docs/step2_plan.md](docs/step2_plan.md), notebooks `notebooks/02` to `07`, shared code in `xai_eval/`, results written to `outputs/step2_explain/`

## Quick start
```bash
pip install -r requirements-dev.txt
python -m anonsynth --n-accounts 5000 --seed 0 --out data/synthetic   # writes data/synthetic/ (not tracked in git)
python -m pytest                                                       # checks the generator
```

## Step 1: fitting the models
```bash
pip install -r requirements-dev.txt -r requirements-models.txt
python -m anonsynth --n-accounts 5000 --seed 0 --out data/synthetic   # if you have not generated the data yet
# then run notebooks/01_fit_models.ipynb; it writes outputs/step1_fit/ (metrics and the saved models)
```
PyTorch Geometric has its own install instructions that depend on your torch/CUDA version (see the note in `requirements-models.txt`).

## Step 2: running the explainability notebooks
```bash
pip install -r requirements-xai.txt
python -m anonsynth --n-accounts 5000 --seed 0 --out data/synthetic   # if you have not generated the data yet
# then run, one cell at a time and in this order:
#   notebooks/02_explain_tabular.ipynb    notebooks/03_explain_sequences.ipynb
#   notebooks/04_explain_text.ipynb       notebooks/05_explain_graph.ipynb
#   notebooks/06_explain_multimodal.ipynb notebooks/07_scorecards.ipynb
```
Each notebook has a `QUICK = True` switch for a few-minute smoke test (its numbers are not meaningful); set `QUICK = False` for the full run. The step-1 models in `outputs/step1_fit/models/` are required.
