# Step 3 plan: understanding and improving the best models

**Status:** draft v1 (2026-10-29). Nothing in this plan is implemented yet. The model list is still open (section 2).

## 1. Goal

Pick a small number of models that still have room to improve, find out what their predictions are actually based on, and then try to retrain them so that they rely on the right variables and perform better, or at least hold up better when the shortcuts are removed.

Questions:
1. **Mechanism.** For each selected model, which inputs does the score really depend on: the planted signal, a confound (post volume, node degree), or something else?
2. **Explanation quality.** Do the explainers from step 2 point at what the interventions in section 3 show the model uses?
3. **Retraining.** Can we change the data or the training so the model relies on the planted signal, and does that help on stress tests where the confounds are broken?

## 2. Choosing the models

**Decisions made**
* LSTM and CNN are dropped. A single number, the peak hourly count, reaches test AUC 1.000 on its own, so they have no headroom. The LSTM autoencoder (AUC 0.988) is also near the ceiling and reads the same timeline input; keep it out unless there is a reason to include it.
* Models at or near chance are out: label spreading (0.499), belief propagation (0.096, inverted), node2vec + logreg (0.591), deepwalk + logreg (0.632).

**Selection criteria**
1. Room to improve (test AUC clearly below the ceiling).
2. Explanations that are messy or disagree, so there is something to learn.
3. A spread of input types, so the findings are not about one data type.

**Candidates (top 10 with headroom, by test AUC)**. See Appendix A. **Selected models: TBD.**

## 3. Phase A: find the mechanism with interventions

The generator is ours, so we can change one property at a time and watch the score. Every test compares against the unchanged input.

| Model family | Hypotheses | Interventions |
|---|---|---|
| Graph networks (GCN, GAT, GraphSAGE, R-GCN) | They mostly track degree; node features matter less than structure | Zero the node features; rewire edges while keeping each node's degree; compare with degree-matched controls; ablate edges by type (ring, bot-follow, background) |
| Fusion | It leans on the text block; the graph block adds little | Already have block Shapley; add block-level interventions on single accounts |
| Text models | They read the marker words | Strip markers; strip `#trending`; swap markers between classes |
| RNN | It reads amplitude, not timing | Normalise each timeline by its total volume; clip peaks; move bursts; swap volumes across accounts |
| Tree models | They use the six informative stats | Already measured with SHAP; add single-feature interventions |

Output: for each model, one table stating which hypotheses held.

## 4. Phase B: define the right variables

Taken from the answer key (`data/synthetic/truth/`), not from the models:
* Profile stats: the six informative features.
* Text: the class marker words.
* Timeline: the planted burst windows.
* Graph: ring and bot-follow edges.
Confounds to break, not to rely on: post volume, node degree, the `#trending` decoy, and `posts_per_week`.

## 5. Phase C: retraining

Ordered from least to most intrusive. Try the earlier ones first.
1. **Fix the data.** Add counterfactual accounts that break a confound (volume-matched, degree-matched).
2. **Reweight groups** so no kind or confound level dominates the loss.
3. **Attribution prior.** Penalise attribution outside the answer-key region during training ("right for the right reasons").
4. **Concept erasure.** Remove the direction in a hidden layer that encodes the confound, then retrain or fine-tune.

**Rule against circularity.** Any answer-key information used in training (steps 3 and 4) must not be the same information used to score the result. Evaluate with the stress tests in section 6, not with the step 2 correctness scores alone.

## 6. Phase D: evaluation

**Stress test sets** (new variants of the generator, with the default output unchanged):
* volume-matched fake and real accounts;
* degree-matched fake and real accounts;
* bursts moved to other hours;
* `#trending` made independent of the label.

**Measure, before and after retraining**
* AUC and accuracy on the original test set and on each stress set;
* the step 2 criteria (correctness, faithfulness, agreement, stability) on the same accounts;
* the intervention tables from Phase A, to check the mechanism actually moved.

**Controls**
* retrain with the same settings but without the intervention, so a gain is not just extra training;
* several seeds, with intervals reported.

**Reporting.** Compare models only within an input group, as in step 2. No composite score and no ranking across groups.

## 7. Code changes this implies (not started)
* `anonsynth`: optional parameters that build the stress sets. Defaults must reproduce today's data exactly.
* `xai_eval`: intervention helpers (feature zeroing, degree-preserving rewiring, volume normalisation).
* New notebooks after 07 for the diagnosis, the retraining and the stress-test comparison.

## 8. Risks and limits
* Everything is synthetic. A model that "relies on the right variables" here is only right relative to our answer key.
* The test set is small: 137 trolls and 188 bots, so per-kind results have wide intervals.
* Public bot benchmarks are known to be easy and to transfer poorly (Hays et al., WWW 2023). The same can be said of this data; keep claims modest.
* Profile-stat models have limited headroom: the Bayes-optimal AUC from profile stats alone is about 0.88 (`docs/synthetic_data.md`), and XGBoost and LightGBM are at about 0.84.
* PCA and SVD are the same model; count them once.
* `#trending` is left as is and is not a focus of this step.

## 9. Optional extensions, if time permits
**Feature clamping.** Hold an input feature or a hidden unit at a fixed value and watch the score. Works for all models at the input level. For neural networks it can also be done on hidden units: LSTM units, CNN channels, GNN node embeddings.

**Feature steering.** Find a concept direction in a hidden layer with a probe (for example "volume" or "degree") and add or subtract it, to test whether the model uses that concept. Neural networks only; trees and SVMs have no hidden layer. The direction is correlational until confirmed with the generator's labels. It is an analysis tool at this scale and also feeds the concept-erasure step in Phase C.

## Appendix A: candidates with headroom

Test AUC and accuracy come from `outputs/step1_fit/metrics.csv`. "Based on" summarises the step 2 results in `outputs/step2_explain/`.

| # | Model | AUC | Acc. | Input | What the predictions rest on |
|---|---|---|---|---|---|
| 1 | GCN | 0.965 | 0.905 | 12 profile stats + graph | Score correlates 0.83 (Spearman) with node degree; degree alone reaches AUC 0.91; half the edges gives 0.886. Explainers weak on ring edges (AUC 0.57 to 0.69) and agree only 0.29. |
| 2 | GAT | 0.949 | 0.887 | same | Degree correlation 0.53; half the edges 0.863. Ring-edge AUC 0.54 to 0.61; explainers agree 0.66. |
| 3 | Multimodal fusion | 0.948 | 0.815 | text embedding + 12 stats + graph | Block credit: text 51 to 56%, graph 26 to 34%, profile 14 to 23%. Text block tracks marker counts (r = 0.77). Graph block separates trolls at AUC 0.51, and the graph alone scores 0.42. |
| 4 | R-GCN | 0.934 | 0.865 | same as GCN | Degree correlation 0.54; half the edges 0.864; ring-edge AUC 0.59 to 0.61. |
| 5 | GraphSAGE | 0.932 | 0.859 | same as GCN | Degree correlation 0.52; half the edges 0.860; ring-edge AUC 0.59 to 0.62. |
| 6 | RNN | 0.886 | 0.846 | 168 hourly counts | Integrated Gradients finds the planted hours (AUC 0.94; GradientSHAP 0.63), but shuffling hours costs only 0.004 AUC, so it reads amplitude, not timing. |
| 7 | Transformer | 0.873 | 0.780 | post text | Marker words: removing them drops post-level AUC by 0.16; removing `#trending` changes it by about 0.01. Explainers still put 53 to 87% of credit on `#trending` in decoy-only posts. |
| 8 | doc2vec + logreg | 0.855 | 0.790 | post text | Marker words (explainer marker credit about 0.83); removing them drops AUC 0.15. `#trending` gets 53 to 56% in decoy-only posts. |
| 9 | XGBoost | 0.844 | 0.773 | 12 profile stats | SHAP credit: 80% informative, 7% proxy, 5% decoys, 7% noise. |
| 10 | LightGBM | 0.843 | 0.765 | 12 profile stats | SHAP credit: 86% informative, 7% proxy, 2% decoys, 5% noise. |

Not in the top 10: random forest (0.829), RBF and linear SVM (0.822), PCA, SVD and NMF + logreg (0.761 to 0.772).
Not yet measured for the graph networks: how much comes from node features versus structure. Phase A fills that gap.
