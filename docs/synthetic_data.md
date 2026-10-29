# Synthetic data: design and data card

Explainability is the focus of this project, so the data is synthetic with a **known ground truth**.
For every modality we know which inputs actually drive the label, which means an explanation
can be scored as right or wrong instead of just looking plausible.

```bash
pip install -r requirements.txt
python -m anonsynth --n-accounts 5000 --seed 0 --out data/synthetic
```

The output is deterministic given the seed. Defaults live in `anonsynth/config.py`.

## Files

| File | Contents |
|---|---|
| `accounts.csv` | `account_id`, 12 tabular features, `label` (1 = bot or troll), `split` |
| `posts.csv` | `post_id`, `account_id`, `text` (6 short posts per account) |
| `edges.csv` | `source`, `target`: undirected, rows shuffled, direction randomized |
| `sequences.npz` | `counts`: hourly post counts, shape (accounts, 168) |
| `style_features.csv` | 16 stylometric columns per account |
| `pairs.csv` | account pairs with `same_author` (50/50), for Siamese/linking models |
| `truth/posts_truth.csv` | per post: `planted_markers`, `has_spurious_token` |
| `truth/edges_truth.csv` | per edge, same row order as `edges.csv`: `edge_type` (`background`, `bot_follow`, `ring`) |
| `truth/accounts_truth.csv` | `kind` (human/bot/troll), `ring_id`, `author_id`, `true_logit`, `true_prob`, per-term logit contributions |
| `truth/sequence_masks.npz` | boolean mask of the hours that carry the planted pattern |
| `truth/ground_truth.json` | config, true formula, feature roles, marker tokens, ring windows, informative style columns |

**Do not feed anything in `truth/` to a model.** It exists to score explanations.

## What is planted, per modality

### Tabular (`accounts.csv`)
The label is `Bernoulli(sigmoid(true_logit))` with

```
logit = intercept + 1.4*z(log posts_per_day) - 1.2*z(reply_fraction)
        + 0.9*z(burstiness)*(-z(log account_age_days))      # interaction
        + 1.0*max(z(night_activity_frac) - 0.5, 0)           # threshold / nonlinearity
        + 0.6*z(url_fraction)
```

| Role | Features | What it tests |
|---|---|---|
| informative, linear | `posts_per_day`, `reply_fraction`, `url_fraction` | basic feature importance |
| interaction | `burstiness` x `account_age_days` | whether PDP/ALE/SHAP interaction values expose it |
| nonlinear threshold | `night_activity_frac` | PDP/ALE should show a flat-then-rising curve |
| redundant proxy | `posts_per_week` (noisy copy of `posts_per_day`, no direct effect) | credit splitting under correlated features, PDP vs ALE |
| distractors | `follower_following_ratio`, `hashtag_rate` | importance assigned to features with zero effect |
| noise | `noise_1..3` | importance floor |

`true_prob` gives the Bayes-optimal probability (AUC about 0.88 at defaults), so the achievable
performance is bounded and known. `term_*` columns give the exact per-account contribution of each
term, which can be compared to SHAP values (for the linear terms, up to a baseline shift).

### Text (`posts.csv`)
Posts are random neutral words. Bots carry bot marker tokens (`giveaway`, `promo`, ...), trolls carry
troll marker tokens (`idiot`, `sheep`, ...), each in 35% of their posts, and authentic accounts
carry either set in 3% of posts. A **spurious token** (`#trending`) appears in 70% of inauthentic
posts in train/val but only 10% in test (10% for authentic accounts everywhere). A model that
leans on it will degrade on test, and a good explanation should show that it does.

### Graph (`edges.csv`)
Trolls sit in coordinated rings of about 25 with dense internal edges (p = 0.25). Bots follow
8 authentic accounts each. Everyone also has sparse background edges (mean degree about 6).
The graph is undirected. Ring membership is in `truth/accounts_truth.csv` and edge types are in `truth/edges_truth.csv`.

### Sequences (`sequences.npz`)
168 hourly counts. Humans: circadian rhythm. Bots: low flat background plus bursts every 6, 8
or 12 hours. Trolls: circadian background plus two 3-hour windows shared by the whole ring.
`truth/sequence_masks.npz` marks the burst hours, so Grad-CAM / saliency / integrated-gradients
maps can be scored against it.

### Style and account linking (`style_features.csv`, `pairs.csv`)
About 15% of accounts are a second account of an existing author. 6 of the 16 style columns carry
author information, the other 10 are pure noise, and column order is shuffled (the informative
columns are listed in `ground_truth.json`). Positive pairs are same-author accounts, negatives are
random different-author accounts from the same split.

## Leakage checks
Model-facing files hold only what a real analyst could observe. `tests/test_leakage.py` enforces this:
- an allowlist of columns per model-facing file (no answer columns can slip in);
- edge row order and edge direction carry no information about edge type (edges are shuffled and
  direction is randomized, because they are generated in blocks by type);
- the order of the two accounts in a pair carries no information about `same_author`;
- account ids, row order, post ids and post length carry no information about the label (planted
  tokens replace neutral words, so they do not change post length);
- decoy and noise features are uninformative, and the label rate is the same in every split.

## Splits
`train` / `val` / `test` = 60 / 20 / 20, stratified on `label`. Linked accounts share their
partner's split, so pairs never straddle splits.

## Known simplifications
- Modalities are conditionally independent given the account kind (human/bot/troll). The only
  cross-modal link is that `posts_per_day` sets the baseline rate of the sequence. Only the
  tabular label is generated from the features; the other signals are generated from the kind.
  Multimodal and graph models are therefore tested on a cleaner problem than real data would give.
- The graph and sequence signals identify trolls and bots differently (rings only exist for
  trolls), so no single modality is sufficient for all inauthentic accounts.
- Real accounts do not have this tidy a structure. The point of the data is to check explanation
  methods against a known answer, not to estimate real-world performance.
