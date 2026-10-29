"""The four evaluation criteria (docs/step2_plan.md section 2) as small, testable functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------------ generic helpers
def rank_corr(a, b) -> float:
    """Spearman correlation; NaN if either vector is constant."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(spearmanr(a, b)[0])


def topk_overlap(a, b, k=3) -> float:
    """Share of the top-k items (by |value|) that two attribution vectors have in common."""
    ta = set(np.argsort(-np.abs(a))[:k])
    tb = set(np.argsort(-np.abs(b))[:k])
    return len(ta & tb) / k


def boot_ci(values, n_boot=1000, seed=0):
    """(mean, lo, hi): percentile bootstrap 95% interval of the mean over instances. NaNs are dropped."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


class Scorecard:
    """Collects one row per (model, method, criterion) with a bootstrap interval over instances."""

    def __init__(self, group: str):
        self.group, self.rows = group, []

    def add(self, model, method, criterion, values, note=""):
        """`values`: one number per explained instance (preferred), or a single number (no interval)."""
        v = np.atleast_1d(np.asarray(values, float))
        m, lo, hi = boot_ci(v) if len(v) > 1 else (float(v[0]), np.nan, np.nan)
        self.rows.append(dict(group=self.group, model=model, method=method, criterion=criterion,
                              value=m, ci_low=lo, ci_high=hi, n=int(np.sum(~np.isnan(v))), note=note))

    @property
    def df(self):
        return pd.DataFrame(self.rows)

    def table(self):
        """Wide view: one row per (model, method), one column per criterion."""
        d = self.df
        if d.empty:
            return d
        return d.pivot_table(index=["model", "method"], columns="criterion", values="value", aggfunc="first")


# ------------------------------------------------------------------ correctness
def mask_auc(attr, mask) -> float:
    """AUC of |attribution| for separating planted positions (mask=True) from the rest, for one instance."""
    mask = np.asarray(mask, bool)
    if mask.all() or not mask.any():
        return np.nan
    return float(roc_auc_score(mask, np.abs(attr)))


def mass_inside(attr, mask) -> float:
    """Share of total |attribution| that falls on planted positions."""
    a = np.abs(attr)
    return float(a[np.asarray(mask, bool)].sum() / max(a.sum(), 1e-12))


def top_k_precision(attr, relevant, k) -> float:
    """Of the k largest |attributions|, the share that are truly relevant (relevant: boolean per item)."""
    top = np.argsort(-np.abs(attr))[:k]
    return float(np.asarray(relevant, bool)[top].mean())


# ------------------------------------------------------------------ faithfulness (deletion test)
def deletion_gain(predict, X, attrs, remove, ks, scale, n_random=5, seed=0, n_items=None):
    """Does deleting the top-attributed inputs move the score more than deleting random ones?

    predict : list/array of instances -> array of scores (the quantity that was explained)
    X       : sequence of instances
    attrs   : (m, d) attributions, one row per instance (any sign; ranked by |value|)
    remove  : remove(x, idx) -> copy of instance x with positions `idx` neutralised
    ks      : list of how many inputs to delete
    scale   : standard deviation of the score over a reference sample, so probabilities and
              anomaly scores are comparable (deletion effect is reported in these units)
    n_items : optional, number of real items per instance when `attrs` is zero-padded (e.g. posts
              with different numbers of words); only the first n_items[i] columns can be deleted

    Returns a DataFrame with one row per (instance, k): change after deleting top-k, change after
    deleting k random inputs (mean of `n_random` draws), and their difference (the "gain").
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(predict(list(X)), float)
    rows = []
    d = attrs.shape[1]
    n_items = np.full(len(X), d) if n_items is None else np.asarray(n_items)
    for k in ks:
        kk = np.minimum(k, n_items)  # cannot delete more items than an instance has
        top_mod = [remove(X[i], np.argsort(-np.abs(attrs[i, : n_items[i]]))[: kk[i]]) for i in range(len(X))]
        d_top = np.abs(np.asarray(predict(top_mod), float) - base) / scale
        d_rand = np.zeros(len(X))
        for _ in range(n_random):
            rand_mod = [remove(X[i], rng.choice(n_items[i], size=kk[i], replace=False)) for i in range(len(X))]
            d_rand += np.abs(np.asarray(predict(rand_mod), float) - base) / scale
        d_rand /= n_random
        rows.append(pd.DataFrame({"i": np.arange(len(X)), "k": k, "top": d_top, "random": d_rand, "gain": d_top - d_rand}))
    return pd.concat(rows, ignore_index=True)


def mean_gain_per_instance(del_df) -> np.ndarray:
    """Average the gain over the deletion sizes: one faithfulness number per instance."""
    return del_df.groupby("i")["gain"].mean().sort_index().to_numpy()


# ------------------------------------------------------------------ stability
def stability(attr_a, attr_b) -> np.ndarray:
    """Per-instance rank correlation between attributions of the original and the perturbed input."""
    return np.array([rank_corr(np.abs(a), np.abs(b)) for a, b in zip(attr_a, attr_b)])


def noisy_copy(X, scale_vec, frac=0.05, seed=0):
    """Add Gaussian noise with std = frac * scale_vec (per feature) to every row of X."""
    rng = np.random.default_rng(seed)
    return np.asarray(X, float) + rng.normal(0, 1, np.shape(X)) * (frac * np.asarray(scale_vec))
