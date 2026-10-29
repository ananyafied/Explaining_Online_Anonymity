"""Tests for the evaluation helpers used by the step-2 notebooks (no models or data files needed)."""
import numpy as np
import pytest

from xai_eval import metrics as xm


def _linear_setup(n=40, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.array([5.0, 3.0, 1.0, 0, 0, 0, 0, 0])
    predict = lambda rows: np.array(rows) @ w  # noqa: E731
    remove = lambda x, idx: np.where(np.isin(np.arange(len(x)), idx), 0.0, x)  # noqa: E731
    return X, w, predict, remove


def test_deletion_gain_rewards_good_attributions_and_punishes_bad_ones():
    X, w, predict, remove = _linear_setup()
    good = np.abs(X * w)                       # exact contribution of each feature
    bad = np.abs(X * (w.max() - w))            # ranks the useless features first
    g_good = xm.deletion_gain(predict, list(X), good, remove, [1, 2], scale=1.0, n_random=20)
    g_bad = xm.deletion_gain(predict, list(X), bad, remove, [1, 2], scale=1.0, n_random=20)
    assert xm.mean_gain_per_instance(g_good).mean() > 0.5
    assert xm.mean_gain_per_instance(g_bad).mean() < 0


def test_deletion_gain_respects_n_items_for_padded_attributions():
    X, w, predict, remove = _linear_setup(n=10)
    attrs = np.abs(X * w)
    attrs[:, 5:] = 0                           # pretend only 5 real items per instance
    out = xm.deletion_gain(predict, list(X), attrs, remove, [3, 9], scale=1.0, n_random=3, n_items=np.full(10, 5))
    assert np.isfinite(out[["top", "random", "gain"]].to_numpy()).all()


def test_mask_auc_and_mass_inside():
    mask = np.zeros(20, bool)
    mask[[3, 4, 5]] = True
    attr = mask * 1.0 + 0.01
    assert xm.mask_auc(attr, mask) == 1.0
    assert 0.9 < xm.mass_inside(attr, mask) < 1.0
    assert np.isnan(xm.mask_auc(attr, np.zeros(20, bool)))


def test_rank_corr_overlap_and_stability():
    a = np.array([5.0, 4, 3, 2, 1])
    assert xm.rank_corr(a, a) == pytest.approx(1.0)
    assert xm.rank_corr(a, a[::-1]) == pytest.approx(-1.0)
    assert np.isnan(xm.rank_corr(a, np.ones(5)))
    assert xm.topk_overlap(a, a[::-1], 2) == 0.0
    assert xm.stability(np.stack([a, a]), np.stack([a, a])) == pytest.approx([1.0, 1.0])


def test_top_k_precision():
    attr = np.array([0.1, 5.0, -4.0, 0.2])
    rel = np.array([False, True, True, False])
    assert xm.top_k_precision(attr, rel, 2) == 1.0


def test_scorecard_interval_contains_mean():
    sc = xm.Scorecard("g")
    sc.add("m", "x", "faithfulness", np.random.default_rng(0).normal(1.0, 0.2, 100))
    sc.add("m", "x", "seconds_per_explanation", 0.5)
    df = sc.df.set_index("criterion")
    assert df.loc["faithfulness", "ci_low"] < df.loc["faithfulness", "value"] < df.loc["faithfulness", "ci_high"]
    assert np.isnan(df.loc["seconds_per_explanation", "ci_low"])
    assert sc.table().shape == (1, 2)


def test_noisy_copy_scale():
    X = np.zeros((2000, 3))
    noisy = xm.noisy_copy(X, np.array([1.0, 2.0, 0.0]), frac=0.1)
    assert abs(noisy[:, 0].std() - 0.1) < 0.01 and abs(noisy[:, 1].std() - 0.2) < 0.02 and noisy[:, 2].std() == 0
