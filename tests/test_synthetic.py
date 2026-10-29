import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from anonsynth import SynthConfig, generate
from anonsynth.tabular import FEATURES


@pytest.fixture(scope="module")
def ds():
    return generate(SynthConfig(n_accounts=4000, seed=1))


def test_deterministic():
    a = generate(SynthConfig(n_accounts=500, seed=3))
    b = generate(SynthConfig(n_accounts=500, seed=3))
    pd.testing.assert_frame_equal(a.accounts, b.accounts)
    pd.testing.assert_frame_equal(a.posts, b.posts)
    pd.testing.assert_frame_equal(a.edges, b.edges)
    assert (a.seq_counts == b.seq_counts).all()
    c = generate(SynthConfig(n_accounts=500, seed=4))
    assert not a.accounts["label"].equals(c.accounts["label"])


def test_prevalence_and_calibration(ds):
    assert abs(ds.accounts["label"].mean() - 0.30) < 0.03
    # labels are Bernoulli(true_prob), so the means agree
    assert abs(ds.accounts["label"].mean() - ds.truth["true_prob"].mean()) < 0.02
    assert set(ds.accounts["split"]) == {"train", "val", "test"}
    assert (ds.accounts["account_id"] == np.arange(len(ds.accounts))).all()


def test_tabular_ground_truth_is_recoverable(ds):
    """A plain logistic model should get the signs of the planted linear effects right,
    and give distractors/noise much smaller weights than the informative features."""
    X = ds.accounts[FEATURES].copy()
    X["posts_per_day"] = np.log(X["posts_per_day"])
    X["posts_per_week"] = np.log(X["posts_per_week"])
    X = (X - X.mean()) / X.std()
    coef = pd.Series(
        LogisticRegression(C=1e6, max_iter=2000).fit(X, ds.accounts["label"]).coef_[0],
        index=FEATURES,
    )
    assert coef["reply_fraction"] < -0.5
    assert coef["url_fraction"] > 0.2
    assert coef[["posts_per_day", "posts_per_week"]].sum() > 0.8  # credit split by proxy
    for noisy in ["noise_1", "noise_2", "noise_3", "follower_following_ratio", "hashtag_rate"]:
        assert abs(coef[noisy]) < 0.25, noisy


def test_true_logit_terms_sum(ds):
    term_cols = [c for c in ds.truth.columns if c.startswith("term_")]
    intercept = ds.ground_truth["tabular"]["intercept"]
    recon = ds.truth[term_cols].sum(axis=1) + intercept
    assert np.allclose(recon, ds.truth["true_logit"])


def test_text_markers_and_spurious_shift(ds):
    posts = ds.posts.merge(ds.posts_truth, on="post_id").merge(
        ds.accounts[["account_id", "label", "split"]], on="account_id"
    )
    has_marker = posts["planted_markers"] != ""
    assert has_marker[posts.label == 1].mean() > 5 * has_marker[posts.label == 0].mean()
    spur = posts.groupby(["split", "label"])["has_spurious_token"].mean()
    assert spur[("train", 1)] > 0.55
    assert spur[("test", 1)] < 0.2
    assert abs(spur[("test", 0)] - spur[("train", 0)]) < 0.05


def test_graph_rings_are_dense_and_only_trolls(ds):
    t = ds.truth
    assert (t.loc[t.ring_id >= 0, "kind"] == "troll").all()
    assert (t.loc[t.kind == "troll", "ring_id"] >= 0).all()
    is_ring = (ds.edges_truth.edge_type == "ring").to_numpy()
    ring_edges = ds.edges[is_ring]
    assert (t.ring_id.to_numpy()[ring_edges.source] == t.ring_id.to_numpy()[ring_edges.target]).all()
    sizes = t[t.ring_id >= 0].groupby("ring_id").size()
    density = len(ring_edges) / (sizes * (sizes - 1) / 2).sum()
    assert 0.15 < density < 0.35
    bg_density = (ds.edges_truth.edge_type == "background").sum() / (len(t) * (len(t) - 1) / 2)
    assert density > 50 * bg_density
    assert (ds.edges.source != ds.edges.target).all()


def test_sequences_masks_mark_planted_bursts(ds):
    counts, masks = ds.seq_counts, ds.seq_masks
    kinds = ds.truth["kind"].to_numpy()
    assert not masks[kinds == "human"].any()
    for k in ("bot", "troll"):
        m = masks[kinds == k]
        c = counts[kinds == k]
        assert c[m].mean() > 2 * c[~m].mean()
    # bot bursts are periodic
    i = int(np.flatnonzero(kinds == "bot")[0])
    hours = np.flatnonzero(masks[i])
    assert len(set(np.diff(hours))) == 1


def test_style_pairs(ds):
    s = ds.style.set_index("account_id")
    inf = ds.ground_truth["style"]["informative_columns"]
    noise = [c for c in s.columns if c not in inf]
    pairs = ds.pairs
    assert pairs["same_author"].mean() == pytest.approx(0.5, abs=0.01)
    a = s.loc[pairs.account_id_a]
    b = s.loc[pairs.account_id_b]
    pos = pairs["same_author"].to_numpy() == 1
    d_inf = ((a[inf].to_numpy() - b[inf].to_numpy()) ** 2).sum(axis=1)
    d_noise = ((a[noise].to_numpy() - b[noise].to_numpy()) ** 2).sum(axis=1)
    assert d_inf[pos].mean() < 0.5 * d_inf[~pos].mean()
    assert abs(d_noise[pos].mean() / d_noise[~pos].mean() - 1) < 0.15
    # pairs never straddle splits
    sp = ds.accounts.set_index("account_id")["split"]
    assert (sp.loc[pairs.account_id_a].to_numpy() == sp.loc[pairs.account_id_b].to_numpy()).all()
    assert (sp.loc[pairs.account_id_a].to_numpy() == pairs["split"].to_numpy()).all()


def test_write_roundtrip(ds, tmp_path):
    ds.write(tmp_path)
    for f in [
        "accounts.csv", "posts.csv", "edges.csv", "style_features.csv", "pairs.csv",
        "sequences.npz", "truth/accounts_truth.csv", "truth/sequence_masks.npz",
        "truth/ground_truth.json", "truth/posts_truth.csv", "truth/edges_truth.csv",
    ]:
        assert (tmp_path / f).exists(), f
    acc = pd.read_csv(tmp_path / "accounts.csv")
    assert "kind" not in acc.columns and "ring_id" not in acc.columns  # no truth leakage
