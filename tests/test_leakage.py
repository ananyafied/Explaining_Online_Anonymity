"""Checks that the files a model is allowed to see carry no information about the answer
beyond the signals that were planted on purpose."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from anonsynth import SynthConfig, generate
from anonsynth.tabular import FEATURES

MODEL_FACING_COLUMNS = {
    "accounts.csv": set(FEATURES) | {"account_id", "label", "split"},
    "posts.csv": {"post_id", "account_id", "text"},
    "edges.csv": {"source", "target"},
    "style_features.csv": {"account_id"} | {f"style_{i:02d}" for i in range(16)},
    "pairs.csv": {"account_id_a", "account_id_b", "same_author", "split"},
}


def strength(y, score):
    """AUC folded around 0.5 (0.5 = no information, 1.0 = perfectly informative)."""
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


@pytest.fixture(scope="module")
def ds():
    return generate(SynthConfig(n_accounts=4000, seed=1))


def test_model_facing_files_contain_only_allowed_columns(ds, tmp_path):
    ds.write(tmp_path)
    for fname, allowed in MODEL_FACING_COLUMNS.items():
        assert set(pd.read_csv(tmp_path / fname, nrows=5).columns) == allowed, fname
    assert set(np.load(tmp_path / "sequences.npz").files) == {"account_id", "counts"}
    assert {p.name for p in (tmp_path / "truth").iterdir()} >= {
        "accounts_truth.csv", "posts_truth.csv", "edges_truth.csv",
        "sequence_masks.npz", "ground_truth.json",
    }


def test_edge_row_order_and_direction_carry_no_information(ds):
    position = np.arange(len(ds.edges))
    etype = ds.edges_truth["edge_type"].to_numpy()
    for t in ("ring", "bot_follow"):
        assert strength(etype == t, position) < 0.55, f"row order reveals {t} edges"
        sel = etype == t
        frac = (ds.edges.source < ds.edges.target).to_numpy()[sel].mean()
        assert 0.45 < frac < 0.55, f"edge direction reveals {t} edges"
    assert len(ds.edges) == len(ds.edges_truth)


def test_pair_order_carries_no_information(ds):
    p = ds.pairs
    assert strength(p.same_author, (p.account_id_a < p.account_id_b).astype(int)) < 0.55
    assert strength(p.same_author, p.account_id_a) < 0.55
    assert strength(p.same_author, np.arange(len(p))) < 0.55


def test_ids_and_row_order_carry_no_information(ds):
    acc = ds.accounts
    y = acc["label"]
    assert strength(y, acc["account_id"]) < 0.55
    assert strength(y, np.arange(len(acc))) < 0.55
    posts = ds.posts.merge(acc[["account_id", "label"]], on="account_id")
    assert strength(posts["label"], posts["post_id"]) < 0.55
    assert strength(posts["label"], posts["account_id"]) < 0.55


def test_post_length_carries_no_information(ds):
    posts = ds.posts.merge(ds.accounts[["account_id", "label"]], on="account_id")
    n_tokens = posts["text"].str.split().str.len()
    assert strength(posts["label"], n_tokens) < 0.52


def test_decoy_and_noise_features_are_uninformative(ds):
    for c in ["follower_following_ratio", "hashtag_rate", "noise_1", "noise_2", "noise_3"]:
        assert strength(ds.accounts["label"], ds.accounts[c]) < 0.54, c


def test_split_label_balance(ds):
    rates = ds.accounts.groupby("split")["label"].mean()
    assert rates.max() - rates.min() < 0.02


def test_posts_truth_and_edges_truth_align(ds):
    assert (ds.posts["post_id"].to_numpy() == ds.posts_truth["post_id"].to_numpy()).all()
    ring = ds.edges_truth["edge_type"].to_numpy() == "ring"
    rid = ds.truth["ring_id"].to_numpy()
    s, t = ds.edges["source"].to_numpy()[ring], ds.edges["target"].to_numpy()[ring]
    assert (rid[s] == rid[t]).all() and (rid[s] >= 0).all()
