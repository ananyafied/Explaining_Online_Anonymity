"""Generate the full synthetic dataset.

    python -m anonsynth --n-accounts 5000 --seed 0 --out data/synthetic
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import graph, sequences, style, tabular, text
from .config import SynthConfig


@dataclass
class SyntheticDataset:
    config: SynthConfig
    accounts: pd.DataFrame  # account_id, features, label, split
    truth: pd.DataFrame  # account_id, kind, ring_id, author_id, true_logit, true_prob, ...
    posts: pd.DataFrame  # post_id, account_id, text
    posts_truth: pd.DataFrame  # post_id, planted_markers, has_spurious_token  (answer key)
    edges: pd.DataFrame  # source, target (undirected, shuffled)
    edges_truth: pd.DataFrame  # edge_type, same row order as edges  (answer key)
    seq_counts: np.ndarray  # (n_accounts, n_hours)
    seq_masks: np.ndarray  # (n_accounts, n_hours) bool
    style: pd.DataFrame  # account_id + style_XX
    pairs: pd.DataFrame
    ground_truth: dict

    def write(self, out_dir) -> Path:
        out = Path(out_dir)
        (out / "truth").mkdir(parents=True, exist_ok=True)
        self.accounts.to_csv(out / "accounts.csv", index=False)
        self.posts.to_csv(out / "posts.csv", index=False)
        self.edges.to_csv(out / "edges.csv", index=False)
        self.style.to_csv(out / "style_features.csv", index=False)
        self.pairs.to_csv(out / "pairs.csv", index=False)
        np.savez_compressed(
            out / "sequences.npz",
            account_id=self.accounts["account_id"].to_numpy(),
            counts=self.seq_counts,
        )
        self.truth.to_csv(out / "truth" / "accounts_truth.csv", index=False)
        self.posts_truth.to_csv(out / "truth" / "posts_truth.csv", index=False)
        self.edges_truth.to_csv(out / "truth" / "edges_truth.csv", index=False)
        np.savez_compressed(out / "truth" / "sequence_masks.npz", masks=self.seq_masks)
        with open(out / "truth" / "ground_truth.json", "w") as f:
            json.dump(self.ground_truth, f, indent=2)
        return out


def _stratified_split(label: np.ndarray, fractions, rng) -> np.ndarray:
    split = np.empty(len(label), dtype=object)
    names = ["train", "val", "test"]
    for y in np.unique(label):
        idx = rng.permutation(np.flatnonzero(label == y))
        cuts = (np.cumsum(fractions)[:-1] * len(idx)).astype(int)
        for name, part in zip(names, np.split(idx, cuts)):
            split[part] = name
    return split


def generate(cfg: SynthConfig | None = None) -> SyntheticDataset:
    cfg = cfg or SynthConfig()
    n = cfg.n_accounts
    seeds = np.random.SeedSequence(cfg.seed).spawn(8)
    r_tab, r_kind, r_split, r_text, r_graph, r_seq, r_style, r_pair = (
        np.random.default_rng(s) for s in seeds
    )

    # 1. tabular features and labels (label is a known function of the features)
    feats = tabular.make_features(n, r_tab)
    label, true_logit, true_prob, tab_info = tabular.make_labels(
        feats, cfg.target_prevalence, r_tab
    )
    terms = tabular.true_logit_terms(feats, tab_info["z_stats"])

    # 2. kind: human / bot / troll
    kind = np.zeros(n, dtype=int)
    inauth = np.flatnonzero(label == 1)
    is_troll = r_kind.random(len(inauth)) < cfg.troll_share
    kind[inauth] = np.where(is_troll, text.KIND_TROLL, text.KIND_BOT)

    # 3. splits, then link some accounts to a shared author (copies partner's split)
    split = _stratified_split(label, cfg.split_fractions, r_split)
    author_id, split = style.link_accounts(split, cfg, r_style)

    # 4. modalities
    account_ids = np.arange(n)
    posts, posts_truth = text.make_posts(account_ids, kind, split, cfg, r_text)
    ring_id = graph.assign_rings(kind, cfg.ring_size, r_graph)
    edges, edges_truth = graph.make_edges(kind, ring_id, cfg, r_graph)
    counts, masks, seq_info = sequences.make_sequences(
        kind, feats["posts_per_day"].to_numpy(), ring_id, cfg, r_seq
    )
    style_df, informative_style_cols = style.make_style(author_id, cfg, r_style)
    pairs = style.make_pairs(author_id, split, r_pair)

    accounts = feats.copy()
    accounts.insert(0, "account_id", account_ids)
    accounts["label"] = label
    accounts["split"] = split
    style_df.insert(0, "account_id", account_ids)

    truth = pd.DataFrame(
        {
            "account_id": account_ids,
            "kind": pd.Series(kind).map({0: "human", 1: "bot", 2: "troll"}),
            "ring_id": ring_id,
            "author_id": author_id,
            "true_logit": true_logit,
            "true_prob": true_prob,
        }
    )
    for c in terms.columns:
        truth[f"term_{c}"] = terms[c].to_numpy()

    ground_truth = {
        "config": cfg.to_dict(),
        "tabular": {
            "intercept": tab_info["intercept"],
            "z_stats": tab_info["z_stats"],
            "coefficients": tabular.COEF,
            "night_hinge": tabular.NIGHT_HINGE,
            "feature_roles": tabular.FEATURE_ROLES,
            "formula": (
                "logit = intercept + 1.4*z(log posts_per_day) - 1.2*z(reply_fraction)"
                " + 0.9*z(burstiness)*(-z(log account_age_days))"
                " + 1.0*max(z(night_activity_frac) - 0.5, 0) + 0.6*z(url_fraction)"
            ),
        },
        "text": {
            "bot_markers": text.BOT_MARKERS,
            "troll_markers": text.TROLL_MARKERS,
            "spurious_token": text.SPURIOUS_TOKEN,
        },
        "graph": {
            "n_rings": int(ring_id.max() + 1) if (ring_id >= 0).any() else 0,
            "note": "ring_id in truth/accounts_truth.csv; truth/edges_truth.csv gives edge_type"
            " (ring/bot_follow/background) in the same row order as edges.csv",
        },
        "sequences": seq_info,
        "style": {"informative_columns": informative_style_cols},
        "notes": [
            "Modalities are conditionally independent given kind (human/bot/troll), except that"
            " posts_per_day sets the baseline rate of each posting sequence.",
            "Only the tabular label is generated from the features; text/graph/sequence/style"
            " signals are generated from kind.",
        ],
    }

    return SyntheticDataset(
        config=cfg,
        accounts=accounts,
        truth=truth,
        posts=posts,
        posts_truth=posts_truth,
        edges=edges,
        edges_truth=edges_truth,
        seq_counts=counts,
        seq_masks=masks,
        style=style_df,
        pairs=pairs,
        ground_truth=ground_truth,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-accounts", type=int, default=SynthConfig.n_accounts)
    p.add_argument("--seed", type=int, default=SynthConfig.seed)
    p.add_argument("--out", default="data/synthetic")
    a = p.parse_args()
    ds = generate(SynthConfig(n_accounts=a.n_accounts, seed=a.seed))
    out = ds.write(a.out)
    acc = ds.accounts
    print(
        f"wrote {out}: {len(acc)} accounts ({acc['label'].mean():.1%} inauthentic), "
        f"{len(ds.posts)} posts, {len(ds.edges)} edges, {len(ds.pairs)} pairs"
    )


if __name__ == "__main__":
    main()
