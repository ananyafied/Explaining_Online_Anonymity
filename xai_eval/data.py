"""Load the synthetic data the same way step 1 did, plus the answer key.

`load_data()` returns a `Data` object. Everything a model sees (`X_df`, `X_seq`, `post_tokens`,
`EDGE_INDEX`, ...) is built exactly as in notebooks/01_fit_models.ipynb; everything in `truth_*`
comes from `truth/` and is used ONLY to score explanations, never as a model input.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def find_repo_root() -> Path:
    p = Path.cwd()
    for cand in (p, *p.parents):
        if (cand / "xai_eval").is_dir() and (cand / "anonsynth").is_dir():
            return cand
    raise FileNotFoundError("Run from inside the repository (it must contain xai_eval/ and anonsynth/).")


# Which profile feature(s) each planted logit term belongs to. The interaction is credited to BOTH
# of its inputs; `posts_per_week` (a noisy copy of posts_per_day) has no direct effect.
TERM_TO_FEATURES = {
    "term_posts": ["posts_per_day"],
    "term_reply": ["reply_fraction"],
    "term_burst_x_young": ["burstiness", "account_age_days"],
    "term_night_hinge": ["night_activity_frac"],
    "term_url": ["url_fraction"],
}
INFORMATIVE = sorted({f for fs in TERM_TO_FEATURES.values() for f in fs})  # the 6 truly used features


@dataclass
class Data:
    root: Path
    data_dir: Path
    accounts: pd.DataFrame
    posts: pd.DataFrame
    edges: pd.DataFrame
    y: np.ndarray
    split: np.ndarray
    is_tr: np.ndarray
    is_va: np.ndarray
    is_te: np.ndarray
    FEATURES: list
    X_df: pd.DataFrame
    X_tab: np.ndarray
    scaler_tab: StandardScaler
    X_tab_std: np.ndarray
    counts: np.ndarray
    X_seq: np.ndarray
    seq_mu: float
    seq_sd: float
    post_tokens: list
    post_account: np.ndarray
    account_tokens: list
    src: np.ndarray
    dst: np.ndarray
    EDGE_INDEX: torch.Tensor
    # ---- answer key (never given to a model)
    truth: pd.DataFrame
    posts_truth: pd.DataFrame
    edges_truth: pd.DataFrame
    seq_masks: np.ndarray
    ground_truth: dict
    bot_markers: list = field(default_factory=list)
    troll_markers: list = field(default_factory=list)
    spurious_token: str = "#trending"

    @property
    def n(self):
        return len(self.accounts)

    @property
    def kind(self):
        return self.truth["kind"].to_numpy()

    @property
    def marker_set(self):
        return set(self.bot_markers) | set(self.troll_markers)

    def sample_accounts(self, per_kind=50, split="test", seed=0, min_degree=0):
        """Fixed sample of accounts (same for every model): `per_kind` each of human / bot / troll.

        min_degree > 0 skips accounts with fewer edges (the graph notebook needs at least 2 so that
        "delete the top edges vs random edges" is defined).
        """
        rng = np.random.default_rng(seed)
        ok = self.split == split
        if min_degree > 0:
            ok = ok & (np.bincount(np.r_[self.src, self.dst], minlength=self.n) >= min_degree)
        out = []
        for k in ("human", "bot", "troll"):
            idx = np.flatnonzero(ok & (self.kind == k))
            out.append(rng.choice(idx, size=min(per_kind, len(idx)), replace=False))
        return np.concatenate(out)


def load_data(root: Path | None = None, data_dir: Path | None = None) -> Data:
    root = Path(root) if root else find_repo_root()
    data_dir = Path(data_dir) if data_dir else root / "data" / "synthetic"
    need = ["accounts.csv", "posts.csv", "edges.csv", "sequences.npz", "truth/accounts_truth.csv"]
    missing = [f for f in need if not (data_dir / f).exists()]
    assert not missing, f"Missing {missing} in {data_dir}. Generate with: python -m anonsynth --n-accounts 5000 --seed 0 --out data/synthetic"

    accounts = pd.read_csv(data_dir / "accounts.csv")
    posts = pd.read_csv(data_dir / "posts.csv")
    edges = pd.read_csv(data_dir / "edges.csv")
    seq = np.load(data_dir / "sequences.npz")
    n = len(accounts)
    assert (accounts["account_id"].to_numpy() == np.arange(n)).all()
    assert (seq["account_id"] == np.arange(n)).all()

    y = accounts["label"].to_numpy().astype(int)
    split = accounts["split"].to_numpy()
    is_tr, is_va, is_te = (split == "train"), (split == "val"), (split == "test")

    FEATURES = [c for c in accounts.columns if c not in ("account_id", "label", "split")]
    X_df = accounts[FEATURES]
    X_tab = X_df.to_numpy(dtype=float)
    scaler_tab = StandardScaler().fit(X_tab[is_tr])
    X_tab_std = scaler_tab.transform(X_tab)

    counts = seq["counts"].astype(float)
    lg = np.log1p(counts)
    seq_mu, seq_sd = lg[is_tr].mean(), lg[is_tr].std()
    X_seq = (lg - seq_mu) / seq_sd

    post_tokens = [t.split() for t in posts["text"]]
    post_account = posts["account_id"].to_numpy()
    account_tokens = [[] for _ in range(n)]
    for toks, a in zip(post_tokens, post_account):
        account_tokens[a].extend(toks)

    src, dst = edges["source"].to_numpy(), edges["target"].to_numpy()
    EDGE_INDEX = torch.as_tensor(np.vstack([np.r_[src, dst], np.r_[dst, src]]), dtype=torch.long)

    gt = json.loads((data_dir / "truth" / "ground_truth.json").read_text())
    return Data(
        root=root, data_dir=data_dir, accounts=accounts, posts=posts, edges=edges, y=y, split=split,
        is_tr=is_tr, is_va=is_va, is_te=is_te, FEATURES=FEATURES, X_df=X_df, X_tab=X_tab,
        scaler_tab=scaler_tab, X_tab_std=X_tab_std, counts=counts, X_seq=X_seq, seq_mu=seq_mu,
        seq_sd=seq_sd, post_tokens=post_tokens, post_account=post_account,
        account_tokens=account_tokens, src=src, dst=dst, EDGE_INDEX=EDGE_INDEX,
        truth=pd.read_csv(data_dir / "truth" / "accounts_truth.csv"),
        posts_truth=pd.read_csv(data_dir / "truth" / "posts_truth.csv"),
        edges_truth=pd.read_csv(data_dir / "truth" / "edges_truth.csv"),
        seq_masks=np.load(data_dir / "truth" / "sequence_masks.npz")["masks"].astype(bool),
        ground_truth=gt,
        bot_markers=gt["text"]["bot_markers"], troll_markers=gt["text"]["troll_markers"],
        spurious_token=gt["text"]["spurious_token"],
    )


def true_feature_importance(D: Data, idx=None) -> pd.Series:
    """Planted importance of each profile feature = mean |its logit term| (interaction credited to both inputs)."""
    t = D.truth if idx is None else D.truth.iloc[idx]
    imp = pd.Series(0.0, index=D.FEATURES)
    for term, feats in TERM_TO_FEATURES.items():
        for f in feats:
            imp[f] += float(np.abs(t[term]).mean())
    return imp


def true_feature_contributions(D: Data, idx) -> np.ndarray:
    """(len(idx), 12) planted |logit contribution| of each profile feature for the given accounts.

    The interaction term is credited to both of its inputs. Decoys, noise and the proxy get 0.
    """
    t = D.truth.iloc[idx]
    out = np.zeros((len(t), len(D.FEATURES)))
    for term, feats in TERM_TO_FEATURES.items():
        for f in feats:
            out[:, D.FEATURES.index(f)] += np.abs(t[term].to_numpy())
    return out
