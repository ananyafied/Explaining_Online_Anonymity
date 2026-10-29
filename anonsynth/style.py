"""Stylometric vectors and account-pair data for authorship / account-linking models.

Each author has a hidden style signature. Some authors run two or more accounts
(sockpuppets). Only `style_dims_informative` of the stylometric columns carry author
information; the rest are noise. Column order is shuffled so position gives nothing away.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def link_accounts(split: np.ndarray, cfg, rng: np.random.Generator):
    """Assign author_id; linked accounts copy their partner's author and split."""
    n = len(split)
    author_id = np.arange(n)
    split = split.copy()
    n_linked = int(cfg.linked_account_frac * n)
    chosen = rng.choice(n, size=n_linked, replace=False)
    rest = np.setdiff1d(np.arange(n), chosen)
    partners = rng.choice(rest, size=n_linked, replace=True)
    author_id[chosen] = author_id[partners]
    split[chosen] = split[partners]
    return author_id, split


def make_style(author_id: np.ndarray, cfg, rng: np.random.Generator):
    n = len(author_id)
    d_inf, d_noise = cfg.style_dims_informative, cfg.style_dims_noise
    signature = rng.normal(size=(n, d_inf))  # indexed by author_id
    informative = signature[author_id] + cfg.style_account_noise * rng.normal(size=(n, d_inf))
    noise = rng.normal(size=(n, d_noise))
    X = np.hstack([informative, noise])
    perm = rng.permutation(d_inf + d_noise)
    X = X[:, perm]
    informative_cols = sorted(int(np.flatnonzero(perm == j)[0]) for j in range(d_inf))
    cols = [f"style_{i:02d}" for i in range(d_inf + d_noise)]
    return pd.DataFrame(X, columns=cols), [cols[i] for i in informative_cols]


def make_pairs(author_id: np.ndarray, split: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    n = len(author_id)
    rows = []
    # positives: every pair of accounts sharing an author
    order = np.argsort(author_id, kind="stable")
    sorted_auth = author_id[order]
    bounds = np.flatnonzero(np.diff(sorted_auth)) + 1
    for grp in np.split(order, bounds):
        for a in range(len(grp)):
            for b in range(a + 1, len(grp)):
                rows.append((int(grp[a]), int(grp[b]), 1, split[grp[a]]))
    n_pos = len(rows)
    # negatives: same number, different author, same split
    neg = []
    for sp in np.unique(split):
        idx = np.flatnonzero(split == sp)
        n_sp = sum(1 for r in rows if r[3] == sp)
        got = 0
        while got < n_sp:
            a = rng.choice(idx, size=n_sp * 2)
            b = rng.choice(idx, size=n_sp * 2)
            ok = author_id[a] != author_id[b]
            for x, y in zip(a[ok], b[ok]):
                neg.append((int(x), int(y), 0, sp))
                got += 1
                if got >= n_sp:
                    break
    rows.extend(neg)
    pairs = pd.DataFrame(rows, columns=["account_id_a", "account_id_b", "same_author", "split"])
    # positives were built with id_a < id_b; randomize the order so it carries no information
    swap = rng.random(len(pairs)) < 0.5
    a = np.where(swap, pairs["account_id_b"], pairs["account_id_a"])
    b = np.where(swap, pairs["account_id_a"], pairs["account_id_b"])
    pairs["account_id_a"], pairs["account_id_b"] = a, b
    return pairs.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
