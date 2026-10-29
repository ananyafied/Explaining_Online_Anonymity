"""Interaction graph (undirected): sparse background, bot follow edges, dense troll rings."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .text import KIND_BOT, KIND_HUMAN, KIND_TROLL


def assign_rings(kind: np.ndarray, ring_size: int, rng: np.random.Generator) -> np.ndarray:
    """ring_id per account (-1 if not in a ring). Only trolls are in rings."""
    ring_id = np.full(len(kind), -1, dtype=int)
    trolls = np.flatnonzero(kind == KIND_TROLL)
    rng.shuffle(trolls)
    if len(trolls) == 0:
        return ring_id
    n_rings = max(1, int(round(len(trolls) / ring_size)))
    for r, chunk in enumerate(np.array_split(trolls, n_rings)):
        ring_id[chunk] = r
    return ring_id


def make_edges(kind: np.ndarray, ring_id: np.ndarray, cfg, rng: np.random.Generator) -> pd.DataFrame:
    n = len(kind)
    parts = []

    # planted ring edges (undirected, stored once)
    for r in np.unique(ring_id[ring_id >= 0]):
        members = np.flatnonzero(ring_id == r)
        iu, ju = np.triu_indices(len(members), k=1)
        keep = rng.random(len(iu)) < cfg.ring_edge_prob
        parts.append(
            pd.DataFrame(
                {"source": members[iu[keep]], "target": members[ju[keep]], "edge_type": "ring"}
            )
        )

    # bots follow authentic accounts
    humans = np.flatnonzero(kind == KIND_HUMAN)
    for b in np.flatnonzero(kind == KIND_BOT):
        targets = rng.choice(humans, size=min(cfg.bot_follow_edges, len(humans)), replace=False)
        parts.append(pd.DataFrame({"source": b, "target": targets, "edge_type": "bot_follow"}))

    # sparse background
    m = int(n * cfg.background_mean_degree / 2)
    u = rng.integers(0, n, size=m)
    v = rng.integers(0, n, size=m)
    ok = u != v
    parts.append(pd.DataFrame({"source": u[ok], "target": v[ok], "edge_type": "background"}))

    edges = pd.concat(parts, ignore_index=True)
    lo = np.minimum(edges["source"], edges["target"])
    hi = np.maximum(edges["source"], edges["target"])
    # the graph is undirected: de-duplicate on the unordered pair
    edges = edges.loc[~pd.Series(lo * n + hi).duplicated().to_numpy()]
    # Edges were generated in blocks by type and ring edges always have source < target, so
    # row order and direction would reveal the edge type. Shuffle rows and randomize direction.
    edges = edges.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    swap = rng.random(len(edges)) < 0.5
    out = pd.DataFrame(
        {
            "source": np.where(swap, edges["target"], edges["source"]),
            "target": np.where(swap, edges["source"], edges["target"]),
        }
    )
    # model-facing edge list vs. answer key (edge_type, same row order)
    return out, edges[["edge_type"]]
