"""Hourly posting-count sequences with planted, maskable temporal patterns.

  humans : circadian rhythm (account-specific peak hour), Poisson counts
  bots   : flat low-level background + periodic bursts every 6/8/12 hours
  trolls : circadian background + two 3-hour coordinated bursts shared by the whole ring

`masks` marks exactly the hours that carry the planted pattern, so saliency or Grad-CAM
heat maps can be scored against it.
"""
from __future__ import annotations

import numpy as np

from .text import KIND_BOT, KIND_TROLL

BOT_PERIODS = (6, 8, 12)
WINDOW_LEN = 3


def make_sequences(kind, posts_per_day, ring_id, cfg, rng: np.random.Generator):
    n, T = len(kind), cfg.n_hours
    hours = np.arange(T)
    hod = hours % 24
    base = np.clip(np.asarray(posts_per_day) / 24.0, 0.02, 3.0)
    peak = rng.normal(19, 3, size=n)

    circ = 1 + 0.8 * np.cos(2 * np.pi * (hod[None, :] - peak[:, None]) / 24.0)
    rate = base[:, None] * circ
    rate[kind == KIND_BOT] = 0.3 * base[kind == KIND_BOT, None]  # flat

    counts = rng.poisson(rate)
    masks = np.zeros((n, T), dtype=bool)

    bot_info = {}
    for i in np.flatnonzero(kind == KIND_BOT):
        period = int(rng.choice(BOT_PERIODS))
        phase = int(rng.integers(0, period))
        burst_hours = hours[(hours - phase) % period == 0]
        counts[i, burst_hours] += rng.poisson(4, size=len(burst_hours))
        masks[i, burst_hours] = True
        bot_info[int(i)] = {"period": period, "phase": phase}

    ring_windows = {}
    for r in np.unique(ring_id[ring_id >= 0]):
        starts = rng.choice(T - WINDOW_LEN, size=2, replace=False)
        ring_windows[int(r)] = [int(s) for s in starts]
    for i in np.flatnonzero(kind == KIND_TROLL):
        for s in ring_windows.get(int(ring_id[i]), []):
            sl = slice(s, s + WINDOW_LEN)
            counts[i, sl] += rng.poisson(5, size=WINDOW_LEN)
            masks[i, sl] = True

    info = {
        "bot_periods": bot_info,
        "ring_windows": ring_windows,
        "window_len": WINDOW_LEN,
    }
    return counts.astype(np.int16), masks, info
