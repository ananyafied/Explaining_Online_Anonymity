"""Tabular account features with a fully known label-generating function.

The label is drawn as Bernoulli(sigmoid(true_logit)), where true_logit is an explicit
sum of terms over the features. Anything an explainability method reports can therefore
be checked against the real function (see `true_logit_terms`).

Feature roles (also written to ground_truth.json):
  informative_linear   : posts_per_day (+), reply_fraction (-), url_fraction (+)
  interaction          : burstiness x account_age_days (bursty AND young => inauthentic)
  nonlinear_threshold  : night_activity_frac (only matters above a threshold)
  redundant_proxy      : posts_per_week (a noisy copy of posts_per_day, no direct effect)
  distractor           : follower_following_ratio, hashtag_rate (realistic but no effect)
  noise                : noise_1..noise_3 (pure N(0,1))
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "account_age_days",
    "posts_per_day",
    "posts_per_week",
    "reply_fraction",
    "burstiness",
    "night_activity_frac",
    "follower_following_ratio",
    "url_fraction",
    "hashtag_rate",
    "noise_1",
    "noise_2",
    "noise_3",
]

FEATURE_ROLES = {
    "posts_per_day": "informative_linear",
    "reply_fraction": "informative_linear",
    "url_fraction": "informative_linear",
    "burstiness": "interaction (with account_age_days)",
    "account_age_days": "interaction (with burstiness)",
    "night_activity_frac": "nonlinear_threshold",
    "posts_per_week": "redundant_proxy (of posts_per_day)",
    "follower_following_ratio": "distractor",
    "hashtag_rate": "distractor",
    "noise_1": "noise",
    "noise_2": "noise",
    "noise_3": "noise",
}

# coefficients of the true logit, on standardized inputs
COEF = {
    "posts": 1.4,  # z(log posts_per_day)
    "reply": -1.2,  # z(reply_fraction)
    "burst_x_young": 0.9,  # z(burstiness) * (-z(log account_age_days))
    "night_hinge": 1.0,  # max(z(night_activity_frac) - HINGE, 0)
    "url": 0.6,  # z(url_fraction)
}
NIGHT_HINGE = 0.5

# which raw feature each standardized quantity comes from
_Z_SOURCES = {
    "posts": ("posts_per_day", np.log),
    "reply": ("reply_fraction", lambda x: x),
    "burst": ("burstiness", lambda x: x),
    "age": ("account_age_days", np.log),
    "night": ("night_activity_frac", lambda x: x),
    "url": ("url_fraction", lambda x: x),
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    z1 = rng.normal(size=n)
    z2 = 0.6 * z1 + 0.8 * rng.normal(size=n)  # burstiness correlated with activity level
    posts_per_day = np.exp(0.9 + 0.9 * z1)
    df = pd.DataFrame(
        {
            "account_age_days": np.clip(np.exp(rng.normal(5.5, 1.0, n)), 1, 4000),
            "posts_per_day": posts_per_day,
            "posts_per_week": 7 * posts_per_day * np.exp(rng.normal(0, 0.1, n)),
            "reply_fraction": sigmoid(rng.normal(-0.5, 1.0, n)),
            "burstiness": np.tanh(0.8 * z2),
            "night_activity_frac": rng.beta(2, 6, n),
            "follower_following_ratio": np.exp(rng.normal(0, 1.0, n)),
            "url_fraction": rng.beta(1.5, 6, n),
            "hashtag_rate": rng.gamma(2.0, 0.5, n),
            "noise_1": rng.normal(size=n),
            "noise_2": rng.normal(size=n),
            "noise_3": rng.normal(size=n),
        }
    )
    return df[FEATURES]


def compute_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for key, (col, fn) in _Z_SOURCES.items():
        v = fn(df[col].to_numpy())
        stats[key] = {"mean": float(v.mean()), "std": float(v.std())}
    return stats


def true_logit_terms(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Per-term contribution to the true logit (without the intercept)."""
    z = {}
    for key, (col, fn) in _Z_SOURCES.items():
        z[key] = (fn(df[col].to_numpy()) - stats[key]["mean"]) / stats[key]["std"]
    return pd.DataFrame(
        {
            "posts": COEF["posts"] * z["posts"],
            "reply": COEF["reply"] * z["reply"],
            "burst_x_young": COEF["burst_x_young"] * z["burst"] * (-z["age"]),
            "night_hinge": COEF["night_hinge"] * np.maximum(z["night"] - NIGHT_HINGE, 0.0),
            "url": COEF["url"] * z["url"],
        },
        index=df.index,
    )


def calibrate_intercept(raw_logit: np.ndarray, prevalence: float) -> float:
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if sigmoid(raw_logit + mid).mean() < prevalence:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def make_labels(df: pd.DataFrame, prevalence: float, rng: np.random.Generator):
    stats = compute_stats(df)
    terms = true_logit_terms(df, stats)
    raw = terms.sum(axis=1).to_numpy()
    intercept = calibrate_intercept(raw, prevalence)
    true_logit = raw + intercept
    true_prob = sigmoid(true_logit)
    label = (rng.random(len(df)) < true_prob).astype(int)
    return label, true_logit, true_prob, {"intercept": intercept, "z_stats": stats}
