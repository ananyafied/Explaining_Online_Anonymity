"""Configuration for the synthetic anonymous-account dataset."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SynthConfig:
    # --- population -------------------------------------------------------
    n_accounts: int = 5000
    seed: int = 0
    target_prevalence: float = 0.30  # share of inauthentic accounts (bot or troll)
    troll_share: float = 0.40  # among inauthentic accounts; the rest are bots
    split_fractions: tuple = (0.6, 0.2, 0.2)  # train / val / test

    # --- text -------------------------------------------------------------
    posts_per_account: int = 6
    marker_prob_inauthentic: float = 0.35  # per post, class-specific marker token
    marker_prob_background: float = 0.03  # same marker tokens, on authentic accounts
    # A spurious token ("#trending") that is correlated with inauthentic accounts in
    # train/val but not in test. Lets you test whether explanations expose shortcuts.
    spurious_prob_train_inauth: float = 0.70
    spurious_prob_test_inauth: float = 0.10
    spurious_prob_authentic: float = 0.10

    # --- graph ------------------------------------------------------------
    ring_size: int = 25  # coordinated troll rings
    ring_edge_prob: float = 0.25
    background_mean_degree: float = 6.0
    bot_follow_edges: int = 8  # bots follow this many authentic accounts

    # --- sequences --------------------------------------------------------
    n_hours: int = 168  # one week of hourly post counts

    # --- stylometry / account linking ------------------------------------
    linked_account_frac: float = 0.15  # accounts that are a second account of an author
    style_dims_informative: int = 6
    style_dims_noise: int = 10
    style_account_noise: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)
