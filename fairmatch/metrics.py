"""Ranking quality and fairness for one job's ranked pool.

`ranking` is an array of row positions, best first. Group B is the
internationally trained group; ratios below 1 mean group B is worse off.
"""

import numpy as np


def _discounts(n):
    return 1.0 / np.log2(np.arange(2, n + 2))


def ndcg_at_k(ranking, relevance, k):
    rel = np.asarray(relevance, dtype=float)
    gains = (2 ** rel[ranking[:k]] - 1) * _discounts(min(k, len(ranking)))
    ideal = (2 ** np.sort(rel)[::-1][:k] - 1) * _discounts(min(k, len(rel)))
    return float(gains.sum() / ideal.sum()) if ideal.sum() > 0 else np.nan


def position_exposure(ranking, k):
    """Attention each candidate gets: 1/log2(position + 1) inside the top k, 0 below it."""
    exposure = np.zeros(len(ranking))
    exposure[ranking[:k]] = _discounts(min(k, len(ranking)))
    return exposure


def rate_ratio(in_top, group_b, mask=None):
    """Group B's rate of being in the top, divided by group A's. Pass arrays pooled over many jobs."""
    return _rate_ratio(np.asarray(in_top), np.asarray(group_b), None if mask is None else np.asarray(mask))


def _rate_ratio(in_top, group_b, mask=None):
    mask = np.ones_like(group_b, dtype=bool) if mask is None else mask
    b, a = mask & (group_b == 1), mask & (group_b == 0)
    if not b.any() or not a.any():
        return np.nan
    rate_a = in_top[a].mean()
    return float(in_top[b].mean() / rate_a) if rate_a > 0 else np.nan


def top_k_mask(ranking, k):
    mask = np.zeros(len(ranking), dtype=bool)
    mask[ranking[:k]] = True
    return mask


def selection_ratio(ranking, group_b, k):
    """Group B's chance of reaching the top k, divided by group A's (1 = parity)."""
    return _rate_ratio(top_k_mask(ranking, k), group_b)


def equal_opportunity_ratio(ranking, group_b, qualified, k):
    """The same ratio, counting only candidates who are truly qualified."""
    return _rate_ratio(top_k_mask(ranking, k), group_b, qualified == 1)


def exposure_ratio(ranking, group_b, k):
    """Average position-discounted attention per member of B, divided by the same for A."""
    exposure = np.zeros(len(ranking))
    exposure[ranking[:k]] = _discounts(min(k, len(ranking)))
    b, a = group_b == 1, group_b == 0
    if not b.any() or not a.any() or exposure[a].mean() == 0:
        return np.nan
    return float(exposure[b].mean() / exposure[a].mean())
