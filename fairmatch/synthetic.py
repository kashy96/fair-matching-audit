"""A synthetic healthcare staffing market with a known ground truth.

Real hiring data never tells you how suitable a candidate *actually* was,
only who got shortlisted, which is exactly where bias hides. So the
generator creates both:

* `true_score`: suitability from specialty, experience, licence,
  certifications, availability and location. Group membership plays no part.
* `shortlisted`: what past recruiters decided. They saw the same merit, but
  penalised internationally trained clinicians (`bias`) and over-rewarded
  *local* experience (`local_bias`), which international candidates have less of.

A model trained on `shortlisted` can then be judged against both.
"""

import numpy as np
import pandas as pd

SPECIALTIES = ["emergency", "icu", "paediatrics", "surgery", "cardiology", "oncology", "general", "psychiatry"]
N_REGIONS = 6
HOURS = [16, 24, 32, 40]


def make_candidates(n, share_b, rng):
    group_b = rng.random(n) < share_b  # True = internationally trained
    total_exp = np.clip(rng.gamma(3.0, 3.0, n), 0, 35)
    # same total experience, but international candidates have less of it locally
    local_frac = np.where(group_b, rng.beta(2, 5, n), rng.beta(8, 2, n))
    return pd.DataFrame({
        "cid": np.arange(n),
        "group_b": group_b.astype(int),
        "specialty": rng.integers(0, len(SPECIALTIES), n),
        "total_exp": total_exp,
        "local_exp": total_exp * local_frac,
        "certs": rng.poisson(2.0, n),
        "licence_ok": (rng.random(n) < 0.9).astype(int),
        "hours": rng.choice(HOURS, n),
        "region": rng.integers(0, N_REGIONS, n),
    })


def make_jobs(n, rng):
    return pd.DataFrame({
        "jid": np.arange(n),
        "job_specialty": rng.integers(0, len(SPECIALTIES), n),
        "min_exp": rng.choice([0, 2, 5, 8], n),
        "job_hours": rng.choice(HOURS, n),
        "job_region": rng.integers(0, N_REGIONS, n),
    })


def make_pairs(candidates, jobs, pool_size, rng):
    """Each job gets a pool of applicants, mostly from its own specialty."""
    by_specialty = dict(tuple(candidates.groupby("specialty")))
    pools = []
    for job in jobs.itertuples(index=False):
        same = by_specialty[job.job_specialty]
        n_same = min(int(pool_size * 0.7), len(same))
        other = candidates[candidates["specialty"] != job.job_specialty]
        pick = np.concatenate([
            rng.choice(same.index, n_same, replace=False),
            rng.choice(other.index, pool_size - n_same, replace=False),
        ])
        pools.append(candidates.loc[pick].assign(jid=job.jid))
    pairs = pd.concat(pools, ignore_index=True).merge(jobs, on="jid")
    pairs["spec_match"] = (pairs["specialty"] == pairs["job_specialty"]).astype(int)
    pairs["exp_gap"] = pairs["total_exp"] - pairs["min_exp"]
    pairs["hours_fit"] = -np.abs(pairs["hours"] - pairs["job_hours"]) / 24
    pairs["same_region"] = (pairs["region"] == pairs["job_region"]).astype(int)
    return pairs


def true_score(p, rng):
    merit = (
        2.0 * p["spec_match"]
        + 1.2 * np.tanh(p["exp_gap"] / 5)  # meeting the requirement matters most, then diminishing returns
        + 0.3 * np.minimum(p["certs"], 5)
        + 1.5 * p["licence_ok"]
        + 0.8 * p["hours_fit"]
        + 0.6 * p["same_region"]
    )
    return merit + rng.normal(0, 0.5, len(p))


def shortlist(p, bias, local_bias, rate, rng):
    judged = (
        p["true_score"]
        - bias * p["group_b"]
        + local_bias * np.tanh(p["local_exp"] / 5)
        + rng.normal(0, 0.7, len(p))
    )
    cutoff = judged.groupby(p["jid"]).transform(lambda x: x.quantile(1 - rate))
    return (judged >= cutoff).astype(int)


def graded_relevance(p):
    """0-3 relevance per job from the true score: top 10% = 3, next 15% = 2, next 25% = 1."""
    pct = p.groupby("jid")["true_score"].rank(pct=True)
    return np.select([pct > 0.90, pct > 0.75, pct > 0.50], [3, 2, 1], 0)


def make_world(cfg, seed, bias=None):
    rng = np.random.default_rng(seed)
    w, h = cfg["world"], cfg["history"]
    candidates = make_candidates(w["n_candidates"], w["share_group_b"], rng)
    jobs = make_jobs(w["n_jobs"], rng)
    pairs = make_pairs(candidates, jobs, w["pool_size"], rng)
    pairs["true_score"] = true_score(pairs, rng)
    pairs["relevance"] = graded_relevance(pairs)
    pairs["qualified"] = (pairs["relevance"] >= 2).astype(int)
    pairs["shortlisted"] = shortlist(
        pairs, h["bias"] if bias is None else bias, h["local_bias"], h["shortlist_rate"], rng
    )
    return pairs
