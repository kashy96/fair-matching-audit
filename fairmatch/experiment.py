"""Run the audit.

    python -m fairmatch.experiment           # main comparison + lambda sweep + bias sweep
    python -m fairmatch.experiment --quick   # one seed, fewer epochs, for a smoke test

Writes results/runs.csv (one row per seed x variant x bias level) and then
calls the report.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import metrics, report
from .model import FEATURES, UNAWARE, train
from .rerank import proportional_rerank
from .synthetic import make_world


def split_jobs(pairs, test_share, seed):
    jobs = pairs["jid"].unique()
    rng = np.random.default_rng(seed + 1000)
    test = set(rng.choice(jobs, int(len(jobs) * test_share), replace=False))
    is_test = pairs["jid"].isin(test)
    return pairs[~is_test], pairs[is_test]


def evaluate(test, scores, k, rerank=False):
    """Rank each test job's pool by `scores`.

    NDCG is averaged over jobs. The fairness ratios are computed on counts
    pooled over all test jobs: each pool holds only about a dozen group-B
    applicants, and averaging many noisy per-job ratios biases the result
    upwards.
    """
    ndcg_true, ndcg_hist = [], []
    in_top, groups, qualified, exposure = [], [], [], []
    for _, g in test.assign(score=scores).groupby("jid"):
        s = g["score"].to_numpy()
        grp = g["group_b"].to_numpy()
        if rerank:
            ranking = proportional_rerank(s, grp, k, grp.mean())
        else:
            ranking = np.argsort(-s, kind="stable")
        ndcg_true.append(metrics.ndcg_at_k(ranking, g["relevance"].to_numpy(), k))
        ndcg_hist.append(metrics.ndcg_at_k(ranking, g["shortlisted"].to_numpy(), k))
        in_top.append(metrics.top_k_mask(ranking, k))
        groups.append(grp)
        qualified.append(g["qualified"].to_numpy())
        exposure.append(metrics.position_exposure(ranking, k))

    in_top, groups = np.concatenate(in_top), np.concatenate(groups)
    qualified, exposure = np.concatenate(qualified), np.concatenate(exposure)
    return {
        "ndcg_true": float(np.nanmean(ndcg_true)),
        "ndcg_hist": float(np.nanmean(ndcg_hist)),
        "selection_ratio": metrics.rate_ratio(in_top, groups),
        "eo_ratio": metrics.rate_ratio(in_top, groups, qualified == 1),
        "exposure_ratio": float(exposure[groups == 1].mean() / exposure[groups == 0].mean()),
        "share_b_top": float(groups[in_top].mean()),
    }


def run_one(cfg, seed, bias, lambdas, include_references=True):
    pairs = make_world(cfg, seed, bias)
    train_pairs, test = split_jobs(pairs, cfg["split"]["test_share"], seed)
    k, tcfg = cfg["eval"]["k"], cfg["train"]
    results = []

    def add(variant, lam, res):
        results.append({"seed": seed, "bias": bias, "variant": variant, "lambda": lam, **res})

    if include_references:
        # what an ideal recruiter would do, and what past recruiters did
        add("oracle", 0.0, evaluate(test, test["true_score"].to_numpy(), k))
        add("historical", 0.0, evaluate(test, test["shortlisted"].to_numpy() + 1e-6 * test["true_score"].to_numpy(), k))

    score = train(train_pairs, FEATURES, tcfg, seed=seed)
    add("baseline", 0.0, evaluate(test, score(test), k))

    score_unaware = train(train_pairs, UNAWARE, tcfg, seed=seed)
    unaware_scores = score_unaware(test)
    add("unaware", 0.0, evaluate(test, unaware_scores, k))
    add("rerank", 0.0, evaluate(test, unaware_scores, k, rerank=True))

    for lam in lambdas:
        if lam == 0:
            continue
        score_fair = train(train_pairs, UNAWARE, tcfg, lam=lam, seed=seed)
        add("penalty", lam, evaluate(test, score_fair(test), k))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.quick:
        cfg["seeds"] = cfg["seeds"][:1]
        cfg["train"]["epochs"] = 3
        cfg["bias_sweep"] = cfg["bias_sweep"][:2]

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    main_bias = cfg["history"]["bias"]
    rows = []
    start = time.time()
    for seed in cfg["seeds"]:
        print(f"seed {seed}: main setting (bias {main_bias})")
        rows += run_one(cfg, seed, main_bias, cfg["fairness"]["lambdas"])
        for bias in cfg["bias_sweep"]:
            if bias == main_bias:
                continue
            print(f"seed {seed}: bias {bias}")
            rows += run_one(cfg, seed, bias, [cfg["fairness"]["sweep_lambda"]])
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.csv", index=False)
    print(f"{len(runs)} rows in {time.time() - start:.0f}s -> {out / 'runs.csv'}")
    report.build(runs, cfg)


if __name__ == "__main__":
    main()
