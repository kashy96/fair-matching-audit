import numpy as np
import yaml

from fairmatch import metrics
from fairmatch.experiment import evaluate, run_one, split_jobs
from fairmatch.rerank import proportional_rerank
from fairmatch.synthetic import make_world


def small_cfg():
    cfg = yaml.safe_load(open("config.yaml"))
    cfg["world"].update(n_candidates=800, n_jobs=40, pool_size=30)
    cfg["train"]["epochs"] = 2
    return cfg


def test_world_has_known_structure():
    pairs = make_world(small_cfg(), seed=0)
    per_job = pairs.groupby("jid")
    assert (per_job.size() == 30).all()
    # about 20% of each pool is shortlisted
    assert abs(pairs["shortlisted"].mean() - 0.2) < 0.03
    # merit does not depend on group, but past shortlists do
    gap_true = pairs.groupby("group_b")["true_score"].mean().diff().iloc[-1]
    gap_hist = pairs.groupby("group_b")["shortlisted"].mean().diff().iloc[-1]
    assert abs(gap_true) < 0.2
    assert gap_hist < -0.05


def test_no_bias_means_no_gap():
    # full-size market: the small one is too noisy for a 5-point tolerance
    cfg = yaml.safe_load(open("config.yaml"))
    cfg["history"]["local_bias"] = 0.0
    pairs = make_world(cfg, seed=1, bias=0.0)
    rates = pairs.groupby("group_b")["shortlisted"].mean()
    assert abs(rates[1] - rates[0]) < 0.05


def test_metrics_on_hand_made_ranking():
    relevance = np.array([3, 2, 0, 0])
    assert metrics.ndcg_at_k(np.array([0, 1, 2, 3]), relevance, 2) == 1.0
    assert metrics.ndcg_at_k(np.array([2, 3, 0, 1]), relevance, 2) == 0.0
    group_b = np.array([0, 0, 1, 1])
    ranking = np.array([0, 1, 2, 3])
    assert metrics.selection_ratio(ranking, group_b, 2) == 0.0
    assert metrics.selection_ratio(np.array([0, 2, 1, 3]), group_b, 2) == 1.0


def test_rerank_meets_floor():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    group_b = np.array([0, 0, 0, 0, 1, 1])
    ranking = proportional_rerank(scores, group_b, k=4, min_share=0.5)
    assert group_b[ranking[:4]].sum() == 2
    assert sorted(ranking) == list(range(6))


def test_oracle_is_best_and_pipeline_runs():
    cfg = small_cfg()
    rows = run_one(cfg, seed=0, bias=1.0, lambdas=[0, 10])
    by_variant = {r["variant"]: r for r in rows}
    assert {"oracle", "historical", "baseline", "unaware", "rerank", "penalty"} <= set(by_variant)
    assert by_variant["oracle"]["ndcg_true"] >= max(r["ndcg_true"] for r in rows) - 1e-9

    pairs = make_world(cfg, seed=0)
    train, test = split_jobs(pairs, 0.3, seed=0)
    assert not set(train["jid"]) & set(test["jid"])
    res = evaluate(test, test["true_score"].to_numpy(), 10)
    assert res["ndcg_true"] == 1.0
