"""How does the unaware model use local experience?

Partial dependence: take the test pairs, set everyone's `local_exp` to the
same value, and average the model's score. Repeating this over a grid shows
how much the score moves with local experience alone. True suitability does
not depend on local experience at all, so any slope is learned from the
biased shortlists.

    python -m fairmatch.inspect_proxy
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from .experiment import split_jobs  # noqa: E402
from .model import UNAWARE, train  # noqa: E402
from .synthetic import make_world  # noqa: E402


def partial_dependence(score, pairs, feature, grid):
    out = []
    for value in grid:
        probe = pairs.copy()
        probe[feature] = value
        out.append(score(probe).mean())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    pairs = make_world(cfg, args.seed)
    train_pairs, test = split_jobs(pairs, cfg["split"]["test_share"], args.seed)
    lam = cfg["fairness"]["sweep_lambda"]
    models = {
        "unaware": train(train_pairs, UNAWARE, cfg["train"], seed=args.seed),
        f"penalty (lambda={lam:g})": train(train_pairs, UNAWARE, cfg["train"], lam=lam, seed=args.seed),
    }

    grid = np.linspace(0, np.percentile(pairs["local_exp"], 95), 25)
    fig, (ax, hx) = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    for name, score in models.items():
        pd_curve = partial_dependence(score, test, "local_exp", grid)
        ax.plot(grid, pd_curve - pd_curve[0], marker="o", ms=3, label=name)
        print(f"{name}: score change from 0 to {grid[-1]:.1f} years of local experience = {pd_curve[-1] - pd_curve[0]:+.2f}")
    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.set_ylabel("Change in model score (logit)")
    ax.set_title("What the model learned about local experience\n(true suitability does not depend on it)")
    ax.grid(alpha=0.3)
    ax.legend()

    bins = np.linspace(0, grid[-1], 30)
    for g, label in ((0, "group A (locally trained)"), (1, "group B (internationally trained)")):
        hx.hist(pairs.loc[pairs["group_b"] == g, "local_exp"], bins=bins, alpha=0.6, density=True, label=label)
    hx.set_xlabel("Years of local experience")
    hx.set_ylabel("Density")
    hx.legend(fontsize=8)
    fig.tight_layout()

    out = Path(cfg["output_dir"]) / "figures" / "proxy_local_experience.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
