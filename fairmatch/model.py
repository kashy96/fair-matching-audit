"""Pointwise candidate-job scorer, optionally with a fairness penalty.

The penalty is lambda * (mean score of group A - mean score of group B)^2,
computed per mini-batch. Group labels are used only in the loss; the
`unaware` feature set never shows them to the model.
"""

import numpy as np
import torch
from torch import nn

FEATURES = ["spec_match", "exp_gap", "total_exp", "local_exp", "certs", "licence_ok", "hours_fit", "same_region", "group_b"]
UNAWARE = [f for f in FEATURES if f != "group_b"]


class Scorer(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Standardiser:
    def fit(self, X):
        self.mean, self.std = X.mean(0), X.std(0) + 1e-8
        return self

    def __call__(self, X):
        return (X - self.mean) / self.std


def train(train_pairs, features, cfg, lam=0.0, seed=0):
    torch.manual_seed(seed)
    X_np = train_pairs[features].to_numpy(dtype=np.float32)
    scale = Standardiser().fit(X_np)
    X = torch.tensor(scale(X_np))
    y = torch.tensor(train_pairs["shortlisted"].to_numpy(dtype=np.float32))
    g = torch.tensor(train_pairs["group_b"].to_numpy(dtype=bool))

    model = Scorer(len(features), cfg["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    bs = cfg["batch_size"]

    model.train()
    for _ in range(cfg["epochs"]):
        for idx in torch.randperm(len(X)).split(bs):
            logits = model(X[idx])
            loss = bce(logits, y[idx])
            gb = g[idx]
            if lam > 0 and gb.any() and (~gb).any():
                p = torch.sigmoid(logits)
                loss = loss + lam * (p[~gb].mean() - p[gb].mean()) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()

    def score(pairs):
        with torch.no_grad():
            x = torch.tensor(scale(pairs[features].to_numpy(dtype=np.float32)))
            return model(x).numpy()

    return score
