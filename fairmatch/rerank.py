import math

import numpy as np


def proportional_rerank(scores, group_b, k, min_share):
    """Greedy re-ranking with a representation floor for group B.

    Walk down the positions. At position i, group B must already hold at least
    floor(min_share * i) of the places; if taking the best remaining candidate
    would break that, take the best remaining group-B candidate instead. A
    simplified version of FA*IR (Zehlike et al., 2017) with a fixed floor
    instead of a significance test. Positions after k keep the score order.
    """
    order = np.argsort(-scores, kind="stable")
    a = [i for i in order if not group_b[i]]
    b = [i for i in order if group_b[i]]
    top, n_b = [], 0
    while len(top) < min(k, len(order)):
        need = math.floor(min_share * (len(top) + 1))
        if b and (n_b < need or not a or scores[b[0]] >= scores[a[0]]):
            top.append(b.pop(0))
            n_b += 1
        else:
            top.append(a.pop(0))
    chosen = set(top)
    return np.array(top + [i for i in order if i not in chosen])
