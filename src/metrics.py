import numpy as np


def challenge_score(labels, scores, alpha=0.05):
    """TPR within the top alpha fraction of the ranking.

    Closed form of the official estimate: the challenge averages over random permutations
    of tied scores, which is the expected value computed here for the tie group at the cutoff.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if not labels.any():
        return float("nan")
    k = int(alpha * len(scores))
    if k == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    cutoff = s[k - 1]
    above, tied = s > cutoff, s == cutoff
    tp = y[above].sum() + y[tied].sum() * (k - above.sum()) / tied.sum()
    return float(tp / labels.sum())
