import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src import ecg

# Band edges in Hz for relative power; content above 150 Hz is lost at the 300 Hz of the hidden sets
BANDS = [(0, 0.5), (0.5, 5), (5, 15), (15, 40), (40, 100), (100, 150), (150, 200)]
FREQS = np.arange(0, 200.01, 0.25)


def _line_ratio(freqs, p, f0, half_width=1.0):
    peak = (freqs >= f0 - half_width) & (freqs <= f0 + half_width)
    side = (np.abs(freqs - f0) > 2 * half_width) & (np.abs(freqs - f0) <= 6 * half_width)
    return np.log10(p[peak].mean() / p[side].mean())


def nuisance_features(x, fs):
    """Acquisition properties of a raw record, with no use of morphology."""
    lead, trail = ecg.zero_padding(x)
    trimmed = x[lead:len(x) - trail]
    if len(trimmed) < fs:
        return {"n_samples": len(x), "lead_zeros": lead, "trail_zeros": trail}
    p = ecg.psd(trimmed, fs, FREQS).mean(axis=1)
    total = np.nansum(p)
    feats = {
        "n_samples": len(x),
        "lead_zeros": lead,
        "trail_zeros": trail,
        "duration_s": len(trimmed) / fs,
        "flat_run_s": ecg.longest_flat_run(trimmed) / fs,
        "abs_p99_mv": np.percentile(np.abs(trimmed), 99),
        "abs_median_mv": np.median(np.abs(trimmed)),
        "lead_residual": np.std(trimmed[:, 2] - (trimmed[:, 1] - trimmed[:, 0])) / (np.std(trimmed[:, 1]) + 1e-9),
        "line_60hz": _line_ratio(FREQS, p, 60),
    }
    for lo, hi in BANDS:
        band = (FREQS >= lo) & (FREQS < hi)
        feats[f"power_{lo}_{hi}hz"] = np.nansum(p[band]) / total
    return feats


def source_probe(X, y, groups, n_splits=5, seed=0):
    """Out-of-fold probabilities of a gradient boosting classifier, grouped by patient."""
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    oof = np.zeros(len(y))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in cv.split(X, y, groups):
        model = HistGradientBoostingClassifier(random_state=seed).fit(X[train], y[train])
        oof[test] = model.predict_proba(X[test])[:, 1]
    return oof


def univariate_auc(X, y):
    """Per-feature ROC AUC, folded so that 0.5 is uninformative and 1.0 separates perfectly."""
    auc = {c: roc_auc_score(y, X[c].fillna(X[c].median())) for c in X.columns}
    return pd.Series({c: max(a, 1 - a) for c, a in auc.items()}).sort_values(ascending=False)
