import numpy as np
from scipy import signal


def zero_padding(x):
    """Leading and trailing rows where all leads are exactly 0 (the rule of the challenge converter)."""
    nonzero = np.flatnonzero(np.any(x != 0, axis=1))
    if nonzero.size == 0:
        return len(x), 0
    return int(nonzero[0]), int(len(x) - 1 - nonzero[-1])


def trim_zero_padding(x):
    lead, trail = zero_padding(x)
    return x[lead:len(x) - trail]


def longest_run(mask):
    if not mask.any():
        return 0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return int((edges[1::2] - edges[::2]).max())


def longest_flat_run(x):
    """Longest stretch (in samples) where every lead stays constant."""
    if len(x) < 2:
        return len(x)
    return longest_run(np.all(np.diff(x, axis=0) == 0, axis=1)) + 1


def psd(x, fs, freqs, seconds=4):
    """Welch PSD per lead, interpolated on `freqs` (NaN above Nyquist)."""
    f, p = signal.welch(x, fs=fs, nperseg=min(len(x), seconds * fs), axis=0)
    return np.stack([np.interp(freqs, f, p[:, i], right=np.nan) for i in range(x.shape[1])], axis=1)
