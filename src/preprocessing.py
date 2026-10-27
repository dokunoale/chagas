import hashlib
from math import ceil

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm

from src import ecg
from src.dataset import sources

# III, aVR, aVL and aVF are linear combinations of I and II
LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_IDX = [sources.LEADS.index(lead) for lead in LEADS]

BAND = (0.5, 40.0)
FS = 80  # Nyquist at the 40 Hz cut-off: nothing above the band survives
WINDOW_S = 7.0
MAX_FLAT_S = 0.5
AGE_CAP = 90


def fingerprint(trimmed):
    q = np.round(trimmed * 1000).astype(np.int32)
    return hashlib.blake2b(q.tobytes(), digest_size=16).hexdigest()


def build_cohort(meta, n_folds=5, n_repeats=3, seed=0):
    """Quality checks, deduplication and repeated patient-level folds on CODE-15%.

    `drop` gives the reason a record is excluded (empty if kept) and `array_index` the row of kept records
    in the processed arrays. `fold_<r>` is the fold of each kept CODE-15% record in repeat r; SaMi-Trop and
    dropped records get -1, since SaMi-Trop only enters training.
    """
    rows = []
    chunks = sources.iter_signals(meta, chunk_size=2000)
    for _, signals in tqdm(chunks, total=ceil(len(meta) / 2000), desc="quality checks"):
        for x in signals:
            trimmed = ecg.trim_zero_padding(x)
            rows.append((len(trimmed) / sources.FS, ecg.longest_flat_run(trimmed) / sources.FS, fingerprint(trimmed)))
    cohort = meta.join(pd.DataFrame(rows, index=meta.index, columns=["duration_s", "flat_run_s", "hash"]))
    cohort["age"] = cohort.age.clip(upper=AGE_CAP)
    cohort["drop"] = np.select(
        [cohort.duration_s == 0, cohort.duration_s < WINDOW_S, cohort.flat_run_s > MAX_FLAT_S,
         cohort.duplicated("hash")],
        ["empty", "short", "flat", "duplicate"], default="")

    keep = (cohort["drop"] == "").to_numpy()
    cohort["array_index"] = np.where(keep, np.cumsum(keep) - 1, -1)

    code15 = cohort[keep & (cohort.source == "CODE-15%").to_numpy()]
    for r in range(n_repeats):
        cohort[f"fold_{r}"] = -1
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        for k, (_, test) in enumerate(cv.split(code15, code15.chagas, code15.patient_id)):
            cohort.loc[code15.index[test], f"fold_{r}"] = k
    return cohort


def load_cohort(path):
    return pd.read_csv(path, dtype={"drop": str}).fillna({"drop": ""})


def _butter(x, fs, cutoff, btype, order=4):
    sos = signal.butter(order, cutoff, btype=btype, fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x - x.mean(axis=0), axis=0)


def bandpass(x, fs, band=BAND):
    return _butter(x, fs, band, "bandpass")


def highpass(x, fs, cutoff=BAND[0]):
    return _butter(x, fs, cutoff, "highpass")


def center_window(x, fs, seconds=WINDOW_S):
    n = int(round(seconds * fs))
    start = (len(x) - n) // 2
    return x[start:start + n]


def filtered(x):
    """Raw record -> 8 independent leads, band-passed at the original rate, whole trimmed record."""
    return bandpass(ecg.trim_zero_padding(x)[:, LEAD_IDX], sources.FS)


def preprocess(x):
    """Raw record -> filtered, resampled to FS, central WINDOW_S window: (560, 8).

    Resampling the whole record before windowing keeps the edge effects out of the window.
    """
    x = signal.resample_poly(filtered(x), FS, sources.FS, axis=0)
    return center_window(x, FS).astype(np.float32)
