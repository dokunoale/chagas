import warnings

import neurokit2 as nk
import numpy as np
from scipy import signal

from src.preprocessing import LEADS

SPECTRAL_BANDS = [(0.5, 5), (5, 15), (15, 40)]
BEAT_WINDOW_S = (0.25, 0.45)  # before and after the R peak
ECTOPIC_CORR = 0.8


def _median_interval_ms(start, end, fs):
    # neurokit2 drops non-positive positions, which only happen on the first beat: align from the end
    n = min(len(start), len(end))
    d = np.asarray(end[len(end) - n:], dtype=float) - np.asarray(start[len(start) - n:], dtype=float)
    return 1000 * np.nanmedian(d) / fs if np.isfinite(d).any() else np.nan


def beats(x, rpeaks, fs):
    before, after = (int(s * fs) for s in BEAT_WINDOW_S)
    return np.stack([x[r - before:r + after] for r in rpeaks if r >= before and r + after <= len(x)])


def ecg_features(x, fs):
    """Physiological features of a preprocessed record of shape (time, 8 leads).

    Rhythm and intervals come from lead II (neurokit2 R peaks and DWT delineation); amplitudes, axis
    and terminal QRS forces from the median beat. Missing values are NaN.
    """
    lead = dict(zip(LEADS, x.T))
    f = {}
    for name in LEADS:
        freqs, p = signal.welch(lead[name], fs=fs, nperseg=2 * fs)
        total = p[(freqs >= SPECTRAL_BANDS[0][0]) & (freqs < SPECTRAL_BANDS[-1][1])].sum()
        for lo, hi in SPECTRAL_BANDS:
            f[f"power_{lo}_{hi}hz_{name}"] = p[(freqs >= lo) & (freqs < hi)].sum() / total

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            rpeaks = np.asarray(nk.ecg_peaks(lead["II"], sampling_rate=fs)[1]["ECG_R_Peaks"])
        except Exception:
            return f
        if len(rpeaks) < 3:
            return f
        rr = np.diff(rpeaks) / fs
        f["hr_bpm"] = 60 / np.median(rr)
        f["rr_cv"] = np.std(rr) / np.mean(rr)
        f["rmssd_ms"] = 1000 * np.sqrt(np.mean(np.diff(rr) ** 2))
        try:
            waves = nk.ecg_delineate(lead["II"], rpeaks, sampling_rate=fs, method="dwt")[1]
            f["pr_ms"] = _median_interval_ms(waves["ECG_P_Onsets"], waves["ECG_R_Onsets"], fs)
            f["qrs_ms"] = _median_interval_ms(waves["ECG_R_Onsets"], waves["ECG_R_Offsets"], fs)
            f["qtc_ms"] = _median_interval_ms(waves["ECG_R_Onsets"], waves["ECG_T_Offsets"], fs) / np.sqrt(np.median(rr))
        except Exception:
            pass

    b = beats(x, rpeaks, fs)
    if len(b) < 2:
        return f
    tpl = np.median(b, axis=0)
    r = int(BEAT_WINDOW_S[0] * fs)
    qrs = slice(r - int(0.06 * fs), r + int(0.06 * fs) + 1)
    late = slice(r + int(0.04 * fs), r + int(0.12 * fs) + 1)  # terminal QRS forces (RBBB)
    i_, ii, v1, v6 = (LEADS.index(n) for n in ("I", "II", "V1", "V6"))

    for k, name in enumerate(LEADS):
        f[f"qrs_pp_mv_{name}"] = np.ptp(tpl[qrs, k])
    net_i = tpl[qrs, i_].max() + tpl[qrs, i_].min()
    net_ii = tpl[qrs, ii].max() + tpl[qrs, ii].min()
    f["axis_deg"] = np.degrees(np.arctan2(net_ii - net_i / 2, net_i))  # aVF = II - I/2
    f["v1_terminal_r_mv"] = tpl[late, v1].max()
    f["i_terminal_s_mv"] = -tpl[late, i_].min()
    f["v6_terminal_s_mv"] = -tpl[late, v6].min()
    f["limb_max_pp_mv"] = max(np.ptp(tpl[qrs, i_]), np.ptp(tpl[qrs, ii]), np.ptp(tpl[qrs, ii] - tpl[qrs, i_]))
    f["precordial_max_pp_mv"] = max(f[f"qrs_pp_mv_V{k}"] for k in range(1, 7))
    corr = np.array([np.corrcoef(beat[:, ii], tpl[:, ii])[0, 1] for beat in b])
    f["ectopic_frac"] = float(np.mean(corr < ECTOPIC_CORR))
    return f
