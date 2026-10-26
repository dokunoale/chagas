import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

from src import ecg
from src.dataset import sources
from src.preprocessing import BAND, LEAD_IDX, center_window, highpass

WIN_S = 0.5
HOP_S = 0.1


def stft(fs, win_s=WIN_S, hop_s=HOP_S):
    # mfft = fs gives a 1 Hz frequency grid
    return ShortTimeFFT(hann(int(win_s * fs), sym=False), hop=int(hop_s * fs), fs=fs, mfft=int(fs), scale_to="psd")


def log_spectrogram(x, fs, win_s=WIN_S, hop_s=HOP_S, band=BAND):
    """Log10 PSD per lead of x (time, leads), full windows only, frequencies within `band`.

    Returns (leads, freqs, frames), the frequencies and the frame centres in seconds.
    """
    sft = stft(fs, win_s, hop_s)
    n = len(x)
    p0, p1 = sft.lower_border_end[1], sft.upper_border_begin(n)[1]
    s = sft.spectrogram(x.T, p0=p0, p1=p1, axis=-1)
    keep = (sft.f >= band[0]) & (sft.f <= band[1])
    return np.log10(s[:, keep] + 1e-8), sft.f[keep], sft.t(n, p0=p0, p1=p1)


def window(x):
    """Raw record -> 8 independent leads, high-passed, central window at the original rate."""
    return center_window(highpass(ecg.trim_zero_padding(x)[:, LEAD_IDX], sources.FS), sources.FS)


def preprocess(x):
    """Raw record -> log spectrogram (8 leads, freqs, frames). The crop to BAND acts as the low-pass."""
    s, _, _ = log_spectrogram(window(x), sources.FS)
    return s.astype(np.float16)
