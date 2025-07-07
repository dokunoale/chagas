import numpy as np
from scipy.signal import iirnotch, butter, filtfilt, lfilter

class FiltfiltNoiseReducer:
    def __init__(self, fs, iir_freq=50.0, butterworth_cutoff=0.5, lowpass_cutoff=None, verbose=False):
        """
        Inizializza i filtri per la riduzione del rumore.

        Parametri:
        - fs: frequenza di campionamento (es: 400 Hz)
        - iir_freq: frequenza di taglio per il filtro notch (default 50.0 Hz)
        - butterworth_cutoff: frequenza di taglio per il filtro high-pass (default 0.5 Hz)
        - lowpass_cutoff: frequenza di taglio per un filtro low-pass opzionale (default None)
        - verbose: se True, mostra una barra di avanzamento durante il filtraggio (default False)
        """
        self.fs = fs
        self.iir_freq = iir_freq
        self.butterworth_cutoff = butterworth_cutoff
        self.lowpass_cutoff = lowpass_cutoff
        self.verbose = verbose

        Q = 30.0
        self.b_notch, self.a_notch = iirnotch(w0=iir_freq, Q=Q, fs=fs)
        self.b_high, self.a_high = butter(N=3, Wn=butterworth_cutoff, btype='highpass', fs=fs)

        if lowpass_cutoff is not None:
            self.b_low, self.a_low = butter(N=3, Wn=lowpass_cutoff, btype='lowpass', fs=fs)
        else:
            self.b_low, self.a_low = None, None

    def __call__(self, inputs):
        """
        Applica i filtri al batch di segnali.

        Parametri:
        - inputs: array numpy di forma (batch_size, time_steps, channels)

        Ritorna:
        - Un array numpy filtrato della stessa forma di inputs.
        """
        filtered_batch = []

        if self.verbose:
            from tqdm import tqdm
            inputs = tqdm(inputs, desc="Filtering records - filtfilt", unit="record")

        for sample in inputs:  # sample shape: (time_steps, channels)
            channels = []
            for i in range(sample.shape[1]):
                x = sample[:, i]
                x = filtfilt(self.b_notch, self.a_notch, x)
                x = filtfilt(self.b_high, self.a_high, x)
                if self.b_low is not None:
                    x = filtfilt(self.b_low, self.a_low, x)
                channels.append(x)
            filtered_sample = np.stack(channels, axis=-1)
            filtered_batch.append(filtered_sample)
        return np.stack(filtered_batch, axis=0).astype(np.float32)


class LfilterNoiseReducer:
    def __init__(self, fs, iir_freq=50.0, butterworth_cutoff=0.5, lowpass_cutoff=None, verbose=False):
        """
        Inizializza i filtri per la riduzione del rumore.

        Parametri:
        - fs: frequenza di campionamento (es: 400 Hz)
        - iir_freq: frequenza di taglio per il filtro notch (default 50.0 Hz)
        - butterworth_cutoff: frequenza di taglio per il filtro high-pass (default 0.5 Hz)
        - lowpass_cutoff: frequenza di taglio per un filtro low-pass opzionale (default None)
        - verbose: se True, mostra una barra di avanzamento durante il filtraggio (default False)
        """
        self.fs = fs
        self.iir_freq = iir_freq
        self.butterworth_cutoff = butterworth_cutoff
        self.lowpass_cutoff = lowpass_cutoff
        self.verbose = verbose

        Q = 30.0
        self.b_notch, self.a_notch = iirnotch(w0=iir_freq, Q=Q, fs=fs)
        self.b_high, self.a_high = butter(N=3, Wn=butterworth_cutoff, btype='highpass', fs=fs)

        if lowpass_cutoff is not None:
            self.b_low, self.a_low = butter(N=3, Wn=lowpass_cutoff, btype='lowpass', fs=fs)
        else:
            self.b_low, self.a_low = None, None

    def __call__(self, inputs):
        """
        Applica i filtri al batch di segnali.

        Parametri:
        - inputs: array numpy di forma (batch_size, time_steps, channels)

        Ritorna:
        - Un array numpy filtrato della stessa forma di inputs.
        """
        filtered_batch = []

        if self.verbose:
            from tqdm import tqdm
            inputs = tqdm(inputs, desc="Filtering records - filtfilt", unit="record")

        for sample in inputs:  # sample shape: (time_steps, channels)
            channels = []
            for i in range(sample.shape[1]):
                x = sample[:, i]
                x = lfilter(self.b_notch, self.a_notch, x)
                x = lfilter(self.b_high, self.a_high, x)
                if self.b_low is not None:
                    x = lfilter(self.b_low, self.a_low, x)
                channels.append(x)
            filtered_sample = np.stack(channels, axis=-1)
            filtered_batch.append(filtered_sample)
        return np.stack(filtered_batch, axis=0).astype(np.float32)

