import numpy as np
from scipy.signal import iirnotch, butter, lfilter, filtfilt

def filtfilt_noise_reduction(inputs, fs=400, iir_freq=60.0, butterworth_cutoff=5., lowpass_cutoff=60.):
    """
    Applica filtri IIR e Butterworth per la riduzione del rumore su un batch di segnali ECG.

    Parametri:
    - inputs: array numpy di forma (batch_size, time_steps, channels)
    - fs: frequenza di campionamento (es: 400 Hz)
    - iir_freq: frequenza di taglio per il filtro notch (default 50.0 Hz)
    - butterworth_cutoff: frequenza di taglio per il filtro high-pass (default 0.5 Hz)
    - lowpass_cutoff: frequenza di taglio per un filtro low-pass opzionale (default None)

    Ritorna:
    - Un array numpy filtrato della stessa forma di inputs.
    """

    # Design notch filter (IIR)
    Q = 30.0
    b_notch, a_notch = iirnotch(w0=iir_freq, Q=Q, fs=fs)

    # Design high-pass Butterworth filter
    b_high, a_high = butter(N=3, Wn=butterworth_cutoff, btype='highpass', fs=fs)

    # Design optional low-pass filter
    if lowpass_cutoff is not None:
        b_low, a_low = butter(N=3, Wn=lowpass_cutoff, btype='lowpass', fs=fs)
    else:
        b_low, a_low = None, None

    def _apply_filters(batch_np):
        # Applica i filtri a ciascun segnale nel batch e a ciascun canale
        filtered_batch = []
        for sample in batch_np:  # sample shape: (time_steps, channels)
            channels = []
            for i in range(sample.shape[1]):
                x = sample[:, i]
                x = filtfilt(b_notch, a_notch, x)
                x = filtfilt(b_high, a_high, x)
                if b_low is not None:
                    x = filtfilt(b_low, a_low, x)
                channels.append(x)
            # Ricostruisci un array (time_steps, channels)
            filtered_sample = np.stack(channels, axis=-1)
            filtered_batch.append(filtered_sample)
        return np.stack(filtered_batch, axis=0).astype(np.float32)

    # Applica i filtri a tutto il batch
    filtered_inputs = _apply_filters(inputs)

    return filtered_inputs