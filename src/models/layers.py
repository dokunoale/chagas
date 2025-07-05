import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class LogSpectrogram(tf.keras.layers.Layer):
    def __init__(self, frame_length=64, frame_step=32, fft_length=64, scale=1000.0, **kwargs):
        super(LogSpectrogram, self).__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.scale = scale

    def call(self, inputs):
        spectrograms = []
        for i in range(inputs.shape[-1]):  # For each lead
            signal = inputs[..., i]
            stft = tf.signal.stft(
                signal,
                frame_length=self.frame_length,
                frame_step=self.frame_step,
                fft_length=self.fft_length,
                window_fn=tf.signal.hann_window,
                pad_end=True
            )
            spectrogram = tf.abs(stft)
            log_spectrogram = tf.math.log1p(self.scale * spectrogram)
            spectrograms.append(log_spectrogram)

        return tf.stack(spectrograms, axis=-1)

    def compute_output_shape(self, input_shape):
        time_dim = (input_shape[1] + self.frame_step - 1) // self.frame_step
        freq_dim = self.fft_length // 2 + 1
        return (input_shape[0], time_dim, freq_dim, 12)
    

@keras.saving.register_keras_serializable()
class LightLogSpectrogram(tf.keras.layers.Layer):
    def __init__(self, frame_length=64, frame_step=32, fft_length=256,
                 scale=128.0, num_log_bins=64, **kwargs):
        super(LightLogSpectrogram, self).__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.scale = scale
        self.num_log_bins = num_log_bins

    def build(self, input_shape):
        # Calcola indici logaritmici una sola volta
        freq_bins = self.fft_length // 2 + 1
        linear_bins = tf.range(freq_bins, dtype=tf.float32)
        log_indices = tf.exp(
            tf.linspace(
                tf.math.log(1.0),
                tf.math.log(tf.cast(freq_bins, tf.float32)),
                self.num_log_bins
            )
        ) - 1.0
        log_indices = tf.clip_by_value(tf.cast(tf.round(log_indices), tf.int32), 0, freq_bins - 1)
        self.log_indices = tf.constant(log_indices)

    def call(self, inputs):
        spectrograms = []
        for i in range(inputs.shape[-1]):  # Per ogni lead
            signal = inputs[..., i]

            # STFT
            stft = tf.signal.stft(
                signal,
                frame_length=self.frame_length,
                frame_step=self.frame_step,
                fft_length=self.fft_length,
                window_fn=tf.signal.hann_window,
                pad_end=True
            )
            spec = tf.abs(stft)

            # log1p scaling
            spec = tf.math.log1p(self.scale * spec)

            # Indicizzazione logaritmica
            spec = tf.gather(spec, self.log_indices, axis=-1)

            spectrograms.append(spec)

        return tf.stack(spectrograms, axis=-1)

    def compute_output_shape(self, input_shape):
        time_dim = (input_shape[1] + self.frame_step - 1) // self.frame_step if input_shape[1] else None
        return (input_shape[0], time_dim, self.num_log_bins, input_shape[-1])
