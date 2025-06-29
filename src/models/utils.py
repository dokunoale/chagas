import tensorflow as tf

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