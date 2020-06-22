import numpy as np
import matplotlib.pyplot as plt
import os
import wave

# from fft import FFT
from feature.fft import FFT


def spectrogram(data, frame_length):
    """calculate spectrogram

    params:
        data:           1D numpy array
        frame_length:   int
    """
    window = np.hamming(frame_length)

    i = 0
    length = data.shape[0]
    res = []
    while i < length:
        # window
        limit = int(i + frame_length)
        if limit < length:
            frame = data[i:limit]
        else:
            frame = data[i:]
            frame = np.pad(frame, (0, int(frame_length - frame.shape[0])), constant_values=(0))

        frame = frame * window

        # fft
        frame = FFT(frame)

        frame = np.power(np.abs(frame), 2)
        res.append(frame[:frame.shape[0] // 2 + 1])
        i += int(frame_length / 2)

    return np.array(res).T


# params
FRAME_TIMES= [0.005, 0.01, 0.015]
DATA_PATH = './data'
RESULT_PATH = './result'

if __name__ == "__main__":
    files = os.listdir(DATA_PATH)

    if not os.path.exists(RESULT_PATH):
        os.mkdir(RESULT_PATH)

    for file in files:
        # read file
        audio = wave.open(os.path.join(DATA_PATH, file), 'rb')

        params = audio.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]

        data = audio.readframes(nframes)
        audio.close()

        data = np.fromstring(data, dtype=np.int16)

        # draw wave
        time = np.arange(0, nframes) * (1.0 / framerate)
        plt.title('The Waveform of ' + file)
        plt.plot(time, data)
        plt.xlabel('Time (s)')
        plt.show()

        # draw spectrogram
        fig, axes = plt.subplots(ncols=1, nrows=len(FRAME_TIMES), figsize=(6, 6))
        fig.suptitle('The Spectrograms of {}'.format(file))
        for i in range(len(FRAME_TIMES)):
            window_size = FRAME_TIMES[i] * framerate
            spec = spectrogram(data, window_size)

            spec = 20 * np.log10(spec)
            # print(spec.shape)

            axes[i].set_title('window length: {}s'.format(FRAME_TIMES[i]))
            axes[i].imshow(spec, aspect=0.5, origin='lower', extent=(0, spec.shape[1]*window_size*5000/framerate, 0, spec.shape[0] * framerate / (2 * spec.shape[0] - 2)))
            axes[i].set_aspect('equal')
            start, end = axes[i].get_xlim()
            axes[i].xaxis.set_ticks(np.arange(start, end, 5000))
            start, end = axes[i].get_ylim()
            axes[i].yaxis.set_ticks(np.arange(start, end, 500))
            axes[i].set_xlabel('Time (1e-4s)')
            axes[i].set_ylabel('Frequency (Hz)')

        plt.show()

