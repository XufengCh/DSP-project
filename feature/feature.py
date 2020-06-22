import numpy as np
# from spectrogram import spectrogram
# from end_point import vad
from feature.spectrogram import spectrogram
from feature.end_point import vad
import wave

def normalize(data, amp=255):
    max_amp = np.max(np.abs(data))
    return amp * data / max_amp


def pre_emphasis(data, alpha=0.97):
    return np.append(data[0], data[1:] - alpha * data[:-1])


def extract_feature(data, window_time=0.02, framerate=8000, ):
    # 预加重
    emp_data = pre_emphasis(data)

    # 端点检测
    left, right = vad(data, framerate, window_time)

    # 将语音信号之外的杂音去除
    data = data.copy()
    i = 0
    while i < left:
        # data[i] = np.finfo(np.float).eps
        data[i] = 0
        i += 1

    i = data.shape[0] - 1
    while i > right:
        # data[i] = np.finfo(np.float).eps
        data[i] = 0
        i -= 1

    # 语谱图
    spec = spectrogram(data, int(framerate * window_time))

    spec = np.where(spec == 0, np.finfo(np.float).eps, spec)

    # time: 2s, framerate: 8000, window_time: 0.02s
    # spec: [129, 192]
    spec = 20 * np.log10(spec)

    # print(spec.shape)
    return spec


if __name__ == "__main__":
    test_file = 'one.wav'
    audio = wave.open(test_file, 'rb')

    params = audio.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]

    data = audio.readframes(nframes)
    audio.close()

    data = np.frombuffer(data, dtype=np.int16)
    # print(data.shape)
    extract_feature(data)


