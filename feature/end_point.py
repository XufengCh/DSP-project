import numpy as np
import os
import wave
# import pyaudio
import matplotlib.pyplot as plt


# 窗口时间
WINDOW = 0.02

# 检测中，相关的参数设置
# 测试得到 MH = average amp / 8 时，效果较好
PARAM_MH = 8

def zero_crossing_rate(data, length):
    """Calculate zero crossing rate of the data

    params:
        data: numpy array
        length: int
    """
    cross_zero_counter = lambda x, y: 1 if (x > 0 and y < 0) or (x < 0 and y > 0) else 0

    data_length = data.shape[0]
    res = []

    i = 0
    while i < data_length:
        # window size:
        limit = int(min(length, data_length - i))

        # 使用矩形窗进行加窗
        frame = data[int(i):int(i+limit)]

        count = 0
        for j in range(limit-1):
            count += cross_zero_counter(frame[j], frame[j + 1])

        res.append(count)
        i += length // 2

    return np.array(res)

def average_energy(data, length):
    """Calculate average energy

    params:
        data: numpy array
        length: int
    """
    # 使用汉明窗
    window = np.hamming(length)

    data_length = data.shape[0]
    res = []

    i = 0
    while i < data_length:
        limit = int(min(length, data_length - i))

        frame = data[int(i):int(i+limit)]
        frame = np.pad(frame, (0, int(length - frame.shape[0])), constant_values=(0))

        frame = frame * window

        energy = np.sum(frame * frame)

        res.append(energy)

        i += length // 2

    return np.array(res)

def average_amplitude(data, length):
    """Calculate average amplitude

    params:
        data: numpy array
        length: int
    """
    # 使用汉明窗
    window = np.hamming(length)

    data_length = data.shape[0]
    res = []

    i = 0
    while i < data_length:
        limit = int(min(length, data_length - i))

        frame = data[int(i):int(i+limit)]
        frame = np.pad(frame, (0, int(length - frame.shape[0])), constant_values=(0))

        frame = frame * window

        amp = np.sum(np.abs(frame))

        res.append(amp)

        i += length // 2

    return np.array(res)

def detect_end_point(amp, zcr, wave_rate, window_time):
    # set MH, ML, Z
    noise_length = 5
    noises_zcr = zcr[:noise_length]
    z_thres = (np.sum(noises_zcr) - np.max(noises_zcr) - np.min(noises_zcr)) / (noise_length - 2)

    mh_thres = np.mean(amp) / 8
    ml_thres = (np.mean(amp[:noise_length]) + mh_thres) * 0.25

    # print('MH: {}'.format(mh_thres))
    # print('ML: {}'.format(ml_thres))
    # print('Z0: {}'.format(z_thres))

    # 因为课程项目是孤立词识别，有理由认为语音信号应是连续的一段
    # 故只检测首尾两个端点，即认为这两个端点之间即是语音信号
    # using mh_thres
    num_frames = amp.shape[0]
    left = 0
    right = num_frames - 1

    while left < right:
        if amp[left] > mh_thres:
            break
        left += 1

    while right > left:
        if amp[right] > mh_thres:
            break
        right -= 1

    # using ml_thres
    while left > 0 and amp[left] > ml_thres:
        left -= 1

    while right < num_frames and amp[right] > ml_thres:
        right += 1

    # using z_thres
    limit = 0.025 / window_time * 2
    i = 0
    while left > 0 and zcr[left] > 3 * z_thres:
        left -= 1

        i += 1
        if i > limit:
            break

    i = 0
    while right < num_frames and zcr[right] > 3 * z_thres:
        right += 1

        i += 1
        if i > limit:
            break

    # voice rigion: [left, right)
    return left, right

def vad(data, framerate, window_time):
    """calculate the end point of wave

    params:
        data: voice data, numpy array
        framerate: int
        window_time: float
    """
    window_length = int(window_time * framerate)

    amp = average_amplitude(data, window_length)
    zcr = zero_crossing_rate(data, window_length)

    # time = np.arange(0, zcr.shape[0] * window_time / 2, window_time / 2)
    # plt.plot(time, zcr)
    # plt.xlim(0, zcr.shape[0] * window_time / 2)
    # plt.xlabel('zero crossing rate (s)')
    # plt.show()

    # time = np.arange(0, amp.shape[0] * window_time / 2, window_time / 2)
    # plt.plot(time, amp)
    # plt.xlim(0, amp.shape[0] * window_time / 2)
    # plt.xlabel('average amplitude (s)')
    # plt.show()

    left, right = detect_end_point(amp, zcr, framerate, window_time)

    return left * window_length // 2, (right+1) * window_length // 2


if __name__ == "__main__":
    test_data = 'data/voice.wav'

    audio = wave.open(test_data, 'rb')

    params = audio.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]

    data = audio.readframes(nframes)
    audio.close()

    data = np.frombuffer(data, dtype=np.int16)

    p = pyaudio.PyAudio()
    out_stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=8000,
                        output=True)

    time = np.arange(0, data.shape[0] / framerate, 1.0 / framerate)

    # end points detection
    left, right = vad(data, framerate, WINDOW)

    out_stream.write(data[left:right])

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(time, data)
    ax.set_xlim(0, 2)
    start, end = ax.get_ylim()
    ax.vlines(left / framerate, start, end, linestyles = "dashed")
    ax.vlines(right / framerate, start, end, linestyles = "dashed")
    ax.set_title('Wave')
    ax.set_xlabel('Time (s)')

    plt.show()

    out_stream.close()
    p.terminate()
