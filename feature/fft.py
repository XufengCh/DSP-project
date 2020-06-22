import numpy as np
import matplotlib.pyplot as plt

CHUNK = 1024
RATE = 16000

def FFT(data):
    l = len(data)
    if l == 1:
        return data

    # padding
    n = np.power(2, np.ceil(np.log2(l))).astype(np.int32)
    data = np.pad(data, (0, n-l), 'constant', constant_values=(0))

    wn = np.exp((2*np.pi*1j) / n)

    d0 = data[0::2]
    d1 = data[1::2]
    y0 = FFT(d0)
    y1 = FFT(d1)

    w = 1
    y = np.empty(n, dtype=complex)
    m = n // 2
    for k in range(0, m):
        tmp = w * y1[k]
        y[k]        = y0[k] + tmp
        y[k + m]    = y0[k] - tmp

        w = w * wn

    return y

if __name__ == "__main__":
    # set plot
    plt.figure(figsize=(8, 4))
    plt.title('FFT')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')

    t = np.arange(0, 1, 1/RATE)
    data = np.sin(2 * np.pi * 100 * t) + 2 * np.sin(2 * np.pi * 500 * t)

    comp_fft = FFT(data[:CHUNK])

    freqs = np.arange(CHUNK) / CHUNK * RATE
    amp = np.abs(comp_fft/CHUNK)

    plt.plot(freqs, amp)
    plt.show()