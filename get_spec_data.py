import os
import wave
import cv2
import librosa
import multiprocessing
from PIL import Image
import numpy as np
from dataset import WINDOW_TIME, BASE_PATH, SPEC_BASE
from feature.feature import extract_feature, normalize

def voice_to_img(path):
    audio = wave.open(path, 'rb')

    params = audio.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]

    data = audio.readframes(nframes)
    audio.close()

    # data = np.fromstring(data, dtype=np.int16)
    data = np.frombuffer(data, dtype=np.int16)

    # 统一采样频率
    data = data.astype(np.float)
    data = librosa.resample(data, framerate, 8000)

    # 得到语谱图
    spec = extract_feature(data, WINDOW_TIME, 8000)
    # print('max: {}'.format(np.max(spec)))

    # 语谱图归一化到0 ~ 1.0的灰度图像
    # spec = normalize(spec, amp=1.0)
    # # spec = spec.astype(np.uint8)
    # print(spec.dtype)
    # spec = spec.astype(np.float32)

    spec = normalize(spec, amp=255.0)
    spec = spec.astype(np.uint8)
    # print(spec.dtype)
    # spec = spec.astype(np.float32)

    # 灰度图转三通道图像
    spec_RGB = cv2.cvtColor(spec, cv2.COLOR_GRAY2RGB)
    # print(type(spec_RGB))
    # print(spec_RGB.dtype)
    # print(spec_RGB.shape)

    spec_RGB = normalize(spec_RGB, amp=255.0)
    spec_RGB = spec_RGB.astype(np.uint8)
    # exit()

    # 对图像进行变换
    # input_tensor = default_transform(Image.fromarray(spec_RGB))
    return Image.fromarray(spec_RGB)


def make_feature_imgs(stu):
    stu_dir = os.path.join(BASE_PATH, stu)

    for word in range(20):
        word_dir = os.path.join(stu_dir, str(word))
        save_dir = os.path.join(SPEC_BASE, stu, str(word))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        files = os.listdir(word_dir)
        for file in files:
            file_path = os.path.join(word_dir, file)
            save_path = os.path.join(save_dir, file[:-4]) + '.jpeg'

            img = voice_to_img(file_path)
            # img = Image.fromarray(img)
            img.save(save_path)

def proccess_stus(stus):
    for stu in stus:
        make_feature_imgs(stu)
        print('{} finished.'.format(stu))

def multi_proccess():
    pnum = 20
    stus = os.listdir(BASE_PATH)

    sub_tasks = [[] for _ in range(pnum)]

    i = 0
    for stu in stus:
        sub_tasks[i].append(stu)

        i += 1
        i %= pnum

    for i in range(pnum):
        p = multiprocessing.Process(target=proccess_stus, args=(sub_tasks[i], ))
        p.start()


if __name__ == "__main__":
    multi_proccess()
