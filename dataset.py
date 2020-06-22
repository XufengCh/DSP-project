import torch
from torchvision import transforms
import numpy as np
import wave
import os
import cv2
import librosa
from PIL import Image
import json
import matplotlib.pyplot as plt
from feature.feature import extract_feature, normalize

BASE_PATH = 'data'
WINDOW_TIME = 0.02
SPEC_BASE = 'spec_data'

default_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

class VoiceDataset(torch.utils.data.Dataset):
    def __init__(self, stu_dirs, transform=default_transform):
        self.num_students = len(stu_dirs)
        self.items = []

        for stu in stu_dirs:
            stu_dir = os.path.join(BASE_PATH, stu)
            for word in range(0, 20):
                word_dir = os.path.join(stu_dir, str(word))
                files = os.listdir(word_dir)
                for file in files:
                    file_path = os.path.join(word_dir, file)
                    self.items.append([file_path, word])
        np.random.shuffle(self.items)

        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # 读取语音文件
        path, word = self.items[index]

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
        input_tensor = self.transform(Image.fromarray(spec_RGB))

        # 输出特征 类别
        return input_tensor, word


def separate_datasets(num_train):
    all_stus = os.listdir(BASE_PATH)
    num_stus = len(all_stus)

    if num_train >= num_stus or num_train < 1:
        print('Error: Don\'t have enough train data or test data. ')
        print('Exit. ')
        exit()

    np.random.shuffle(all_stus)
    num_test = num_stus - num_train

    sep = {
        'train': all_stus[:num_train],
        'test': all_stus[-num_test:]
    }
    file_name = 'data_' + str(num_train) + '_' + str(num_test) + '.json'

    # write
    with open(file_name, 'w') as f:
        json.dump(sep, f)


if __name__ == "__main__":
    # test = VoiceDataset(['16307130080'])

    # img, word = test.__getitem__(0)

    # print(test.__len__())

    # img = img.permute(1, 2, 0)

    # # img = 20 * np.log10(img)

    # plt.imshow(img, cmap='gray')
    # plt.show()

    train_sets = [15, 18, 21]

    for num in train_sets:
        separate_datasets(num)
