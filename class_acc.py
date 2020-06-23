import pandas as pd
import numpy as np
import os


MATRIX_BASE = 'matrix'
word_list = [
    '数字', '语音', '识别', '上海', '北京',
    '考试', '课程', '可测', '科创', '客车',
    'Digital', 'Speech', 'Voice', 'Shanghai', 'Beijing',
    'China', 'Course', 'Test', 'Coding', 'Code'
]

def cal_class_acc(path):
    df = pd.read_csv(path)
    # save_path = path[:-4]+'_cls_acc.csv'

    result = df.to_numpy()
    # print(result.shape)
    # print(result)

    acc = np.zeros(len(word_list))

    for index in range(result.shape[0]):
        num_total = sum(result[index][1:])

        acc[index] = result[index][index + 1] / num_total
    return acc


if __name__ == "__main__":
    csvs = os.listdir(MATRIX_BASE)
    acc_dict = {}
    for csv in csvs:
        path = os.path.join(MATRIX_BASE, csv)
        name = csv[:-4]

        acc_dict[name] = cal_class_acc(path)

    df = pd.DataFrame(acc_dict, index=word_list)

    df.to_csv('class_accuracy.csv', float_format='%.2f', encoding='utf-8-sig')
