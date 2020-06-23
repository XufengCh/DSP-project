import torch
import numpy
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
import os
import pandas

from dataset import VoiceDataset
from model import build_model

test_set = {
    'train15': 'data_15_12.json',
    'train18': 'data_18_9.json',
    'train21': 'data_21_6.json',
}

word_list = [
    '数字', '语音', '识别', '上海', '北京',
    '考试', '课程', '可测', '科创', '客车',
    'Digital', 'Speech', 'Voice', 'Shanghai', 'Beijing',
    'China', 'Course', 'Test', 'Coding', 'Code'
]

parser = argparse.ArgumentParser()

parser.add_argument('--model', default=None, type=str)
parser.add_argument('--save', default='matrix', type=str)

args = parser.parse_args()

if args.model is None:
    print('No model for Evaluation. \nExit. ')
    exit()

# select GPU
gpu_id = 2
device = torch.device('cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')


def get_testset(model_name:str):
    # [dataset_name, epoch, iteration]
    params = model_name[:-4].split('_')
    params[0] = params[0][params[0].find('train'):]
    # return test_set[params[0]]
    return params[0]


def evaluate():
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    dataset_name = get_testset(args.model)

    with open(test_set[dataset_name], 'r') as f:
        sep = json.load(f)

    dataset = VoiceDataset(sep['test'])

    data_loader = DataLoader(dataset, batch_size=1)

    # model
    model, _ = build_model(args.model)
    model.to(device)
    model.eval()
    print('Model created... ')

    # result
    matrix = np.zeros((20, 20), dtype=int)
    pos = 0
    total = 0

    # Testing
    print('Start testing... ')
    with torch.no_grad():
        for img, gt in data_loader:
            # print(img.shape)
            # print(gt.shape)

            img, gt = img.to(device), gt.to(device)

            output = model(img)
            # print(output.shape)
            # print(output[0].shape)
            output = output[0]
            output = torch.argmax(output)
            # print(output)
            matrix[gt.item(), output.item()] += 1

            if gt.item() == output.item():
                pos += 1
            total += 1

            if total % 100 == 0:
                print('Finish {} cases '.format(total))

    table = pandas.DataFrame(matrix, index=word_list, columns=word_list)

    table_path = os.path.join(args.save, dataset_name) + '.csv'
    table.to_csv(table_path, encoding='utf-8-sig')
    print("\nFinish test {}. ".format(dataset_name))
    print('Total accuracy: {}. '.format(pos / total))


if __name__ == "__main__":
    evaluate()
