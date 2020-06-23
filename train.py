import torchvision
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import argparse
import os
import json
from datetime import datetime

from dataset import VoiceDataset
from model import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='logs/', type=str)
parser.add_argument('--save_dir', default='models/', type=str)
parser.add_argument('--max_iter', default=500000, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--dataset', default='data_15_12.json', type=str)
parser.add_argument('--batch_size', default=16, type=int)

args = parser.parse_args()

# select GPU
gpu_id = 2
device = torch.device('cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')


# def build_model(path=None):
#     model = torchvision.models.resnet101(num_classes=20)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

#     if path is not None:
#         checkpoint = torch.load(path)
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         print('Loading {}...'.format(path))

#     return model, optimizer


def get_iter_epoch(name:str):
    if name.endswith('.pth'):
        name = name[:-4]

    params = name.split('_')
    return int(params[-1]), int(params[-2])


def save_state(model, optimizer, dataset, epoch, iteration):
    checkpointer = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    file_name = dataset + '_' + str(epoch) + '_' + str(iteration) + '.pth'
    file_name = os.path.join(args.save_dir, file_name)
    torch.save(checkpointer, file_name)


def train():
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if args.dataset is None:
        print("ERROR: No dataset selected. ")
        exit()

    with open(args.dataset, 'r') as f:
        sep = json.load(f)

    dataset_name = 'train' + str(len(sep['train']))

    dataset = VoiceDataset(sep['train'])

    data_loader = DataLoader(dataset, args.batch_size)

    # Logging into tensorboard
    log_dir = os.path.join(args.log_dir, dataset_name + '_' + datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(log_dir)

    # model
    model, optimizer = build_model(args.resume)
    model.to(device)
    model.train()
    print('Model created...')

    # loss
    criterion = torch.nn.CrossEntropyLoss()
    losses = 0

    # resume
    iteration = 0
    start_epoch = 0

    if args.resume is not None:
        iteration, start_epoch = get_iter_epoch(args.resume)

    epoch_size = len(dataset) // args.batch_size
    num_epoches = int(np.ceil(args.max_iter / epoch_size))

    print('Begin training... ')
    for epoch in range(start_epoch, num_epoches):
        for imgs, gts in data_loader:
            if iteration == (epoch + 1) * epoch_size:
                break

            if iteration == args.max_iter:
                break

            imgs.requires_grad_()
            imgs, gts = imgs.to(device), gts.to(device)

            output = model(imgs)
            loss = criterion(output, gts)

            losses += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration += 1

            if iteration % 10 == 0:
                avg_loss = losses / 10

                writer.add_scalar('loss', avg_loss, iteration)
                print('[Epoch: %d || iteration: %d || Loss: %f]' % (epoch, iteration, avg_loss))
                losses = 0

            # save model
            if iteration % 2000 == 0:
                print('Saving state, iter: ', iteration)
                save_state(model, optimizer, dataset_name, epoch, iteration)

    save_state(model, optimizer, dataset_name, epoch, iteration)
    writer.close()


if __name__ == "__main__":
    train()
