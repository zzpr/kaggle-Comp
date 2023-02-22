"""
@Auth ： zhang-zhang
@Time ： 2023/2/1 11:57
@IDE  ： PyCharm
"""

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torchvision
from torch import nn
from train import *
import warnings
warnings.filterwarnings("ignore")

# 处理图像和csv数据

train = pd.read_csv('../classify-leaves/train.csv')
# 类别 -> 下标
class2num = dict(zip(list(train.loc[:, 'label'].unique()), range(len(train.label.unique()))))
# 下标 -> 类别
num2class = {b: a for a, b in class2num.items()}
# print(len(num2class))  # 176

class myDataset(Dataset):

    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.25, resize=(256, 256)):
        self.resize_height = resize[0]
        self.resize_width = resize[1]
        self.file_path = file_path
        self.mode = mode
        self.data = pd.read_csv(csv_path)
        self.data_len = len(self.data.index)
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == 'train':
            self.train_image = np.asarray(self.data.loc[:self.train_len-1, 'image'])
            self.train_label = np.asarray(self.data.loc[:self.train_len-1, 'label'])
            self.images = self.train_image
            self.labels = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data.loc[self.train_len:, 'image'])
            self.valid_label = np.asarray(self.data.loc[self.train_len:, 'label'])
            self.images = self.valid_image
            self.labels = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data.loc[:, 'image'])
            self.images = self.test_image

        self.images_len = len(self.images)

        print('Finished reading %s dataset. %d number samples found.' % (mode, self.images_len))

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(os.path.join(self.file_path, image_path))

        if self.mode == 'train':
            transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            transform = transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        image = transform(image)

        if self.mode == 'test':
            return image

        label = self.labels[index]
        label_num = class2num[label]
        return image, label_num

    def __len__(self):
        return self.images_len

train_path = '../classify-leaves/train.csv'
test_path = '../classify-leaves/test.csv'
image_path = '../classify-leaves/'

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

train_dataset = myDataset(train_path, image_path, mode='train')
valid_dataset = myDataset(train_path, image_path, mode='valid')
test_dataset = myDataset(test_path, image_path, mode='test')

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(len(train_loader), len(valid_loader), len(test_loader))  # 一共分成对应个batch：1377 459 880

# 显示前两块GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
decive = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.resnet34(pretrained=True, progress=True)
# # 显卡大于1块时，device_ids选择模型载入数据对应的显卡
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
# model.fc = nn.Linear(in_features=2048, out_features=176, bias=True)
model.to(decive)

criterion = nn.CrossEntropyLoss().to(decive)

LR = 1e-3
optim = torch.optim.Adam(model.fc.parameters(), lr=LR)
optim.state_dict()
Epochs = 50

def train_model(Epochs, device, model, criterion, optim, DataLoaders, ValidLoaders, ValidLen):
    glo_step = 0
    for epoch in range(1, Epochs+1):
        tik = time()
        print(f'=====================第{epoch}轮=====================')

        model.train()
        tik_train = time()
        for data in DataLoaders:
            img, label = data
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = criterion(pred, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if glo_step % 100 == 0:
                print(f'Loss:{loss.item():.2f}, Time:{time()- tik_train}')
            tik_train = time()

        # 每一个epoch都保存一个断点
        checkpoint = {'model_state_dict': model.state_dict(),
                      'optim_state_dict': optim.state_dict()}
        # dir = '/checkpoint'
        # if not os.path.exists(dir):
        #     os.mkdir(dir)
        torch.save(checkpoint, f'./checkpoint{epoch}.cp')

        model.eval()
        right_num = 0
        with torch.no_grad():
            for data in ValidLoaders:
                img, label = data
                img, label = img.to(device), label.to(device)
                pred = model(img)
                right_num += (pred.argmax(1)).sum()

            accuracy = right_num / ValidLen
            print(f'Accuracy:{accuracy}, Time:{time() - tik}')

        torch.save(model.state_dict(), f'./model{accuracy:.3f}.pth')
train_model(Epochs, decive, model, criterion, optim, train_loader, valid_loader, len(valid_dataset))
print('================END================')






