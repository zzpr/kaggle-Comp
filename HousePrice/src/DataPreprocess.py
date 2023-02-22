"""
@Auth ： zhang-zhang
@Time ： 2023/1/28 10:26
@IDE  ： PyCharm
"""

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')
# print(train_data.shape, test_data.shape)  (1460, 81) (1459, 80)

# 查看一下数据特征, id不需要，SalePrice需要预测
five = train_data.iloc[:2, [0, 1, 2, 3, -3, -2, -1]]
#    Id  MSSubClass MSZoning  LotFrontage SaleType SaleCondition  SalePrice
# 0   1          60       RL         65.0       WD        Normal     208500
# 1   2          20       RL         80.0       WD        Normal     181500

# 合并数据
all_data = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# print(all_data.shape)  # (2919, 79)

# 处理数值型数据
# 获取所有的数值型数据
numberic_feature = all_data.dtypes[all_data.dtypes != 'object'].index
# 1.将所有的数值型数据变为正态分布
all_data[numberic_feature] = all_data[numberic_feature].apply(
    lambda x: (x - x.mean()) / (x.max() - x.min())
)
# 2.填充Nan
all_data[numberic_feature] = all_data[numberic_feature].fillna(0)

# 处理非数值型数据
# print(all_data.shape)  # (2919, 79)
all_data = pd.get_dummies(all_data, dummy_na=True)

# 将处理好的数据分成训练集和测试集，以及标签
train_feature = all_data.iloc[:len(train_data)]
test_feature = all_data.iloc[len(train_data):]
train_label = train_data.iloc[:, -1]

class myDataset(Dataset):
    def __init__(self, dataset, labels, mode='train', valid_ration=0.25):
        # 确定是训练集还是验证集
        self.mode = mode
        # 确定数据集
        self.data = dataset
        self.label = labels
        # 数据集长度
        self.data_len = len(self.data)
        # 训练集长度
        self.train_len = int(self.data_len * (1 - valid_ration))

        if self.mode == 'train':
            # 从data和lable中获取对应的数据
            self.train_data = np.asarray(self.data.loc[:self.train_len])
            self.train_label = np.asarray(self.label.loc[:self.train_len])
            self.datas = self.train_data
            self.labels = self.train_label
        elif self.mode == "valid":
            self.valid_data = np.asarray(self.data.loc[self.train_len:])
            self.valid_label = np.asarray(self.label.loc[self.train_len:])
            self.datas = self.valid_data
            self.labels = self.valid_label

        self.datas_len = len(self.datas)
        print(f'Finished reading {self.mode} dataset. {self.datas_len} number samples found.')

    def __getitem__(self, index):
        data = self.datas[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return self.datas_len

train_dataset = myDataset(train_feature, train_label, mode='train')
valid_dataset = myDataset(train_feature, train_label, mode='valid')
# print(train_dataset, valid_dataset)

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
# print(len(train_loader), len(valid_loader))




# class Network(object):
#
#     def __init__(self, num_of_weights):  # 初始化权重值
#         # 随机产生w的初始值，为了保持程序每次运行结果的一致性，设置固定的随机数种子
#         np.random.seed(0)
#         self.w = np.random.randn(num_of_weights, 1)  #初始参数一般符合标准正态分布
#         self.b = 0.
#
#     def forward(self, x):
#         z = np.dot(x, self.w) + self.b
#         return z
#
#     def loss(self, z, y):
#         error = z - y
#         cost = error * error
#         cost = np.mean(cost)
#         return cost
#
#     def gradient(self, x, y, z):
#         gradient_w = (z - y) * x
#         gradient_w = np.mean(gradient_w, axis=0)
#         gradient_w = gradient_w[:, np.newaxis]
#         gradient_b = (z - y)
#         gradient_b = np.mean(gradient_b)
#         return gradient_w, gradient_b
#
#     def update(self, gradient_w, gradient_b, eta=0.01):
#         self.w = self.w - eta * gradient_w
#         self.b = self.b - eta * gradient_b
#
#     def train(self, train_data, train_label, valid_data, valid_label, epoch_num=100, batch_size=10, lr=0.01):
#         n = len(train_data)
#         # losses = []
#         step = 0
#         for epoch in range(1, epoch_num+1):
#             print(f'=============第{epoch}轮=============')
#             # 打乱数据集并分批
#             np.random.shuffle(train_data)
#             for k in range(0, n, batch_size):
#                 mini_batches = [train_data[k:k+batch_size]]
#                 for iter_id, mini_batch in enumerate(mini_batches):
#                     x = mini_batch[:, :-1]
#                     y = mini_batch[:, -1:]
#                     z = self.forward(x)
#                     L = self.loss(z, y)
#                     gradient_w, gradient_b = self.gradient(x, y, z)
#                     self.update(gradient_w, gradient_b, lr)
#                     # losses.append(L)
#                     step += 1
#
#                     if step % 100 == 0:
#                         print(f'Loss:{L.item()}')
# model = Network(331)
#
# x_train, x_valid, y_train, y_valid = train_test_split(train[0], train[1], test_size=.20)
# print(x_train.shape, x_valid.shape)  # torch.Size([1168, 331]) torch.Size([292, 331])



if __name__ == '__main__':
    pass







