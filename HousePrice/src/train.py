"""
@Auth ： zhang-zhang
@Time ： 2023/1/28 11:25
@IDE  ： PyCharm
"""

from DataPreprocess import *
from model import *
import torch
import numpy as np
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler

EPOCHS = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = haonet()
model.to(device)

# lossfc = nn.L1Loss()
# lossfc.to(device)

def loss(z, y):
    error = z - y
    cost = (error * error).mean()
    # cost = np.mean(cost)
    return cost


learning_rate = 1e-3
optim = optim.Adam(model.parameters(), lr=learning_rate)

step = 0
right_num = 0
for epoch in range(1, EPOCHS+1):

    print(f'================第{epoch}轮================')

    model.train()
    for x, y in train_loader:
        x, y = x.to(torch.float32), y.to(torch.float32)
        output = model(x)
        L = loss(output, y)
        optim.zero_grad()
        L.backward()
        optim.step()
        step += 1

        if step % 100 == 0:
            print(f'step:{step}, loss:{L.item()}')

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(torch.float32), y.to(torch.float32)
            output = model(x)
            # print(output)
            # right_num += (output.item() == y).sum()

        # accuracy = right_num / len(valid_dataset)
        # print(f'accuracy:{accuracy}')

torch.save(model.state_dict(), '../model.pth')