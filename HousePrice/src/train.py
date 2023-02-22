"""
@Auth ： zhang-zhang
@Time ： 2023/1/28 11:25
@IDE  ： PyCharm
"""

from DataPreprocess import *
from model import *
import torch
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler

batch_size = 64
EPOCHS = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train, x_test, y_train, y_test = train_test_split(train[0], train[1], test_size=.20)
# print(x_train.shape)
# train_loader = DataLoader(train, batch_size=batch_size, sampler=)
# test_loader = DataLoader(test_feature, batch_size=batch_size)

# model = haonet()
def get_model():
    model = nn.Sequential(nn.Linear(331, 1))
    return model
model = get_model()
model.to(device)

lossfc = nn.MSELoss()
lossfc.to(device)

learning_rate = 1e-3
optim = optim.Adam(model.parameters(), lr=learning_rate)

step = 0
right_num = 0
for epoch in range(1, EPOCHS+1):

    print(f'================第{epoch}轮================')

    model.train()

    for x, y in zip(x_train, y_train):
        x = torch.tensor(x, dtype=float).to(torch.float32)
        y = torch.tensor(y).to(torch.float32)
        output = model(x)
        loss = lossfc(output, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        step += 1

        if step % 100 == 0:
            print(f'step:{step}, loss:{loss.item()}')

    model.eval()
    with torch.no_grad():
        for x, y in zip(x_test, y_test):
            x = torch.tensor(x, dtype=float).to(torch.float32)
            y = torch.tensor(y).to(torch.float32)
            output = model(x)
            # print(output)
            right_num += (output.item() == y).sum()

        accuracy = right_num / len(x_test)
        print(f'accuracy:{accuracy}')

torch.save(model.state_dict(), '../model.pth')