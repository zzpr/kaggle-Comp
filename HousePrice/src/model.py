"""
@Auth ： zhang-zhang
@Time ： 2023/1/28 10:32
@IDE  ： PyCharm
"""
from torch import nn

class haonet(nn.Module):
    def __init__(self):
        super(haonet, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(331, 128),
            nn.Linear(128, 1)
        )

    def forward(self, input):
        output = self.module(input)
        return output

if __name__ == '__main__':
    model = haonet()
    print(model)