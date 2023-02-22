"""
@Auth ： zhang-zhang
@Time ： 2023/1/28 10:26
@IDE  ： PyCharm
"""

import pandas as pd
import torch
from torch import nn

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
# print(all_data.shape)  # (2919, 331)

# 将处理好的数据分成训练集和测试集，以及标签
train_feature = torch.tensor(all_data.iloc[:len(train_data)].values, dtype=torch.float64)
test_feature = torch.tensor(all_data.iloc[len(train_data):].values, dtype=torch.float64)
train_label = torch.tensor(train_data.iloc[:, -1].values.reshape(-1, 1), dtype=torch.float64)

train = [train_feature, train_label]

if __name__ == '__main__':
    print(type(train))







