import numpy as np
import pandas as pd
from d2l import torch as d2l
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt
import torch
import hashlib
import os
import tarfile
import zipfile
import requests

def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)



DATA_HUB = {} # 二元组存储URL和完整性密钥（hash)
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

# 定义脚本下载
DATA_HUB['kaggle_house_train'] = (DATA_URL + 'kaggle_house_pred_train.csv', '585e9cc93e70b39160e7921475f9bcd7d31219ce')
DATA_HUB['kaggle_house_test'] = (DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

'''
print(train_data.shape, test_data.shape)

print(test_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

print(all_features)
'''
# 数据预处理
# 总表
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:-1]))

# 按列筛选出非类型列
numeric_features = all_features.columns[all_features.dtypes != 'object']
# 取值变为高斯分布
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
# 缺失值置零
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 处理离散特征
all_features = pd.get_dummies(all_features, dummy_na=True)

# 提取numpy格式，转换为张量表示
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_label = torch.tensor(train_data['SalePrice'].values.reshape(-1, 1), dtype=torch.float32)

# 训练

loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net

# 相对误差作为损失
# 先把预测值设限
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))

    return rmse.item() # 转化提取出一个python数字（原本是张量）

def load_array(train, batchsize, is_train = True):
    dataset = data.TensorDataset(*train)
    return data.DataLoader(dataset, batchsize, shuffle=is_train)

def train(net, train_features, train_labels, test_features, test_labels, num_epochs, lr, wd, batchsize):
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batchsize)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr, weight_decay=wd)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y) #?
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

# K折交叉验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k 
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train = X_part
            y_train = y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

# k折训练
def k_fold(k, X_train, y_train, num_epochs, lr, wd, batchsize):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train) # 得到第i个分割数据
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, lr, wd, batchsize) # 开始训练
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls], xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs], ylim=[0.1, 10], legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f},' f'验证log rmse{float(valid_ls[-1]):f}')   # x轴的显示范围                                   # y的刻度类型
    return train_l_sum / k, valid_l_sum / k

# 模型选择
k, num_epochs, lr, wd, batchsize = 12, 100, 20, 0, 64
"""
train_l, valid_l = k_fold(k, train_features, train_label, num_epochs, lr, wd, batchsize)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
plt.show() # python 显示
"""
# 选好模型后全部训练
def train_and_pred(train_features, train_labels, test_features, test_data, num_epochs, lr, wd, batchsize):
    net = get_net()
    train_ls, _= train(net, train_features, train_labels, None, None, num_epochs, lr, wd, batchsize)
    d2l.plot(np.arange(1, num_epochs+1), [train_ls], xlabel='epoch', ylabel='log rmse', xlim = [1, num_epochs], yscale='log')   
    # 应用于数据集,解除计算图，不需要梯度了
    pred = net(test_features).detach().numpy()
    # 导出到kaggle，保证传递一个一维数组
    test_data['SalePrice'] = pd.Series(pred.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis = 1)
    submission.to_csv('submission.csv', index = False)

train_and_pred(train_features, train_label, test_features, test_data, num_epochs, lr, wd, batchsize)
