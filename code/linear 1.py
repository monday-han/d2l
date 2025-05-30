import numpy as np
import torch
from torch.utils import data # 有处理数据的模块
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True): # 表示数据迭代器是否要在每个迭代周期打乱数据
    dataset = data.TensorDataset(*data_arrays) # 解包元组，feature、label分别作为单独的张量传递
    return data.DataLoader(dataset, batch_size, shuffle=is_train) # shuffle表示在每个训练周期epoch中打乱顺序

batch_size = 10
data_iter = load_array((features, labels), batch_size)

#next(iter(data_iter)) # 转为python迭代器

# 模型定义
from torch import nn
net = nn.Sequential(nn.Linear(2, 1)) # 构建神经网络层 后面为输入维度和输出维度
# net[0]是网络的第一个Layer
# 初始化参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr = 0.03)

# 训练
epochs = 3 # 迭代周期
for epoch in range(epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad() # 梯度清零
        l.backward() # 反向传播求梯度
        trainer.step() # 利用梯度更新参数
        
w = net[0].weight#.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias#.data
print('b的估计误差：', true_b - b)

'''
# 框架: 数据流水线 模型选择 损失函数选择 优化器选择 训练 验证
# %matplotlib inline # plot时默认嵌入mat
import random # 随机化函数
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples): #@save
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.mv(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4]) # 设置参数w b并生成x y集合
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000) # 占用大量内存
batch_size = 10

def data_iter(batch_size, features, labels): # sgd分批随机选择
    num = len(labels)
    indices = list(range(num))
    random.shuffle(indices)
    for i in range(0, num, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num)])
        # print(batch_indices)
        yield features[batch_indices], labels[batch_indices] # 调用函数时返回生成器，每次到这返回一组数据后再继续执行，防止内存不足

# 初始化模型参数
w = torch.normal(0, 0.01, size = (2, 1), requires_grad = True)
b = torch.zeros(1, requires_grad=True)

# 选择模型
def linreg(X, w, b): #@save
    return torch.matmul(X, w) + b

# 选择损失函数
def squared_loss(y_hat, y): #@save
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 # 防止行列向量不同，所以reshape

# 选择优化器
def sgd(params, lr, batch_size): #@save
    with torch.no_grad(): # 在参数更新时禁用梯度
        for param in params:
            param -= lr * param.grad / batch_size # 均方损失时未除size
            param.grad.zero_() # 置零

# 开始训练
# 参数设置
lr = 0.03 # 学习率
num_epochs = 3 # 迭代周期
net = linreg # 模型
loss = squared_loss # 损失函数

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels): # 分批次处理
        l = loss(net(X, w, b), y) # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size) # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {train_l.mean():f}')

# 评估模型
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
'''