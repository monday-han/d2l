import torch
from torch import nn
from d2l import torch as d2l
# 数据分批，得到优化器梯度下降批次
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
lr = 0.1
epochs = 10


net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

def init_weight(a):
    if type(a) == nn.Linear:
        nn.init.normal_(a.weight, std = 0.01)
        
net.apply(init_weight)

loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

d2l.train_ch3(net, train_iter, test_iter, loss, epochs, trainer)

'''
import torch
from torch import nn
from d2l import torch as d2l
# 设置批次，读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256
# 建立隐藏层
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.randn(num_hiddens, requires_grad=True) * 0.01)
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.randn(num_outputs, requires_grad=True) * 0.01)
# 当参数设置全1或全0时，会发生所有梯度更新相同（对称性灾难），梯度消失（参数无法更新）
params = [W1, b1, W2, b2]
# relu函数设置
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(a, X)
# 建立模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1) # @代表矩阵乘法
    return (H @ W2 + b2)
# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')
# 设置学习参数
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
'''