import torch
from torch import nn
from d2l import torch as d2l
# 使用两个隐藏层
# 设置两个dropout值
dropout1 , dropout2 = 0.2, 0.5

batch_size, lr, num_epochs = 256, 0.5, 10
loss = nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 建立模型
net = nn.Sequential(
    nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
    nn.Dropout(dropout1), nn.Linear(256, 256), nn.ReLU(), 
    nn.Dropout(dropout2), nn.Linear(256, 10)
)

def init_weight(a):
    if type(a) == nn.Linear:
        nn.init.normal_(a.weight, std = 0.01)
        
net.apply(init_weight);

trainer = torch.optim.SGD(net.parameters(), lr = lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)