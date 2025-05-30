# concise
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10)) # nn将多维向量展为向量（reshape） 

# 初始化layer的参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std = 0.01) # 初始化

net.apply(init_weights); # 应用到net中

loss = nn.CrossEntropyLoss() # 设置损失函数

trainer = torch.optim.SGD(net.parameters(), lr = 0.1) # 设置优化器

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)





'''
# from 0
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256 # 批量大小
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) # 得到训练、测试集

num_inputs = 784 # 特征数量
num_outputs = 10 # 类别数量

# 参数初始化
W = torch.normal(0, 0.01, size = (num_inputs, num_outputs), requires_grad=True) 
b = torch.zeros(num_outputs, requires_grad=True)

# softmax算法定义
def softmax(X): # 行是n个样本，列是10个预测类别
    X_exp = torch.exp(X)
    partition = X_exp.sum(axis = 1, keepdim = True)# keepdim保持原维度，帮助广播
    return X_exp / partition# 这里应用了广播机制

# 定义模型
def net(X):
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b.reshape(-1, W.shape[1])) # 均根据W的形状转化，b维度不对但是加减pytorch帮你广播（乘除不允许）

# 定义损失函数
def cross_entropy(y_hat, y): # 计算交叉熵
    return - torch.log(y_hat[range(len(y_hat)), y])

# 计算分类精度
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if y_hat.shape[0] > 1 and y_hat.shape[1] > 1:
        y_pre = y_hat.argmax(axis = 1) # 得到预测类别(得到最大元素索引)
    cmp = y_pre.type(y.dtype) == y
    return cmp.type(y.dtype).sum() # pytorch要求布尔值要转化为数值类型才能进行计算

# 对任意数据迭代器评估net的精度
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):# 如果net是nn模型
          net.eval()# 将模型设置为评估模式（还有训练模式）
    metric = Accumulator(2)  # 累加两个值 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel()) # 加入正确数和总数
    return metric[0] / metric[1] # 得到精度
    
# 定义累加器
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n): # 初始化函数
        self.data = [0.0] * n

    def add(self, *args): # 更新列表（累加预测数）
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self): 
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
# 训练
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y) 
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward() # 反向传播需要操作标量，l为损失的向量
            updater(batch_size) # 接收批次大小，计算平均损失梯度
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
     
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

# 动画类
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        
# 最终训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    
lr = 0.1 # 学习率设置

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 20
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
'''