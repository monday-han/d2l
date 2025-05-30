trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd}, # 在优化器中的w参数增加正则项
        {"params":net[0].bias}], lr=lr)