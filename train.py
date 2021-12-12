from builder.model import build_model
from builder.optimizer import build_optimizer
from builder.scheduler import build_lr_scheduler
from builder.dataloader import build_train_loader
from utils.model_utils import resume_or_load,save_model
import torch
import numpy as np
import random

from config import cfg

def set_seed(cfg):
    torch.manual_seed(cfg.SEED)     # 为torch框架设置seed种子
    np.random.seed(cfg.SEED)        # 为np设置seed种子
    random.seed(cfg.SEED)           # 为random设置seed种子

#-------------------------------接着上一次的模型训练----------------------------------#
if __name__ == '__main__':
    set_seed(cfg)               # 设置随机数种子
    model = build_model(cfg)    # 构建模型

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 计算模型中训练参数的总数
    print('train number of params:', n_parameters)

    data_loader = build_train_loader(cfg)           # 构建data_loader

    optimizer = build_optimizer(cfg,model)          # 构建SGDoptimizer

    scheduler = build_lr_scheduler(cfg,optimizer)   # 构建lr_scheduler

    resume_or_load(cfg,model)                       # 加载部分resnet预训练权重

    #---------train----------------#
    d_2 = model.state_dict()
    for epoch in range(cfg.TRAIN.START_EPOCH,cfg.TRAIN.MAX_EPOCH):   # 大概要迭代 36个epoch   (2160000*batch_size)/118287 = 36
        for iter, data in enumerate(data_loader):
            model.train()

            loss_dict = model(data)
            losses = sum([
                metrics_value for metrics_value in loss_dict.values()
                if metrics_value.requires_grad
            ])
            losses /= 1
            optimizer.zero_grad()   # 清空梯度
            losses.backward()       # 计算反向求导
            optimizer.step()        # 更新参数
            scheduler.step()        # 更新lr

            if (iter % 20 == 0):
                print("(epoch", epoch, ")",
                      iter, "/", len(data_loader),
                      "lr:{:.6f}".format(scheduler.get_lr()[0]),
                      "loss:", [{s: "{:.3f}".format(loss_dict[s].cpu().detach().numpy().tolist())} for s in loss_dict]
                      )

        save_model(cfg, model, epoch)  # 每个epoch保存一次模型


