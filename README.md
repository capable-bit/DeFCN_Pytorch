# DeFCN_Pytorch
DeFCN是在PyTorch上实现了“端到端的完全卷积网络对象检测”，基于一种全卷积神经网络，不需要任何NMS等手工后处理，最终实现真正意义上的端到端的目标检测网络。

此项目实现了对原DeFCN作者代码的改写，原作者是基于Cvpods框架进行训练。本项目去掉了的Cvpods框架，实现了纯PyTorch代码去训练DeFCN模型。此项目不需要安装任何框架，下载一些基本的库之后即可开始训练。

以下是训练步骤：

git clone git://github.com/capable-bit/DeFCN_Pytorch.git

cd DeFCN_Pytorch

ln -s your_path/datasets/coco ./datasets

python train.py
