from monitor import check_gpu_usage
import time
import math
from functools import partial

import torch

import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import models_mae
import Baseline_SPEB
from early_stopping import EarlyStopping
# load data
import pickle


for k in range(0,10):
    print("正在进行第",k+1,"折模型训练......")
    with open('/home/manjianzhi/jinjin/ADNI_GM_1/ADNI_auto_train_test/pkl_train/ADNI_train'+str(k+1)+'.pkl', 'rb') as f:
        train_loader = pickle.load(f)


    # 初始化resnetgan网络，用于特征提取
    mae = Baseline_SPEB.mae_vit_base_patch16_dec512d8b()
    mae = mae.cuda()

    eff_batch_size = 1e-3 * 32 / 256

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 欧式距离损失函数
    optimizer_g = torch.optim.AdamW(mae.parameters(), lr=0.0002, betas=(0.9, 0.95), weight_decay=0.05)

    # 用数组保存每一轮迭代中，在测试数据上测试的损失值和精确度，也是为了通过画图展示出来。
    eval_losses = []
    eval_acces = []
    save_path = "./" #当前目录下
    early_stopping = EarlyStopping(save_path)

    num_epochs = 200
    # 训练生成对抗网络
    for epoch in range(num_epochs):
        epoch_loss = 0.
        num_batch = 0
        for batch_idx, (real_data, y) in enumerate(train_loader):
            real_data = real_data[:, :, 1:, 5:-4, 1:] #(112,128,112)
            real_data = real_data.type(torch.FloatTensor)
            real_data = real_data.to(device)
            # 更新生成器
            optimizer_g.zero_grad()
            loss, pred, mask = mae(real_data)

            loss.backward()
            optimizer_g.step()

            iter_loss = loss.item()

            epoch_loss += iter_loss

            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\t生成器Loss: {:.6f}'.format(
                epoch, batch_idx * len(real_data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), iter_loss), end='')
            num_batch = batch_idx
        mean_epoch_loss = epoch_loss / (num_batch + 1)
        print("    epoch:", epoch, "生成器loss:", mean_epoch_loss)

        # 早停止
        early_stopping(epoch_loss, mae)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练


    # torch.save(mae.state_dict(), './weights/mae3d_gan_wegihts22.pth')
    torch.save(mae.state_dict(), './weights/SPEB_10fold/Baseline_SPEB_wegihts_fold_'+str(k+1)+'.pth')

    print("第",str(k+1),"折Baseline_SPEB模型训练完毕！模型权重已保存！")