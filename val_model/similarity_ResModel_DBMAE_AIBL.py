import math
from functools import partial

import torch

import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import numpy as np
from torch.utils.data import TensorDataset
import sys
sys.path.append('/home/bci/pycharm-project/MRI/mae-main/generator_model')
import Gen_ResModel_DBMAE
# load data
import pickle

for k in range(0, 10):
    print("正在进行第", k + 1, "折图像生成......")
    with open('/home/bci/pycharm-project/MRI/AIBL/pkl/AIBL_AD_NC_train_batch'+str(k+1)+'.pkl', 'rb') as f:
       train_loader = pickle.load(f)
    # HCP NACC OASIS
    with open('/home/bci/pycharm-project/MRI/AIBL/pkl/AIBL_AD_NC_test'+str(k+1)+'.pkl', 'rb') as f:
       test_loader = pickle.load(f)

    # 初始化resnetgan网络，用于特征提取
    mae = Gen_ResModel_DBMAE.mae_vit_base_patch16_dec512d8b()
    mae = mae.cuda()

    # load model
    mae.load_state_dict(torch.load('../weights/pretrain_10fold_new/ResModel_DBMAE_wegihts_fold_'+str(k+1)+'.pth'))

    mae.eval()
    real_datas = []
    fake_datas = []
    label = []
    with torch.no_grad():
        for batch_idx, (real_data, y) in enumerate(train_loader):
            # real_data = real_data.squeeze(1)
            real_data = real_data[:, :, 1:, 5:-4, 1:]
            real_data = real_data.type(torch.FloatTensor)
            real_data = real_data.to(device)
            fake_data = mae(real_data)

            real_data = real_data[0][0]
            fake_data = fake_data[0][0]

            real_datas.append(real_data.cpu().numpy())
            fake_datas.append(fake_data.cpu().numpy())

            y = y[0].numpy()
            label.append(y)

    # 保存数组到文件（使用np.save函数）
    np.save('/home/bci/pycharm-project/MRI/MAE/ResModel_DBMAE_images_new/AIBL_AD_NC_train_fake_datas'+str(k+1)+'.npy', np.array(fake_datas))
    np.save('/home/bci/pycharm-project/MRI/MAE/ResModel_DBMAE_images_new/AIBL_AD_NC_train_label'+str(k+1)+'.npy', np.array(label))

    # 转换为pkl文件
    X_train = np.array(fake_datas)
    X_train = torch.tensor(X_train, dtype=torch.float)
    X_train = X_train.unsqueeze(1)

    y_train = torch.tensor(np.array(label), dtype=torch.int)

    train_dataset = TensorDataset(X_train, y_train)

    # 数据读取
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0,
                                                   drop_last=True)
    # test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    train_file = '/home/bci/pycharm-project/MRI/MAE/ResModel_DBMAE_images_new/AIBL_AD_NC_train_fake_datas'+str(k+1)+'.pkl'

    with open(train_file, 'wb') as f:
        pickle.dump(train_data_loader, f)

    real_datas = []
    fake_datas = []
    label = []
    with torch.no_grad():
        for batch_idx, (real_data, y) in enumerate(test_loader):
            # real_data = real_data.squeeze(1)
            real_data = real_data[:, :, 1:, 5:-4, 1:]
            real_data = real_data.type(torch.FloatTensor)
            real_data = real_data.to(device)
            fake_data = mae(real_data)

            real_data = real_data[0][0]
            fake_data = fake_data[0][0]

            real_datas.append(real_data.cpu().numpy())
            fake_datas.append(fake_data.cpu().numpy())

            y = y[0].numpy()
            label.append(y)

    # 保存数组到文件（使用np.save函数）
    np.save('/home/bci/pycharm-project/MRI/MAE/ResModel_DBMAE_images_new/AIBL_AD_NC_test_fake_datas'+str(k+1)+'.npy',
            np.array(fake_datas))
    np.save('/home/bci/pycharm-project/MRI/MAE/ResModel_DBMAE_images_new/AIBL_AD_NC_test_label'+str(k+1)+'.npy', np.array(label))

    # 转换为pkl文件
    X_test = np.array(fake_datas)
    X_test = torch.tensor(X_test, dtype=torch.float)
    X_test = X_test.unsqueeze(1)

    y_test = torch.tensor(np.array(label), dtype=torch.int)

    test_dataset = TensorDataset(X_test, y_test)

    # 数据读取
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0,
                                                    drop_last=True)
    # test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    test_file = '/home/bci/pycharm-project/MRI/MAE/ResModel_DBMAE_images_new/AIBL_AD_NC_test_fake_datas'+str(k+1)+'.pkl'

    with open(test_file, 'wb') as f:
        pickle.dump(test_data_loader, f)



    print("第",str(k+1),"折ResModel-DBMAE图像生成完毕！文件已保存！")