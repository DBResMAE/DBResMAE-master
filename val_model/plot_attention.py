import math
from functools import partial

import torch

import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import numpy as np
from torch.utils.data import TensorDataset
import sys
sys.path.append('/home/manjianzhi/jinjin/MRI/mae-main/generator_model')
import Gen_ResModel_DBMAE
# load data
import pickle

for k in range(0, 10):
    print("正在进行第", k + 1, "折图像生成......")
    with open('/home/manjianzhi/jinjin/ADNI_GM_1/ADNI_auto_train_test/pkl_test/ADNI_test'+str(k+1)+'.pkl', 'rb') as f:
       test_loader = pickle.load(f)
    # HCP NACC OASIS
    with open('/home/manjianzhi/jinjin/HCP_pkl/HCP_test'+str(k+1)+'.pkl', 'rb') as f:
       HCP_test_loader = pickle.load(f)
    with open('/home/manjianzhi/jinjin/NACC_pkl/NACC_test'+str(k+1)+'.pkl', 'rb') as f:
       NACC_test_loader = pickle.load(f)
    with open('/home/manjianzhi/jinjin/OASIS_pkl/OASIS_test'+str(k+1)+'.pkl', 'rb') as f:
       OASIS_test_loader = pickle.load(f)

    # 初始化resnetgan网络，用于特征提取
    mae = Gen_ResModel_DBMAE.mae_vit_base_patch16_dec512d8b()
    mae = mae.cuda()

    # load model
    mae.load_state_dict(torch.load('../weights/pretrain_10fold_new/ResModel_DBMAE_wegihts_fold_'+str(k+1)+'.pth'))

    mae.eval()
    real_datas = []
    fake_datas = []
    label = []
    nn = 1
    with torch.no_grad():
        for batch_idx, (real_data, y) in enumerate(test_loader):
            if nn < 12:
                nn = nn + 1
                continue
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

    # HCP
    real_datas = []
    fake_datas = []
    label = []
    with torch.no_grad():
        for batch_idx, (real_data, y) in enumerate(HCP_test_loader):
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


    # NACC
    real_datas = []
    fake_datas = []
    label = []
    with torch.no_grad():
        for batch_idx, (real_data, y) in enumerate(NACC_test_loader):
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


    # OASIS
    real_datas = []
    fake_datas = []
    label = []
    with torch.no_grad():
        for batch_idx, (real_data, y) in enumerate(OASIS_test_loader):
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
