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

    # with open('/home/manjianzhi/jinjin/ADNI_GM_1/ADNI_auto_train_test/pkl_train/ADNI_train1.pkl', 'rb') as f:
    #     train_loader = pickle.load(f)
    # with open('/home/manjianzhi/jinjin/ADNI_GM_1/ADNI_auto_train_test/pkl_val/ADNI_val1.pkl', 'rb') as f:
    #     val_loader = pickle.load(f)


    # 初始化resnetgan网络，用于特征提取
    mae = Gen_ResModel_DBMAE.mae_vit_base_patch16_dec512d8b()
    mae = mae.cuda()

    # load model
    # mae.load_state_dict(torch.load('../weights/mae3d_gan_wegihts24.pth'))
    mae.load_state_dict(torch.load('../weights/pretrain_10fold_new/ResModel_DBMAE_wegihts_fold_'+str(k+1)+'.pth'))

    mae.eval()
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
    np.save('/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/ADNI_test_real_datas_'+str(k+1)+'.npy', np.array(real_datas))
    np.save('/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/ADNI_test_fake_datas_'+str(k+1)+'.npy', np.array(fake_datas))
    np.save('/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/ADNI_test_label_'+str(k+1)+'.npy', np.array(label))

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

    train_file = '/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/ADNI_test_fake_datas_'+str(k+1)+'.pkl'

    with open(train_file, 'wb') as f:
        pickle.dump(test_data_loader, f)

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

    # 保存数组到文件（使用np.save函数）
    np.save('/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/HCP_test_real_datas_' + str(k + 1) + '.npy',
            np.array(real_datas))
    np.save('/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/HCP_test_fake_datas_' + str(k + 1) + '.npy',
            np.array(fake_datas))
    np.save('/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/HCP_test_label_' + str(k + 1) + '.npy', np.array(label))

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

    train_file = '/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/HCP_test_fake_datas_' + str(k + 1) + '.pkl'

    with open(train_file, 'wb') as f:
        pickle.dump(test_data_loader, f)

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

    # 保存数组到文件（使用np.save函数）
    np.save('/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/NACC_test_real_datas_' + str(k + 1) + '.npy',
            np.array(real_datas))
    np.save('/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/NACC_test_fake_datas_' + str(k + 1) + '.npy',
            np.array(fake_datas))
    np.save('/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/NACC_test_label_' + str(k + 1) + '.npy', np.array(label))

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

    train_file = '/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/NACC_test_fake_datas_' + str(k + 1) + '.pkl'

    with open(train_file, 'wb') as f:
        pickle.dump(test_data_loader, f)

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

    # 保存数组到文件（使用np.save函数）
    np.save('/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/OASIS_test_real_datas_' + str(k + 1) + '.npy',
            np.array(real_datas))
    np.save('/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/OASIS_test_fake_datas_' + str(k + 1) + '.npy',
            np.array(fake_datas))
    np.save('/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/OASIS_test_label_' + str(k + 1) + '.npy', np.array(label))

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

    train_file = '/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/OASIS_test_fake_datas_' + str(k + 1) + '.pkl'

    with open(train_file, 'wb') as f:
        pickle.dump(test_data_loader, f)

    print("第",str(k+1),"折ResModel-DBMAE图像生成完毕！文件已保存！")