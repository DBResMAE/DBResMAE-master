import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from torch.utils.data import TensorDataset
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# from cyclegan_pytorch import CycleGAN
from network_pytorch import Generator, Discriminator
# load data
import pickle

for k in range(0,1):
    print("正在进行第", k + 1, "折图像生成......")
    with open('/home/lijinjin22/ADNI_GM_1/ADNI_auto_train_test/pkl_test/ADNI_test' + str(k + 1) + '.pkl',
              'rb') as f:
        test_loader = pickle.load(f)
    # HCP NACC OASIS
    # with open('/home/manjianzhi/jinjin/HCP_pkl/HCP_test' + str(k + 1) + '.pkl', 'rb') as f:
    #     HCP_test_loader = pickle.load(f)
    # with open('/home/manjianzhi/jinjin/NACC_pkl/NACC_test' + str(k + 1) + '.pkl', 'rb') as f:
    #     NACC_test_loader = pickle.load(f)
    # with open('/home/manjianzhi/jinjin/OASIS_pkl/OASIS_test' + str(k + 1) + '.pkl', 'rb') as f:
    #     OASIS_test_loader = pickle.load(f)

    img_shape = (1, 112, 128, 112)
    # 加载模型
    generator_AB = os.path.join('./weights/', 'generator_AB_wegihts_fold_' + str(k + 1) + '.pt')
    generator_BA = os.path.join('./weights/', 'generator_BA_wegihts_fold_' + str(k + 1) + '.pt')
    generator_AB = torch.load(generator_AB)
    generator_AB.eval()
    generator_BA = torch.load(generator_BA)
    generator_BA.eval()

    # ADNI
    fake_datas = []
    label = []
    with torch.no_grad():
        for batch_idx, (real_data, y) in enumerate(test_loader):
            # real_data = real_data.squeeze()
            real_data = real_data[:, :, 1:, 5:-4, 1:]  # (112,128,112)
            # real_data = real_data[:, :, 8:-8, :, 3:-13]  #(96,128,9real_data = real_data[0][0].cpu().numpy()6)
            real_data = real_data.type(torch.FloatTensor)
            real_data = real_data.to(device)
            y = y.to(device)

            ####################################################
            #  Generator (GAN)
            ####################################################
            fake_B = generator_AB(real_data)
            rec_A = generator_BA(fake_B)
            rec_A = rec_A[0][0]
            fake_datas.append(rec_A.cpu().numpy())

            y = y[0].cpu().numpy()
            label.append(y)
    # 保存数组到文件（使用np.save函数）
    np.save('/home/lijinjin22/MAE/GAN_images/ADNI_test_fake_datas_' + str(k + 1) + '.npy',
            np.array(fake_datas))
    np.save('/home/lijinjin22/MAE/GAN_images/ADNI_test_label_' + str(k + 1) + '.npy',
            np.array(label))
    # 转换为pkl文件
    X_test = np.array(fake_datas)
    X_test = torch.tensor(X_test, dtype=torch.float)
    X_test = X_test.unsqueeze(1)
    y_test = torch.tensor(np.array(label), dtype=torch.int)

    test_dataset = TensorDataset(X_test, y_test)
    # 数据读取
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0,
                                                   drop_last=True)
    train_file = '/home/lijinjin22/MAE/GAN_images/ADNI_test_fake_datas_' + str(k + 1) + '.pkl'
    with open(train_file, 'wb') as f:
        pickle.dump(test_data_loader, f)

    exit(0)

    # HCP
    fake_datas = []
    label = []
    with torch.no_grad():
        for batch_idx, (real_data, y) in enumerate(HCP_test_loader):
            # real_data = real_data.squeeze()
            real_data = real_data[:, :, 1:, 5:-4, 1:]  # (112,128,112)
            # real_data = real_data[:, :, 8:-8, :, 3:-13]  #(96,128,9real_data = real_data[0][0].cpu().numpy()6)
            real_data = real_data.type(torch.FloatTensor)
            real_data = real_data.to(device)
            y = y.to(device)

            ####################################################
            #  Generator (GAN)
            ####################################################
            fake_B = generator_AB(real_data)
            rec_A = generator_BA(fake_B)
            rec_A = rec_A[0][0]
            fake_datas.append(rec_A.cpu().numpy())
            y = y[0].cpu().numpy()
            label.append(y)
    # 保存数组到文件（使用np.save函数）
    np.save('/home/manjianzhi/jinjin/MAE/GAN_images/HCP_test_fake_datas_' + str(k + 1) + '.npy',
            np.array(fake_datas))
    np.save('/home/manjianzhi/jinjin/MAE/GAN_images/HCP_test_label_' + str(k + 1) + '.npy',
            np.array(label))

    # 转换为pkl文件
    X_test = np.array(fake_datas)
    X_test = torch.tensor(X_test, dtype=torch.float)
    X_test = X_test.unsqueeze(1)
    y_test = torch.tensor(np.array(label), dtype=torch.int)

    test_dataset = TensorDataset(X_test, y_test)
    # 数据读取
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0,
                                                   drop_last=True)
    train_file = '/home/manjianzhi/jinjin/MAE/GAN_images/HCP_test_fake_datas_' + str(k + 1) + '.pkl'
    with open(train_file, 'wb') as f:
        pickle.dump(test_data_loader, f)


    # NACC
    fake_datas = []
    label = []
    with torch.no_grad():
        for batch_idx, (real_data, y) in enumerate(NACC_test_loader):
            # real_data = real_data.squeeze()
            real_data = real_data[:, :, 1:, 5:-4, 1:]  # (112,128,112)
            # real_data = real_data[:, :, 8:-8, :, 3:-13]  #(96,128,9real_data = real_data[0][0].cpu().numpy()6)
            real_data = real_data.type(torch.FloatTensor)
            real_data = real_data.to(device)
            y = y.to(device)

            ####################################################
            #  Generator (GAN)
            ####################################################
            fake_B = generator_AB(real_data)
            rec_A = generator_BA(fake_B)
            rec_A = rec_A[0][0]
            fake_datas.append(rec_A.cpu().numpy())

            y = y[0].cpu().numpy()
            label.append(y)
    # 保存数组到文件（使用np.save函数）
    np.save('/home/manjianzhi/jinjin/MAE/GAN_images/NACC_test_fake_datas_' + str(k + 1) + '.npy',
            np.array(fake_datas))
    np.save('/home/manjianzhi/jinjin/MAE/GAN_images/NACC_test_label_' + str(k + 1) + '.npy',
            np.array(label))

    # 转换为pkl文件
    X_test = np.array(fake_datas)
    X_test = torch.tensor(X_test, dtype=torch.float)
    X_test = X_test.unsqueeze(1)
    y_test = torch.tensor(np.array(label), dtype=torch.int)

    test_dataset = TensorDataset(X_test, y_test)
    # 数据读取
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0,
                                                   drop_last=True)
    train_file = '/home/manjianzhi/jinjin/MAE/GAN_images/NACC_test_fake_datas_' + str(k + 1) + '.pkl'
    with open(train_file, 'wb') as f:
        pickle.dump(test_data_loader, f)

    # OASIS
    fake_datas = []
    label = []
    with torch.no_grad():
        for batch_idx, (real_data, y) in enumerate(OASIS_test_loader):
            # real_data = real_data.squeeze()
            real_data = real_data[:, :, 1:, 5:-4, 1:]  # (112,128,112)
            # real_data = real_data[:, :, 8:-8, :, 3:-13]  #(96,128,9real_data = real_data[0][0].cpu().numpy()6)
            real_data = real_data.type(torch.FloatTensor)
            real_data = real_data.to(device)
            y = y.to(device)

            ####################################################
            #  Generator (GAN)
            ####################################################
            fake_B = generator_AB(real_data)
            rec_A = generator_BA(fake_B)
            rec_A = rec_A[0][0]
            fake_datas.append(rec_A.cpu().numpy())

            y = y[0].cpu().numpy()
            label.append(y)
    # 保存数组到文件（使用np.save函数）
    np.save('/home/manjianzhi/jinjin/MAE/GAN_images/OASIS_test_fake_datas_' + str(k + 1) + '.npy',
            np.array(fake_datas))
    np.save('/home/manjianzhi/jinjin/MAE/GAN_images/OASIS_test_label_' + str(k + 1) + '.npy',
            np.array(label))

    # 转换为pkl文件
    X_test = np.array(fake_datas)
    X_test = torch.tensor(X_test, dtype=torch.float)
    X_test = X_test.unsqueeze(1)
    y_test = torch.tensor(np.array(label), dtype=torch.int)

    test_dataset = TensorDataset(X_test, y_test)
    # 数据读取
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0,
                                                   drop_last=True)
    train_file = '/home/manjianzhi/jinjin/MAE/GAN_images/OASIS_test_fake_datas_' + str(k + 1) + '.pkl'
    with open(train_file, 'wb') as f:
        pickle.dump(test_data_loader, f)

    print("第",str(k+1),"折GAN图像生成完毕！文件已保存！")