import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from torch.utils.data import TensorDataset
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from torch.nn import functional as F
from VAE import VAE
from VAE_modules import Discriminator
from nn_misc import \
    KLD_loss, kl_conditional_and_marg, discriminator_loss, train_dataloader
# from early_stopping import EarlyStopping
# load data
import pickle

def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


for k in range(0,10):
    print("正在进行第", k + 1, "折图像生成......")
    with open('/home/lijinjin22/ADNI_GM_1/ADNI_auto_train_test/pkl_test/ADNI_test' + str(k + 1) + '.pkl',
              'rb') as f:
        test_loader = pickle.load(f)

    # 加载模型
    model_path = os.path.join('./weights/', 'cVAE_Gen_wegihts_fold_' + str(k + 1) + '.pt')
    net_G = torch.load(model_path)
    net_G.eval()

    # ADNI
    fake_datas = []
    label = []
    with torch.no_grad():
        for batch_idx, (real_data, y) in enumerate(test_loader):
            # real_data = real_data.squeeze()
            batch_x = real_data[:, :, 1:, 5:-4, 1:] #(112,128,112)
            batch_x = batch_x.reshape(-1,112*128*112)
            # real_data = real_data[:, :, 8:-8, :, 3:-13]  #(96,128,9real_data = real_data[0][0].cpu().numpy()6)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            batch_y = y.to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.int64)

            batch_y = F.one_hot(batch_y, num_classes=2)

            ####################################################
            #  Generator (VAE)
            ####################################################
            [x_hat, x, z_mean, z_logvar] = net_G(batch_x, batch_y, 0)
            x_hat = x_hat.reshape(-1, 112 , 128 , 112)
            x_hat = x_hat[0]
            fake_datas.append(x_hat.cpu().numpy())

            y = y[0].numpy()
            label.append(y)
    # 保存数组到文件（使用np.save函数）
    np.save('/home/lijinjin22/MAE/CVAE_images/ADNI_test_fake_datas_' + str(k + 1) + '.npy',
            np.array(fake_datas))
    np.save('/home/lijinjin22/MAE/CVAE_images/ADNI_test_label_' + str(k + 1) + '.npy',
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
    train_file = '/home/lijinjin22/MAE/CVAE_images/ADNI_test_fake_datas_' + str(k + 1) + '.pkl'
    with open(train_file, 'wb') as f:
        pickle.dump(test_data_loader, f)

    exit(0)



    # HCP
    fake_datas = []
    label = []
    with torch.no_grad():
        for batch_idx, (real_data, y) in enumerate(HCP_test_loader):
            # real_data = real_data.squeeze()
            batch_x = real_data[:, :, 1:, 5:-4, 1:]  # (112,128,112)
            batch_x = batch_x.reshape(-1, 112 * 128 * 112)
            # real_data = real_data[:, :, 8:-8, :, 3:-13]  #(96,128,9real_data = real_data[0][0].cpu().numpy()6)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            y = y -2
            batch_y = y.to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.int64)

            batch_y = F.one_hot(batch_y, num_classes=2)

            ####################################################
            #  Generator (VAE)
            ####################################################
            [x_hat, x, z_mean, z_logvar] = net_G(batch_x, batch_y, 0)
            x_hat = x_hat.reshape(-1, 112, 128, 112)
            x_hat = x_hat[0]
            fake_datas.append(x_hat.cpu().numpy())
            y = y[0].numpy()
            label.append(y)
    # 保存数组到文件（使用np.save函数）
    np.save('/home/manjianzhi/jinjin/MAE/CVAE_images/HCP_test_fake_datas_' + str(k + 1) + '.npy',
            np.array(fake_datas))
    np.save('/home/manjianzhi/jinjin/MAE/CVAE_images/HCP_test_label_' + str(k + 1) + '.npy',
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
    train_file = '/home/manjianzhi/jinjin/MAE/CVAE_images/HCP_test_fake_datas_' + str(k + 1) + '.pkl'
    with open(train_file, 'wb') as f:
        pickle.dump(test_data_loader, f)


    # NACC
    fake_datas = []
    label = []
    with torch.no_grad():
        for batch_idx, (real_data, y) in enumerate(NACC_test_loader):
            # real_data = real_data.squeeze()
            batch_x = real_data[:, :, 1:, 5:-4, 1:]  # (112,128,112)
            batch_x = batch_x.reshape(-1, 112 * 128 * 112)
            # real_data = real_data[:, :, 8:-8, :, 3:-13]  #(96,128,9real_data = real_data[0][0].cpu().numpy()6)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            y = y - 3
            batch_y = y.to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.int64)

            batch_y = F.one_hot(batch_y, num_classes=2)

            ####################################################
            #  Generator (VAE)
            ####################################################
            [x_hat, x, z_mean, z_logvar] = net_G(batch_x, batch_y, 0)
            x_hat = x_hat.reshape(-1, 112, 128, 112)
            x_hat = x_hat[0]
            fake_datas.append(x_hat.cpu().numpy())

            y = y[0].numpy()
            label.append(y)
    # 保存数组到文件（使用np.save函数）
    np.save('/home/manjianzhi/jinjin/MAE/CVAE_images/NACC_test_fake_datas_' + str(k + 1) + '.npy',
            np.array(fake_datas))
    np.save('/home/manjianzhi/jinjin/MAE/CVAE_images/NACC_test_label_' + str(k + 1) + '.npy',
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
    train_file = '/home/manjianzhi/jinjin/MAE/CVAE_images/NACC_test_fake_datas_' + str(k + 1) + '.pkl'
    with open(train_file, 'wb') as f:
        pickle.dump(test_data_loader, f)

    # OASIS
    fake_datas = []
    label = []
    with torch.no_grad():
        for batch_idx, (real_data, y) in enumerate(OASIS_test_loader):
            # real_data = real_data.squeeze()
            batch_x = real_data[:, :, 1:, 5:-4, 1:]  # (112,128,112)
            batch_x = batch_x.reshape(-1, 112 * 128 * 112)
            # real_data = real_data[:, :, 8:-8, :, 3:-13]  #(96,128,9real_data = real_data[0][0].cpu().numpy()6)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            y = y - 4
            batch_y = y.to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.int64)

            batch_y = F.one_hot(batch_y, num_classes=2)

            ####################################################
            #  Generator (VAE)
            ####################################################
            [x_hat, x, z_mean, z_logvar] = net_G(batch_x, batch_y, 0)
            x_hat = x_hat.reshape(-1, 112, 128, 112)
            x_hat = x_hat[0]
            fake_datas.append(x_hat.cpu().numpy())

            y = y[0].numpy()
            label.append(y)
    # 保存数组到文件（使用np.save函数）
    np.save('/home/manjianzhi/jinjin/MAE/CVAE_images/OASIS_test_fake_datas_' + str(k + 1) + '.npy',
            np.array(fake_datas))
    np.save('/home/manjianzhi/jinjin/MAE/CVAE_images/OASIS_test_label_' + str(k + 1) + '.npy',
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
    train_file = '/home/manjianzhi/jinjin/MAE/CVAE_images/OASIS_test_fake_datas_' + str(k + 1) + '.pkl'
    with open(train_file, 'wb') as f:
        pickle.dump(test_data_loader, f)

    print("第",str(k+1),"折CVAE图像生成完毕！文件已保存！")