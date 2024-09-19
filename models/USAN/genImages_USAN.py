import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from unet_2d_model import UNet, segmenter, domain_predictor
from utils import Args, EarlyStopping_unlearning
from confusion_loss import confusion_loss
from dice_loss import dice_loss
from torch.utils.data import TensorDataset
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
    # Load the model
    unet = UNet()
    segmenter = segmenter()
    unet = unet.cuda()
    segmenter = segmenter.cuda()

    unet.load_state_dict(torch.load('./weights/unet_wegihts_fold_' + str(k + 1) + '.pth'))
    segmenter.load_state_dict(torch.load('./weights/segmenter_wegiht_fold_' + str(k + 1) + '.pth'))

    unet.eval()
    segmenter.eval()

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
            #  Generator (USAN)
            ####################################################
            features = unet(real_data)
            output_pred = segmenter(features)
            output_pred = output_pred[0][0]
            fake_datas.append(output_pred.cpu().numpy())

            y = y[0].cpu().numpy()
            label.append(y)
    # 保存数组到文件（使用np.save函数）
    np.save('/home/lijinjin22/MAE/USAN_images/ADNI_test_fake_datas_' + str(k + 1) + '.npy',
            np.array(fake_datas))
    np.save('/home/lijinjin22/MAE/USAN_images/ADNI_test_label_' + str(k + 1) + '.npy',
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
    train_file = '/home/lijinjin22/MAE/USAN_images/ADNI_test_fake_datas_' + str(k + 1) + '.pkl'
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
            #  Generator (USAN)
            ####################################################
            features = unet(real_data)
            output_pred = segmenter(features)
            output_pred = output_pred[0][0]
            fake_datas.append(output_pred.cpu().numpy())
            y = y[0].cpu().numpy()
            label.append(y)
    # 保存数组到文件（使用np.save函数）
    np.save('/home/manjianzhi/jinjin/MAE/USAN_images/HCP_test_fake_datas_' + str(k + 1) + '.npy',
            np.array(fake_datas))
    np.save('/home/manjianzhi/jinjin/MAE/USAN_images/HCP_test_label_' + str(k + 1) + '.npy',
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
    train_file = '/home/manjianzhi/jinjin/MAE/USAN_images/HCP_test_fake_datas_' + str(k + 1) + '.pkl'
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
            #  Generator (USAN)
            ####################################################
            features = unet(real_data)
            output_pred = segmenter(features)
            output_pred = output_pred[0][0]
            fake_datas.append(output_pred.cpu().numpy())

            y = y[0].cpu().numpy()
            label.append(y)
    # 保存数组到文件（使用np.save函数）
    np.save('/home/manjianzhi/jinjin/MAE/USAN_images/NACC_test_fake_datas_' + str(k + 1) + '.npy',
            np.array(fake_datas))
    np.save('/home/manjianzhi/jinjin/MAE/USAN_images/NACC_test_label_' + str(k + 1) + '.npy',
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
    train_file = '/home/manjianzhi/jinjin/MAE/USAN_images/NACC_test_fake_datas_' + str(k + 1) + '.pkl'
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
            #  Generator (USAN)
            ####################################################
            features = unet(real_data)
            output_pred = segmenter(features)
            output_pred = output_pred[0][0]
            fake_datas.append(output_pred.cpu().numpy())

            y = y[0].cpu().numpy()
            label.append(y)
    # 保存数组到文件（使用np.save函数）
    np.save('/home/manjianzhi/jinjin/MAE/USAN_images/OASIS_test_fake_datas_' + str(k + 1) + '.npy',
            np.array(fake_datas))
    np.save('/home/manjianzhi/jinjin/MAE/USAN_images/OASIS_test_label_' + str(k + 1) + '.npy',
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
    train_file = '/home/manjianzhi/jinjin/MAE/USAN_images/OASIS_test_fake_datas_' + str(k + 1) + '.pkl'
    with open(train_file, 'wb') as f:
        pickle.dump(test_data_loader, f)

    print("第",str(k+1),"折USAN图像生成完毕！文件已保存！")