

import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import model_vit_ResModel_DBMAE
from early_stopping import EarlyStopping

import pickle

train_acc = []
val_acc = []
fake_test_acc = []
real_test_acc = []

for k in range(0, 10):
    print("正在计算第", k + 1, "折病理分类......")
    with open('/home/manjianzhi/jinjin/MAE/white_norm/ADNI_NyulNormalize_'+str(k+1)+'.pkl', 'rb') as f:
        fake_test_loader = pickle.load(f)

    # with open('/home/manjianzhi/jinjin/ADNI_GM_1/ADNI_auto_train_test/pkl_test/ADNI_test'+str(k+1)+'.pkl', 'rb') as f:
    #     real_test_loader = pickle.load(f)


    mae = model_vit_ResModel_DBMAE.vit_base_patch16()
    mae = mae.cuda()

    # load model
    mae.load_state_dict(torch.load('./weights/finetune_ResModel_DBMAE/finetune_fold_'+str(k+1)+'.pth'), strict=False)


    print("CVAE模型生成图像测试集精度：")
    # check prediction
    m = len(fake_test_loader.dataset)
    batch_size = fake_test_loader.batch_size

    y_pred_eegnet = np.zeros(m)
    y_true = np.zeros(m)
    mae.eval()
    with torch.no_grad():
        for batch_idx, (data, y) in tqdm(enumerate(fake_test_loader, 0), total=int(np.ceil(m / batch_size))):
            data = data[:, :, 1:, 5:-4, 1:]
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            # eegnet prediction
            outputs_eegnet = mae(data)
            _, y_pred = torch.max(outputs_eegnet.data, 1)
            y_pred_eegnet[batch_idx * batch_size:(batch_idx + 1) * batch_size] = y_pred.cpu().numpy()

            # labels
            y_true[batch_idx * batch_size:(batch_idx + 1) * batch_size] = y.numpy()

    print("   accuracy {:.5f}%".format((y_true == y_pred_eegnet).sum() / m * 100))
    fake_test_a = (y_true == y_pred_eegnet).sum() / m * 100
    fake_test_acc.append(fake_test_a)




    # print("原始图像测试集精度：")
    # # check prediction
    # m = len(real_test_loader.dataset)
    # batch_size = real_test_loader.batch_size
    #
    # y_pred_eegnet = np.zeros(m)
    # y_true = np.zeros(m)
    # mae.eval()
    # with torch.no_grad():
    #     for batch_idx, (data, y) in tqdm(enumerate(real_test_loader, 0), total=int(np.ceil(m / batch_size))):
    #         data = data[:, :, 1:, 5:-4, 1:]
    #         data = data.type(torch.FloatTensor)
    #         data = data.to(device)
    #         # eegnet prediction
    #         outputs_eegnet = mae(data)
    #         _, y_pred = torch.max(outputs_eegnet.data, 1)
    #         y_pred_eegnet[batch_idx * batch_size:(batch_idx + 1) * batch_size] = y_pred.cpu().numpy()
    #
    #         # labels
    #         y_true[batch_idx * batch_size:(batch_idx + 1) * batch_size] = y.numpy()
    #
    # print("   accuracy {:.5f}%".format((y_true == y_pred_eegnet).sum() / m * 100))
    # real_test_a = (y_true == y_pred_eegnet).sum() / m * 100
    # real_test_acc.append(real_test_a)

    print("第", str(k + 1), "折病理分类结束！")


print("Nyul模型10折病理分类结果：")

print("生成图像测试集：")
print(fake_test_acc)
print("平均分类准确率：")
all_acc = np.array(fake_test_acc)
avg_acc = np.mean(all_acc)
print(avg_acc)
# print("=================================")
#
# print("原始图像测试集：")
# print(real_test_acc)
# print("平均分类准确率：")
# all_acc = np.array(real_test_acc)
# avg_acc = np.mean(all_acc)
# print(avg_acc)
