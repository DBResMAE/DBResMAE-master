
import math
from functools import partial

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
from train_utils_segmentation import train_encoder_domain_unlearn_semi, val_encoder_domain_unlearn_semi, train_unlearn_semi, val_unlearn_semi

# load data
import pickle
args = Args()
args.channels_first = True
args.epochs = 50
args.batch_size = 4
args.diff_model_flag = False
args.alpha = 50
args.patience = 100
im_size = (112, 128, 112)

for k in range(0,10):
    print("正在进行第",k+1,"折模型训练......")
    with open('/home/lijinjin22/ADNI_GM_1/ADNI_auto_train_test/pkl_train/ADNI_train'+str(k+1)+'.pkl', 'rb') as f:
        train_loader = pickle.load(f)
    img_shape = (1, 112, 128, 112)

    # Load the model
    unet = UNet()
    segmenter = segmenter()
    domain_pred = domain_predictor(2)

    unet = unet.cuda()
    segmenter = segmenter.cuda()
    domain_pred = domain_pred.cuda()

    criteron = dice_loss()
    criteron.cuda()
    domain_criterion = nn.BCELoss()
    domain_criterion.cuda()
    conf_criterion = confusion_loss()
    conf_criterion.cuda()

    # optimizer_step1 = optim.Adam(
    #     list(unet.parameters()) + list(segmenter.parameters()) + list(domain_pred.parameters()), lr=args.learning_rate)
    optimizer = optim.Adam(list(unet.parameters()) + list(segmenter.parameters()), lr=1e-4)
    # optimizer_conf = optim.Adam(list(unet.parameters()), lr=1e-4)
    # optimizer_dm = optim.Adam(list(domain_pred.parameters()), lr=1e-4)  # Lower learning rate for the unlearning bit

    # Initalise the early stopping
    early_stopping = EarlyStopping_unlearning(args.patience, verbose=False)
    loss_store = []
    num_epochs = 50
    # 训练生成对抗网络
    for epoch in range(num_epochs):
        regressor_loss = 0.
        epoch_loss = 0.
        num_batch = 0
        for batch_idx, (real_data, y) in enumerate(train_loader):
            # real_data = real_data.squeeze()
            real_data = real_data[:, :, 1:, 5:-4, 1:] #(112,128,112)
            # real_data = real_data[:, :, 8:-8, :, 3:-13]  #(96,128,9real_data = real_data[0][0].cpu().numpy()6)
            real_data = real_data.type(torch.FloatTensor)
            real_data = real_data.to(device)
            y = y.to(device)
            # 更新生成器
            optimizer.zero_grad()
            features = unet(real_data)
            # print(features.shape)
            output_pred = segmenter(features)
            # print(output_pred.shape)
            loss_1 = criteron(output_pred, real_data)
            loss_1.backward()
            optimizer.step()

            regressor_loss += loss_1
            num_batch += 1
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\t生成器Loss: {:.6f}'.format(
                epoch, batch_idx * len(real_data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss_1), end='')

        av_loss = regressor_loss / num_batch
        print("    epoch:", epoch, "生成器loss:", av_loss)




    torch.save(unet.state_dict(), './weights/unet_wegihts_fold_'+ str(k + 1) +'.pth')
    torch.save(segmenter.state_dict(), './weights/segmenter_wegiht_fold_'+ str(k + 1) +'.pth')


    print("第",str(k+1),"折USAN模型训练完毕！模型权重已保存！")
    exit(0)