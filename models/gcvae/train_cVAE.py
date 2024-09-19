
import torch
import torch.optim as optim
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

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
    print("正在进行第",k+1,"折模型训练......")
    with open('/home/manjianzhi/jinjin/ADNI_GM_1/ADNI_auto_train_test/pkl_train/ADNI_train'+str(k+1)+'.pkl', 'rb') as f:
        train_loader = pickle.load(f)

        # build model, Generator(VAE) and Discriminator (D)
    net_G = VAE(
        in_dim=1605632,
        nb_classes=2,
        latent_dim=32,
        p_dropout=0.1,
        hidden_dims=[99,64])
    net_D = Discriminator(
        in_dim=1605632,
        nb_classes=2,  # x or x_hat
        p_dropout=0.1,
        hidden_dims=[32,32])
    # move to device
    net_G.to(device)
    net_D.to(device)
    # optimizers for G and D
    optimizer_G = torch.optim.Adam(
        params=net_G.parameters(),
        lr=1e-2,
        betas=(0.9, 0.999),
        eps=1e-7,
        amsgrad=False)
    optimizer_D = torch.optim.Adam(
        params=net_D.parameters(),
        lr=1e-2,
        betas=(0.9, 0.999),
        eps=1e-7,
        amsgrad=False)
    scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_G, milestones=[100], gamma=0.1)
    scheduler_D = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_D, milestones=[100, 800], gamma=0.1)
    # val mse logger
    best_G = None
    best_D = None
    best_val_mse = 1e5
    val_mse_logger = []
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 1


    # 训练生成对抗网络
    for epoch in range(num_epochs):
        net_G.train()
        net_D.train()
        epoch_loss = 0.
        num_batch = 0
        for batch_idx, (real_data, y) in enumerate(train_loader):
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
            # Train Discriminator k times (k=args.nb_adv_epochs)
            ####################################################
            for _ in range(1):
                net_D.zero_grad()
                # train with real samples: batch_x
                real_labels = torch.ones((batch_x.shape[0]), device=device)
                real_pred = net_D(batch_x, 0)
                real_loss = discriminator_loss(real_pred, real_labels)
                real_loss.backward()
                # train with fake samples : reconstructed x_hat
                [x_hat, _, _, _] = net_G(batch_x, batch_y, 0)
                fake_labels = torch.zeros((x_hat.shape[0]), device=device)
                # using detach() to avoid accumulating gradient on Generator
                fake_pred = net_D(x_hat.detach(), 0)
                fake_loss = discriminator_loss(fake_pred, fake_labels)
                fake_loss.backward()
                optimizer_D.step()

            ####################################################
            # Train Generator (VAE)
            ####################################################
            net_G.zero_grad()
            [x_hat, x, z_mean, z_logvar] = net_G(batch_x, batch_y, 0)
            mse = F.mse_loss(x, x_hat)
            prior_loss = KLD_loss(z_mean, z_logvar)
            margin_loss = \
                kl_conditional_and_marg(z_mean, z_logvar, 32)
            real_labels = torch.ones((batch_x.shape[0]), device=device)
            real_pred = net_D(batch_x, 0)
            real_loss = discriminator_loss(real_pred, real_labels)
            fake_labels = torch.zeros((x_hat.shape[0]), device=device)
            fake_pred = net_D(x_hat, 0)
            fake_loss = discriminator_loss(fake_pred, fake_labels)
            d_fake_loss = discriminator_loss(fake_pred, real_labels)
            # loss for Generator
            G_loss = 1 * mse + 0.001 * prior_loss + \
                     0.001 * (margin_loss + mse) + 1. * d_fake_loss
            G_loss.backward()
            optimizer_G.step()

            scheduler_G.step()
            scheduler_D.step()


            best_G = net_G
            best_D = net_D


    print("第",str(k+1),"折CVAE模型训练完毕！模型权重已保存！")