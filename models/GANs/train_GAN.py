


import torch
import torch.optim as optim
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# from cyclegan_pytorch import CycleGAN
from network_pytorch import Generator, Discriminator
# load data
import pickle


for k in range(0,10):
    print("正在进行第",k+1,"折模型训练......")
    with open('/home/lijinjin22/ADNI_GM_1/ADNI_auto_train_test/pkl_train/ADNI_train'+str(k+1)+'.pkl', 'rb') as f:
        train_loader = pickle.load(f)
    img_shape = (1, 112, 128, 112)


    # 初始化resnetgan网络，用于特征提取
    generator_AB = Generator(img_shape, gf=32, depth=3).cuda()
    generator_BA = Generator(img_shape, gf=32, depth=3).cuda()
    discriminator_A = Discriminator(img_shape, df=64, depth=3).cuda()
    discriminator_B = Discriminator(img_shape, df=64, depth=3).cuda()

    gen_optimizer = optim.Adam(
        list(generator_AB.parameters()) + list(generator_BA.parameters()), lr=2e-4)
    disc_optimizer = optim.Adam(
        list(discriminator_A.parameters()) + list(discriminator_B.parameters()), lr=2e-4)

    criterion_gan = nn.MSELoss().to(device)
    criterion_cycle = nn.L1Loss().to(device)
    criterion_identity = nn.L1Loss().to(device)
    checkpoint_dir = os.path.join('./logs', 'checkpoint')

    num_epochs = 50
    # 训练生成对抗网络
    for epoch in range(num_epochs):
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
            gen_optimizer.zero_grad()

            fake_B = generator_AB(real_data)
            rec_A = generator_BA(fake_B)

            z1 = discriminator_B(fake_B)
            loss_gan_AB = criterion_gan(z1, torch.ones_like(z1))
            loss_cycle_A = criterion_cycle(rec_A, real_data) * 10.0
            loss_identity_A = criterion_identity(generator_BA(real_data), real_data) * 5.0

            gen_loss = loss_gan_AB  + loss_cycle_A  + loss_identity_A
            gen_loss.backward()
            gen_optimizer.step()

            # 训练判别器
            disc_optimizer.zero_grad()
            z2 = discriminator_A(real_data)
            real_loss_A = criterion_gan(z2, torch.ones_like(z2))
            z3 = discriminator_A(fake_B.detach())
            fake_loss_A = criterion_gan(z3, torch.zeros_like(z3))
            disc_A_loss = (real_loss_A + fake_loss_A) / 2

            disc_loss = disc_A_loss
            disc_loss.backward()
            disc_optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], "
                      f"Gen Loss: {gen_loss.item()}, Disc Loss: {disc_loss.item()}")


    # torch.save(mae.state_dict(), './weights/mae3d_gan_wegihts22.pth')
    torch.save(generator_AB, os.path.join('./weights/', 'generator_AB_wegihts_fold_' + str(k + 1) + '.pt'))
    torch.save(generator_BA,
               os.path.join('./weights/', 'generator_BA_wegihts_fold_' + str(k + 1) + '.pt'))
    torch.save(discriminator_A,
             os.path.join('./weights/', 'discriminator_A_wegihts_fold_' + str(k + 1) + '.pt'))
    torch.save(discriminator_B,
             os.path.join('./weights/', 'discriminator_B_wegihts_fold_' + str(k + 1) + '.pt'))

    print("第",str(k+1),"折GAN模型训练完毕！模型权重已保存！")
    exit(0)