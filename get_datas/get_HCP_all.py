import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset

from pkl_getdata2 import MyDataset


path = '/home/manjianzhi/jinjin/CSV_100'
path2 = "/home/manjianzhi/jinjin/HCP_T1_prosessed"

# 这个地方控制运行那几折，现在k取值是0-9，对应1-10折
for k in range(1, 2):
    print("正在读取第", k+1, "折数据，请稍等......")
    y_val = []


    # 下面这句是训练集和测试集的csv文件路径，你就弄一个训练集得就行，测试集删了
    val_path = path + "/HCP_data.csv"


    file_path = path2 + "/"


    val_data = MyDataset(val_path, file_path)

    n = len(val_data.T1_data)
    for i in range(n):
        T1_imgs_1, val_label = val_data[i]
        if i == 0:
            X_val = T1_imgs_1
        else:
            X_val = torch.cat((X_val,T1_imgs_1), dim=0)
        print(X_val.shape)
        y_val.append(val_label)


    # X_train = torch.tensor(X_train, dtype=torch.float)
    y_val = torch.tensor(y_val, dtype=torch.int)


    X_val = X_val.unsqueeze(1)


    val_dataset = TensorDataset(X_val, y_val)


    # 数据读取
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

    train_file = '/home/manjianzhi/jinjin/HCP_pkl/HCP_all.pkl'

    with open(train_file, 'wb') as f:
        pickle.dump(val_data_loader, f)


    print(X_val.shape)
    print(y_val.shape)

    print("第", k + 1, "折数据读取完毕。")


