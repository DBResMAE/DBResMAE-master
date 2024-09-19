import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset

from pkl_getdata import MyDataset


path = '/home/lijinjin22/ADNI_GM_1/ADNI_auto_train_test/fold_10'
path2 = "/home/lijinjin22/ADNI_GM_1/ADNI_GM_1"

# 这个地方控制运行那几折，现在k取值是0-9，对应1-10折
for k in range(0, 10):
    print("正在读取第", k+1, "折数据，请稍等......")
    y_train = []
    y_test = []


    # 下面这句是训练集和测试集的csv文件路径，你就弄一个训练集得就行，测试集删了
    test_path = path + "/fold_"+str(k+1)+"_train.csv"


    file_path = path2 + "/"


    test_data = MyDataset(test_path, file_path)

    n = len(test_data.T1_data)
    for i in range(n):
        T1_imgs_1, test_label = test_data[i]
        if i == 0:
            X_test = T1_imgs_1
        else:
            X_test = torch.cat((X_test,T1_imgs_1), dim=0)
        print(X_test.shape)
        y_test.append(test_label)


    y_test = torch.tensor(y_test, dtype=torch.int)


    X_test = X_test.unsqueeze(1)


    test_dataset = TensorDataset(X_test, y_test)


    # 数据读取
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

    train_file = '/home/lijinjin22/ADNI_GM_1/ADNI_auto_train_test/pkl_train/ADNI_train_fold_' +str(k+1)+ '.pkl'

    with open(train_file, 'wb') as f:
        pickle.dump(test_data_loader, f)

    print(X_test.shape)
    print(y_test.shape)

    print("第", k + 1, "折数据读取完毕。")
    exit(0)


