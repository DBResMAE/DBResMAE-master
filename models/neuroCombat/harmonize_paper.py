from neuroCombat import neuroCombat
import pandas as pd
import numpy as np
import pickle

data_list1 = np.load('/home/manjianzhi/jinjin/ADNI_GM_1/ADNI_auto_train_test/pkl_test/ADNI_test_no_norm1.npy')
print(data_list1.shape)

data_list2 = np.load('/home/manjianzhi/jinjin/HCP_pkl/HCP_test1.npy')
print(data_list2.shape)

data_list3 = np.load('/home/manjianzhi/jinjin/NACC_pkl/NACC_test1.npy')
print(data_list3.shape)

data_list4 = np.load('/home/manjianzhi/jinjin/OASIS_pkl/OASIS_test1.npy')
print(data_list4.shape)


# 合并所有数据集成一个大数据集
feature_data = np.vstack((data_list1[:100].reshape(-1, 113 * 137 * 113), data_list2[:].reshape(-1, 113 * 137 * 113),
                 data_list3[:].reshape(-1, 113 * 137 * 113), data_list4[:].reshape(-1, 113 * 137 * 113)))
# 转置数组
feature_data = np.transpose(feature_data)

#创建标签数组
labels = np.concatenate((np.ones(len(data_list1)-1,dtype=int),
                         2 * np.ones(len(data_list2),dtype=int),
                         3 * np.ones(len(data_list3),dtype=int),
                         4 * np.ones(len(data_list4),dtype=int)))

# 使用tolist()方法将数组转换为列表
labels = labels.tolist()


# Getting example data
# 200 rows (features) and 10 columns (scans)
data = feature_data

# Specifying the batch (scanner variable) as well as a biological covariate to preserve:
covars = {'batch':labels}
covars = pd.DataFrame(covars)

# To specify names of the variables that are categorical:
# categorical_cols = ['gender']

# To specify the name of the variable that encodes for the scanner/batch covariate:
batch_col = 'batch'

#Harmonization step:
data_combat = neuroCombat(dat=data,
    covars=covars,
    batch_col=batch_col)["data"]

# 转置数组
data_combat = np.transpose(data_combat)

# 保存数组到文件（使用np.save函数）
np.save('/home/manjianzhi/jinjin/Combat/Harmonize_ADNI_HCP_NACC_OASIS_data.npy', data_combat)
