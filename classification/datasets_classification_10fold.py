import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE

all_acc = []
for k in range(0, 10):
    print("正在计算第", k + 1, "折数据集分类......")
    load_ADNI_array = np.load('/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/ADNI_test_fake_datas_'+str(k+1)+'.npy')
    load_HCP_array = np.load('/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/HCP_test_fake_datas_'+str(k+1)+'.npy')
    load_NACC_array = np.load('/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/NACC_test_fake_datas_'+str(k+1)+'.npy')
    load_OASIS_array = np.load('/home/manjianzhi/jinjin/MAE/ResModel_DBMAE_images_new/OASIS_test_fake_datas_'+str(k+1)+'.npy')

    fake_all_data = np.vstack((load_ADNI_array.reshape(-1, 112 * 128 * 112),
                          load_HCP_array.reshape(-1, 112 * 128 * 112),
                          load_NACC_array.reshape(-1, 112 * 128 * 112),
                          load_OASIS_array.reshape(-1, 112 * 128 * 112)))
    all_data = fake_all_data
    print(all_data.shape)

    # tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=0)
    # reduced_data2 = tsne.fit_transform(all_data)

    #创建标签数组
    labels = np.concatenate((np.ones(load_ADNI_array.shape[0],dtype=int),
                             2 * np.ones(load_HCP_array.shape[0],dtype=int),
                             3 * np.ones(load_NACC_array.shape[0],dtype=int),
                             4 * np.ones(load_OASIS_array.shape[0],dtype=int)))

    labels = labels - 1
    labels2 = labels.reshape(-1, 1)
    # 创建 OneHotEncoderz
    encoder = OneHotEncoder(sparse=False)
    # 进行 one-hot 编码
    one_hot_encoded = encoder.fit_transform(labels2)
    # print(one_hot_encoded)
    # exit(0)

    #划分数据集
    X_train, X_test, y_train, y_test = train_test_split(all_data, one_hot_encoded, stratify=one_hot_encoded, test_size=0.2,random_state=42,shuffle=True)

    #选择模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    #模型训练
    model.fit(X_train, y_train)

    #模型评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(accuracy)
    print(report)
    all_acc.append(accuracy)
    print("第", str(k + 1), "折数据集分类结束！")

print("ResModel_DBMAE模型10折数据集分类结果：")
print(all_acc)
print("平均分类准确率：")
all_acc = np.array(all_acc)
avg_acc = np.mean(all_acc)
print(avg_acc)
