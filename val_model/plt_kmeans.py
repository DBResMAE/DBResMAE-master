import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

load_ADNI_array = np.load('/home/bci/pycharm-project/MRI/MAE/ResModel_DBMAE_images_new/ADNI_test_fake_datas_1.npy')
load_HCP_array = np.load('/home/bci/pycharm-project/MRI/MAE/ResModel_DBMAE_images_new/HCP_test_fake_datas_1.npy')
load_NACC_array = np.load('/home/bci/pycharm-project/MRI/MAE/ResModel_DBMAE_images_new/NACC_test_fake_datas_1.npy')
load_OASIS_array = np.load('/home/bci/pycharm-project/MRI/MAE/ResModel_DBMAE_images_new/OASIS_test_fake_datas_1.npy')


print(load_ADNI_array.shape)
# 合并四个数据集并创建标签数组
fake_all_data = np.vstack((load_ADNI_array.reshape(-1, 113 * 137 * 113), load_HCP_array.reshape(-1, 113 * 137 * 113),
                      load_NACC_array.reshape(-1, 113 * 137 * 113), load_OASIS_array.reshape(-1, 113 * 137 * 113)))



adni_label = np.array(adni_label)


labels = np.concatenate((adni_label,
                         2 * np.ones(len(load_HCP_array)),
                         3 * np.ones(len(load_NACC_array)),
                         4 * np.ones(len(load_OASIS_array))))

# 使用t-SNE降维到2维ne

tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
reduced_data = tsne.fit_transform(fake_all_data)
print(reduced_data[:, 0])
print(reduced_data[:, 1])
# 绘制二维散点图，使用不同颜色表示每个数据集
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', marker='o', s=50)
plt.title('t-SNE_real_datas')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

plt.xlim(-10,10,5)
plt.ylim(-15,15)
plt.xticks(np.arange(-10,11,5))

# 添加图例
legend_labels = ['ADNI_pMCI Dataset', 'ADNI_sMCI Dataset', 'HCP Dataset', 'NACC Dataset', 'OASIS Dataset']
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)

plt.show()
