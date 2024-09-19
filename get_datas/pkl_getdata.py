import numpy as np
import pandas as pd
import nibabel as nib
import torch

from torch.utils.data import Dataset


def normalization(img):
    img = img / (img.max() - img.min())
    return img

class MyDataset(Dataset):
    def __init__(self,txt,file_path):
        fh = pd.read_csv(txt)  # 读进来
        self.T1_data = np.array(fh['test_data']).tolist()
        self.label = np.array(fh['test_label']).tolist()
        self.file_path = file_path

    def __getitem__(self, index):
        T1_path = self.T1_data[index]
        label = self.label[index]
        if label == 0:
            T1_imgs = nib.load(self.file_path + "pMCI/" + T1_path)
        if label == 1:
            T1_imgs = nib.load(self.file_path + "sMCI/" + T1_path)
        T1_imgs = T1_imgs.get_fdata()
        T1_imgs = normalization(T1_imgs)
        T1_imgs1 = torch.from_numpy(T1_imgs)
        T1_imgs_1 = torch.reshape(T1_imgs1, (1, 113, 137, 113))

        return T1_imgs_1,label

    def __len__(self):
        return len(self.T1_data)