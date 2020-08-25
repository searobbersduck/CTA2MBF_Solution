import os
import sys

from glob import glob
from tqdm import tqdm
import numpy as np

import SimpleITK as sitk

from torch.utils.data import Dataset, DataLoader
import nibabel
from scipy import ndimage
import time
import torch
import torch.nn as nn

import fire

class CTA2MBF_DS(Dataset):
    def __init__(self, root_dirs, config_xxx_files, config_yyy_files, phase,crop_size, scale_size):
        self.root_dirs = root_dirs
        self.config_xxx_files = config_xxx_files
        self.config_yyy_files = config_yyy_files
        self.phase = phase
        self.crop_size = crop_size
        self.scale_size = scale_size

        self.xxx_info_list = []
        self.yyy_info_list = []

        self.xxx_srcs_list = []
        self.xxx_dsts_list = []

        for i in range(len(config_xxx_files)):
            with open(config_xxx_files[i]) as f:
                for line in f.readlines():
                    line = line.strip()
                    if line is None or len(line) == 0:
                        continue
                    ss = line.split('\t')
                    if len(ss) != 2:
                        continue
                    self.xxx_srcs_list.append(os.path.join(root_dirs[i], ss[0]))
                    self.xxx_dsts_list.append(os.path.join(root_dirs[i], ss[1]))

        assert len(self.xxx_srcs_list) == len(self.xxx_dsts_list)
        self.xxx_srcs_list = self.xxx_srcs_list

        self.yyy_srcs_list = []
        self.yyy_dsts_list = []

        for i in range(len(config_yyy_files)):
            with open(config_yyy_files[i]) as f:
                for line in f.readlines():
                    line = line.strip()
                    if line is None or len(line) == 0:
                        continue
                    ss = line.split('\t')
                    if len(ss) != 2:
                        continue
                    self.yyy_srcs_list.append(os.path.join(root_dirs[i], ss[0]))
                    self.yyy_dsts_list.append(os.path.join(root_dirs[i], ss[1]))

        assert len(self.yyy_srcs_list) == len(self.yyy_dsts_list)

    def __len__(self):
        return len(self.xxx_srcs_list)


    def __getitem__(self, idx):
        if self.phase == 'train':
            if np.random.rand() < 0.85:
                src_path = self.xxx_srcs_list[idx]
                dst_path = self.xxx_dsts_list[idx]
            else:
                rand_idx = np.random.randint(0, len(self.yyy_srcs_list))
                src_path = self.yyy_srcs_list[rand_idx]
                dst_path = self.yyy_dsts_list[rand_idx]
            
            src_data = np.load(src_path)
            dst_data = np.load(dst_path)

            src_tensor = torch.from_numpy(src_data).float()
            src_tensor = torch.unsqueeze(src_tensor, axis=0)

            # dst_data = np.array(dst_data, dtype=np.int32)
            dst_tensor = torch.from_numpy(dst_data).float()
            dst_tensor = torch.unsqueeze(dst_tensor, axis=0)

            return src_tensor, dst_tensor, src_path, dst_path


def test_CTA2MBF_DS():
    root_dirs = ['../../data/toy_data/2.slice_2d/train']
    config_xxx_files = ['../../data/toy_data/2.slice_2d/config/config_2d_copd_xxx_train.txt']
    config_yyy_files = ['../../data/toy_data/2.slice_2d/config/config_2d_copd_yyy_train.txt']
    crop_size = [512, 512]
    ds = CTA2MBF_DS(root_dirs, config_xxx_files, config_yyy_files, 'train', crop_size, crop_size)
    data_loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=1, pin_memory=False)
    for i, (images, masks, _, _) in tqdm(enumerate(data_loader)):
        print(images.shape)
        print('hello world')
        # break


if __name__ == '__main__':
    test_CTA2MBF_DS()
    
