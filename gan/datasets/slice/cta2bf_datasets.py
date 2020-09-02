import os
import sys

from glob import glob
from tqdm import tqdm
import numpy as np

import SimpleITK as sitk
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import nibabel
from scipy import ndimage
import time
import torch
import torch.nn as nn

import fire


class CTA2MBF_DS(Dataset):
    def __init__(self, root_dirs, config_xxx_files, config_yyy_files, phase,crop_size,pad_size):
        self.root_dirs = root_dirs
        self.config_xxx_files = config_xxx_files
        self.config_yyy_files = config_yyy_files
        self.phase = phase
        self.pad_size = pad_size
        self.crop_size = crop_size
        # self.scale_size = scale_size

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
                    # ss = line.split(' ')
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
                    # ss = line.split(' ')
                    if len(ss) != 2:
                        continue
                    self.yyy_srcs_list.append(os.path.join(root_dirs[i], ss[0]))
                    self.yyy_dsts_list.append(os.path.join(root_dirs[i], ss[1]))

        assert len(self.yyy_srcs_list) == len(self.yyy_dsts_list)

    def __len__(self):
        return len(self.xxx_srcs_list)

    # def __pad_data (self,cta,mbf,size):
    #     transform0 = transforms.CenterCrop(size)
    #     cta2 = transform0(cta) 
    #     mbf2 = transform0(mbf) 
    #     return cta2,mbf2
    
    def __pad_data (self,cta,mbf,size):
        # [img_h, img_w] = cta.shape
        # [input_h, input_w] = size
        cta2 = np.zeros(size) + cta
        mbf2 = np.zeros(size) + mbf
        return cta2,mbf2
        
    # def __center_crop_data (self,cta,mbf,size):
    #     transform1 = transforms.CenterCrop(size)
    #     cta2 = transform1(cta) 
    #     mbf2 = transform1(mbf) 
    #     return cta2,mbf2
    def __center_crop_data (self,cta,mbf,size):
        [img_h, img_w] = cta.shape
        [input_h, input_w] = size
        # assert np.all(np.less_equal(size, dwi_data.shape))

        Y_min = img_h//2-input_h//2
        X_min = img_w//2-input_w//2

        Y_max = Y_min + input_h
        X_max = X_min + input_w        

        cta2 = cta[Y_min: Y_max, X_min: X_max]
        mbf2 = mbf[Y_min: Y_max, X_min: X_max]

        return cta2,mbf2

    # def __random_crop_data (self,cta,mbf,size):
    #     transform2 = transforms.RandomCrop(size, padding=0, pad_if_needed=False, fill=0, padding_mode='constant')
    #     cta2 = transform2(cta) 
    #     mbf2 = transform2(mbf) 
    #     return cta2,mbf2

    def __random_crop_data (self,cta,mbf,size):

        [img_h, img_w] = cta.shape
        [input_h, input_w] = size
        # assert np.all(np.less_equal(size, dwi_data.shape))
 
        y_min_upper = img_h - input_h
        x_min_upper = img_w - input_w


        Y_min = np.random.randint(0, y_min_upper)
        X_min = np.random.randint(0, x_min_upper)

        Y_max = Y_min + input_h
        X_max = X_min + input_w

        cta2 = cta[Y_min: Y_max, X_min: X_max]
        mbf2 = mbf[Y_min: Y_max, X_min: X_max]

        return cta2,mbf2 

    # def __randomresized_crop_data (self,cta,mbf,size):
    #     transform3 = transforms.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
    #     cta2 = transform3(cta) 
    #     mbf2 = transform3(mbf) 
    #     return cta2,mbf2

    def __getitem__(self, idx):
        if self.phase == 'train':
            # if np.random.rand() < 0.85:
            if np.random.rand() < 0.80:
                src_path = self.xxx_srcs_list[idx]
                dst_path = self.xxx_dsts_list[idx]
            else:
                rand_idx = np.random.randint(0, len(self.yyy_srcs_list))
                src_path = self.yyy_srcs_list[rand_idx]
                dst_path = self.yyy_dsts_list[rand_idx]
            
            src_data = np.load(src_path)

            dst_data = np.load(dst_path)

            src_data, dst_data = self.__pad_data(src_data, dst_data, self.pad_size)

            if self.crop_size is not None:
                if np.random.rand() < 0.8:
                    src_data, dst_data = self.__random_crop_data(src_data, dst_data, self.crop_size)
                # elif 0.4 < np.random.rand() < 0.8:
                #     src_data, dst_data = self.__randomresized_crop_data(src_data, dst_data, self.crop_size)
                else:
                    src_data, dst_data = self.__center_crop_data(src_data, dst_data, self.crop_size)                



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
    
