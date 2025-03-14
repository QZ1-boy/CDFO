import pandas as pd
from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys
from PIL import Image
import random

class Vimeo90K_Dataset(Dataset):
    """compressed yuv with side info dataset"""
    def __init__(self, csv_file, transform=None, QP=32, only_I_frame=True, random_start=False, max_len=7, only_1_GT=False,
                 HR_dir="/share3/home/zqiang/Vimeo90K/sequences/", 
                 LR_dir_prefix="/share3/home/zqiang/Vimeo90K/sequences_CompressedFrame/"):  
        self.QP = str(QP)
        self.data_path_details = pd.read_csv(csv_file)
        self.HR_dir = HR_dir
        self.LR_dir_prefix = LR_dir_prefix + 'QP' + self.QP + '/'
        self.transform = transform
        self.max_len = max_len
        self.only_I_frame = only_I_frame
        self.random_start = (not only_I_frame) and random_start
        self.only_1_GT = only_1_GT
        self.dir_all = []

        ####
        self.LR_imgs_ = np.zeros([len(self.data_path_details), 7, 64,  112, 3], dtype = np.uint8)
        self.HR_imgs_ = np.zeros([len(self.data_path_details), 7, 256, 448, 3], dtype = np.uint8)
        ####

        for d_i in range(len(self.data_path_details)):
            seq_name = self.data_path_details.iloc[d_i, 0]
            lr_imgs_folder = self.LR_dir_prefix + seq_name + "/"
            hr_imgs_folder = self.HR_dir + seq_name + "/"

            seq_tmp = []
            for f_i in range(7):
                img_idx = "%01d" % (f_i + 1)
                one_tmp = []
                lr_img_name = lr_imgs_folder + 'im' + img_idx + '.png'
                one_tmp.append(lr_img_name)
                ####
                lr_img_tmp = io.imread(lr_img_name)
                self.LR_imgs_[d_i, f_i, :, :, :] = lr_img_tmp
                ####
                hr_img_name = hr_imgs_folder + 'im' + img_idx + '.png'
                one_tmp.append(hr_img_name)

                seq_tmp.append(one_tmp)
            self.dir_all.append(seq_tmp)

            if (d_i + 1) % 5000 == 0:
                print('reading lr sequences (' + str(d_i + 1) + '/' + str(len(self.data_path_details)) +')')

    def __len__(self):
        return len(self.data_path_details)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        first_poc = 0
        lr_imgs = self.LR_imgs_[idx, first_poc:first_poc+self.max_len, :, :]
        center_idx = self.max_len//2 + first_poc
        ####  hr
        if self.only_1_GT: 
            hr_img = io.imread(self.dir_all[idx][center_idx][1])
            hr_imgs = hr_img[np.newaxis, :, :]
        else:
            pass
            exit(0)

        sample = {'lr_imgs': lr_imgs,
                  'hr_imgs': hr_imgs}

        if self.transform:
            sample = self.transform(sample)
        
        return sample





class Vimeo90K_Dataset_ETC(Dataset):
    """compressed yuv with side info dataset"""
    def __init__(self, csv_file, transform=None, QP=32, only_I_frame=True, random_start=False, max_len=13, only_1_GT=False,
                 HR_dir="/share3/home/zqiang/REDS/train_sharp/", 
                 LR_dir_prefix="/share3/home/zqiang/REDS/train_sharp_VC_BI_CompressedFrame/"):  
        self.QP = str(QP)
        self.data_path_details = pd.read_csv(csv_file)
        self.HR_dir = HR_dir
        self.LR_dir_prefix = LR_dir_prefix + 'QP' + self.QP + '/'
        self.transform = transform
        self.max_len = max_len
        self.only_I_frame = only_I_frame
        self.random_start = (not only_I_frame) and random_start
        self.only_1_GT = only_1_GT
        self.dir_all = []

        ####
        self.LR_imgs_ = np.zeros([len(self.data_path_details), 100, 176, 320,  3], dtype = np.uint8)
        self.HR_imgs_ = np.zeros([len(self.data_path_details), 100, 704, 1080, 3], dtype = np.uint8)
        ####

        for d_i in range(len(self.data_path_details)):
            seq_name = "%03d" % (self.data_path_details.iloc[d_i, 0])
            lr_imgs_folder = self.LR_dir_prefix + seq_name + "/"
            hr_imgs_folder = self.HR_dir + seq_name + "/"

            seq_tmp = []
            for f_i in range(100):
                img_idx = "%08d" % f_i
                one_tmp = []
                lr_img_name = lr_imgs_folder + img_idx + '.png'
                one_tmp.append(lr_img_name)
                ####
                lr_img_tmp = io.imread(lr_img_name)
                self.LR_imgs_[d_i, f_i, :, :, :] = lr_img_tmp
                ####
                hr_img_name = hr_imgs_folder + img_idx + '.png'
                one_tmp.append(hr_img_name)

                seq_tmp.append(one_tmp)
            self.dir_all.append(seq_tmp)

            if (d_i + 1) % 50 == 0:
                print('reading lr sequences (' + str(d_i + 1) + '/' + str(len(self.data_path_details)) +')')

    def __len__(self):
        return len(self.data_path_details)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.only_I_frame:
            first_poc = 0
        else:
            if self.random_start:
                first_poc = random.randint(0,86)
            else:
                first_poc = random.randint(0,43) * 2

        lr_imgs = self.LR_imgs_[idx, first_poc:first_poc+self.max_len, :, :]
        center_idx = self.max_len//2 + first_poc
        ####  hr
        if self.only_1_GT: 
            hr_img = io.imread(self.dir_all[idx][center_idx][1])
            hr_imgs = hr_img[np.newaxis, :, :]
        else:
            pass
            exit(0)

        sample = {'lr_imgs': lr_imgs,
                  'hr_imgs': hr_imgs}

        if self.transform:
            sample = self.transform(sample)
        
        return sample


class RandomCrop(object):
    """Crop randomly the images in a sample"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        # read
        lr_imgs = sample['lr_imgs']
        hr_imgs = sample['hr_imgs']
        # crop
        t, h, w, c = lr_imgs.shape
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        lr_imgs = lr_imgs[:, top: top+new_h, left: left+new_w, :]
        hr_imgs = hr_imgs[:, top*4: (top+new_h)*4, left*4: (left+new_w)*4, :]


        return {'lr_imgs': lr_imgs,
                'hr_imgs': hr_imgs}


class ToTensor(object):
    """Convert ndarrays in samples to Tensors."""

    def __call__(self, sample):

        lr_imgs = sample['lr_imgs']
        hr_imgs = sample['hr_imgs']
        lr_imgs = lr_imgs[np.newaxis, :, :, :] # np.expand_dims(, axis=0) # (chn, frames, h, w)
        hr_imgs = hr_imgs[np.newaxis, :, :, :] # np.expand_dims(, axis=0)


        return {  'lr_imgs': torch.from_numpy(lr_imgs).float() / 255.0,
                  'hr_imgs': torch.from_numpy(hr_imgs).float() / 255.0
                }


class Augment(object):
    def __call__(self, sample, hflip=True, rot=True):


        lr_imgs = sample['lr_imgs']
        hr_imgs = sample['hr_imgs']

        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        # # (272, 480, 2, 32)  [h, w, chn, f]

        # for imgs [f, h, w]
        if hflip:
            lr_imgs = lr_imgs[:, :, ::-1, :]
            hr_imgs = hr_imgs[:, :, ::-1, :]
        if vflip:
            lr_imgs = lr_imgs[:, ::-1, :, :]
            hr_imgs = hr_imgs[:, ::-1, :, :]
        if rot90: 
            lr_imgs = lr_imgs.transpose(0, 2, 1, 3)
            hr_imgs = hr_imgs.transpose(0, 2, 1, 3)
        
        return {'lr_imgs': lr_imgs.copy(),
                'hr_imgs': hr_imgs.copy()}
