import pandas as pd
from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys
from PIL import Image
import random

class CDVL_Dataset(Dataset):
    """compressed yuv with side info dataset"""
    def __init__(self, csv_file, transform=None, QP=32, only_I_frame=True, random_start=False, qp_flag=False, max_len=8, only_1_GT=False, 
                 need_bi_flag=False,
                 HR_dir="/share3/home/zqiang/CVCP/Uncompressed_HR/",  #   "/data/cpl/sideInfoVSRdata/VC600_GT_grey_npy/"
                 LR_dir_prefix="/share3/home/zqiang/CVCP/Decoded_LR/RA/",    #  "/data/cpl/sideInfoVSRdata/LR/RA/"
                 SideInfo_dir_prefix="/share3/home/zqiang/CVCP/Coding_Priors/RA/"):   #   "/data/cpl/sideInfoVSRdata/side/RA/"
        self.QP = str(QP)
        self.data_path_details = pd.read_csv(csv_file)
        self.HR_dir = HR_dir ### HDD
        # self.HR_dir = "/database/VC600_GT_grey_npy/"   ### SSD
        self.LR_dir_prefix = LR_dir_prefix + 'QP' + self.QP + '/RA_'
        self.LR_dir_postfix = '_32F_' +  'QP' + self.QP + '.yuv/'  #  '_480x270_32F_' + 
        self.transform = transform
        self.qp_flag = qp_flag
        self.max_len = max_len
        self.only_I_frame = only_I_frame
        self.random_start = (not only_I_frame) and random_start
        self.only_1_GT = only_1_GT

        self.need_bi_flag = need_bi_flag
        if self.need_bi_flag:
            self.LR_bi_prefix = '/data/cpl/lr_uncompressed/bicubic/'
            self.LR_bi_postfix = '/'
            self.lr_bi_imgs_ = np.zeros([len(self.data_path_details), 32, 270, 480], dtype = np.uint8)
        else:
            self.LR_bi_prefix = ''
            self.LR_bi_postfix = ''
            self.lr_bi_imgs_ = None

        self.dir_all = []

        ####
        self.LR_imgs_ = np.zeros([len(self.data_path_details), 32, 270, 480], dtype = np.uint8)
        self.QPs = np.zeros([len(self.data_path_details), 32], dtype = np.int8)
        ####

        for d_i in range(len(self.data_path_details)):
            seq_name = self.data_path_details.iloc[d_i, 0]
            lr_imgs_folder = self.LR_dir_prefix + seq_name + self.LR_dir_postfix
            hr_imgs_folder = self.HR_dir + seq_name + "/"
            lr_bi_folder = self.LR_bi_prefix + seq_name + self.LR_bi_postfix

            seq_tmp = []
            for f_i in range(32):
                img_idx = "%05d" % f_i
                one_tmp = []

                lr_img_name = lr_imgs_folder + img_idx + '.png'
                one_tmp.append(lr_img_name)
                ####
                lr_img_tmp = io.imread(lr_img_name)
                self.LR_imgs_[d_i, f_i, :, :] = lr_img_tmp
                ####
                hr_img_name = hr_imgs_folder + img_idx + '.png'
                one_tmp.append(hr_img_name)
                lr_img_bi = lr_bi_folder + img_idx + '.png'
                one_tmp.append(lr_img_bi)
                ####
                if self.need_bi_flag:
                    lr_bi_tmp = io.imread(lr_img_bi)
                    self.lr_bi_imgs_[d_i, f_i, :, :] = lr_bi_tmp
                ####
                seq_tmp.append(one_tmp)
            self.dir_all.append(seq_tmp)

            if (d_i + 1) % 100 == 0:
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
                first_poc = random.randint(0,25)
            else:
                first_poc = random.randint(0,6) * 4

        lr_imgs = self.LR_imgs_[idx, first_poc:first_poc+7, :, :]
        center_idx = self.max_len // 2 + first_poc
        ####  hr
        if self.only_1_GT: 
            hr_img = io.imread(self.dir_all[idx][center_idx][1])
            hr_imgs = hr_img[np.newaxis, :, :] # lr single chn -> (h, w)
        else:
            pass
            exit(0)
        
        # print('hr_imgs.shape',hr_imgs.shape)

        if self.qp_flag:
            qp = self.QPs[idx, first_poc:first_poc+7]
        else:
            qp = None

        if self.need_bi_flag:
            lr_bi_imgs = self.lr_bi_imgs_[idx, center_idx, :, :]
            lr_bi_imgs = lr_bi_imgs[np.newaxis, :, :]
        else:
            lr_bi_imgs = None

        sample = {'lr_imgs': lr_imgs,
                  'hr_imgs': hr_imgs,
                  'qp': qp,
                  'lrbi': lr_bi_imgs}

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
        qp = sample['qp']

        lrbi = sample['lrbi']
        # crop
        h, w = lr_imgs.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        lr_imgs = lr_imgs[:, top: top+new_h, left: left+new_w]
        hr_imgs = hr_imgs[:, top*4: (top+new_h)*4, left*4: (left+new_w)*4]

        if lrbi is not None:
            lrbi = lrbi[:, top: top+new_h, left: left+new_w]

        return {'lr_imgs': lr_imgs,
                'hr_imgs': hr_imgs,
                'qp': qp, 
                'lrbi': lrbi}


class ToTensor(object):
    """Convert ndarrays in samples to Tensors."""

    def __call__(self, sample):

        lr_imgs = sample['lr_imgs']
        hr_imgs = sample['hr_imgs']
        qp = sample['qp']
        lrbi = sample['lrbi']
        lr_imgs = lr_imgs[np.newaxis, :, :, :] # np.expand_dims(, axis=0) # (chn, frames, h, w)
        hr_imgs = hr_imgs[np.newaxis, :, :, :] # np.expand_dims(, axis=0)
        lrbi = lrbi[np.newaxis, :, :, :] if lrbi is not None else np.zeros(1)
        qp = qp if qp is not None else np.zeros(1)

        return {  'lr_imgs': torch.from_numpy(lr_imgs).float() / 255.0,
                  'hr_imgs': torch.from_numpy(hr_imgs).float() / 255.0,
                  'qp': torch.from_numpy(qp).float() / 52.0,
                  'lrbi':torch.from_numpy(lrbi).float() / 255.0
                }


class Augment(object):
    def __call__(self, sample, hflip=True, rot=True):


        lr_imgs = sample['lr_imgs']
        hr_imgs = sample['hr_imgs']
        qp = sample['qp']

        lrbi = sample['lrbi']
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        # # (272, 480, 2, 32)  [h, w, chn, f]

        # for imgs [f, h, w]
        if hflip:
            lr_imgs = lr_imgs[:, :, ::-1]
            hr_imgs = hr_imgs[:, :, ::-1]
        if vflip:
            lr_imgs = lr_imgs[:, ::-1, :]
            hr_imgs = hr_imgs[:, ::-1, :]
        if rot90:
            lr_imgs = lr_imgs.transpose(0, 2, 1)
            hr_imgs = hr_imgs.transpose(0, 2, 1)
        
        if lrbi is not None:
            if hflip:
                lrbi = lrbi[:, :, ::-1]
            if vflip:
                lrbi = lrbi[:, ::-1, :]
            if rot90:
                lrbi = lrbi.transpose(0, 2, 1)
            lrbi = lrbi.copy()

        return {'lr_imgs': lr_imgs.copy(),
                'hr_imgs': hr_imgs.copy(),
                'qp': qp, 
                'lrbi': lrbi}
