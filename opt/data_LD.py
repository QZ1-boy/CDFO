import pandas as pd
from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys
from PIL import Image
import random

class CDVL_sideInfo_Dataset(Dataset):
    """compressed yuv with side info dataset"""
    def __init__(self, csv_file, transform=None, QP=32, only_I_frame=True, random_start=False,
                 mv_flag=False, res_flag=False, part_flag=False, qp_flag=False, max_len=8, only_1_GT=False, 
                 unflt_flag=False, pred_flag=False, need_bi_flag=False,
                 HR_dir="/share3/home/zqiang/CVCP/Uncompressed_HR/",  #   "/data/cpl/sideInfoVSRdata/VC600_GT_grey_npy/"
                 LR_dir_prefix="/share3/home/zqiang/CVCP/Decoded_LR/LD/",    #  "/data/cpl/sideInfoVSRdata/LR/LD/"
                 SideInfo_dir_prefix="/share3/home/zqiang/CVCP/Coding_Priors/LD/"):   #   "/data/cpl/sideInfoVSRdata/side/LD/"

        self.QP = str(QP)
        self.data_path_details = pd.read_csv(csv_file)
        
        self.HR_dir = HR_dir ### HDD
        # self.HR_dir = "/database/VC600_GT_grey_npy/"   ### SSD

        self.LR_dir_prefix = LR_dir_prefix + 'QP' + self.QP + '/LD_'
        self.LR_dir_postfix = '_32F_' +  'QP' + self.QP + '.yuv/'  #  '_480x270_32F_' + 

        self.SideInfo_dir_prefix = SideInfo_dir_prefix + 'QP' + self.QP + '/LD_'
        self.SideInfo_dir_prefix_unfiltered = "/share3/home/zqiang/CVCP/pred_unfiltered_LD/" + 'QP' + self.QP + '/LD_'
        self.SideInfo_dir_postFix =  '_32F_' + 'QP' + self.QP + ".priors/"  #  "_480x272_32F_" +

        self.transform = transform

        self.mv_flag = mv_flag
        self.res_flag = res_flag
        self.part_flag = part_flag
        self.qp_flag = qp_flag
        self.unflt_flag = unflt_flag
        self.pred_flag = pred_flag

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
        self.PMs_ = np.zeros([len(self.data_path_details), 32, 270, 480], dtype = np.uint8)
        self.RMs_ = np.zeros([len(self.data_path_details), 32, 270, 480], dtype = np.int8)
        self.UFs_ = np.zeros([len(self.data_path_details), 32, 272, 480], dtype = np.uint8)
        self.MVl0s_ = np.zeros([len(self.data_path_details), 32, 270, 480, 3], dtype = np.int8)
        # self.MVl1s_ = np.zeros([len(self.data_path_details), 32, 272, 480], dtype = np.uint8)
        self.QPs = np.zeros([len(self.data_path_details), 32], dtype = np.int8)
        ####

        for d_i in range(len(self.data_path_details)):
            seq_name = self.data_path_details.iloc[d_i, 0]

            lr_imgs_folder = self.LR_dir_prefix + seq_name + self.LR_dir_postfix
            hr_imgs_folder = self.HR_dir + seq_name + "/"
            side_paths = self.SideInfo_dir_prefix + seq_name + self.SideInfo_dir_postFix
            side_paths_forunfiltered = self.SideInfo_dir_prefix_unfiltered + seq_name + self.SideInfo_dir_postFix

            # qp_name = side_paths + 'LD_' + seq_name + '_4_QP_v.npy'
            # qp_tmp = np.load(qp_name)
            # self.QPs[d_i, :] = qp_tmp[0, :]

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
                # hr_img_name = hr_imgs_folder + img_idx + '.npy'
                one_tmp.append(hr_img_name)
                # print('[side_paths]',side_paths)

                mvl0_name = side_paths + 'MV_l0/' + img_idx + '_mvl0.npy'
                one_tmp.append(mvl0_name)
                ####
                mvl0_tmp = np.clip(np.load(mvl0_name), -128, 127).astype(np.int8)
                self.MVl0s_[d_i, f_i, :, :, :] = mvl0_tmp
                ####

                mvl1_name = side_paths + 'MV_l1/' + img_idx + '_mvl1.npy'
                one_tmp.append(mvl1_name)

                res_name = side_paths + 'Residue/' + img_idx + '_res.npy'
                one_tmp.append(res_name)
                if self.res_flag:
                    res_tmp = np.clip(np.load(res_name), -128, 127).astype(np.int8)
                    self.RMs_[d_i, f_i, :, :] = res_tmp # [:, :, 0]

                mpm_name = side_paths + 'Partition_Map/' + img_idx + '_M_mask.png'
                one_tmp.append(mpm_name)
                ####
                if self.part_flag:
                    mpm_tmp = io.imread(mpm_name)
                    self.PMs_[d_i, f_i, :, :] = mpm_tmp
                ####

                unflt_f_name = side_paths_forunfiltered + 'pred_unfiltered/' + img_idx + '_unflt.png'
                one_tmp.append(unflt_f_name)
                ####
                if self.unflt_flag:
                    unflt_tmp = io.imread(unflt_f_name)
                    self.UFs_[d_i, f_i, :, :] = unflt_tmp
                ####

                pred_f_name = side_paths + 'Prediction_Signal/' + img_idx + '_pred.png'
                one_tmp.append(pred_f_name)

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

        """
        lr = (frames, h, w)
        hr = (frames, h, w)
        mvl0 = (frames, h, w, chn)
        mvl1 = (frames, h, w, chn)
        res = (frames, h, w, 1)
        m_partionM = (frames, h, w)
        unflt = (frames, h, w)
        pred = (frames, h, w)
        mv_status = (frames, h, w)
        qp = (1, 32)
        """

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
            # hr_img = np.load(self.dir_all[idx][center_idx][1], mmap_mode='r',allow_pickle=True)
            hr_imgs = hr_img[np.newaxis, :, :] # lr single chn -> (h, w)
        else:
            pass
            exit(0)

        ####  mv
        if self.mv_flag:
            mvl0s = []
            mvl1s = []
            if self.only_1_GT:
                mvl0 = self.MVl0s_[idx, center_idx, :, :, :]
                mvl0s.append(mvl0)

                # mvl1 = np.load(self.dir_all[idx][center_idx][3])
                mvl1 = mvl0
                mvl1s.append(mvl1)
            else:
                print("??")
                exit(0)

            mvl0s = np.stack(mvl0s, axis=0)
            mvl1s = np.stack(mvl1s, axis=0)
        else:
            mvl0s = None
            mvl1s = None

        if self.res_flag:
            res_s = self.RMs_[idx, first_poc:first_poc+7, :, :]
        else:
            res_s = None

        if self.part_flag:
            mpm_s = self.PMs_[idx, first_poc:first_poc+7, :, :]
        else:
            mpm_s = None

        if self.unflt_flag:
            unflt_fs = self.UFs_[idx, first_poc:first_poc+7, :, :]
        else:
            unflt_fs = None

        if self.pred_flag: # TODO
            pred_f = io.imread(self.dir_all[idx][center_idx][7])
            pred_fs = pred_f[np.newaxis, :, :]
        else:
            pred_fs = None


        if self.qp_flag:
            qp = self.QPs[idx, first_poc:first_poc+7]
        else:
            qp = None

        if self.need_bi_flag:
            # lr_bi_imgs = io.imread(self.dir_all[idx][center_idx][8])
            lr_bi_imgs = self.lr_bi_imgs_[idx, center_idx, :, :]
            lr_bi_imgs = lr_bi_imgs[np.newaxis, :, :]
        else:
            lr_bi_imgs = None

        sample = {'lr_imgs': lr_imgs,
                  'hr_imgs': hr_imgs,
                  'mvl0s': mvl0s,
                  'mvl1s': mvl1s,
                  'mpm_s':mpm_s,
                  'pred_fs':pred_fs,
                  'unflt_fs':unflt_fs,
                  'res_s': res_s,
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
        mvl0s = sample['mvl0s']
        mvl1s = sample['mvl1s']

        res_s = sample['res_s']
        mpm_s = sample['mpm_s']
        pred_fs = sample['pred_fs']
        unflt_fs = sample['unflt_fs']
        qp = sample['qp']

        lrbi = sample['lrbi']
        # crop
        h, w = lr_imgs.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        lr_imgs = lr_imgs[:, top: top+new_h, left: left+new_w]
        hr_imgs = hr_imgs[:, top*4: (top+new_h)*4, left*4: (left+new_w)*4]

        if mvl0s is not None:
            mvl0s = mvl0s[:, top: top+new_h, left: left+new_w, :]
        if mvl1s is not None:
            mvl1s = mvl1s[:, top: top+new_h, left: left+new_w, :]
        if res_s is not None:
            res_s = res_s[:, top: top+new_h, left: left+new_w]
        if mpm_s is not None:
            mpm_s = mpm_s[:, top: top+new_h, left: left+new_w]
        if pred_fs is not None:
            pred_fs = pred_fs[:, top: top+new_h, left: left+new_w]
        if unflt_fs is not None:
            unflt_fs = unflt_fs[:, top: top+new_h, left: left+new_w]
        if lrbi is not None:
            lrbi = lrbi[:, top: top+new_h, left: left+new_w]

        return {'lr_imgs': lr_imgs,
                'hr_imgs': hr_imgs,
                'mvl0s': mvl0s,
                'mvl1s': mvl1s,

                'mpm_s':mpm_s,
                'pred_fs':pred_fs,
                'unflt_fs':unflt_fs,
                'res_s': res_s,
                'qp': qp, 

                'lrbi': lrbi}


class ToTensor(object):
    """Convert ndarrays in samples to Tensors."""

    def __call__(self, sample):

        lr_imgs = sample['lr_imgs']
        hr_imgs = sample['hr_imgs']

        mvl0s = sample['mvl0s']
        mvl1s = sample['mvl1s']

        res_s = sample['res_s']
        mpm_s = sample['mpm_s']
        pred_fs = sample['pred_fs']
        unflt_fs = sample['unflt_fs']
        qp = sample['qp']

        lrbi = sample['lrbi']

        lr_imgs = lr_imgs[np.newaxis, :, :, :] # np.expand_dims(, axis=0) # (chn, frames, h, w)
        hr_imgs = hr_imgs[np.newaxis, :, :, :] # np.expand_dims(, axis=0)

        mvl0s = mvl0s.transpose((3, 0, 1, 2)) if mvl0s is not None else np.zeros(1)
        mvl1s = mvl1s.transpose((3, 0, 1, 2)) if mvl1s is not None else np.zeros(1)

        res_s = np.expand_dims(res_s, axis=0) if res_s is not None else np.zeros(1)
        mpm_s = np.expand_dims(mpm_s, axis=0) if mpm_s is not None else np.zeros(1)
        pred_fs = pred_fs[np.newaxis, :, :, :] if pred_fs is not None else np.zeros(1)
        unflt_fs = unflt_fs[np.newaxis, :, :, :] if unflt_fs is not None else np.zeros(1)
        lrbi = lrbi[np.newaxis, :, :, :] if lrbi is not None else np.zeros(1)
        qp = qp if qp is not None else np.zeros(1)

        return {  'lr_imgs': torch.from_numpy(lr_imgs).float() / 255.0,
                  'hr_imgs': torch.from_numpy(hr_imgs).float() / 255.0,
                  'mvl0s': torch.from_numpy(mvl0s).float(),
                  'mvl1s': torch.from_numpy(mvl1s).float(),

                  'res_s': torch.from_numpy(res_s).float() / 255.0,
                  'mpm_s': torch.from_numpy(mpm_s).float() / 255.0,
                  'pred_fs': torch.from_numpy(pred_fs).float() / 255.0,
                  'unflt_fs': torch.from_numpy(unflt_fs).float() / 255.0,
                  'qp': torch.from_numpy(qp).float() / 52.0,

                  'lrbi':torch.from_numpy(lrbi).float() / 255.0
                }


class Augment(object):
    def __call__(self, sample, hflip=True, rot=True):


        lr_imgs = sample['lr_imgs']
        hr_imgs = sample['hr_imgs']

        mvl0s = sample['mvl0s']
        mvl1s = sample['mvl1s']

        res_s = sample['res_s']
        mpm_s = sample['mpm_s']
        pred_fs = sample['pred_fs']
        unflt_fs = sample['unflt_fs']
        qp = sample['qp']

        lrbi = sample['lrbi']
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        # # (272, 480, 2, 32)  [h, w, chn, f]

        # [f, h, w, chn]
        if mvl0s is not None:
            mvl0s = mvl0s.astype(np.float32)
            mvl1s = mvl1s.astype(np.float32)

            mvl0s[:,:,:,[0,1]] = mvl0s[:,:,:,[1,0]]
            mvl1s[:,:,:,[0,1]] = mvl1s[:,:,:,[1,0]]
            
        if mvl0s is not None:
            if hflip:
                mvl0s = mvl0s[:,:,::-1,:]
                mvl0s[:,:,:,0] *= -1

                mvl1s = mvl1s[:,:,::-1,:]
                mvl1s[:,:,:,0] *= -1
            if vflip:
                mvl0s = mvl0s[:,::-1,:,:]
                mvl0s[:,:,:,1] *= -1

                mvl1s = mvl1s[:,::-1,:,:]
                mvl1s[:,:,:,1] *= -1
            if rot90:
                mvl0s = mvl0s.transpose(0, 2, 1, 3)
                mvl0s[:,:,:,[0,1]] = mvl0s[:,:,:,[1,0]]

                mvl1s = mvl1s.transpose(0, 2, 1, 3)
                mvl1s[:,:,:,[0,1]] = mvl1s[:,:,:,[1,0]]

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

        # for partition map
        if mpm_s is not None:
            if hflip:
                mpm_s = mpm_s[:, :, ::-1]
            if vflip:
                mpm_s = mpm_s[:, ::-1, :]
            if rot90:
                mpm_s = mpm_s.transpose(0, 2, 1)
            mpm_s = mpm_s.copy()
        
        if lrbi is not None:
            if hflip:
                lrbi = lrbi[:, :, ::-1]
            if vflip:
                lrbi = lrbi[:, ::-1, :]
            if rot90:
                lrbi = lrbi.transpose(0, 2, 1)
            lrbi = lrbi.copy()
        ### residual map
        if res_s is not None:
            if hflip:
                res_s = res_s[:, :, ::-1]
            if vflip:
                res_s = res_s[:, ::-1, :]
            if rot90:
                res_s = res_s.transpose(0, 2, 1)
            res_s = res_s.copy()
        
        ### unfiltered frame
        if unflt_fs is not None:
            if hflip:
                unflt_fs = unflt_fs[:, :, ::-1]
            if vflip:
                unflt_fs = unflt_fs[:, ::-1, :]
            if rot90:
                unflt_fs = unflt_fs.transpose(0, 2, 1)
            unflt_fs = unflt_fs.copy()
            
        # 1 mv -> 7 mvs
        if mvl0s is not None:
            mvl0s_7 = np.zeros([7, mvl0s.shape[1], mvl0s.shape[2], 2]).astype(np.float32)
            ### frame 2
            pre_f_x = mvl0s[0,:,:,0] / (mvl0s[0,:,:,2] * -1.0)
            pre_f_y = mvl0s[0,:,:,1] / (mvl0s[0,:,:,2] * -1.0)

            mvl0s_7[2, :, :, 0] = np.where(~np.isnan(pre_f_x), pre_f_x, 0)
            mvl0s_7[2, :, :, 1] = np.where(~np.isnan(pre_f_y), pre_f_y, 0)

            mvl0s_7[1, :, :, :] = mvl0s_7[2, :, :, :] * 2.0
            mvl0s_7[0, :, :, :] = mvl0s_7[2, :, :, :] * 3.0

            mvl0s_7[4, :, :, :] = mvl0s_7[2, :, :, :] * -1.0
            mvl0s_7[5, :, :, :] = mvl0s_7[2, :, :, :] * -2.0
            mvl0s_7[6, :, :, :] = mvl0s_7[2, :, :, :] * -3.0

            mvl1s_7 = np.zeros([7, mvl0s.shape[1], mvl0s.shape[2], 2]).astype(np.float32)
            
            ##### TBD
            # pre_f_x = mvl1s[0,:,:,0] / (mvl1s[0,:,:,2] * -1.0)
            # pre_f_y = mvl1s[0,:,:,1] / (mvl1s[0,:,:,2] * -1.0)

            # mvl1s_7[2, :, :, 0] = np.where(~np.isnan(pre_f_x), pre_f_x, 0)
            # mvl1s_7[2, :, :, 1] = np.where(~np.isnan(pre_f_y), pre_f_y, 0)

            # mvl1s_7[1, :, :, :] = mvl1s_7[2, :, :, :] * 2.0
            # mvl1s_7[0, :, :, :] = mvl1s_7[2, :, :, :] * 3.0

            # mvl1s_7[4, :, :, :] = mvl1s_7[2, :, :, :] * -1.0
            # mvl1s_7[5, :, :, :] = mvl1s_7[2, :, :, :] * -2.0
            # mvl1s_7[6, :, :, :] = mvl1s_7[2, :, :, :] * -3.0
            mvl0s_7 = mvl0s_7 / 4.0
            mvl1s_7 = mvl1s_7 / 4.0
        else:
            mvl0s_7 = None
            mvl1s_7 = None

        return {'lr_imgs': lr_imgs.copy(),
                'hr_imgs': hr_imgs.copy(),
                'mvl0s': mvl0s_7,
                'mvl1s': mvl1s_7,

                'mpm_s':mpm_s,
                'pred_fs':pred_fs,
                'unflt_fs':unflt_fs,
                'res_s': res_s,
                'qp': qp, 

                'lrbi': lrbi}
