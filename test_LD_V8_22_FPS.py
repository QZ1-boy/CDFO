import os 
import cv2
import sys
import torch
from PIL import Image
import numpy as np
import argparse
from arch.SIDECVSR_our import CVSR_V8 
from metric.psnr_ssim import cal_psnr_ssim
import warnings 
# warnings.filterwarnings('ignore')
import time

def generate_input_index(center_index, frame_number, max_index):
    o_list = np.array(range(frame_number)) - (frame_number // 2) + center_index
    o_list = np.clip(o_list, 0, max_index)
    return o_list


def generate_input(frame_number, path, filelist):
    inputF = []
    for i in frame_number:
        img = cv2.imread(path + filelist[i], 0)
        y = np.expand_dims(np.expand_dims(img, axis=0), axis=0).astype(np.float32)
        if img.shape[0] == 270:
            y = np.concatenate([y,y[:, :, -2:,:]],axis=2)
            y[:, :,-2:,:] = 0
        y_pyt = torch.from_numpy(y).float() / 255.0
        inputF.append(y_pyt)
    
    return inputF


def generate_PM_input(frame_number, path):
    inputF = []
    for i in frame_number:
        ii = max(1, i)
        idx = "%05d" % ii
        pm_img = cv2.imread(path + idx + '_M_mask.png', 0)
        y = np.expand_dims(np.expand_dims(pm_img, axis=0), axis=0).astype(np.float32)
        if pm_img.shape[0] == 270:
            y = np.concatenate([y, y[:, :, -2:,:]], axis=2)
            y[:, :,-2:,:] = 0
        y_pyt = torch.from_numpy(y).float() / 255.0
        inputF.append(y_pyt)
    
    return inputF


def generate_UF_input(frame_number, path):
    inputF = []
    for i in frame_number:
        ii = max(1, i)
        idx = "%05d" % ii
        pm_img = cv2.imread(path + idx + '_unflt.png', 0)
        y = np.expand_dims(np.expand_dims(pm_img, axis=0), axis=0).astype(np.float32)
        y_pyt = torch.from_numpy(y).float() / 255.0
        inputF.append(y_pyt)

    return inputF


def generate_RM_input(frame_number, path):
    inputF = []
    for i in frame_number:
        ii = max(1, i)
        idx = "%05d" % ii
        res_map = np.load(path + idx + '_res.npy')
        res_map = res_map[:,:,0]
        y = np.expand_dims(np.expand_dims(res_map, axis=0), axis=0).astype(np.float32)
        y_pyt = torch.from_numpy(y).float() / 255.0
        inputF.append(y_pyt)
    
    return inputF


def generate_DSF_input(frame_number, path):
    inputF = []
    for i in frame_number:
        ii = max(1, i)
        idx = "%05d" % ii
        pm_img = cv2.imread(path + idx + '.png', 0)
        y = np.expand_dims(np.expand_dims(pm_img, axis=0), axis=0).astype(np.float32)
        if pm_img.shape[0] == 270:
            y = np.concatenate([y, y[:, :, -2:,:]], axis=2)
            y[:, :,-2:,:] = 0
        y_pyt = torch.from_numpy(y).float() / 255.0
        inputF.append(y_pyt)
    
    return inputF


def read_one_pic(img_name):
    img = cv2.imread(img_name, 0)
    y = np.expand_dims(np.expand_dims(img, axis=0), axis=0).astype(np.float32)
    y_pyt = torch.from_numpy(y).float() / 255.0
    return y_pyt


def mv2mvs(mv):
    mv_ = mv.astype(np.float32)
    mv_ = mv_[np.newaxis, :, :, :]
    mv_[:,:,:,[0,1]] = mv_[:,:,:,[1,0]]

    mvl0s_7 = np.zeros([7, mv_.shape[1], mv_.shape[2], 2]).astype(np.float32)
    # # frame 2
    pre_f_x = mv_[0,:,:,0] / (mv_[0,:,:,2] * -1.0)
    pre_f_y = mv_[0,:,:,1] / (mv_[0,:,:,2] * -1.0)

    mvl0s_7[2, :, :, 0] = np.where(~np.isnan(pre_f_x), pre_f_x, 0)
    mvl0s_7[2, :, :, 1] = np.where(~np.isnan(pre_f_y), pre_f_y, 0)

    mvl0s_7[1, :, :, :] = mvl0s_7[2, :, :, :] * 2.0
    mvl0s_7[0, :, :, :] = mvl0s_7[2, :, :, :] * 3.0

    mvl0s_7[4, :, :, :] = mvl0s_7[2, :, :, :] * -1.0
    mvl0s_7[5, :, :, :] = mvl0s_7[2, :, :, :] * -2.0
    mvl0s_7[6, :, :, :] = mvl0s_7[2, :, :, :] * -3.0

    mvl0s_7 = mvl0s_7 / (4.0 * 32.0)

    return torch.from_numpy(mvl0s_7).float()

####################

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# torch.cuda.current_device()
# torch.cuda._initialized = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_FRAME = 7

def eval_seq(tst_list, gt_list, methods_name = "J", coding_cfg = "LD", testing=True, cal_metric=True):
    
    if testing:
        QP = 22
        model = CVSR_V8()
        print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
        model_path = '/share3/home/zqiang/CVSR_train/training_results/train_LD_V8_22/ckpt/epoch-10.pth'  #  18000  16400
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(device)

        tst_path = "./test_data/%s/qp%s/lr_grey/" % (coding_cfg, QP)
        sideInfo_path = "./test_data/%s/qp%s/sideInfo_QP%s/" % (coding_cfg, QP, QP)
        for img_set, one_gt in zip(tst_list, gt_list):

            tmp_path = tst_path + img_set + '/'
            tmp_side_path = sideInfo_path + img_set[:-4] + '/'
    
            print(tmp_path)
            for _, _, f in os.walk(tmp_path):
                f.sort()
                Sumtime = 0
                print(len(f))
                for i in range(len(f)):
                    o_list = generate_input_index(i, INPUT_FRAME, len(f)-1)
                    input_imgY = generate_input(o_list, tmp_path, f)
                    lrs = torch.unsqueeze(torch.cat(input_imgY, 0).to(device), 0)

                    pm_Y = generate_PM_input(o_list, tmp_side_path + 'part_m/')
                    pms = torch.unsqueeze(torch.cat(pm_Y, 0).to(device), 0)

                    rm_Y = generate_RM_input(o_list, tmp_side_path + 'res/')
                    rms = torch.unsqueeze(torch.cat(rm_Y, 0).to(device), 0)

                    uf_Y = generate_UF_input(o_list, tmp_side_path + 'unfiltered/')
                    ufs = torch.unsqueeze(torch.cat(uf_Y, 0).to(device), 0)

                    idx = "%05d" % max(1, i)

                    mvl0 = np.load(tmp_side_path + 'mvl0/' + idx + '_mvl0.npy')  #  mvl0_ehc  mvl0
                    mvs0 = mv2mvs(mvl0).to(device)
                    mvs0 = torch.unsqueeze(mvs0, 0)
                    mvs0 = mvs0.permute(0,1,4,2,3)

                    mvl1 = np.load(tmp_side_path + 'mvl1/' + idx + '_mvl1.npy')  #  mvl0_ehc  mvl0
                    mvs1 = mv2mvs(mvl1).to(device)
                    mvs1 = torch.unsqueeze(mvs1, 0)
                    mvs1 = mvs1.permute(0,1,4,2,3)

                    modify_mv_for_end_frames(i, mvs0, len(f))
                    modify_mv_for_end_frames(i, mvs1, len(f))

                    with torch.no_grad():
                        strT = time.time()
                        if i == 0:
                            cur_sr, L1_fea = model(lrs, mvs0, mvs1, pms, rms, ufs)
                        else:
                            cur_sr, L1_fea = model(lrs, mvs0, mvs1, pms, rms, ufs, L1_fea)
                        Sumtime += time.time()-strT
                    
                    print(i,'...', end="\r")
                FPS =  len(f) / Sumtime
                print('Ours at', one_gt, 'FPS:', FPS)
                f1 = open("./FPS/log_Ours_FPS.txt", 'a+')
                f1.write('# Seq [%s] FPS: %s:\n' % ( one_gt, FPS ))
                f1.write('\n')
                f1.close()
        

def modify_mv_for_end_frames(i, mvs, max_idx):
    if i == 0:
        mvs[:,0,:,:,:] = 0.0
        mvs[:,1,:,:,:] = 0.0
        mvs[:,2,:,:,:] = 0.0

    if i == 1:
        mvs[:,0,:,:,:] = mvs[:,2,:,:,:]   
        mvs[:,1,:,:,:] = mvs[:,2,:,:,:]

    if i == 2:
        mvs[:,0,:,:,:] = mvs[:,1,:,:,:]

    if i == max_idx-1:
        mvs[:,4,:,:,:] = 0.0
        mvs[:,5,:,:,:] = 0.0
        mvs[:,6,:,:,:] = 0.0

    if i == max_idx-2:
        mvs[:,5,:,:,:] = mvs[:,4,:,:,:]
        mvs[:,6,:,:,:] = mvs[:,4,:,:,:]

    if i == max_idx-3:
        mvs[:,6,:,:,:] = mvs[:,5,:,:,:]

    return mvs


def main():
    res_vid_name = [
            'PeopleOnStreet_640x400_150F.yuv', 
            'Johnny_320x184_600F.yuv',
            'Kimono1_fps24_480x272_240F.yuv', 
            ]
    gt_vid_name = [
               'PeopleOnStreet_2560x1600_30.yuv', 
               'Johnny_1280x720_60.yuv',
               'Kimono1_1920x1080_24_240F.yuv', 
               ]

    eval_seq(res_vid_name, gt_vid_name, methods_name = "J", coding_cfg = "LD", testing=True, cal_metric=True)


if __name__ == '__main__':
    main()
