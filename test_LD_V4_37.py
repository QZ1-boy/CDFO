import os 
import cv2
import sys
import torch
from PIL import Image
import numpy as np
import argparse
from arch.SIDECVSR_J_L_our import CVSR_V8  # SIDECVSR
from metric.psnr_ssim import cal_psnr_ssim
import warnings 
# warnings.filterwarnings('ignore')

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# torch.cuda.current_device()
# torch.cuda._initialized = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_FRAME = 7

def eval_seq(tst_list, gt_list, methods_name = "J", coding_cfg = "LD", testing=True, cal_metric=True):
    
    if testing:
        QP = 37
        # for QP in ['37']:  #  ['22', '27', '32', '37']
        model = CVSR_V8()
        # print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
        # sys.exit(0)
        model_path = './Models/%s_QP%s_%s_epoch-9500.pth' % (coding_cfg, QP, methods_name)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(device)
        # print(model)

        tst_path = "./test_data/%s/qp%s/lr_grey/" % (coding_cfg, QP)
        sideInfo_path = "./test_data/%s/qp%s/sideInfo_QP%s/" % (coding_cfg, QP, QP)

        for img_set in tst_list:

            tmp_path = tst_path + img_set + '/'
            tmp_side_path = sideInfo_path + img_set[:-4] + '/'
            save_path = './results_evl/%s_QP%s_%s/%s/' % (coding_cfg, QP, methods_name, img_set)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
    
            print(tmp_path)
            for _, _, f in os.walk(tmp_path):
                f.sort()
                
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
                    mvl0 = np.load(tmp_side_path + 'mvl0/' + idx + '_mvl0.npy')
                    mvs = mv2mvs(mvl0).to(device)
                    mvs = torch.unsqueeze(mvs, 0)
                    mvs = mvs.permute(0,1,4,2,3)

                    modify_mv_for_end_frames(i, mvs, len(f))

                    with torch.no_grad():
                        if i == 0:
                            cur_sr, L1_fea = model(lrs, mvs, pms, rms, ufs)
                        else:
                            cur_sr, L1_fea = model(lrs, mvs, pms, rms, ufs, L1_fea)
                        # cur_sr = model(lrs, mvs, pms, rms, ufs)

                    if cur_sr.shape[2] == 1088:
                        out_sr = cur_sr[:,:,:-8,:]
                    elif cur_sr.shape[2] == 736:
                        out_sr = cur_sr[:,:,:-16,:]
                    else:
                        out_sr = cur_sr
                    out_sr = out_sr.cpu().squeeze(0)
                    out_sr = torch.clamp(out_sr,0,1).numpy() * 255.0   
                    cv2.imwrite(save_path + f[i], out_sr[0].astype(np.uint8))
                    
                    print(i,'...', end="\r")
    if cal_metric:
        f1 = open("./log/%s_%s.txt" % (coding_cfg, methods_name), 'a+')
        for one_t, one_gt in zip(tst_list, gt_list):
            psnr_s = []
            ssim_s = []
            QP = 37
            # for QP in ['37']:  #  ['22', '27', '32', '37']
            psnr, ssim = cal_psnr_ssim(
                        './results_evl/%s_QP%s_%s/' % (coding_cfg, QP, methods_name),
                        [one_t],
                        [one_gt],
                        './test_data/gt_Y/')
            psnr_s.append(psnr)
            ssim_s.append(ssim)
            f1.write('# M(%s_%s) Seq(%s) [QP22-QP37] PSNR/SSIM:\n' % (coding_cfg, methods_name, one_t))
            for p_i in psnr_s:
                print(p_i)
                f1.write(p_i + '\n')
            for s_i in ssim_s:
                print(s_i)
                f1.write(s_i + '\n')
            print('***')
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
            'BasketballDrive_fps50_480x272_500F.yuv', 
            'Kimono1_fps24_480x272_240F.yuv', 
            'BQTerrace_fps60_480x272_600F.yuv', 
            'ParkScene_fps24_480x272_240F.yuv',
            'Traffic_640x400_300F.yuv', 
            'PeopleOnStreet_640x400_150F.yuv', 
            'KristenAndSara_320x184_600F.yuv', 
            'Johnny_320x184_600F.yuv',
            'FourPeople_320x184_600F.yuv',
            'Cactus_480x272_500F.yuv',
            ]
    gt_vid_name = [
               'BasketballDrive_1920x1080_50_500F.yuv', 
               'Kimono1_1920x1080_24_240F.yuv', 
               'BQTerrace_1920x1080_60_600F.yuv', 
               'ParkScene_1920x1080_24_240F.yuv',
               'Traffic_2560x1600_30.yuv', 
               'PeopleOnStreet_2560x1600_30.yuv', 
               'KristenAndSara_1280x720_60.yuv', 
               'Johnny_1280x720_60.yuv',
               'FourPeople_1280x720_60.yuv',
               'Cactus_1920x1080_50.yuv',
               ]

    eval_seq(res_vid_name, gt_vid_name, methods_name = "J", coding_cfg = "LD", testing=True, cal_metric=True)


if __name__ == '__main__':
    main()
