from __future__ import print_function, division
import argparse
import sys
import os
import torch
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from math import log10
from datetime import datetime
import cv2 
import sys
import torch
from PIL import Image
import numpy as np
import random
import argparse
from arch.SIDECVSR_our import CVSR_V8 
from metric.psnr_ssim import cal_psnr_ssim
import warnings 
warnings.filterwarnings('ignore')
# ... 
from opt.data_LD_bi import CDVL_sideInfo_Dataset, RandomCrop, ToTensor, Augment
from opt.loss import CharbonnierLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)
INPUT_FRAME = 7

####################

def parse_args():
    parser = argparse.ArgumentParser(description='FIGHT')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--warm_start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=30000, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--max_len', default=7, type=int)
    parser.add_argument('--val_itv',  default=200, type=int)
    parser.add_argument('--weight_decay', default=0.00001, type=float)
    parser.add_argument('--model_name', default=sys.argv[0][:-6], type=str)
    parser.add_argument('--qp', default=37,type=int)  
    return parser.parse_args()


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


def eval_seq(tst_list, gt_list, epoch, coding_cfg = "LD", testing=True, cal_metric=True):  
    if testing:
        # for QP in (args.qp):  #  ['37', '37', '32', '37']  ['37']
        QP = args.qp
        model = CVSR_V8(SCGs=8)
        print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
        # sys.exit(0)
        model_path = './training_results/train_LD_V8_%s/ckpt/epoch-%s.pth' % (args.qp, epoch)
        print('model_path',model_path)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(device)

        tst_path = "./test_data/%s/qp%s/lr_grey/" % (coding_cfg, QP)
        sideInfo_path = "./test_data/%s/qp%s/sideInfo_QP%s/" % (coding_cfg, QP, QP)
        sideInfo_path37 = "./test_data/%s/qp37/sideInfo_QP37/" % (coding_cfg)

        for img_set in tst_list:
            tmp_path = tst_path + img_set + '/'
            tmp_side_path = sideInfo_path + img_set[:-4] + '/'
            tmp_side_path37 = sideInfo_path37 + img_set[:-4] + '/'
            save_path = './train_evl_results/%s_QP%s_V8/%s/' % (coding_cfg, QP, img_set)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            print('evl_seq',tmp_path)
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
                    # ufs = ufs.permute(0,2,1,3,4)

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
                        if i == 0:
                            cur_sr, L1_fea = model(lrs, mvs0, mvs1, pms, rms, ufs)
                        else:
                            cur_sr, L1_fea = model(lrs, mvs0, mvs1, pms, rms, ufs, L1_fea)

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
        f1 = open("./train_evl_results/log/%s_our_V8_%s.txt" % (coding_cfg, QP), 'a+')
        for one_t, one_gt in zip(tst_list, gt_list):
            psnr_s = []
            ssim_s = []
            
            # for QP in str(args.qp):  #  ['37', '37', '32', '37']  [ '37']
            QP = args.qp
            psnr, ssim, psnr_n, ssim_n = cal_psnr_ssim(
                        './train_evl_results/%s_QP%s_V8/' % (coding_cfg, QP),
                        [one_t],
                        [one_gt],
                        './test_data/gt_Y/')
            psnr_s.append(psnr)
            ssim_s.append(ssim)
            f1.write('# Epoch: %s M(%s) Seq(%s) [QP37] PSNR/SSIM:\n' % (epoch, coding_cfg, one_t))
            for p_i in psnr_s:
                print(p_i)
                f1.write(p_i + '\n')
            for s_i in ssim_s:
                print(s_i)
                f1.write(s_i + '\n')
            print('***')
            f1.write('\n')
        f1.close()
        return p_i, s_i   


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


def setup_seed(seed=20):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True


def train(args, model):
    avg_train_loss_list = np.array([])
    important_str = "*"*20 + "\n"
    # dataloader
    composed = transforms.Compose([RandomCrop(64), Augment(), ToTensor()])
    side_dataset = CDVL_sideInfo_Dataset(csv_file='./misc/sequences_list.csv',
                                         transform=composed,
                                         only_I_frame=False,
                                         random_start=True,
                                         max_len=args.max_len,
                                         QP=args.qp,
                                         only_1_GT=True,
                                         part_flag=True,
                                         res_flag=True,
                                         unflt_flag=True,
                                         need_bi_flag=False,
                                         mv_flag=True,
                                         HR_dir="/share3/home/zqiang/CVCP/Uncompressed_HR/",
                                         LR_dir_prefix="/share3/home/zqiang/CVCP/Decoded_LR/LD/",
                                         SideInfo_dir_prefix="/share3/home/zqiang/CVCP/Coding_Priors/LD/"
                                         )
    dataloader = DataLoader(side_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    
    # optimizer
    milestones = [2000]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    # result folder
    print('*'*20)
    res_folder_name = './training_results/%s_%d' % (args.model_name, args.qp) 
    nowtime = time.time()
    timestr = time.strftime('%m%d_%H-%M',time.localtime(nowtime))
    res_folder_logger_name = res_folder_name + '/' + timestr
    

    if not os.path.exists(res_folder_name):
        os.makedirs(res_folder_name)
        os.mkdir(res_folder_name + '/ckpt/')
    else:
        print("WARNING: RESULT PATH ALREADY EXISTED -> " + res_folder_name)
    
    if not os.path.exists(res_folder_logger_name):
        os.makedirs(res_folder_logger_name)

    print('find models here: ', res_folder_name)
    print("QP is " + str(args.qp))
    print('*'*20)
    writer = SummaryWriter(res_folder_logger_name)
    f1 = open(res_folder_name + "/training_log.csv", 'a+')

    # fang dai
    arch_file_name = os.popen('grep arch. '+ sys.argv[0][:-3] +'.py | head -1').read()
    data_type_name = os.popen('grep opt.data '+ sys.argv[0][:-3] +'.py | head -1').read()
    important_str += "*** " + res_folder_name + '\n'
    important_str += "*** QP is " + str(args.qp) + '\n'
    important_str += "*"*20 + "\n"
    
    # print(important_str)
    # training
    model.train()
    for epoch in range(args.warm_start_epoch, args.epochs):
        batch_train_losses = []
        scheduler.step()
        for num, data in enumerate(dataloader):
            optimizer.zero_grad()
            frames = data['lr_imgs'].permute(0,2,1,3,4).to(device)  # [batch, chn, frames, h, w]
            mv0s = data['mvl0s'].permute(0,2,1,3,4)
            mv0s = mv0s.contiguous().to(device) / 32.0
            mv1s = data['mvl1s'].permute(0,2,1,3,4)
            mv1s = mv1s.contiguous().to(device) / 32.0
            res_s = data['res_s'].to(device)
            mpm_s = data['mpm_s'].to(device)
            mpm_s = mpm_s.permute(0,2,1,3,4).contiguous()
            hr = data['hr_imgs'].to(device)
            uf_s = data['unflt_fs'].to(device)

            sr, _ = model(frames, mv0s, mv1s, mpm_s, res_s, uf_s)
            loss = CharbonnierLoss(sr, hr[:,:,0,:,:])

            batch_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # output log 
        now_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        avg_train_loss = round(sum(batch_train_losses) / len(batch_train_losses), 5)
        avg_train_loss_list = np.append(avg_train_loss_list, avg_train_loss)
        log_msg = '[%s] Epoch: %d/%d | average epoch loss: %f' % (now_time, epoch + 1, args.epochs, avg_train_loss)
        writer.add_scalar('Train/Loss', avg_train_loss, epoch)
        print(log_msg)
        f1.write(log_msg)
        f1.write('\n')

        if (epoch + 1) % args.val_itv == 0:
            # print(log_msg)
            # save model 
            torch.save(model.state_dict(), res_folder_name + '/ckpt/' +
                'epoch-%d.pth' % (epoch + 1 + args.warm_start_epoch)) 
            np.save(res_folder_name + '/avg_train_loss_list.np', avg_train_loss_list)
            cur_learning_rate = [param_group['lr'] for param_group in optimizer.param_groups]
            print('Saved model. lr %f' % cur_learning_rate[0])
            f1.write('Saved model. lr %f' % cur_learning_rate[0])
            f1.write('\n')

            # evaluate model
            res_vid_name = ['ParkScene_fps24_480x272_240F.yuv',]
            gt_vid_name = ['ParkScene_1920x1080_24_240F.yuv',]
            psnr_s, ssim_s = eval_seq(res_vid_name, gt_vid_name, coding_cfg = "LD", testing=True, cal_metric=True, epoch=epoch+1)
            # print('[psnr ssim]',psnr_s,ssim_s)
            writer.add_scalar('Train/PSNR', float(psnr_s), epoch)
            writer.add_scalar('Train/SSIM', float(ssim_s), epoch)
            f1.write('PSNR:%f, SSIM: %f' %  (float(psnr_s), float(ssim_s)))
            f1.write('\n')
            print(important_str)
            
    f1.close()


def main(args):
    setup_seed(4)
    model = CVSR_V8(SCGs=8)
    model = model.to(device)
    # model.load_state_dict(torch.load('/share3/home/zqiang/CVSR_train/training_results/train_LD_V8_37/ckpt/epoch-1600.pth', map_location='cpu'))
    print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
    train(args, model)


if __name__ == '__main__':
    args = parse_args()
    main(args)
