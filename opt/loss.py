import torch
import torch.nn.functional as F
import torch.nn as nn
from .lpips import LPIPS
from pytorch_wavelets import DWTForward
import random
def total_variation(x, reduction='mean', mean_res=False):
    hor = x[..., :-1, :] - x[..., 1:, :]
    ver = x[..., :-1] - x[..., 1:]
    if mean_res:
        batch_num = hor.shape[0]
        hor_mean = torch.abs(hor).view(batch_num, -1).mean(1, keepdim=True)
        ver_mean = torch.abs(ver).view(batch_num, -1).mean(1, keepdim=True)
        tot_var = torch.sum(hor_mean) + torch.sum(ver_mean)
    else:
        tot_var = torch.sum(torch.abs(hor)) + torch.sum(torch.abs(ver))
    return tot_var 


def CharbonnierLoss(x, y, mean_res=False):
    if x.shape != y.shape:
        print("!!!")
        print(x.shape, y.shape)
    eps = 1e-4
    diff = x - y
    if mean_res:
        batch_num = x.shape[0]
        diff = diff.view(batch_num, -1).mean(1, keepdim=True)
    loss = torch.sum(torch.sqrt(diff * diff + eps))

    return loss


def Charbonnier_ETCLoss(x, y, mean_res=False):
    batch, frames, channel, height, width = x.shape
    if x.shape != y.shape:
        print("!!!")
        print(x.shape, y.shape)
    eps = 1e-4
    diff = x - y
    if mean_res:
        batch_num = x.shape[0]
        diff = diff.view(batch_num, -1).mean(1, keepdim=True)

    x_energy = torch.fft.rfft2(x, norm='backward').real
    y_energy = torch.fft.rfft2(y, norm='backward').real
    SR_ene_loss = 0
    GT_ene_loss = 0
    for i in range(frames-1):
        SR_ene_loss = GT_ene_loss + (torch.sqrt(x_energy[:,i+1,:,:,:] *  x_energy[:,i+1,:,:,:]) - torch.sqrt(x_energy[:,i,:,:,:] *  x_energy[:,i,:,:,:]))
        GT_ene_loss = GT_ene_loss + (torch.sqrt(y_energy[:,i+1,:,:,:] *  y_energy[:,i+1,:,:,:]) - torch.sqrt(y_energy[:,i,:,:,:] *  y_energy[:,i,:,:,:]))
    loss_spa = torch.sum(torch.sqrt(diff * diff + eps))
    loss_energy = torch.sum(GT_ene_loss) - torch.sum(SR_ene_loss)
    loss = loss_spa + 0.1 * loss_energy

    return loss


def Charbonnier_FCLLoss(x, y, z, mean_res=False):
    if x.shape != y.shape:
        print("!!!")
        print(x.shape, y.shape)
    eps = 1e-4
    diff = x - y
    if mean_res:
        batch_num = x.shape[0]
        diff = diff.view(batch_num, -1).mean(1, keepdim=True)
    loss_spa = torch.sum(torch.sqrt(diff * diff + eps))
    loss = loss_spa + MultiWaveContrastiveLoss(x, y, z)

    return loss




class Char_FCLLoss(nn.Module):
    def __init__(self, mean_res=False):
        super().__init__()
        self.mean_res = mean_res
        self.MWaveContrastiveLoss = MultiWaveContrastiveLoss()
    
    def forward(self, sr, lr_up, hr):
        batch, frames, channel, height, width = sr.shape
        if sr.shape != hr.shape:
            print("!!!")
            print(sr.shape, hr.shape)
        eps = 1e-4
        diff = sr - hr
        if self.mean_res:
            batch_num = sr.shape[0]
            diff = diff.view(batch_num, -1).mean(1, keepdim=True)
        
        loss_spa = torch.sum(torch.sqrt(diff * diff + eps))
        loss = loss_spa + self.MWaveContrastiveLoss(sr, lr_up, hr)

        return loss







class Char_ETC_FCLLoss(nn.Module):
    def __init__(self, mean_res=False):
        super().__init__()
        self.mean_res = mean_res
        self.MWaveContrastiveLoss = MultiWaveContrastiveLoss()
    
    def forward(self, sr, lr_up, hr):
        batch, frames, channel, height, width = sr.shape
        if sr.shape != hr.shape:
            print("!!!")
            print(sr.shape, hr.shape)
        eps = 1e-4
        diff = sr - hr
        if self.mean_res:
            batch_num = sr.shape[0]
            diff = diff.view(batch_num, -1).mean(1, keepdim=True)
        
        x_energy = torch.fft.rfft2(sr, norm='backward').real
        y_energy = torch.fft.rfft2(hr, norm='backward').real
        SR_ene_loss = abs((x_energy[:,:frames-1,:,:,:]-x_energy[:,1:,:,:,:] + eps))
        GT_ene_loss = abs((y_energy[:,:frames-1,:,:,:]-y_energy[:,1:,:,:,:] + eps))
        loss_spa = torch.sum(torch.sqrt(diff * diff + eps))
        loss_energy = torch.log(torch.sum(GT_ene_loss) - torch.sum(SR_ene_loss) + eps)
        loss = loss_spa + 0.01 * loss_energy + self.MWaveContrastiveLoss(sr, lr_up, hr)

        return loss





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class MultiWaveContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips = LPIPS(net='vgg', spatial=False, weight=False).to(device)
        self.neg = 3 # args.mcl_neg  #  3
        self.cl_loss_type = 'l1' 
        self.DWT2 = DWTForward(J=1, wave='haar', mode='reflect').to(device)

    def forward(self, sr, lr, hr):
        # print('sr',sr.device, hr.device, lr.device)
        # sr = sr.squeeze(2)
        # hr = hr.squeeze(2)
        # lr = lr.squeeze(2)
        sr = sr[:,7//2,:,:,:].squeeze(2)
        hr = hr[:,7//2,:,:,:].squeeze(2)
        lr = lr[:,7//2,:,:,:].squeeze(2)
        b, c, h, w = sr.shape
        b_, c_, h_, w_ = lr.shape
        if h_ != h or w_ != w:
            lr = F.interpolate(lr, (h, w), mode='bicubic',  align_corners=True)

        sr_H, sr_L  = self.WaveDecompose(sr)
        hr_H, hr_L  = self.WaveDecompose(hr)
        lr_H, lr_L  = self.WaveDecompose(lr)
        sr_H.append(sr)
        hr_H.append(hr)
        lr_H.append(lr)
        L_list = hr_L + lr_L
        
        if not isinstance(sr_H, list):
            sr_H = [sr_H]
        if not isinstance(hr_H, list):
            hr_H = [hr_H]
        if not isinstance(lr_H, list):
            lr_H = [lr_H]
        if not isinstance(L_list, list):
            L_list = [L_list]

        with torch.no_grad():
            pos_loss1 = self.cl_pos1(sr_H, hr_H)
            pos_loss2 = self.cl_pos2(sr_L, L_list)
            neg_loss1 = self.cl_neg(sr_H, lr_H)
            neg_loss2 = self.cl_neg(sr_H, lr_H)
        loss = self.cl_loss(pos_loss1, neg_loss1) + self.cl_loss(pos_loss2, neg_loss2)

        return loss

    def WaveDecompose(self, x, norm=True):
        waveL = []
        waveH = []
        LL, Hc = self.DWT2(x)
        LH, HL, HH = Hc[0][:, :, 0, :, :], Hc[0][:, :, 1, :, :], Hc[0][:, :, 2, :, :]
        if norm:
            LH, HL, HH = LH * 0.5 + 0.5, HL * 0.5 + 0.5, HH * 0.5 + 0.5
        waveL.append(LL)
        waveH.append(HH)
        waveH.append(HL)
        waveH.append(LH)

        return waveH, waveL 
  
    def cl_pos1(self, sr_list, hr_list):
        pos_loss = 0
        for num in range(len(hr_list)):
            # print('sr_list[num]',sr_list[num].shape, hr_list[num].shape)
            pos_lpips = self.lpips(sr_list[num], hr_list[num]).mean()
            pos_loss += pos_lpips
        pos_loss /= len(hr_list)
        return pos_loss

    def cl_pos2(self, sr_list, hr_list):
        pos_loss = 0
        for num in range(len(hr_list)):
            pos_lpips = self.lpips(sr_list[0], hr_list[num]).mean()
            pos_loss += pos_lpips
        pos_loss /= len(hr_list)
        return pos_loss

    def cl_neg(self, sr_list, lr_list):
        b, c, h, w = sr_list[0].shape
        batch_list = list(range(b))
        neg_lpips = 0
        for num in range(len(lr_list)):
            neg_lpips += self.lpips(sr_list[num], lr_list[num]).mean()

            for neg_times in range(self.neg):
                random.shuffle(batch_list)
                neg_lpips_shuffle = self.lpips(sr_list[num][batch_list, :, :, :], lr_list[num][batch_list, :, :, :]).mean()
                neg_lpips += neg_lpips_shuffle
        neg_lpips /= ((self.neg+1)*len(lr_list))
        return neg_lpips

    def cl_loss(self, pos_loss, neg_loss):

        if self.cl_loss_type in ['l2', 'cosine']:
            cl_loss = pos_loss - neg_loss

        elif self.cl_loss_type == 'l1':
            cl_loss = pos_loss / (neg_loss + 3e-7)
        else:
            raise TypeError(f'{self.args.cl_loss_type} not fount in cl_loss')

        return cl_loss



def MSELoss(x, y):
    loss = nn.MSELoss()
    return loss(x, y)


def CharbonnierLoss_g(x, y, gt_fg):
    eps = 1e-4
    diff = x - y  ### B 1 H W
    diff = diff * gt_fg
    #### X B 1 1 1
    #### = ok
    loss = torch.sum(torch.sqrt(diff * diff + eps))
    return loss


def sobel_loss(img1, img2):
    filter_x = torch.tensor([[1., 2. , 1.], [0., 0., 0.], [-1., -2. , -1.]]).to(img2.device)
    filter_y = torch.tensor([[1., 0 , -1.], [2., 0., -2.], [1., 0. , -1.]]).to(img2.device)
    filter_xy = torch.tensor([[0., 1. , 2.], [-1., 0., 1.], [-2., -1. , 0.]]).to(img2.device)
    filter_yx = torch.tensor([[2., 1. , 0.], [1., 0., -1.], [0., -1. , -2.]]).to(img2.device)

    filter_x = filter_x.view(1,1,3,3)
    filter_y = filter_y.view(1,1,3,3)
    filter_xy = filter_xy.view(1,1,3,3)
    filter_yx = filter_yx.view(1,1,3,3)

    sobel_x_img1 = F.conv2d(img1, filter_x, stride=1, padding=1)
    sobel_y_img1 = F.conv2d(img1, filter_y, stride=1, padding=1)
    sobel_xy_img1 = F.conv2d(img1, filter_xy, stride=1, padding=1)
    sobel_yx_img1 = F.conv2d(img1, filter_yx, stride=1, padding=1)
    

    sobel_x_img2 = F.conv2d(img2, filter_x, stride=1, padding=1)
    sobel_y_img2 = F.conv2d(img2, filter_y, stride=1, padding=1)
    sobel_xy_img2 = F.conv2d(img2, filter_xy, stride=1, padding=1)
    sobel_yx_img2 = F.conv2d(img2, filter_yx, stride=1, padding=1)

    loss = torch.sum((torch.abs(sobel_x_img1 - sobel_x_img2) + torch.abs(sobel_y_img1 - sobel_y_img2) + \
            torch.abs(sobel_xy_img1 - sobel_xy_img2) + torch.abs(sobel_yx_img1 - sobel_yx_img2))) / 4.0

    return loss