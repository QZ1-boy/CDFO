import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
import torchvision
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from arch.ops.dcn import ModulatedDeformConvPack
from einops import rearrange
from einops.layers.torch import Rearrange
import numbers
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
# from ops.dcn.deform_conv import ModulatedDeformConv
import functools
import cv2 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple, trunc_normal_
import matplotlib.pylab as plt
import warnings

class DP_conv(nn.Module):  # dense pointwise convolution
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super(DP_conv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channel,   
            out_channels=in_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channel
        )  # depthwise(DW)conv
        self.point_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel, 
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        ) # pointwise(PW)conv

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class deformable_SKConv(nn.Module):
    def __init__(self, in_fea, out_fea, in_nc, branches=3, reduce=16):
        super(deformable_SKConv, self).__init__()
        self.in_nc = in_nc
        self.branches = branches
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_fea, out_channels=in_nc, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=out_fea, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.offset_mask = nn.ModuleList([])
        self.deform_conv = nn.ModuleList([])
        # self.offset_mask = DP_conv(in_channel=in_nc, out_channel=in_nc * 3 * 1, kernel_size=2 + 1, stride=1)
        # self.deform_conv = ModulatedDeformConv(in_nc, in_nc, kernel_size=3, stride=1, padding=3 // 2, deformable_groups=in_nc)
        for i in range(branches):
            d_size = (2 * i + 1) ** 2
            self.offset_mask.append(DP_conv(in_channel=in_nc, out_channel=in_nc * 3 * d_size, kernel_size=2 * i + 1, stride=1))
            self.deform_conv.append(ModulatedDeformConv(in_nc, in_nc, kernel_size=2 * i + 1, stride=1, padding=(2 * i + 1) // 2, deformable_groups=in_nc))
            # self.deform_conv.append(ModulatedDeformConv(in_nc, in_nc, kernel_size=3, stride=1, padding=3 // 2, deformable_groups=in_nc))
        self.conv_attention = nn.Sequential(
            nn.Conv2d(in_channels=in_nc, out_channels=in_nc, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(in_nc, in_nc, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_nc * branches, out_channels=out_fea, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, fea, inputs, MV):
        out = [] 
        out_att = []
        for i in range(self.branches):
            d_size = (2 * i + 1) ** 2
            offset_mask = self.offset_mask[i](self.input_conv(fea))
            offset = offset_mask[:, :self.in_nc * 2 * d_size, ...]
            offset = offset + MV.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
            mask = torch.sigmoid(offset_mask[:, self.in_nc * 2 * d_size:, ...])
            fused_feat = F.relu(self.deform_conv[i](self.input_conv(inputs), offset, mask), inplace=True)
            attention = self.conv_attention(fused_feat)
            attention = self.gap(attention)
            attention = self.fc(attention)
            out.append(fused_feat)
            out_att.append(attention)

        # for i in range(self.branches):
        #     d_size = 1
        #     offset_mask = self.offset_mask(self.input_conv(fea))
        #     offset = offset_mask[:, :self.in_nc * 2 * d_size, ...]
        #     offset = offset + MV.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        #     mask = torch.sigmoid(offset_mask[:, self.in_nc * 2 * d_size:, ...])
        #     fused_feat = F.relu(self.deform_conv(self.input_conv(inputs_list[i]), offset, mask), inplace=True)
        #     attention = self.conv_attention(fused_feat)
        #     attention = self.gap(attention)
        #     attention = self.fc(attention)
        #     out.append(fused_feat)
        #     out_att.append(attention)

        out = torch.stack(out, dim=1)
        out_att = torch.stack(out_att, dim=1)
        out = out * out_att  # b, 3, c, h, w
        b, _, c, h, w = out.shape
        out = out.view(b, -1, h, w)
        out = self.conv(out)
        return out


# ==========
# Spatio-temporal deformable fusion module
# ==========
class STDF(nn.Module):
    def __init__(self, in_nc=32, out_nc=64, nf=64, nb=3, base_ks=3, deform_ks=3):
        """in_nc: num of input channels. out_nc: num of output channels. nf: num of channels (filters) of each conv layer.
            nb: num of conv layers. deform_ks: size of the deformable kernel.
        """
        super(STDF, self).__init__()
        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        dim = 64
        self.num_heads = 8
        bias = False

        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.project_out_1 = nn.Conv2d(nf, nf, 1, stride=1, padding=1//2)
        self.project_out_3 = nn.Conv2d(nf, nf, 3, stride=1, padding=3//2)
        self.project_out_5 = nn.Conv2d(nf, nf, 5, stride=1, padding=5//2)
        self.d_SKConv = deformable_SKConv(in_fea=nf, out_fea=out_nc, in_nc=in_nc, branches=3, reduce=4)

    def forward(self, inputs, extra_feat, pred_feat, MV):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks
        warped_feat = flow_warp(extra_feat, MV.permute(0, 2, 3, 1))
        ####
        b, c, h, w = warped_feat.shape
        q = warped_feat
        k = extra_feat
        v = pred_feat
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        init_out_list = []
        init_out1 = self.project_out_1(out)
        init_out_list.append(init_out1)
        init_out3 = self.project_out_3(out)
        init_out_list.append(init_out3)
        init_out5 = self.project_out_5(out)
        init_out_list.append(init_out5)
        # compute offset and mask
        out = self.out_conv(init_out3)
        out = self.d_SKConv(out, init_out1, MV)
        return out



def nd_meshgrid(h, w, device):
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)
    id_flow = np.expand_dims(np.stack([xv, yv], axis=-1), axis=0)
    return torch.from_numpy(id_flow).float().to(device)


class STN(nn.Module):
    def __init__(self, mode='bilinear', padding_mode='zeros', normalize=False):
        super(STN, self).__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.norm = normalize
    def forward(self, inputs, u, v):
        mesh = nd_meshgrid(h = inputs.shape[2], w = inputs.shape[3], device = inputs.device)
        if not self.norm:
            h, w = inputs.shape[-2:]
            _u = (u / w * 2) * 32
            _v = (v / h * 2) * 32
        flow = torch.stack([_u, _v], dim=-1).to('cuda')
        mesh = (mesh + flow).clamp(-1,1)
        # warped_img = F.grid_sample(inputs, mesh, mode=self.mode, padding_mode=self.padding_mode) ### original 1.1.0
        warped_img = F.grid_sample(inputs, mesh, mode=self.mode, padding_mode=self.padding_mode, align_corners=True)
        return warped_img


class MV_LOCAL_ATTN(nn.Module):

    def __init__(self, nf=64, p_k=3):
        super(MV_LOCAL_ATTN, self).__init__()
        self.nf = nf
        self.make_fea_patches = torch.nn.Unfold(kernel_size=(p_k, p_k), padding=p_k//2, stride=1)
        self.warp_module = STN(padding_mode='border', normalize=False)

        self.kernel_pred_module = nn.Sequential(
            nn.Conv2d(nf * p_k * p_k * 2, 2*nf, 1, 1, 0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(2*nf, p_k * p_k, 1, 1, 0, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, nbh_fea, cen_fea, mv):
        B, C, H, W = cen_fea.shape
        nbh_fea_p = self.make_fea_patches(nbh_fea)
        nbh_fea_p = nbh_fea_p.view(B, -1, H, W)
        
        cen_fea_p = self.make_fea_patches(cen_fea)
        cen_fea_p = cen_fea_p.view(B, -1, H, W)

        aligned_nbh_fea_p = self.warp_module(nbh_fea_p, mv[:,0,:,:], mv[:,1,:,:])  # aligned_nbh_fea_p.shape = (B, 64*9, H, W)
        fuse_fea = torch.cat([aligned_nbh_fea_p, cen_fea_p], 1)
        local_attn_map = self.kernel_pred_module(fuse_fea)   # (B, 9, H, W)
        
        aligned_nbh_fea_p = aligned_nbh_fea_p.view(B, C, -1, H, W) 
        local_attn_map = torch.unsqueeze(local_attn_map, 1)
        alg_attn_nbh_fea = torch.mean(aligned_nbh_fea_p * local_attn_map, 2)

        return alg_attn_nbh_fea.view(B, -1, H, W)



class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out



def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)



class fea_fusion(nn.Module):
    def __init__(self, nf=64):
        super(fea_fusion, self).__init__()

        self.q = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.p = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.N = 7
        self.nf = nf
    
    def forward(self, feas):
        B, _, H, W = feas.size()
        emb = self.q(feas.view(-1, self.nf, H, W)).view(B, self.N, -1, H, W)
        emb_ref = self.p(emb[:, self.N//2, :, :, :].contiguous())  #  center features
        cor_l = []
        for i in range(self.N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W  # 
            cor_l.append(cor_tmp)

        # obtain the weight-- attention map
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W   
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, self.nf, 1, 1).view(B, -1, H, W)
        feas_ = feas * cor_prob

        return feas_



class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x



class Block(nn.Module):
    def __init__(self,
               num_residual_units,
               kernel_size,
               width_multiplier=1,
               group=4):
        super(Block, self).__init__()

        body = []
        conv = nn.Conv2d( num_residual_units, int(num_residual_units * width_multiplier), kernel_size, padding=kernel_size // 2)
        body.append(conv)
        body.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        conv = nn.Conv2d( int(num_residual_units * width_multiplier), num_residual_units, kernel_size, padding=kernel_size // 2)
        body.append(conv)
        initialize_weights(body, 0.1)
        self.body = nn.Sequential(*body)

        down = []
        down.append(nn.Conv2d(num_residual_units, num_residual_units, 1))
        down.append(Interpolate(scale_factor=0.5))
        self.down = nn.Sequential(*down)

        up = []
        up.append(nn.Conv2d(num_residual_units, num_residual_units, 1))
        up.append(Interpolate(scale_factor=2.0))
        self.up = nn.Sequential(*up)
        initialize_weights([self.up, self.down], 0.1)

    def forward(self, x_list):
        res_list = [self.body(x) for x in x_list]
        down_res_list = [res_list[0]] + [self.down(x) for x in res_list[:-1]]
        up_res_list = [self.up(x) for x in res_list[1:]] + [res_list[-1]]
        x_list = [
            x + r + d + u
            for x, r, d, u in zip(x_list, res_list, down_res_list, up_res_list)
        ]
        return x_list




class Block_(nn.Module):
    def __init__(self,num_residual_units,kernel_size, width_multiplier=1, group=4):
        super(Block_, self).__init__()
        body = []
        conv = nn.Conv2d( num_residual_units, int(num_residual_units * width_multiplier), kernel_size, padding=kernel_size // 2)
        body.append(conv)
        body.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        conv = nn.Conv2d( int(num_residual_units * width_multiplier), num_residual_units, kernel_size, padding=kernel_size // 2)
        body.append(conv)
        initialize_weights(body, 0.1)
        self.body = nn.Sequential(*body)

        down = []
        down.append(nn.Conv2d(num_residual_units, num_residual_units, 1))
        down.append(Interpolate(scale_factor=0.5))
        self.down = nn.Sequential(*down)

        up = []
        up.append(nn.Conv2d(num_residual_units, num_residual_units, 1))
        up.append(Interpolate(scale_factor=2.0))
        self.up = nn.Sequential(*up)
        initialize_weights([self.up, self.down], 0.1)

    def forward(self, x):
        r = self.body(x) 
        down_res = self.up(self.body(self.down(x)))
        up_res = self.down(self.body(self.up(x)))
        out = x + r + down_res + up_res
        return out


class SCGroup(nn.Module):
    def __init__(self, nf=64, back_RBs=3):
        super(SCGroup, self).__init__()
        self.nf = nf
        self.conv = nn.Conv2d(nf, nf, 3, padding=1)
        body = []
        for _ in range(back_RBs):
            body.append(Block(nf,kernel_size=3, width_multiplier=4))
        self.body = nn.Sequential(*body)
    
    def forward(self, x_list):
        res_list = self.body(x_list)
        res_list = [self.conv(x) for x in res_list]
        x_list = [
            x + r
            for x, r in zip(x_list, res_list)
        ]
        return x_list



class SCGroup_(nn.Module):
    def __init__(self, nf=64, back_RBs=3):
        super(SCGroup_, self).__init__()
        self.nf = nf
        self.conv = nn.Conv2d(nf, nf, 3, padding=1)
        body = []
        for _ in range(back_RBs):
            body.append(Block_(nf,kernel_size=3, width_multiplier=4))
        self.body = nn.Sequential(*body)
    
    def forward(self, x):
        r = self.body(x)
        r = self.conv(r) 
        out  = x + r
        return out



class SCNet(nn.Module):
    def __init__(self, nf=64, SCGroupN=4):
        super(SCNet, self).__init__()
        self.nf = nf
        body = []
        for _ in range(SCGroupN):
            body.append(SCGroup(nf=nf))
        self.body = nn.Sequential(*body)
    
    def forward(self, x_list):
        res_list = self.body(x_list)
        x_list = [
            x + r
            for x, r in zip(x_list, res_list)
        ]
        return x_list




class SCNet_(nn.Module):
    def __init__(self, nf=64, SCGroupN=4):
        super(SCNet_, self).__init__()
        self.nf = nf
        body = []
        for _ in range(SCGroupN):
            body.append(SCGroup_(nf=nf))
        self.body = nn.Sequential(*body)
    
    def forward(self, x):
        r = self.body(x)
        out = r + x
        return out






class AGGBlock(nn.Module):

    def __init__(self,
               num_residual_units,
               kernel_size,
               width_multiplier=1,
               group=4):
        super(AGGBlock, self).__init__()

        body = []
        conv = nn.Conv2d( num_residual_units, int(num_residual_units * width_multiplier), kernel_size, padding=kernel_size // 2)
        body.append(conv)
        body.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        conv = nn.Conv2d(int(num_residual_units * width_multiplier), num_residual_units, kernel_size, padding=kernel_size // 2)
        body.append(conv)
        initialize_weights(body, 0.1)
        self.body = nn.Sequential(*body)

        down = []
        down.append(nn.Conv2d(num_residual_units, num_residual_units, 1))
        down.append(Interpolate(scale_factor=0.5))
        self.down = nn.Sequential(*down)

        up = []
        up.append(nn.Conv2d(num_residual_units, num_residual_units, 1))
        up.append(Interpolate(scale_factor=2.0))
        self.up = nn.Sequential(*up)
        initialize_weights([self.up, self.down], 0.1)

    def forward(self, x_list):
        res_list = [self.body(x) for x in x_list]
        down_res_list = [res_list[0]] + [self.down(x) for x in res_list[:-1]]
        up_res_list = [self.up(x) for x in res_list[1:]] + [res_list[-1]]
        x_list = [
            x + r + d + u
            for x, r, d, u in zip(x_list, res_list, down_res_list, up_res_list)
        ]
        return x_list



class AGGSCGroup(nn.Module):
    def __init__(self, nf=64, back_RBs=3):
        super(AGGSCGroup, self).__init__()
        self.nf = nf
        self.conv = nn.Conv2d(nf, nf, 3, padding=1)
        body = []
        for _ in range(back_RBs):
            body.append(
                AGGBlock( nf, kernel_size=3,width_multiplier=4))
        self.body = nn.Sequential(*body)
    
    def forward(self, x_list):
        res_list = self.body(x_list)
        res_list = [self.conv(x) for x in res_list]
        x_list = [
            x + r
            for x, r in zip(x_list, res_list)
        ]
        return x_list



class AGGSCNet(nn.Module):
    def __init__(self, nf=64, SCGroupN=4):
        super(AGGSCNet, self).__init__()
        self.nf = nf
        body = []
        for _ in range(SCGroupN):
            body.append(SCGroup(nf=nf))
        self.body = nn.Sequential(*body)
    
    def forward(self, x_list):
        res_list = self.body(x_list)
        x_list = [
            x + r
            for x, r in zip(x_list, res_list)
        ]
        return x_list



class RiRGroup(nn.Module):
    def __init__(self, nf=64, back_RBs=3):
        super(RiRGroup, self).__init__()
        self.nf = nf
        self.conv = nn.Conv2d(nf, nf, 3, padding=1)
        body = []
        for _ in range(back_RBs):
            body.append(nn.Conv2d(nf, nf*4, 3, 1, padding=1))
            body.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            body.append(nn.Conv2d(nf*4, nf, 3, 1, padding=1))
            body.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.body = nn.Sequential(*body)
    
    def forward(self, x_list):
        res_list = self.body(x_list)
        res_list = self.conv(res_list)
        x_list = x_list + res_list

        return x_list



class RinRNet(nn.Module):
    def __init__(self, nf=64, SCGroupN=4):
        super(RinRNet, self).__init__()
        self.nf = nf
        body = []
        for _ in range(SCGroupN):
            body.append(RiRGroup(nf=nf))
        self.body = nn.Sequential(*body)
    
    def forward(self, x):
        res_ = self.body(x)
        x_list = x + res_
        
        return x_list



class SFTLayer(nn.Module):
    def __init__(self, nf=64):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(nf//2+nf, nf, 1)
        self.SFT_scale_conv1 = nn.Conv2d(nf, nf, 1)
        self.SFT_shift_conv0 = nn.Conv2d(nf//2+nf, nf, 1)
        self.SFT_shift_conv1 = nn.Conv2d(nf, nf, 1)

    def forward(self, feas, side_feas):
        x_in = torch.cat([feas, side_feas],1)
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x_in), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x_in), 0.1, inplace=True))
        return feas * (scale + 1) + shift



class ResBlock_SFT(nn.Module):
    def __init__(self, nf=64):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer(nf=nf)
        self.conv0 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.sft1 = SFTLayer(nf=nf)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)

    def forward(self, feas, side_feas):
        fea = self.sft0(feas, side_feas)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1(fea, side_feas)
        fea = self.conv1(fea)
        return feas + fea  # return a tuple containing features and conditions



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows



def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x



class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = 8
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
            # print('[....]')
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        # print('[self.input_resolution]',self.input_resolution)
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        # print('[self.window_size]',img_mask.shape, self.window_size)
        self.window_size = 8
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1  self.window_size
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        # print('[x]',x.shape)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # self.drop_path

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops



class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


class side_embeded_feature_extract_block(nn.Module):
    def __init__(self, nf=64):
        super(side_embeded_feature_extract_block, self).__init__()

        self.RB_wSide_1 = ResBlock_SFT(nf=nf)
        self.RB_wSide_2 = ResBlock_SFT(nf=nf)
        self.RB_wSide_3 = ResBlock_SFT(nf=nf)
        self.RB_wSide_4 = ResBlock_SFT(nf=nf)
        self.RB_wSide_5 = ResBlock_SFT(nf=nf)
        self.RB_wSide_6 = ResBlock_SFT(nf=nf)
        self.RB_wSide_7 = ResBlock_SFT(nf=nf)


    def forward(self, img_feas, side_feas):
        fea1_o = self.RB_wSide_1(img_feas, side_feas)
        fea2_o = self.RB_wSide_2(fea1_o, side_feas)
        fea3_o = self.RB_wSide_3(fea2_o, side_feas)
        fea4_o = self.RB_wSide_4(fea3_o, side_feas)
        fea5_o = self.RB_wSide_5(fea4_o, side_feas)
        fea6_o = self.RB_wSide_6(fea5_o, side_feas)
        fea7_o = self.RB_wSide_7(fea6_o, side_feas)
        
        return fea7_o


class BackBoneBlock(nn.Module):
    def __init__(self, num, fm, **args):
        super().__init__()
        self.arr = nn.ModuleList([])
        for _ in range(num):
            self.arr.append(fm(**args))

    def forward(self, x):
        for block in self.arr:
            x = block(x)
        return x



class PAIBackBoneBlock(nn.Module):
    def __init__(self, num, fm, **args):
        super().__init__()
        self.arr = nn.ModuleList([])
        for _ in range(num):
            self.arr.append(fm(**args))

    def forward(self, x1, X2):
        for block in self.arr:
            x = block(x1, X2)
        return x


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)




def featuremap_visual(feature, iter = 1,  out_dir=None,  # 特征图保存路径文件
                    save_feature=True,  # 是否以图片形式保存特征图
                    feature_title=None,  # 特征图名字，默认以shape作为title
                    num_ch=-1,  # 显示特征图前几个通道，-1 or None 都显示
                    nrow=8,  # 每行显示多少个特征图通道
                    padding=0,  # 特征图之间间隔多少像素值
                    pad_value=1  # 特征图之间的间隔像素
                    ):
    # feature = feature.detach().cpu()
    b, c, h, w = feature.shape
    feature = feature[0][29:30,:,:]   #  30:31
    # feature = feature.unsqueeze(1)
    
    img = feature.detach().cpu()
    img = img.numpy()   #  .squeeze(1)
    # print('[img]',img.shape)
    images = img.transpose((1, 2, 0))

    # title = str(images.shape) if feature_title is None else str(feature_title)
    title = str(h) + '-' + str(w) + '-' + str(c) + '-' + feature_title + '-'  +  str(iter) 

    plt.title(title)
    min_val = np.amin(images)
    max_val = np.amax(images)    
    images =  (images - min_val)/(max_val-min_val)
    # images = images/ 255.
    
    out_root = '/share3/home/zqiang/CVSR_train/viz_feat_EGLA/' + title + '.png' 
    # print('[out_root]',out_root)
    plt.figure()
    plt.imshow(images)
    plt.axis('off')
    fig = plt.gcf()
    # fig.set_size_inches(7.0/3,7.0/3) #dpi = 300, output = 700*700 pixels
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(out_root, cmap='cmap',bbox_inches='tight', transparent=True, dpi=100)  #  , pad_inches = 0






def featuremap_visual_0(feature, iter = 1,  out_dir=None,  # 特征图保存路径文件
                    save_feature=True,  # 是否以图片形式保存特征图
                    feature_title=None,  # 特征图名字，默认以shape作为title
                    num_ch=-1,  # 显示特征图前几个通道，-1 or None 都显示
                    nrow=8,  # 每行显示多少个特征图通道
                    padding=0,  # 特征图之间间隔多少像素值
                    pad_value=1  # 特征图之间的间隔像素
                    ):
    # feature = feature.detach().cpu()
    b, c, h, w = feature.shape
    feature = feature[0][0:1,:,:]   #  30:31
    # feature = feature.unsqueeze(1)
    
    img = feature.detach().cpu()
    img = img.numpy()   #  .squeeze(1)
    # print('[img]',img.shape)
    images = img.transpose((1, 2, 0))

    # title = str(images.shape) if feature_title is None else str(feature_title)
    title = str(h) + '-' + str(w) + '-' + str(c) + '-' + feature_title + '-'  +  str(iter) 

    plt.title(title)
    min_val = np.amin(images)
    max_val = np.amax(images)    
    images =  (images - min_val)/(max_val-min_val)
    # images = images/ 255.
    
    out_root = '/share3/home/zqiang/CVSR_train/viz_feat/' + title + '.png' 
    # print('[out_root]',out_root)
    plt.figure()
    plt.imshow(images)
    plt.axis('off')
    fig = plt.gcf()
    # fig.set_size_inches(7.0/3,7.0/3) #dpi = 300, output = 700*700 pixels
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(out_root, cmap='cmap',bbox_inches='tight', transparent=True, dpi=100)  #  , pad_inches = 0





class TransformerBlock(nn.Module):
    def __init__(self, dim=48, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=WithBias_LayerNorm):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        # self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        # self.ffn = GFeedForward(dim, ffn_expansion_factor, bias)
        # self.mlp = Mlp(in_features=dim, hidden_features=dim, act_layer=nn.GELU, drop=0.)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        # x = x + self.conv(self.norm2(x))
        # x = x + self.mlp(self.norm2(x))
        # x = x + self.attn(self.norm1(x))
        # x = x + self.attn(self.norm1(x))
        # x = x + self.ffn(self.norm2(x))

        return x



class PartitionTransformerBlock(nn.Module):
    def __init__(self, dim=48, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=WithBias_LayerNorm):
        super(PartitionTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        # self.side_to_feaoneUDK = side_to_feaoneUDK(dim, nf=16)
        self.SA = SpatialAttention()

    def forward(self, x1, x2):
        x2 = self.SA(x2) # self.side_to_feaoneUDK(x2)
        x1 = x1 + self.attn(self.norm1(x1)) + x2
        x1 = x1 + self.conv(self.norm2(x1))

        x2 = self.SA(x2) # x2 = self.side_to_feaoneUDK(x2)
        x1 = x1 + self.attn(self.norm1(x1)) + x2
        x1 = x1 + self.conv(self.norm2(x1))

        x2 = self.SA(x2) # x2 = self.side_to_feaoneUDK(x2)
        x1 = x1 + self.attn(self.norm1(x1)) + x2
        x1 = x1 + self.conv(self.norm2(x1))

        x2 = self.SA(x2) # x2 = self.side_to_feaoneUDK(x2)
        x1 = x1 + self.attn(self.norm1(x1)) + x2
        x1 = x1 + self.conv(self.norm2(x1))

        return x1



class PartitionTransformerSA(nn.Module):
    def __init__(self, dim=48, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=WithBias_LayerNorm):
        super(PartitionTransformerSA, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.side_to_feaoneUDSA = side_to_feaoneUDSA(dim, nf=16)
        # self.SA = SpatialAttention()

    def forward(self, x1, x2):
        x2 = self.side_to_feaoneUDSA(x2) # self.side_to_feaoneUDK(x2)
        x1 = x1 + self.attn(self.norm1(x1)) + x2
        x1 = x1 + self.conv(self.norm2(x1))

        x2 = self.side_to_feaoneUDSA(x2) # x2 = self.side_to_feaoneUDK(x2)
        x1 = x1 + self.attn(self.norm1(x1)) + x2
        x1 = x1 + self.conv(self.norm2(x1))

        x2 = self.side_to_feaoneUDSA(x2) # x2 = self.side_to_feaoneUDK(x2)
        x1 = x1 + self.attn(self.norm1(x1)) + x2
        x1 = x1 + self.conv(self.norm2(x1))

        x2 = self.side_to_feaoneUDSA(x2) # x2 = self.side_to_feaoneUDK(x2)
        x1 = x1 + self.attn(self.norm1(x1)) + x2
        x1 = x1 + self.conv(self.norm2(x1))

        return x1




class PartitionTransformerSA_1(nn.Module):
    def __init__(self, dim=48, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=WithBias_LayerNorm):
        super(PartitionTransformerSA_1, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.side_to_feaoneUDSA = side_to_feaoneUDSA(dim, nf=16)
        # self.SA = SpatialAttention()

    def forward(self, x1, x2):
        x2 = self.side_to_feaoneUDSA(x2) # self.side_to_feaoneUDK(x2)
        # featuremap_visual(x2, feature_title='UDSA_1st' )
        x1 = x1 + self.attn(self.norm1(x1)) + x2
        x1 = x1 + self.conv(self.norm2(x1))
        # featuremap_visual(x1, feature_title='PAISA_1st' )

        x2 = self.side_to_feaoneUDSA(x2) # x2 = self.side_to_feaoneUDK(x2)
        # featuremap_visual(x2, feature_title='UDSA_2st' )
        x1 = x1 + self.attn(self.norm1(x1)) + x2
        x1 = x1 + self.conv(self.norm2(x1))
        # featuremap_visual(x1, feature_title='PAISA_2st' )

        x2 = self.side_to_feaoneUDSA(x2) # x2 = self.side_to_feaoneUDK(x2)
        # featuremap_visual(x2, feature_title='UDSA_3st' )
        x1 = x1 + self.attn(self.norm1(x1)) + x2
        x1 = x1 + self.conv(self.norm2(x1))
        # featuremap_visual(x1, feature_title='PAISA_3st' )

        # x2 = self.side_to_feaoneUDSA(x2) # x2 = self.side_to_feaoneUDK(x2)
        # x1 = x1 + self.attn(self.norm1(x1)) + x2
        # x1 = x1 + self.conv(self.norm2(x1))

        return x1




class PartitionTransformerSA_2(nn.Module):
    def __init__(self, dim=48, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=WithBias_LayerNorm):
        super(PartitionTransformerSA_2, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.side_to_feaoneUDSA = side_to_feaoneUDSA_2(dim, nf=16)
        # self.SA = SpatialAttention()

    def forward(self, x1, x2):
        # print('[x1]',x1.shape, x2.shape)
        # featuremap_visual_0(x2, feature_title='UDSA_0st')
        # featuremap_visual(x1, feature_title='PAISA_0st' )
        x2 = self.side_to_feaoneUDSA(x2) + x1 # self.side_to_feaoneUDK(x2)
        # featuremap_visual_0(x2, feature_title='UDSA_1st' )
        x1 = x1 + self.attn(self.norm1(x1))
        x1 = x1 + self.conv(self.norm2(x1)) + x2
        # featuremap_visual(x1, feature_title='PAISA_1st' )
        x2 = self.side_to_feaoneUDSA(x2) + x2 # x2 = self.side_to_feaoneUDK(x2)
        # featuremap_visual_0(x2, feature_title='UDSA_2st' )
        x1 = x1 + self.attn(self.norm1(x1)) 
        x1 = x1 + self.conv(self.norm2(x1))  + x2
        # featuremap_visual(x1, feature_title='PAISA_2st' )
        x2 = self.side_to_feaoneUDSA(x2) + x2 # x2 = self.side_to_feaoneUDK(x2)
        # featuremap_visual_0(x2, feature_title='UDSA_3st' )
        x1 = x1 + self.attn(self.norm1(x1))
        x1 = x1 + self.conv(self.norm2(x1)) + x2
        # featuremap_visual(x1, feature_title='PAISA_3st' )

        # x2 = self.side_to_feaoneUDSA(x2) # x2 = self.side_to_feaoneUDK(x2)
        # x1 = x1 + self.attn(self.norm1(x1)) + x2
        # x1 = x1 + self.conv(self.norm2(x1))

        return x1




class PartitionTransformerSA_woPAB(nn.Module):
    def __init__(self, dim=48, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=WithBias_LayerNorm):
        super(PartitionTransformerSA_woPAB, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        # self.side_to_feaoneUDSA = side_to_feaoneUDSA_2(dim, nf=16)
        # self.SA = SpatialAttention()

    def forward(self, x1):
        # print('[x1]',x1.shape, x2.shape)
        # featuremap_visual_0(x2, feature_title='UDSA_0st')
        # featuremap_visual(x1, feature_title='PAISA_0st' )
        # x2 = self.side_to_feaoneUDSA(x2) + x1 # self.side_to_feaoneUDK(x2)
        # featuremap_visual_0(x2, feature_title='UDSA_1st' )
        x1 = x1 + self.attn(self.norm1(x1))
        x1 = x1 + self.conv(self.norm2(x1)) # + x2
        # featuremap_visual(x1, feature_title='PAISA_1st' )
        # x2 = self.side_to_feaoneUDSA(x2) + x2 # x2 = self.side_to_feaoneUDK(x2)
        # featuremap_visual_0(x2, feature_title='UDSA_2st' )
        x1 = x1 + self.attn(self.norm1(x1)) 
        x1 = x1 + self.conv(self.norm2(x1))  # + x2
        # featuremap_visual(x1, feature_title='PAISA_2st' )
        # x2 = self.side_to_feaoneUDSA(x2) + x2 # x2 = self.side_to_feaoneUDK(x2)
        # featuremap_visual_0(x2, feature_title='UDSA_3st' )
        x1 = x1 + self.attn(self.norm1(x1))
        x1 = x1 + self.conv(self.norm2(x1)) # + x2
        # featuremap_visual(x1, feature_title='PAISA_3st' )

        # x2 = self.side_to_feaoneUDSA(x2) # x2 = self.side_to_feaoneUDK(x2)
        # x1 = x1 + self.attn(self.norm1(x1)) + x2
        # x1 = x1 + self.conv(self.norm2(x1))

        return x1







## Gated-Dconv Feed-Forward Network (GDFN)
class GFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GFeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



class transformer_feat_extract(nn.Module):
    def __init__(self, hiddenDim=64,):
        super(transformer_feat_extract, self).__init__()
        lamRes = torch.nn.Parameter(torch.ones(1))
        lamX = torch.nn.Parameter(torch.ones(1))
        self.adaptiveWeight = (lamRes, lamX)
        num_heads = 8

        self.path1 = nn.Sequential(
            BackBoneBlock(1, TransformerBlock,
                          dim=hiddenDim, num_heads=num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=WithBias_LayerNorm),
            nn.Conv2d(hiddenDim, hiddenDim, kernel_size=3, padding=1),
        )
       

    def forward(self, x):
        fea_o = self.path1(x)
        
        return fea_o



class PAItransformer_feat_extract(nn.Module):
    def __init__(self, hiddenDim=64, num_heads=8):
        super(PAItransformer_feat_extract, self).__init__()
        lamRes = torch.nn.Parameter(torch.ones(1))
        lamX = torch.nn.Parameter(torch.ones(1))
        self.adaptiveWeight = (lamRes, lamX)
        self.path1 = PartitionTransformerBlock(dim=hiddenDim, num_heads=num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=WithBias_LayerNorm)

    def forward(self, x1, x2):
        fea_o = self.path1(x1, x2)       
        return fea_o



class PAItransformerSA(nn.Module):
    def __init__(self, hiddenDim=64, num_heads=8):
        super(PAItransformerSA, self).__init__()
        lamRes = torch.nn.Parameter(torch.ones(1))
        lamX = torch.nn.Parameter(torch.ones(1))
        self.adaptiveWeight = (lamRes, lamX)
        self.path1 = PartitionTransformerSA(dim=hiddenDim, num_heads=num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=WithBias_LayerNorm)

    def forward(self, x1, x2):
        fea_o = self.path1(x1, x2)       
        return fea_o



class PAItransformerSA_1(nn.Module):
    def __init__(self, hiddenDim=64, num_heads=8):
        super(PAItransformerSA_1, self).__init__()
        lamRes = torch.nn.Parameter(torch.ones(1))
        lamX = torch.nn.Parameter(torch.ones(1))
        self.adaptiveWeight = (lamRes, lamX)
        self.path1 = PartitionTransformerSA_1(dim=hiddenDim, num_heads=num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=WithBias_LayerNorm)

    def forward(self, x1, x2):
        fea_o = self.path1(x1, x2)       
        return fea_o


class PAItransformerSA_2(nn.Module):
    def __init__(self, hiddenDim=64, num_heads=8):
        super(PAItransformerSA_2, self).__init__()
        lamRes = torch.nn.Parameter(torch.ones(1))
        lamX = torch.nn.Parameter(torch.ones(1))
        self.adaptiveWeight = (lamRes, lamX)
        self.path1 = PartitionTransformerSA_2(dim=hiddenDim, num_heads=num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=WithBias_LayerNorm)

    def forward(self, x1, x2):
        fea_o = self.path1(x1, x2)       
        return fea_o



class PAItransformerSA_woPAB(nn.Module):
    def __init__(self, hiddenDim=64, num_heads=8):
        super(PAItransformerSA_woPAB, self).__init__()
        lamRes = torch.nn.Parameter(torch.ones(1))
        lamX = torch.nn.Parameter(torch.ones(1))
        self.adaptiveWeight = (lamRes, lamX)
        self.path1 = PartitionTransformerSA_woPAB(dim=hiddenDim, num_heads=num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=WithBias_LayerNorm)

    def forward(self, x1):
        fea_o = self.path1(x1)       
        return fea_o





class transformer_feat_extract_1(nn.Module):
    def __init__(self, hiddenDim=64,):
        super(transformer_feat_extract_1, self).__init__()
        lamRes = torch.nn.Parameter(torch.ones(1))
        lamX = torch.nn.Parameter(torch.ones(1))
        self.adaptiveWeight = (lamRes, lamX)
        num_heads = 8

        self.path1 = nn.Sequential(
            BackBoneBlock(1, TransformerBlock,
                          dim=hiddenDim, num_heads=num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type=WithBias_LayerNorm),
            nn.Conv2d(hiddenDim, hiddenDim, kernel_size=3, padding=1),
        )
       

    def forward(self, x):
        fea_o = self.path1(x)
        fea_o = self.path1(fea_o)
        
        return fea_o



class side_to_fea(nn.Module):
    def __init__(self, nf=32):
        super(side_to_fea, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, side):
        
        return self.body(side)



class side_to_feaone(nn.Module):
    def __init__(self, nf=32):
        super(side_to_feaone, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, side):
        
        return self.body(side)



class side_to_feaoneUD(nn.Module):
    def __init__(self, nf=32):
        super(side_to_feaoneUD, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, nf, 3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ConvTranspose2d(nf, nf, 3, stride=2, padding=2),  # , output_padding=1
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ConvTranspose2d(nf, nf, 3, stride=2, padding=2, output_padding=1),  #  , output_padding=1
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, 1, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, side):
        
        return self.body(side)



class side_to_feaoneUDK(nn.Module):
    def __init__(self, in_f, nf=32):
        super(side_to_feaoneUDK, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_f, nf, 3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ConvTranspose2d(nf, nf, 3, stride=2, padding=2),  # , output_padding=1
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ConvTranspose2d(nf, nf, 3, stride=2, padding=2, output_padding=1),  #  , output_padding=1
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, in_f, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, side):
        
        return self.body(side)





class side_to_feaoneUDSA(nn.Module):
    def __init__(self, in_f, nf=32):
        super(side_to_feaoneUDSA, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_f, nf, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Conv2d(nf, nf, 3, stride=1, padding=1, bias=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            SpatialAttention(),
            nn.ConvTranspose2d(nf, nf, 3, stride=2, padding=2),  # , output_padding=1
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ConvTranspose2d(nf, nf, 3, stride=2, padding=2, output_padding=1),  #  , output_padding=1
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, in_f, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, side):
        
        return self.body(side)





class side_to_feaoneUDSA_2(nn.Module):
    def __init__(self, in_f, nf=32):
        super(side_to_feaoneUDSA_2, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_f, nf, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, 3, stride=2, padding=2, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Conv2d(nf, nf, 3, stride=1, padding=1, bias=True),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            SpatialAttention(),
            nn.ConvTranspose2d(nf, nf, 3, stride=2, padding=2),  # , output_padding=1
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ConvTranspose2d(nf, nf, 3, stride=2, padding=2, output_padding=1),  #  , output_padding=1
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, in_f, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        # self.body1 = nn.Sequential(
        #     nn.Conv2d(in_f, nf, 3, stride=1, padding=1, bias=True),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True), )
        # self.body2 = nn.Sequential(
        #     nn.Conv2d(nf, nf, 3, stride=2, padding=2, bias=True),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),)
        # self.body3 = nn.Sequential(
        #     nn.Conv2d(nf, nf, 3, stride=2, padding=2, bias=True),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),)
        # self.body3_1 = nn.Sequential(
        #     nn.Conv2d(nf, nf, 3, stride=1, padding=1, bias=True),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),)
        # self.body4 = nn.Sequential(
        #     nn.ConvTranspose2d(nf, nf, 3, stride=2, padding=2), 
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True), )
        # self.body5 = nn.Sequential(
        #     nn.ConvTranspose2d(nf, nf, 3, stride=2, padding=2, output_padding=1),  #  , output_padding=1
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True), )
        # self.body6 = nn.Sequential(
        #     nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),)
        # self.body7 = nn.Sequential(
        #     nn.Conv2d(nf, in_f, 3, 1, 1, bias=True),
        #     # nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     )
        
        # self.SA = SpatialAttention()

    def forward(self, side):
        x = self.body(side)
        # x1 = self.body1(side)
        # x2 = self.body2(x1)
        # x3 = self.body3(x2)
        # x3_1 = self.body3_1(x3)
        # x4 = self.SA(x3_1) # + x3_1
        # x5 = self.body4(x4) # + x2
        # x6 = self.body5(x5) 
        # x7 = self.body6(x6)
        # x8 = self.body7(x7)

        return x







class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )



class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


class PAM(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x1, x2):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        
        m_batchsize, C, height, width = x1.size()
        proj_query = self.query_conv(x1).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x2).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x2).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x2
        return out



class CAM_(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out



class CAM(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x0, x1, x2):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x1.size()
        x = x0 + x2 
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x2.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x2.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x2
        return out



## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=1, bias=bias))
        modules_body.append(nn.ReLU(True))
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=1, bias=bias))
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=1, bias=bias))
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # print('[res]',res.shape, x.shape)
        res += x
        return res


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



class NonLocalAttention(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, softmax_scale=10, average=True, res_scale=1):
        super(NonLocalAttention, self).__init__()
        self.res_scale = res_scale
        self.conv_match1 = nn.Sequential(nn.Conv2d( channel, channel//reduction, 1),
                                        nn.PReLU(),)
        self.conv_match2 = nn.Sequential(nn.Conv2d( channel, channel//reduction, 1),
                                        nn.PReLU(),)
        self.conv_assembly = nn.Sequential(nn.Conv2d( channel, channel, 1),
                                        nn.PReLU(),)
    def forward(self, input):
        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input)

        N,C,H,W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0,2,3,1).view((N,H*W,C))
        x_embed_2 = x_embed_2.view(N,C,H*W)
        score = torch.matmul(x_embed_1, x_embed_2)
        score = torch.softmax(score, dim=2)
        x_assembly = x_assembly.view(N,-1,H*W).permute(0,2,1)
        x_final = torch.matmul(score, x_assembly)
        return x_final.permute(0,2,1).view(N,-1,H,W)+self.res_scale*input



def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)



class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim=64):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, res, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)  # (BxW) x H x C 
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)  # (BxH) x C x W 
        
        res_mask = res.masked_fill(res != 0, 1.0)

        proj_key = self.key_conv(res_mask*x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)  # (BxW) x C x H
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)  # (BxH) x C x W
        
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)  # (BxW) x C x H
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)  # (BxH) x C x W

        # O((H+W)HWC)  
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)  # B x H x W x H
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)  # B x H x W x W
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)  # (BxW) x H x H
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)  # (BxH) x W x W 

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)  # B x C x H x W
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)  # B x C x H x W
        OUT = self.gamma*(out_H + out_W) + x

        return OUT



class ConvUnit(nn.Module):
    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, num_groups=1,
                 use_act=True, act_type='relu'):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_in,
                              out_channels=num_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=num_groups,
                              bias=False,
                              padding_mode='zeros')
        self.use_act = use_act

    def forward(self, x):
        """ forward of conv unit """
        out = self.conv(x)
        return out


class LLongRangAttention(nn.Module):
    """ Long-Rang Attention Module"""
    def __init__(self, in_dim=64):
        super(LLongRangAttention,self).__init__()
        self.input_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim*2, kernel_size=1)
        # self.mask_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.window_size = 8
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du_re = nn.Sequential(
                nn.Conv2d(in_dim, in_dim , 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim , in_dim, 3, stride=2, padding=2, bias=True),
                nn.ReLU(inplace=True),)
        self.conv_du_re2 = nn.Sequential(
                nn.Conv2d(in_dim, in_dim , 1, padding=0, bias=True),
                nn.ReLU(inplace=True),)
        # self.spatial = nn.Conv2d(2, 1, 3, stride=1, padding=(3-1) // 2)
        # self.upconv = nn.Sequential(nn.ConvTranspose2d(in_dim, in_dim, 3, stride=2, padding=2, output_padding=1))
        self.sigmoid = nn.Sigmoid()
        self.fuse = nn.Conv2d(in_dim*2, in_dim , 1, padding=0, bias=True)
        self.directW1_conv = nn.Conv2d(1, 1, kernel_size=(1, 9), stride=1, padding=(0, (9-1)//2), bias=True).cuda()
        self.directH1_conv = nn.Conv2d(1, 1, kernel_size=(9, 1), stride=1, padding=((9-1)//2, 0), bias=True).cuda()

        # self.directH_conv = nn.Conv2d(1, 1, kernel_size=(1, 9), stride=1, padding=(0, (9-1)//2), bias=True).cuda()
        # self.directW_conv = nn.Conv2d(1, 1, kernel_size=(9, 1), stride=1, padding=((9-1)//2, 0), bias=True).cuda()
        

    def gumbel_softmax(self, x, dim, tau):
        gumbels = torch.rand_like(x)
        while bool((gumbels == 0).sum() > 0):
            gumbels = torch.rand_like(x)

        gumbels = -(-gumbels.log()).log()
        gumbels = (x + gumbels) / tau
        x = gumbels.softmax(dim)

        return x

    def forward(self, res, x):
        # Residual mask generator
        # featuremap_visual(100*res, feature_title='Res_init')
        # featuremap_visual(x, feature_title='feat_org')
        v_max = self.conv_du_re(res)
        v_max = self.avg_pool(v_max)
        v_max = self.conv_du_re2(v_max)
        v_max = F.interpolate(v_max, (res.size(2), res.size(3)), mode='bilinear', align_corners=False)
        # v_max = self.upconv(v_max) 
        R_M = self.gumbel_softmax(v_max,1,1)
        # R_M = self.sigmoid(v_max)
        # featuremap_visual(x, feature_title='curr_feat')
        # featuremap_visual(R_M, feature_title='Res_Mask')
        # featuremap_visual(R_M_, feature_title='Res_Mask_')
        # print('max min', torch.max(R_M), torch.min(R_M))
        res_mask = R_M.masked_fill(R_M < 0.5, 0.0)
        res_mask = res_mask.masked_fill(res_mask >= 0.5, 1.0)
        # res_masksave = res_mask[0,31:32,:,:].squeeze(0).cpu().numpy() # cv2.cvtColor(res_mask[0,0,:,:].squeeze(0),cv2.COLOR_BGR2GRAY)
        # res_masksave = np.where(res_masksave[...,:] < 0.5, 0, 255)
        # res_masksave = np.array(res_masksave, dtype='uint8')
        # cv2.imwrite('/share3/home/zqiang/CVSR_train/viz_feat/res_masksave.png', res_masksave)
        
        # print('max min', torch.max(res_mask), torch.min(res_mask))
        # featuremap_visual(res_mask, feature_title='Res_Mask_org')
        # featuremap_visual(((1+x*res_mask)), feature_title='Res_Mask_1')
        res_mask_inv = 1.0 - res_mask
        # featuremap_visual(((10+x*res_mask_inv)/torch.max(x)), feature_title='Res_Mask_inv')
        x_ = self.input_conv(x)
        b, c, h, w = x.shape
        
        # long-range attentin
        # for row horizational
        q, v = rearrange(x_, 'b (qv c) h w -> qv (b h) w c', qv=2) 
        # q = q.view(b*h, w, c) # rearrange(q, '(b h c) w 1 -> (b h 1) w c')   #  (BxHxC) x 1 x W 
        # v = v.view(b*h, w, c) # rearrange(v, '(b h c) w 1 -> (b h 1) w c')   #  (BxHxC) x 1 x W
        res_mask = rearrange(res_mask, 'b c h w -> (b h) w c', b=b)   #  (BxH) x (1xW) x C
        sparse_q = res_mask*q
        sparse_q = self.directW1_conv(sparse_q.unsqueeze(1)).squeeze(1)
        # sparse_q = self.directW_conv(sparse_q.unsqueeze(1)).squeeze(1)
        # v = self.directW_conv(v.unsqueeze(1)).squeeze(1)
        v = self.directW1_conv(v.unsqueeze(1)).squeeze(1)
        atn = (sparse_q @ sparse_q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        v = (atn @ v)
        # for column verital
        q = rearrange(sparse_q, '(b h) w c -> (b w) h c', b=b)
        q = self.directH1_conv(q.unsqueeze(1)).squeeze(1)
        # q = self.directH_conv(q.unsqueeze(1)).squeeze(1)
        v = rearrange(v, '(b h) w c -> (b w) h c', b=b)
        atn = (q @ q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        long_out = (atn @ v)
        long_out = rearrange(long_out, '(b w) h c-> b c h w', b=b)

        # local attention (window attention)
        wsize = self.window_size
        q, v = rearrange(x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c', qv=2, dh=wsize, dw=wsize) 
        res_invb = rearrange(res_mask_inv, 'b (r c) (h dh) (w dw) -> r (b h w) (dh dw) c', r=1, dh=wsize, dw=wsize)  
        # res_invb = res_invb.view(-1,h,w) # squeeze(0) # .permute(1,2,0)
        res_invb = res_invb.squeeze(0)
        sparse_q = res_invb*q 
        atn = (sparse_q @ sparse_q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        loc_out= (atn @ v)
        loc_out = rearrange(loc_out, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)', h=h // wsize, w=w // wsize, dh=wsize, dw=wsize)
        # featuremap_visual(long_out, feature_title='long_out EGLA')
        # featuremap_visual(loc_out, feature_title='loc_out EGLA')
        out = self.fuse(torch.cat([long_out, loc_out],1))
        # featuremap_visual(out, feature_title='Out EGLA')

        return out + x





class LLongRangAttention_woLA(nn.Module):
    """ Long-Rang Attention Module"""
    def __init__(self, in_dim=64):
        super(LLongRangAttention_woLA,self).__init__()
        self.input_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim*2, kernel_size=1)
        self.window_size = 8
        # self.upconv = nn.Sequential(nn.ConvTranspose2d(in_dim, in_dim, 3, stride=2, padding=2, output_padding=1))
        self.sigmoid = nn.Sigmoid()
        # self.fuse = nn.Conv2d(in_dim*2, in_dim , 1, padding=0, bias=True)
        self.directW1_conv = nn.Conv2d(1, 1, kernel_size=(1, 9), stride=1, padding=(0, (9-1)//2), bias=True).cuda()
        self.directH1_conv = nn.Conv2d(1, 1, kernel_size=(9, 1), stride=1, padding=((9-1)//2, 0), bias=True).cuda()

        # self.directH_conv = nn.Conv2d(1, 1, kernel_size=(1, 9), stride=1, padding=(0, (9-1)//2), bias=True).cuda()
        # self.directW_conv = nn.Conv2d(1, 1, kernel_size=(9, 1), stride=1, padding=((9-1)//2, 0), bias=True).cuda()
        
    def gumbel_softmax(self, x, dim, tau):
        gumbels = torch.rand_like(x)
        while bool((gumbels == 0).sum() > 0):
            gumbels = torch.rand_like(x)

        gumbels = -(-gumbels.log()).log()
        gumbels = (x + gumbels) / tau
        x = gumbels.softmax(dim)

        return x

    def forward(self, x):
        # Residual mask generator
        x_ = self.input_conv(x)
        b, c, h, w = x.shape
        
        # long-range attentin
        # for row horizational
        q, v = rearrange(x_, 'b (qv c) h w -> qv (b h) w c', qv=2) 
        # q = q.view(b*h, w, c) # rearrange(q, '(b h c) w 1 -> (b h 1) w c')   #  (BxHxC) x 1 x W 
        # v = v.view(b*h, w, c) # rearrange(v, '(b h c) w 1 -> (b h 1) w c')   #  (BxHxC) x 1 x W
        sparse_q = rearrange(x_, 'b c h w -> (b h) w c', b=b)   #  (BxH) x (1xW) x C
        sparse_q = self.directW1_conv(sparse_q.unsqueeze(1)).squeeze(1)
        # sparse_q = self.directW_conv(sparse_q.unsqueeze(1)).squeeze(1)
        # v = self.directW_conv(v.unsqueeze(1)).squeeze(1)
        v = self.directW1_conv(v.unsqueeze(1)).squeeze(1)
        atn = (sparse_q @ sparse_q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        v = (atn @ v)
        # for column verital
        q = rearrange(sparse_q, '(b h) w c -> (b w) h c', b=b)
        q = self.directH1_conv(q.unsqueeze(1)).squeeze(1)
        # q = self.directH_conv(q.unsqueeze(1)).squeeze(1)
        v = rearrange(v, '(b h) w c -> (b w) h c', b=b)
        atn = (q @ q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        long_out = (atn @ v)
        long_out = rearrange(long_out, '(b w) h c-> b c h w', b=b)

        # local attention (window attention)
        # wsize = self.window_size
        # q, v = rearrange(x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c', qv=2, dh=wsize, dw=wsize) 
        # res_invb = rearrange(res_mask_inv, 'b (r c) (h dh) (w dw) -> r (b h w) (dh dw) c', r=1, dh=wsize, dw=wsize)  
        # # res_invb = res_invb.view(-1,h,w) # squeeze(0) # .permute(1,2,0)
        # res_invb = res_invb.squeeze(0)
        # sparse_q = res_invb*q 
        # atn = (sparse_q @ sparse_q.transpose(-2, -1))
        # atn = atn.softmax(dim=-1)
        # loc_out= (atn @ v)
        # loc_out = rearrange(loc_out, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)', h=h // wsize, w=w // wsize, dh=wsize, dw=wsize)
        # out = self.fuse(torch.cat([long_out, loc_out],1))
        
        out = long_out

        return out + x





class LLongRangAttention_woGA(nn.Module):
    """ Long-Rang Attention Module"""
    def __init__(self, in_dim=64):
        super(LLongRangAttention_woGA,self).__init__()
        self.input_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim*2, kernel_size=1)
        # self.mask_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.window_size = 8
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv_du_re = nn.Sequential(
        #         nn.Conv2d(in_dim, in_dim , 1, padding=0, bias=True),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(in_dim , in_dim, 3, stride=2, padding=2, bias=True),
        #         nn.ReLU(inplace=True),)
        # self.conv_du_re2 = nn.Sequential(
        #         nn.Conv2d(in_dim, in_dim , 1, padding=0, bias=True),
        #         nn.ReLU(inplace=True),)
        # self.avg_pool = nn.AvgPool2d((2, 2), stride=(2, 2)) # nn.AdaptiveAvgPool2d(1)
        # self.conv_du_re = nn.Sequential(
        #         nn.Conv2d(in_dim, in_dim , 3, padding=1, bias=True),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(in_dim , in_dim, 3, stride=1, padding=1, bias=True),
        #         nn.ReLU(inplace=True),
        #         # nn.AvgPool2d(1, stride=1, padding=0),
        #         nn.Conv2d(in_dim, in_dim , 3, padding=1, bias=True),
        #         # nn.ReLU(inplace=True),
        #         # nn.ConvTranspose2d(in_dim, in_dim, 3, stride=1, padding=1, output_padding=0)
        #         # nn.ConvTranspose2d(in_dim, in_dim, 3, stride=2, padding=2, output_padding=1)
        #         )
        
        # self.conv_du_re2 = nn.Sequential(
        #         nn.Conv2d(in_dim, in_dim , 1, padding=0, bias=True),
        #         nn.ReLU(inplace=True),
        #         nn.ConvTranspose2d(in_dim, in_dim, 3, stride=2, padding=2, output_padding=1),
        #         nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #         nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True),
        #         nn.LeakyReLU(negative_slope=0.1, inplace=True),)

        # self.spatial = nn.Conv2d(2, 1, 3, stride=1, padding=(3-1) // 2)
        # self.upconv = nn.Sequential(nn.ConvTranspose2d(in_dim, in_dim, 3, stride=2, padding=2, output_padding=1))
        self.sigmoid = nn.Sigmoid()
        # self.fuse = nn.Conv2d(in_dim*2, in_dim , 1, padding=0, bias=True)
        # self.directW1_conv = nn.Conv2d(1, 1, kernel_size=(1, 9), stride=1, padding=(0, (9-1)//2), bias=True).cuda()
        # self.directH1_conv = nn.Conv2d(1, 1, kernel_size=(9, 1), stride=1, padding=((9-1)//2, 0), bias=True).cuda()

        # self.directH_conv = nn.Conv2d(1, 1, kernel_size=(1, 9), stride=1, padding=(0, (9-1)//2), bias=True).cuda()
        # self.directW_conv = nn.Conv2d(1, 1, kernel_size=(9, 1), stride=1, padding=((9-1)//2, 0), bias=True).cuda()
        

    def gumbel_softmax(self, x, dim, tau):
        gumbels = torch.rand_like(x)
        while bool((gumbels == 0).sum() > 0):
            gumbels = torch.rand_like(x)

        gumbels = -(-gumbels.log()).log()
        gumbels = (x + gumbels) / tau
        x = gumbels.softmax(dim)

        return x

    def forward(self, res, x):
        # Residual mask generator
        # featuremap_visual(res, feature_title='Res_init')
        # v_max = self.conv_du_re(res)
        # v_max = self.avg_pool(v_max)
        # v_max = self.conv_du_re2(v_max)
        # v_max = F.interpolate(v_max, (res.size(2), res.size(3)), mode='bilinear', align_corners=False)
        # # v_max = self.upconv(v_max) 
        # R_M = self.gumbel_softmax(v_max,1,1)
        # R_M = self.sigmoid(v_max)
        # featuremap_visual(x, feature_title='curr_feat')
        # featuremap_visual(R_M, feature_title='Res_Mask')
        # featuremap_visual(R_M_, feature_title='Res_Mask_')
        # print('max min', torch.max(R_M), torch.min(R_M))
        # res_mask = R_M.masked_fill(R_M < 0.5, 0.0)
        # res_mask = res_mask.masked_fill(res_mask >= 0.5, 1.0)
        # res_masksave = res_mask[0,31:32,:,:].squeeze(0).cpu().numpy() # cv2.cvtColor(res_mask[0,0,:,:].squeeze(0),cv2.COLOR_BGR2GRAY)
        # res_masksave = np.where(res_masksave[...,:] < 0.5, 0, 255)
        # res_masksave = np.array(res_masksave, dtype='uint8')
        # cv2.imwrite('/share3/home/zqiang/CVSR_train/viz_feat/res_masksave.png', res_masksave)
        
        # print('max min', torch.max(res_mask), torch.min(res_mask))
        # featuremap_visual(res_mask, feature_title='Res_Mask_org')
        # featuremap_visual(x*res_mask, feature_title='Res_Mask_1')
        # res_mask_inv = 1.0 - res_mask
        # featuremap_visual(x*res_mask_inv, feature_title='Res_Mask_inv')
        x_ = self.input_conv(x)
        b, c, h, w = x.shape
        
        # long-range attentin
        # for row horizational
        # q, v = rearrange(x_, 'b (qv c) h w -> qv (b h) w c', qv=2) 
        # # q = q.view(b*h, w, c) # rearrange(q, '(b h c) w 1 -> (b h 1) w c')   #  (BxHxC) x 1 x W 
        # # v = v.view(b*h, w, c) # rearrange(v, '(b h c) w 1 -> (b h 1) w c')   #  (BxHxC) x 1 x W
        # res_mask = rearrange(res_mask, 'b c h w -> (b h) w c', b=b)   #  (BxH) x (1xW) x C
        # sparse_q = res_mask*q
        # sparse_q = self.directW1_conv(sparse_q.unsqueeze(1)).squeeze(1)
        # # sparse_q = self.directW_conv(sparse_q.unsqueeze(1)).squeeze(1)
        # # v = self.directW_conv(v.unsqueeze(1)).squeeze(1)
        # v = self.directW1_conv(v.unsqueeze(1)).squeeze(1)
        # atn = (sparse_q @ sparse_q.transpose(-2, -1))
        # atn = atn.softmax(dim=-1)
        # v = (atn @ v)
        # # for column verital
        # q = rearrange(sparse_q, '(b h) w c -> (b w) h c', b=b)
        # q = self.directH1_conv(q.unsqueeze(1)).squeeze(1)
        # # q = self.directH_conv(q.unsqueeze(1)).squeeze(1)
        # v = rearrange(v, '(b h) w c -> (b w) h c', b=b)
        # atn = (q @ q.transpose(-2, -1))
        # atn = atn.softmax(dim=-1)
        # long_out = (atn @ v)
        # long_out = rearrange(long_out, '(b w) h c-> b c h w', b=b)

        # local attention (window attention)
        wsize = self.window_size
        q, v = rearrange(x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c', qv=2, dh=wsize, dw=wsize) 
        # res_invb = rearrange(x_, 'b (r c) (h dh) (w dw) -> r (b h w) (dh dw) c', r=1, dh=wsize, dw=wsize)  
        # res_invb = res_invb.view(-1,h,w) # squeeze(0) # .permute(1,2,0)
        # res_invb = res_invb.squeeze(0)
        # sparse_q = res_invb*q 
        atn = (q @ q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        loc_out= (atn @ v)
        loc_out = rearrange(loc_out, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)', h=h // wsize, w=w // wsize, dh=wsize, dw=wsize)
        # out = self.fuse(torch.cat([long_out, loc_out],1))
        out = loc_out

        return out + x






class LLongRangAttention_1(nn.Module):
    """ Long-Rang Attention Module"""
    def __init__(self, in_dim=64):
        super(LLongRangAttention_1,self).__init__()
        self.input_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim*2, kernel_size=1)
        # self.mask_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.window_size = 8
        self.avg_pool = nn.AvgPool2d((2, 2), stride=(2, 2)) # nn.AdaptiveAvgPool2d(1)
        self.conv_du_re = nn.Sequential(
                nn.Conv2d(in_dim, in_dim , 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim , in_dim, 3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                # nn.AvgPool2d(1, stride=1, padding=0),
                nn.Conv2d(in_dim, in_dim , 3, padding=1, bias=True),
                # nn.ReLU(inplace=True),
                # nn.ConvTranspose2d(in_dim, in_dim, 3, stride=1, padding=1, output_padding=0)
                # nn.ConvTranspose2d(in_dim, in_dim, 3, stride=2, padding=2, output_padding=1)
                )
        
        # self.conv_du_re2 = nn.Sequential(
        #         nn.Conv2d(in_dim, in_dim , 1, padding=0, bias=True),
        #         nn.ReLU(inplace=True),
        #         nn.ConvTranspose2d(in_dim, in_dim, 3, stride=2, padding=2, output_padding=1),
        #         nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #         nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True),
        #         nn.LeakyReLU(negative_slope=0.1, inplace=True),)

        # self.spatial = nn.Conv2d(2, 1, 3, stride=1, padding=(3-1) // 2)
        # self.upconv = nn.Sequential(nn.ConvTranspose2d(in_dim, in_dim, 3, stride=2, padding=2, output_padding=1))
        self.sigmoid = nn.Sigmoid()
        self.fuse = nn.Conv2d(in_dim*2, in_dim , 1, padding=0, bias=True)
        self.directH_conv = nn.Conv2d(1, 1, kernel_size=(1, 9), stride=1, padding=(0, (9-1)//2), bias=True).cuda()
        self.directW_conv = nn.Conv2d(1, 1, kernel_size=(9, 1), stride=1, padding=((9-1)//2, 0), bias=True).cuda()
        

    def gumbel_softmax(self, x, dim, tau):
        gumbels = torch.rand_like(x)
        while bool((gumbels == 0).sum() > 0):
            gumbels = torch.rand_like(x)

        gumbels = -(-gumbels.log()).log()
        gumbels = (x + gumbels) / tau
        x = gumbels.softmax(dim)

        return x

    def forward(self, res, x):
        # Residual mask generator
        # featuremap_visual(res, feature_title='Res_init')
        v_max = self.conv_du_re(res)
        # v_max = self.avg_pool(v_max)
        # v_max = self.conv_du_re2(v_max)
        # v_max = F.interpolate(v_max, (res.size(2), res.size(3)), mode='bilinear', align_corners=False)
        # v_max = self.upconv(v_max) 
        # R_M = self.gumbel_softmax(v_max,1,1)
        R_M = self.sigmoid(v_max)
        # featuremap_visual(x, feature_title='curr_feat')
        # featuremap_visual(R_M, feature_title='Res_Mask')
        # featuremap_visual(R_M_, feature_title='Res_Mask_')
        # print('max min', torch.max(R_M), torch.min(R_M))
        res_mask = R_M.masked_fill(R_M < 0.5, 0.0)
        res_mask = res_mask.masked_fill(res_mask >= 0.5, 1.0)
        # res_masksave = res_mask[0,31:32,:,:].squeeze(0).cpu().numpy() # cv2.cvtColor(res_mask[0,0,:,:].squeeze(0),cv2.COLOR_BGR2GRAY)
        # res_masksave = np.where(res_masksave[...,:] < 0.5, 0, 255)
        # res_masksave = np.array(res_masksave, dtype='uint8')
        # cv2.imwrite('/share3/home/zqiang/CVSR_train/viz_feat/res_masksave.png', res_masksave)
        
        # print('max min', torch.max(res_mask), torch.min(res_mask))
        # featuremap_visual(res_mask, feature_title='Res_Mask_org')
        # featuremap_visual(x*res_mask, feature_title='Res_Mask_1')
        res_mask_inv = 1.0 - res_mask
        # featuremap_visual(x*res_mask_inv, feature_title='Res_Mask_inv')
        x_ = self.input_conv(x)
        b, c, h, w = x.shape
        
        # long-range attentin
        # for row
        q, v = rearrange(x_, 'b (qv c) h w -> qv (b h) w c', qv=2) 
        # q = q.view(b*h, w, c) # rearrange(q, '(b h c) w 1 -> (b h 1) w c')   #  (BxHxC) x 1 x W 
        # v = v.view(b*h, w, c) # rearrange(v, '(b h c) w 1 -> (b h 1) w c')   #  (BxHxC) x 1 x W
        # q, v = rearrange(x_, 'b (qv c) h w -> qv (b h) w c', qv=2)   #  (BxH) x (1xW) x C
        res_mask = rearrange(res_mask, 'b c h w -> (b h) w c', b=b)   #  (BxH) x (1xW) x C
        # res_mask = res_mask.view(h,w,-1).squeeze(0) # .permute(1,2,0)
        sparse_q = res_mask*q
        sparse_q = self.directW_conv(sparse_q.unsqueeze(1)).squeeze(1)
        atn = (sparse_q @ sparse_q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        v = (atn @ v)
        # for column
        q = rearrange(sparse_q, '(b h) w c -> (b w) h c', b=b)
        q = self.directH_conv(q.unsqueeze(1)).squeeze(1)
        v = rearrange(v, '(b h) w c -> (b w) h c', b=b)
        atn = (q @ q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        long_out = (atn @ v)
        long_out = rearrange(long_out, '(b w) h c-> b c h w', b=b)

        # local attention (window attention)
        wsize = self.window_size
        q, v = rearrange(x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c', qv=2, dh=wsize, dw=wsize) 
        res_invb = rearrange(res_mask_inv, 'b (r c) (h dh) (w dw) -> r (b h w) (dh dw) c', r=1, dh=wsize, dw=wsize)  
        # res_invb = res_invb.view(-1,h,w) # squeeze(0) # .permute(1,2,0)
        res_invb = res_invb.squeeze(0)
        sparse_q = res_invb*q 
        atn = (sparse_q @ sparse_q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        loc_out= (atn @ v)
        loc_out = rearrange(loc_out, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)', h=h // wsize, w=w // wsize, dh=wsize, dw=wsize)
        out = self.fuse(torch.cat([long_out, loc_out],1))

        return out + x



class LongRangAttention(nn.Module):
    """ Long-Rang Attention Module"""
    def __init__(self, in_dim=64):
        super(LongRangAttention,self).__init__()
        self.input_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim*2, kernel_size=1)
        # self.mask_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.window_size = 8
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du_re = nn.Sequential(
                nn.Conv2d(in_dim, in_dim , 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim , in_dim, 3, stride=2, padding=2, bias=True),
                nn.ReLU(inplace=True),)
        self.conv_du_re2 = nn.Sequential(
                nn.Conv2d(in_dim, in_dim , 1, padding=0, bias=True),
                nn.ReLU(inplace=True),)
        # self.spatial = nn.Conv2d(2, 1, 3, stride=1, padding=(3-1) // 2)
        self.sigmoid = nn.Sigmoid()
        self.fuse = nn.Conv2d(in_dim*2, in_dim , 1, padding=0, bias=True)
        self.short_conv = nn.Sequential(
                        nn.Conv2d(num_in, num_out, kernel_size=kernel_size, stride=stride,
                                padding=kernel_size//2, num_groups=1),
                        nn.Conv2d(num_out, num_out, kernel_size=(1, 5), stride=1,
                                padding=(0, 2), num_groups=num_out),
                        nn.Conv2d(num_out, num_out, kernel_size=(5, 1), stride=1,
                                padding=(2, 0), num_groups=num_out),)

    def gumbel_softmax(self, x, dim, tau):
        gumbels = torch.rand_like(x)
        while bool((gumbels == 0).sum() > 0):
            gumbels = torch.rand_like(x)

        gumbels = -(-gumbels.log()).log()
        gumbels = (x + gumbels) / tau
        x = gumbels.softmax(dim)

        return x

    def forward(self, res, x):
        # Residual mask generator
        r_f = self.conv_du_re(res)
        v_max = self.avg_pool(r_f)
        v_max = self.conv_du_re2(v_max)
        v_max = F.interpolate(v_max, (res.size(2), res.size(3)), mode='bilinear', align_corners=False)
        R_M = self.gumbel_softmax(v_max,1,1)
        res_mask = res.masked_fill(R_M != 0, 1.0)
        res_mask_inv = 1 - res_mask
        x_ = self.input_conv(x)
        b, c, h, w = x.shape
        # print('x res_mask', res_mask.shape, x.shape)
        # long-range attentin
        # for row
        q, v = rearrange(x_, 'b (qv c) h w -> qv (b h) w c', qv=2)   #  (BxH) x (1xW) x C
        res_mask = rearrange(res_mask, 'b c h w -> (b h) w c', b=b)   #  (BxH) x (1xW) x C
        # res_mask = res_mask.view(h,w,-1).squeeze(0) # .permute(1,2,0)
        # print('res_mask', res_mask.shape, q.shape)
        sparse_q = res_mask*q
        atn = (sparse_q @ sparse_q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        v = (atn @ v)
        # for column
        q = rearrange(sparse_q, '(b h) w c -> (b w) h c', b=b)
        v = rearrange(v, '(b h) w c -> (b w) h c', b=b)
        atn = (q @ q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        long_out = (atn @ v)
        long_out = rearrange(long_out, '(b w) h c-> b c h w', b=b)

        # local attention (window attention)
        wsize = self.window_size
        q, v = rearrange(x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c', qv=2, dh=wsize, dw=wsize) 
        res_invb = rearrange(res_mask_inv, 'b (r c) (h dh) (w dw) -> r (b h w) (dh dw) c', r=1, dh=wsize, dw=wsize)  
        # res_invb = res_invb.view(-1,h,w) # squeeze(0) # .permute(1,2,0)
        res_invb = res_invb.squeeze(0)
        # print('q',q.shape, sparse_q.shape, res_invb.shape) 
        sparse_q = res_invb*q 
        atn = (sparse_q @ sparse_q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        loc_out= (atn @ v)
        loc_out = rearrange(loc_out, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)', h=h // wsize, w=w // wsize, dh=wsize, dw=wsize)
        out = self.fuse(torch.cat([long_out, loc_out],1))

        return out + x


class RSNLAttention(nn.Module):
    def __init__(self, channel=64, reduction=2, ksize=1, scale=3, stride=1, softmax_scale=10, average=True, res_scale=1):
        super(RSNLAttention, self).__init__()
        self.res_scale = res_scale
        self.conv_match1 = nn.Sequential(nn.Conv2d(channel, channel, 1),
                                        nn.PReLU(),)
        # self.conv_match2 = nn.Sequential(nn.Conv2d(channel, channel, 1),
        #                                 nn.PReLU(),)
        self.conv_assembly = nn.Sequential(nn.Conv2d(channel, channel, 1),
                                        nn.PReLU(),)
        self.conv_du_re = nn.Sequential( nn.Conv2d(channel, channel , 1, padding=0, bias=True),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(channel , channel, 3, stride=2, padding=2, bias=True),
                                        nn.ReLU(inplace=True),)
        self.conv_du_re2 = nn.Sequential(nn.Conv2d(channel, channel , 1, padding=0, bias=True),
                                        nn.ReLU(inplace=True),)
    
    def gumbel_softmax(self, x, dim, tau):
        gumbels = torch.rand_like(x)
        while bool((gumbels == 0).sum() > 0):
            gumbels = torch.rand_like(x)

        gumbels = -(-gumbels.log()).log()
        gumbels = (x + gumbels) / tau
        x = gumbels.softmax(dim)

        return x
    
    def forward(self, res, x_com):
        # Residual mask generator
        r_f = self.conv_du_re(res)
        v_max = F.max_pool2d(r_f, kernel_size=3, stride=1)
        v_max = self.conv_du_re2(v_max)
        v_max = F.interpolate(v_max, (res.size(2), res.size(3)), mode='bilinear', align_corners=False)
        R_M = self.gumbel_softmax(v_max,1,1)

        x_embed_1 = self.conv_match1(x_com)
        # x_embed_2 = self.conv_match2(x_com)
        x_assembly = self.conv_assembly(x_com)

        N,C,H,W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0,2,3,1).view((N,H*W,C)) 
        # x_embed_2 = x_embed_2.view(N,C,H*W)
        x_embed_2 = R_M.view(N,C,H*W)
        score = torch.matmul(x_embed_1, x_embed_2)
        score = torch.softmax(score, dim=2)
        # R_SM = R_M.view(C*N,H*W).to_sparse().requires_grad_()#  .view(C*N,H*W)
        # score = torch.sparse.mm(R_SM, x_embed_1).view(N,H*W)

        x_assembly = x_assembly.view(N,-1,H*W).permute(0,2,1)
        x_final = torch.matmul(score, x_assembly)
        out = x_final.permute(0,2,1).view(N,-1,H,W)+self.res_scale*x_com

        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out) # broadcasting
        return x * scale


## Residual Map gudied Attention Block 
class RDAB_S(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(RDAB_S, self).__init__()
        # global average pooling: feature --> point
        # feature channel downscale and upscale --> channel weight
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        # c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        self.conv_du_re = nn.Sequential(
                nn.Conv2d(channel, channel , 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel , channel, 3, stride=2, padding=2, bias=True),
                nn.ReLU(inplace=True),)
        self.conv_du_re2 = nn.Sequential(
                nn.Conv2d(channel, channel , 1, padding=0, bias=True),
                nn.ReLU(inplace=True),)
        self.conv_du_am = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid())
        self.conv_dc = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),)

        self.sigmoid = nn.Sigmoid()
        self.conv_df = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),)

    def gumbel_softmax(self, x, dim, tau):
        gumbels = torch.rand_like(x)
        while bool((gumbels == 0).sum() > 0):
            gumbels = torch.rand_like(x)

        gumbels = -(-gumbels.log()).log()
        gumbels = (x + gumbels) / tau
        x = gumbels.softmax(dim)

        return x

    def forward(self, res, x_c):
        # Residual mask generator
        r_f = self.conv_du_re(res)
        v_max = F.max_pool2d(r_f, kernel_size=3, stride=1)
        v_max = self.conv_du_re2(v_max)
        v_max = F.interpolate(v_max, (res.size(2), res.size(3)), mode='bilinear', align_corners=False)
        R_M = self.gumbel_softmax(v_max,1,1)

        # Attention mask generator
        att_M = self.conv_du_am(self.avg_pool(x_c))
        x_f = self.conv_dc(x_c)
        out = x_f * (R_M + att_M)
        out = self.conv_df(out)

        return out



## Residual Map gudied Attention Block 
class RDAB(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(RDAB, self).__init__()
        # global average pooling: feature --> point
        # feature channel downscale and upscale --> channel weight
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du_re = nn.Sequential(
                nn.Conv2d(channel, channel , 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel , channel, 3, stride=2, padding=2, bias=True),
                nn.ReLU(inplace=True),)
        self.conv_du_re2 = nn.Sequential(
                nn.Conv2d(channel, channel , 1, padding=0, bias=True),
                nn.ReLU(inplace=True),)
        self.conv_dc = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),)
        self.spatial = nn.Conv2d(2, 1, 3, stride=1, padding=(3-1) // 2)
        self.sigmoid = nn.Sigmoid()
        self.conv_df = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),)

    def gumbel_softmax(self, x, dim, tau):
        gumbels = torch.rand_like(x)
        while bool((gumbels == 0).sum() > 0):
            gumbels = torch.rand_like(x)

        gumbels = -(-gumbels.log()).log()
        gumbels = (x + gumbels) / tau
        x = gumbels.softmax(dim)

        return x

    def forward(self, res, x_c):
        # Residual mask generator
        r_f = self.conv_du_re(res)
        # v_max = F.max_pool2d(r_f, kernel_size=3, stride=1)
        v_max = self.avg_pool(r_f)
        v_max = self.conv_du_re2(v_max)
        v_max = F.interpolate(v_max, (res.size(2), res.size(3)), mode='bilinear', align_corners=False)
        R_M = self.gumbel_softmax(v_max,1,1)

        # Attention mask generator
        x_w = torch.cat((torch.max(x_c,1)[0].unsqueeze(1), torch.mean(x_c,1).unsqueeze(1)), dim=1)
        att_M = self.sigmoid(self.spatial(x_w))
        x_f = self.conv_dc(x_c)
        out = x_f * (R_M + att_M)
        out = self.conv_df(out)

        return out




## Residual Map gudied Attention  Block 
class RDAB_(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(RDAB_, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

        self.conv_dc = nn.Sequential(
                nn.Conv2d(channel, channel*4, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel*4, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_df = nn.Sequential(
                nn.Conv2d(2*channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),)

    def forward(self, res, x, x_c):
        y = self.avg_pool(res)
        y = self.conv_du(y)
        x_c = self.conv_dc(x_c)
        out = x_c * y + x
        out = self.conv_df(torch.cat([out, x],1))

        return out



## Residual Map gudied Position Attention Block 
class RPAB(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(RPAB, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.PAM = PAM(in_dim=channel)
        self.conv_dc = nn.Sequential(
                nn.Conv2d(channel, channel*4, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel*4, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_df = nn.Sequential(
                nn.Conv2d(2*channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),)

    def forward(self, res, x, x_c):
        y = self.PAM(res, x)
        # y = self.avg_pool(res)
        # y = self.conv_du(y)
        # x_c = self.conv_dc(x_c)
        # out = x_c * y + x
        out = self.conv_df(torch.cat([y, x_c],1))

        return out


##########################################################################
##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3,reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)
        # self.conv_reduce = nn.Conv2d(4*in_channels, in_channels, kernel_size=1, stride=1,bias=bias)

    def forward(self, inp_feat1,inp_feat2):
        
        inp_feats = torch.cat([inp_feat1,inp_feat2], dim=1)
        batch_size = inp_feat1.shape[0]
        n_feats = inp_feat1.shape[1]
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
      
        return feats_V



class ResBlock_3d(nn.Module):
    def __init__(self, nf):
        super(ResBlock_3d, self).__init__()
        # self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        
        self.dcn0 = nn.Conv3d(1, nf, kernel_size=3, stride=1, padding=1)
        self.dcn1 = nn.Conv3d(nf, 1, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        x0 = x.unsqueeze(1)
        x1 = self.lrelu(self.dcn0(x0))
        # print('[x1]',x1.shape)
        out = self.dcn1(x1) + x0
        # print('[out]',out.shape)
        out = out.view(m_batchsize, -1, height, width)
        return out



class Calib_ResBlock_3d(nn.Module):
    def __init__(self, nf):
        super(Calib_ResBlock_3d, self).__init__()
        # self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        
        self.dcn0 = nn.Conv3d(4, nf, kernel_size=3, stride=1, padding=1)
        self.dcn1 = nn.Conv3d(nf, 4, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x_):
        b, c, height, width = x_.size()
        # x0 = x.unsqueeze(1)
        p = 4
        # print('[x_]',x_.shape)
        x = rearrange(x_, 'b c (h h1) (w w2) -> b h1 w2 c (h w) ', h1=p, w2=p)
        # x0 = x.unsqueeze(1)
        x0 = x
        # print('[x]',x.shape)
        x1 = self.lrelu(self.dcn0(x0))
        # print('[x1]',x1.shape)
        out = self.dcn1(x1) + x0
        # print('[out]',out.shape)
        out = out.view(b, -1, height, width)
        out = out + x_
        return out



class ContextBlock(nn.Module):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()

        # self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, padding=0, groups=1, bias=False)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, groups=1, bias=False),
            # nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, groups=1, bias=False)
            # nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )
        # self.conv_reduce = nn.Conv2d(n_feat, 8, kernel_size=1, padding=0, groups=1, bias=False)   #  8
        # self.conv_increase= nn.Conv2d(8, n_feat, kernel_size=1, padding=0, groups=1, bias=False)   #  8

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        # print('[height]',height, width)
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        x0 = x
        context = self.modeling(x)
        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term
        out = x

        return out



def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output



class MVDeformableAlignment(ModulatedDeformConvPack):
    """deformable alignment module.
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(MVDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1):
        warped_feat = flow_warp(extra_feat, flow_1.permute(0, 2, 3, 1))
        extra_feat = torch.cat([warped_feat, flow_1, flow_1], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        # offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        # offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        # offset_2 = offset_2 + flow_1.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        # offset = torch.cat([offset_1, offset_2], dim=1)
        offset = offset + flow_1.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        # mask
        mask = nn.Sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                             self.dilation, mask)



class MVSelfAttDeformableAlignment(ModulatedDeformConvPack):
    """deformable alignment module.
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(MVSelfAttDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),
        )
        dim = 64
        self.num_heads = 8
        bias = False

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

        # self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.init_offset()

    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, pred_feat, flow_1):
        warped_feat = flow_warp(extra_feat, flow_1.permute(0, 2, 3, 1))
        # extra_feat = torch.cat([warped_feat, flow_1, flow_1], dim=1)
        # print('[warped_feat]',x.shape, warped_feat.shape,pred_feat.shape)

        ####
        b, c, h, w = warped_feat.shape
        # qkv = self.qkv_dwconv(self.qkv(warped_feat))
        # q, k, v = qkv.chunk(3, dim=1)
        q = warped_feat
        k = extra_feat
        v = pred_feat
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        ###

        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        # offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        # offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        # offset_2 = offset_2 + flow_1.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        # offset = torch.cat([offset_1, offset_2], dim=1)
        offset = offset + flow_1.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        # mask
        mask = nn.Sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                             self.dilation, mask)



class MVDualAttAlignment(ModulatedDeformConvPack):
    """deformable alignment module.
    """
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        super(MVDualAttAlignment, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),
        )
        dim = 64
        self.num_heads = 8
        bias = False
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(self.out_channels, self.out_channels // 16, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.out_channels // 16, self.out_channels, 1, padding=0, bias=True),
                nn.Sigmoid())

        self.fusion_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.sigmoid = nn.Sigmoid()

        self.init_offset()

    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, pred_feat, flow_1):
        warped_feat = flow_warp(extra_feat, flow_1.permute(0, 2, 3, 1))
        fused_feat = self.fusion_out(torch.cat([warped_feat,pred_feat],dim=1))

        #### MSA 1
        b, c, h, w = warped_feat.shape
        q = x
        k = fused_feat 
        v = warped_feat * self.conv_du(self.avg_pool(warped_feat))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_1 = self.project_out(out)

        #### MSA 2
        b, c, h, w = pred_feat.shape
        q = x
        k = fused_feat 
        v = pred_feat * self.conv_du(self.avg_pool(pred_feat))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_2 = self.project_out(out)

        out_1 = self.conv_offset(out_1)
        out_2 = self.conv_offset(out_2)
        o1_1, o2_1, mask_1 = torch.chunk(out_1, 3, dim=1)
        o1_2, o2_2, mask_2 = torch.chunk(out_2, 3, dim=1)

        # offset
        offset_1 = self.max_residue_magnitude * torch.tanh(torch.cat((o1_1, o2_1), dim=1))
        offset_2 = self.max_residue_magnitude * torch.tanh(torch.cat((o1_2, o2_2), dim=1))
        offset = offset_1 + offset_2 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)

        # mask
        mask = self.sigmoid(mask_1 + mask_2)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask)



'''
class DualAttAlignment(ModulatedDeformConvPack):
    """deformable alignment module.
    """
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        super(DualAttAlignment, self).__init__(*args, **kwargs)

        dim = 64
        self.num_heads = 4
        bias = False
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(self.out_channels, self.out_channels // 16, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.out_channels // 16, self.out_channels, 1, padding=0, bias=True),
                nn.Sigmoid())

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.fusion_in = nn.Sequential(
                nn.Conv2d(dim*2, dim, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 1, padding=0, bias=True))
        self.fusion_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)

    def forward(self, x, extra_feat, pred_feat, flow_1):
        warped_feat = flow_warp(extra_feat, flow_1.permute(0, 2, 3, 1))
        fused_feat = self.fusion_out(torch.cat([warped_feat,pred_feat],dim=1))

        #### MSA 1
        b, c, h, w = warped_feat.shape
        q = x
        k = fused_feat 
        v = warped_feat * self.conv_du(self.avg_pool(warped_feat))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_1 = self.project_out(out)
        ####

        #### MSA 2
        b, c, h, w = pred_feat.shape
        q = x
        k = fused_feat 
        v = pred_feat * self.conv_du(self.avg_pool(pred_feat))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_2 = self.project_out(out)
        ####
        out = self.fusion_out(torch.cat([out_1,out_2],dim=1)) + x 

        return out

'''



class DualAttAlignment(nn.Module):
    """deformable alignment module.
    """
    def __init__(self,):
        super().__init__()
        dim = 64
        self.out_channels = dim
        self.num_heads = 4
        bias = False
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                        nn.Conv2d(self.out_channels, self.out_channels // 16, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(self.out_channels // 16, self.out_channels, 1, padding=0, bias=True),
                        nn.Sigmoid())

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.fusion_in = nn.Sequential(
                            nn.Conv2d(dim*2, dim, 1, padding=0, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(dim, dim, 1, padding=0, bias=True))
        self.fusion_out = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias),
                            nn.ReLU(inplace=True), )
        self.CALayer = CALayer(dim)
        self.ResidualBlock = ResidualBlock_noBN(dim)
        self.ResidualBlock1 = ResidualBlock_noBN(dim)

    def forward(self, x, extra_feat, pred_feat, flow_1):
        warped_feat = flow_warp(extra_feat, flow_1.permute(0, 2, 3, 1))
        fused_feat = self.fusion_out(torch.cat([warped_feat,pred_feat],dim=1))

        #### MSA 1
        b, c, h, w = warped_feat.shape
        q = x
        k = fused_feat 
        v = warped_feat * self.conv_du(self.avg_pool(warped_feat))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_1 = self.project_out(out)
        ####

        #### MSA 2
        b, c, h, w = pred_feat.shape
        q = x
        k = fused_feat 
        v = pred_feat * self.conv_du(self.avg_pool(pred_feat))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_2 = self.project_out(out)
        #### Aggregation
        out = self.fusion_out(torch.cat([out_1 + out_2, x],dim=1)) 
        # out = self.fusion_out(torch.cat([out_1,out_2],dim=1)) 
        out = self.ResidualBlock1(self.ResidualBlock(self.CALayer(out))) + x 

        return out





class DualAttAlignment_woPd(nn.Module):
    """deformable alignment module.
    """
    def __init__(self,):
        super().__init__()
        dim = 64
        self.out_channels = dim
        self.num_heads = 4
        bias = False
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                        nn.Conv2d(self.out_channels, self.out_channels // 16, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(self.out_channels // 16, self.out_channels, 1, padding=0, bias=True),
                        nn.Sigmoid())

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.fusion_in = nn.Sequential(
                            nn.Conv2d(dim*2, dim, 1, padding=0, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(dim, dim, 1, padding=0, bias=True))
        self.fusion_out = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias),
                            nn.ReLU(inplace=True), )
        self.CALayer = CALayer(dim)
        self.ResidualBlock = ResidualBlock_noBN(dim)
        self.ResidualBlock1 = ResidualBlock_noBN(dim)

    def forward(self, x, extra_feat, flow_1):
        warped_feat = flow_warp(extra_feat, flow_1.permute(0, 2, 3, 1))
        # fused_feat = self.fusion_out(torch.cat([warped_feat,pred_feat],dim=1))

        #### MSA 1
        b, c, h, w = warped_feat.shape
        q = x
        k = warped_feat 
        v = warped_feat * self.conv_du(self.avg_pool(warped_feat))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_1 = self.project_out(out)
        ####

        #### MSA 2
        # b, c, h, w = pred_feat.shape
        # q = x
        # k = fused_feat 
        # v = pred_feat * self.conv_du(self.avg_pool(pred_feat))
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)
        # attn = (q @ k.transpose(-2, -1)) * self.temperature
        # attn = attn.softmax(dim=-1)
        # out = (attn @ v)
        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # out_2 = self.project_out(out)
        #### Aggregation
        out = self.fusion_out(torch.cat([out_1, x],dim=1)) 
        # out = self.fusion_out(torch.cat([out_1,out_2],dim=1)) 
        out = self.ResidualBlock1(self.ResidualBlock(self.CALayer(out))) + x 

        return out





class DualAttAlignment_woMV(nn.Module):
    """deformable alignment module.
    """
    def __init__(self,):
        super().__init__()
        dim = 64
        self.out_channels = dim
        self.num_heads = 4
        bias = False
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                        nn.Conv2d(self.out_channels, self.out_channels // 16, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(self.out_channels // 16, self.out_channels, 1, padding=0, bias=True),
                        nn.Sigmoid())

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.fusion_in = nn.Sequential(
                            nn.Conv2d(dim*2, dim, 1, padding=0, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(dim, dim, 1, padding=0, bias=True))
        self.fusion_out = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias),
                            nn.ReLU(inplace=True), )
        self.CALayer = CALayer(dim)
        self.ResidualBlock = ResidualBlock_noBN(dim)
        self.ResidualBlock1 = ResidualBlock_noBN(dim)

    def forward(self, x, extra_feat, pred_feat):
        # warped_feat = flow_warp(extra_feat, flow_1.permute(0, 2, 3, 1))
        # fused_feat = self.fusion_out(torch.cat([warped_feat,pred_feat],dim=1))

        #### MSA 1
        # b, c, h, w = warped_feat.shape
        # q = x
        # k = fused_feat 
        # v = warped_feat * self.conv_du(self.avg_pool(warped_feat))
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)
        # attn = (q @ k.transpose(-2, -1)) * self.temperature
        # attn = attn.softmax(dim=-1)
        # out = (attn @ v)
        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # out_1 = self.project_out(out)
        ####

        #### MSA 2
        b, c, h, w = pred_feat.shape
        q = x
        k = pred_feat 
        v = pred_feat * self.conv_du(self.avg_pool(pred_feat))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_2 = self.project_out(out)
        #### Aggregation
        out = self.fusion_out(torch.cat([out_2, x],dim=1)) 
        # out = self.fusion_out(torch.cat([out_1,out_2],dim=1)) 
        out = self.ResidualBlock1(self.ResidualBlock(self.CALayer(out))) + x 

        return out






class MViterativeDeformableAlignment(ModulatedDeformConvPack):
    """deformable alignment module.
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(MViterativeDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()
        self.scaleing = torch.nn.Sequential(
            nn.Conv2d(self.out_channels*2, self.out_channels, 3, 1, 1, bias=True),
            torch.nn.Sigmoid(),
        )
        self.off2flow = torch.nn.Sequential(
            nn.Conv2d(self.out_channels, 4, 3, 1, 1, bias=True),
            torch.nn.Sigmoid(),
        )

        self.offset_oc = torch.nn.Sequential(
            nn.Conv2d(self.out_channels*4 + self.out_channels//2, self.out_channels, 3, 1, 1, bias=True),
            torch.nn.Sigmoid(),
        )

    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, pre_offset_fea=None):

        warped_feat = flow_warp(extra_feat, flow_1.permute(0, 2, 3, 1))

        # extra_feat = torch.cat([warped_feat, flow_1, flow_1], dim=1)
        if pre_offset_fea is None:
            # offset_fea = torch.cat([_offset,_offset],1)
            extra_feat = torch.cat([warped_feat, flow_1, flow_1], dim=1)
        else:
            offset_fea_init = torch.cat([warped_feat, pre_offset_fea],1)
            pre_offset_fea = self.off2flow(pre_offset_fea * self.scaleing(offset_fea_init))
            extra_feat = torch.cat([warped_feat, pre_offset_fea], dim=1) 

        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset_0 = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset_0 + flow_1.flip(1).repeat(1, offset_0.size(1) // 2, 1, 1)
        offset_out = self.offset_oc(offset_0)

        # mask
        mask = torch.sigmoid(mask)
        align_fea = torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                             self.dilation, mask)

        return align_fea, offset_out


# Flow-Guided Sparse Window-based Multi-head Self-Attention
class FGSW_MSA(nn.Module):
    def __init__(self, dim=64, window_size=(3,8,8),  dim_head=32, heads=4, shift=False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.shift = shift
        inner_dim = dim_head * heads
        # position embedding
        q_l = self.window_size[1]*self.window_size[2]
        kv_l = self.window_size[0]*self.window_size[1]*self.window_size[2]
        self.static_a = nn.Parameter(torch.Tensor(1, heads, q_l , kv_l))
        trunc_normal_(self.static_a)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Conv2d(dim, inner_dim, 3, 1, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 3, 1, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 3, 1, 1, bias=False)

    def forward(self, q_inp, k_f, k_r, flow):
        """
        :param q_inp: [n,1,c,h,w]
        :param k_inp: [n,2r+1,c,h,w]  (r: temporal radius of neighboring frames)
        :param flow: list: [[n,2,h,w],[n,2,h,w]]
        :return: out: [n,1,c,h,w]
        """
        b,_,h,w = q_inp.shape
        q_inp = q_inp.view(b,1,-1,h,w)
        b,f_q,c,h,w = q_inp.shape
        fb,hb,wb = self.window_size
        flow_f = flow
        # sliding window
        # if self.shift:
        #     q_inp, k_inp = map(lambda x: torch.roll(x, shifts=(-hb//2, -wb//2), dims=(-2, -1)), (q_inp, k_inp))
        #     if flow_f is not None:
        #         flow_f = torch.roll(flow_f, shifts=(-hb // 2, -wb // 2), dims=(-2, -1))
        # k_f, k_r = k_inp[:, 0], k_inp[:, 1] # , k_inp[:, 2]

        # retrive key elements
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
        grid.requires_grad = False
        grid = grid.type_as(k_f)
        if flow_f is not None:
            vgrid = grid + flow_f.permute(0, 2, 3, 1)
            vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
            vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
            # index the nearest token
            # k_f = F.grid_sample(k_f.float(), vgrid_scaled, mode='bilinear')
            k_f = F.grid_sample(k_f.float(), vgrid_scaled, mode='nearest')
            k_r = F.grid_sample(k_r.float(), vgrid_scaled, mode='nearest')

        k_inp = torch.stack([k_f, k_r], dim=1)
        # norm
        q = self.norm_q(q_inp.permute(0,1,3,4,2)).permute(0,1,4,2,3)
        kv = self.norm_kv(k_inp.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        q = self.to_q(q.flatten(0, 1))
        k, v = self.to_kv(kv.flatten(0, 1)).chunk(2, dim=1)

        # split into (B,N,C)
        q, k, v = map(lambda t: rearrange(t, '(b f) c (h p1) (w p2)-> (b h w) (f p1 p2) c', p1=hb, p2=wb, b=b), (q, k, v))
        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        # scale
        q *= self.scale
        # attention
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        # sim = sim + self.static_a
        attn = sim.softmax(dim=-1)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        # merge windows back to original feature map
        out = rearrange(out, '(b h w) (f p1 p2) c -> (b f) c (h p1) (w p2)', b=b, h=(h // hb), w=(w // wb), p1=hb, p2=wb)
        # combine heads
        out = self.to_out(out).view(b, c, h, w)

        # inverse shift
        # if self.shift:
        #     out = torch.roll(out, shifts=(hb//2, wb//2), dims=(-2, -1))

        return out




class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
        
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None



class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=0.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = ((self.beta_min + self.reparam_offset**2)**0.5)
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch)+self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size() 
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs



class LocalCorr(torch.nn.Module):
    def __init__(self,nf,nbr_size = 3,alpha=-1.0):
        super(LocalCorr,self).__init__()
        self.nbr_size = nbr_size
        self.alpha = alpha
        pass 
    def forward(self,nbr_list,ref):
        mean = torch.stack(nbr_list,1).mean(1).detach().clone()
        # print(mean.shape)
        b,c,h,w = ref.size()
        ref_clone = ref.detach().clone()
        ref_flat = ref_clone.view(b,c,-1,h*w).permute(0,3,2,1).contiguous().view(b*h*w,-1,c)
        ref_flat = torch.nn.functional.normalize(ref_flat,p=2,dim=-1)
        pad = self.nbr_size // 2
        afea_list = []
        for i in range(len(nbr_list)):
            nbr = nbr_list[i]
            weight_diff = (nbr - mean)**2
            weight_diff = torch.exp(self.alpha*weight_diff)
            
            nbr_pad = torch.nn.functional.pad(nbr,(pad,pad,pad,pad),mode='reflect')
            nbr = torch.nn.functional.unfold(nbr_pad,kernel_size=self.nbr_size).view(b,c,-1,h*w)
            nbr = torch.nn.functional.normalize(nbr,p=2,dim=1)
            nbr = nbr.permute(0,3,1,2).contiguous().view(b*h*w,c,-1)
            d = torch.matmul(ref_flat,nbr).squeeze(1)
            weight_temporal = torch.nn.functional.softmax(d,-1)
            agg_fea = torch.einsum('bc,bnc->bn',weight_temporal,nbr).view(b,h,w,c).contiguous().permute(0,3,1,2)

            agg_fea = agg_fea * weight_diff
            
            afea_list.append(agg_fea)
        al_fea = torch.stack(afea_list+[ref],1)
        return al_fea



class Motion_FeaFusion(nn.Module):
    def __init__(self, nf=64):
        super(Motion_FeaFusion, self).__init__()
        self.scaleing = torch.nn.Sequential(
            nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True),
            torch.nn.Sigmoid(),
        )
        self.conv_out = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
    def forward(self,m0,m1):
        m_init = torch.cat([m0,m1],1)
        weighting = self.scaleing(m_init)
        # print('we',weighting.shape,m0.shape,m1.shape)
        mf = torch.cat([weighting*m0,(1.0-weighting)*m1],1)
        return self.lrelu( self.conv_out(mf) )



class EMVNet(torch.nn.Module):
    def __init__(self):
        super(EMVNet, self).__init__()
        ##### encoder
        out_channel_N = 64
        self.conv1 = nn.Conv2d(2, out_channel_N, 3, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(out_channel_N)

        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 3, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(out_channel_N)

        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 3, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(out_channel_N)

        self.conv4 = nn.Conv2d(out_channel_N, out_channel_N, 3, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (2 + out_channel_N) / (out_channel_N + out_channel_N))))
        # torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)

        ##### decoder
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 3, stride=2, padding=2, output_padding=1) #  
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 )))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(out_channel_N, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 3, stride=2, padding=2, output_padding=1) #  , output_padding=1
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(out_channel_N, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 3, stride=2, padding=2, output_padding=1) #  , output_padding=1
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN(out_channel_N, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(out_channel_N, 2, 3, stride=2, padding=2, output_padding=1) # , output_padding=1
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)


    def forward(self, tenFirst): 
       
        n, c, h, w = tenFirst.shape
        x = self.gdn1(self.conv1(tenFirst))
        # x = self.gdn2(self.conv2(x))
        # x = self.gdn3(self.conv3(x))
        # x = self.conv4(x)

        # x = self.igdn1(self.deconv1(x))
        # x = self.igdn2(self.deconv2(x))
        # x = self.igdn3(self.deconv3(x))
        tenFlow = self.deconv4(x)
        n_, c_, h_, w_ = tenFlow.shape
        # tenFlow = tenFlow[:,:,:h,:w]
        # print('[tenFirst.shape]',tenFirst.shape,tenFlow.shape)

        
        return tenFlow



class GhostModuleMul(nn.Module):
    """
    GhostModuleMul warpper definition.
    Args:
        ratio (int): Reduction ratio.
        dw_size (int): kernel size of cheap operation.
        use_act (bool): Used activation or not.
        act_type (string): Activation type.
    Returns:
        Tensor, output tensor.
    Examples:
    """

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, ratio=2, dw_size=3,
                 use_act=True, act_type='relu'):
        super(GhostModuleMul, self).__init__()
        self.avgpool2d = nn.AvgPool2d(kernel_size=1, stride=1)
        self.gate_fn = Activation('sigmoid')
        init_channels = math.ceil(num_out / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
                        nn.Conv2d(num_in, init_channels, kernel_size=3, stride=1, padding=3//2, num_groups=1),
                        nn.LeakyReLU(negative_slope=0.1, inplace=True),)

        self.cheap_operation = nn.Sequential(
                        nn.Conv2d(num_in, init_channels, kernel_size=3, stride=1, padding=3//2, num_groups=1),
                        nn.LeakyReLU(negative_slope=0.1, inplace=True),)
        self.short_conv = nn.Sequential(
                        nn.Conv2d(num_in, num_out, kernel_size=kernel_size, stride=stride,
                                padding=kernel_size//2, num_groups=1),
                        nn.Conv2d(num_out, num_out, kernel_size=(1, 5), stride=1,
                                padding=(0, 2), num_groups=num_out),
                        nn.Conv2d(num_out, num_out, kernel_size=(5, 1), stride=1,
                                padding=(2, 0), num_groups=num_out),)

    def forward(self, x):
        """ ghost module forward """
        res = self.avgpool2d(x)
        res = self.short_conv(res)
        res = self.gate_fn(res)
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = out * res
        return out





class SIDECVSR(nn.Module):
    def __init__(self, nf=64, nframes=7, fea_ext_RBs=7, SCGs=4, istraining=False):
        super(SIDECVSR, self).__init__()
        self.nf = nf
        self.center = nframes // 2
        self.istraining = istraining

        #### extraction
        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)   
        self.feature_extraction = side_embeded_feature_extract_block(nf=nf)

        #### fusion
        self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        #### reconstruction
        self.recon_trunk = SCNet(nf=nf, SCGroupN=SCGs)

        #### upsampling
        self.upconv1 = nn.Conv2d(nf + nf // 4 + nf // 16, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        # self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, 1, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.mv_patch_attn = MV_LOCAL_ATTN(nf=nf)

        #### fea fusion attn
        self.tmp_fea_attn = fea_fusion(nf=nf)

        #### multi-scale
        self.down = Interpolate(scale_factor=0.5)

        #### fea pyramid fuse 
        self.upconv1_L2 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
        self.upconv1_L3 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)

        #### 
        self.side_fea_ext = side_to_fea(nf=nf//2)


    def forward(self, x, mvs, pms, rms, ufs, pre_L1_fea=None):
        B, N, C, H, W = x.size()  # N video frames C=1 # mvs.shape = (b, n, 2, h, w)
        x_center = x[:, self.center, :, :, :].contiguous()
        # sides = torch.cat([rms.view(-1, C, H, W), pms.view(-1, C, H, W), ufs.view(-1, C, H, W)], 1)
        # sides_fea = self.side_fea_ext(sides)
        
        # 1.GCPI
        feas_pyr = []
        # imgs -> feas # multi-scale
        if pre_L1_fea is None:
            L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
            # sides = torch.cat([rms.view(-1, C, H, W), pms.view(-1, C, H, W), ufs.view(-1, C, H, W)], 1)
            sides = pms.view(-1, C, H, W)
            sides_fea = self.side_fea_ext(sides)
            L1_fea = self.feature_extraction(L1_fea, sides_fea)
        else:
            need_add_fea = self.lrelu(self.conv_first(x[:,-1,:,:,:]))
            need_add_sides = torch.cat([rms[:,-1,:,:,:], pms[:,-1,:,:,:], ufs[:,-1,:,:,:]], 1)
            need_add_sides_fea = self.side_fea_ext(need_add_sides)

            need_add_L1_fea = self.feature_extraction(need_add_fea, need_add_sides_fea)
            need_add_L1_fea = torch.unsqueeze(need_add_L1_fea, 1)
            pre_L1_fea = pre_L1_fea.view(B, N, -1, H, W)
            L1_fea = torch.cat([pre_L1_fea[:,1:,:,:,:], need_add_L1_fea], 1)

            L1_fea = L1_fea.view(B*N, -1, H, W)
        
        feas_pyr.append(L1_fea)
        L2_fea = self.down(L1_fea)
        feas_pyr.append(L2_fea)
        L3_fea = self.down(L2_fea)
        feas_pyr.append(L3_fea)  #  obtain the sequence 7 frames features
        
        # 2.MV-GSA
        fuse_fea_pyr = []
        for pyr_i in range(3):
            fea_one_lv = feas_pyr[pyr_i].view(B, N, -1, H//(2**pyr_i), W//(2**pyr_i))
            # local attention
            aligned_fea = []
            for i in range(N):
                if i != N // 2:
                    if pyr_i == 0:
                        tmp_mv = mvs[:,i,:,:,:].clone()
                    if pyr_i == 1:
                        tmp_mv = F.interpolate(mvs[:,i,:,:,:].clone(), scale_factor=0.5, mode='bilinear', align_corners=False) / 2.0
                    if pyr_i == 2:
                        tmp_mv = F.interpolate(mvs[:,i,:,:,:].clone(), scale_factor=0.25, mode='bilinear', align_corners=False) / 4.0
                    
                    # MV-GSA   alignment  obtain the multi-scale aligned features
                    alg_nbh_fea = self.mv_patch_attn(fea_one_lv[:,i,:,:,:].clone(), fea_one_lv[:, N//2,:,:,:].clone(), tmp_mv) ### original mv
                    aligned_fea.append(alg_nbh_fea)
                else:
                    aligned_fea.append(fea_one_lv[:,i,:,:,:].clone())

            # feature fusion 
            aligned_fea = torch.stack(aligned_fea, dim=1)                      # [B, N, C, H, W]
            aligned_fea = aligned_fea.view(B, -1, H//(2**pyr_i), W//(2**pyr_i))

            # 3.ATFM
            fea = self.lrelu(self.tsa_fusion(self.tmp_fea_attn(aligned_fea)))  ### tmp_attn + fusion 

            fuse_fea_pyr.append(fea)

        # reconstruct CSSR 
        # 4.CSSR
        out = self.recon_trunk(fuse_fea_pyr)

        out_L3 = self.lrelu(self.upconv1_L3(out[2]))
        out_L3 = self.pixel_shuffle(self.pixel_shuffle(out_L3))
        out_L2 = self.lrelu(self.upconv1_L2(out[1]))
        out_L2 = self.pixel_shuffle(out_L2)
        out_fuse = torch.cat([out[0], out_L2, out_L3], 1)
        
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out_fuse)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.conv_last(out)
        
        # skip connection + output
        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base
        return out, L1_fea



class CVSR_V7(nn.Module):
    def __init__(self, nf=64, nframes=7, fea_ext_RBs=7, SCGs=4, istraining=False):
        super(CVSR_V7, self).__init__()
        self.nf = nf
        self.center = nframes // 2
        self.istraining = istraining
        self.stride = 4
        #### extraction
        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)   
        self.conv_second = nn.Conv2d(1, nf, 3, 1, 1, bias=True) 
        self.transformer_feature_extraction = PAItransformer_feat_extract()
        self.conv_expand_fea_r = nn.Conv2d(2*nf, nf, 3, 1, 1, bias=True)
        self.conv_expand_ufs = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.conv_expand_rms = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.fb_fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)
        #### fusion
        self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        #### reconstruction
        self.recon_trunk = SCNet(nf=nf, SCGroupN=7) 
        #### upsampling
        self.upconv1 = nn.Conv2d(nf + nf // 4 + nf // 16, nf * 4, 1, 1, 0, bias=True)  
        self.upconv2 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_last = nn.Conv2d(nf, 1, 1, 1, 0, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.MV_deform_align = MVDualAttAlignment(64, 64, 3, padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.RDAB = RDAB()
        #### multi-scale
        self.down = Interpolate(scale_factor=0.5)
        self.up = Interpolate(scale_factor=2.0)

        #### fea pyramid fuse 
        self.upconv1_L2 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
        self.upconv1_L3 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)

    def forward(self, x, mvs0, mvs1, pms, rms, ufs, pre_L1_fea=None):
        B, N, C, H, W = x.size()  # N video frames C=1 # mvs.shape = (b, n, 2, h, w)
        x_center = x[:, self.center, :, :, :].contiguous()
        # 1.GCPI
        feas_pyr = []
        # imgs -> feas # multi-scale
        if pre_L1_fea is None:
            L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
            sides_fea = self.conv_second(pms.view(-1, C, H, W))
            L1_fea = self.transformer_feature_extraction(L1_fea, sides_fea)  #   
        else:
            need_add_fea = self.lrelu(self.conv_first(x[:,-1,:,:,:]))
            need_add_sides_fea = self.conv_second(pms[:,-1,:,:,:])
            need_add_L1_fea = self.transformer_feature_extraction(need_add_fea, need_add_sides_fea)  
            need_add_L1_fea = torch.unsqueeze(need_add_L1_fea, 1)
            pre_L1_fea = pre_L1_fea.view(B, N, -1, H, W)
            L1_fea = torch.cat([pre_L1_fea[:,1:,:,:,:], need_add_L1_fea], 1)
            L1_fea = L1_fea.view(B*N, -1, H, W)
        
        feas_pyr.append(L1_fea)
        L2_fea = self.down(L1_fea)
        feas_pyr.append(L2_fea)
        L3_fea = self.down(L2_fea)
        feas_pyr.append(L3_fea)  #  obtain the sequence 7 frames features
        # 2.MV-GSA
        fuse_fea_pyr = [] 
        for pyr_i in range(2,-1,-1):  #  L3 L2 L1
            fea_one_lv = feas_pyr[pyr_i].view(B, N, -1, H//(2**pyr_i), W//(2**pyr_i))
            BB, CC, NN, HH, WW = ufs.shape
            if CC != 1:
                ufs = ufs.transpose(1, 2)
                rms = rms.transpose(1, 2)
            # local attention
            for_aligned_fea = []
            aligned_fea = []
            for i in range(N - 1, -1, -1):     
                if i != N // 2:
                    if pyr_i == 0:
                        tmp_mv = (mvs0[:,i,:,:,:].clone())  
                        ufs_prior = self.conv_expand_ufs(ufs[:,:,i,:,:].clone())
                        rms_prior = self.conv_expand_rms(rms[:,:,i,:,:].clone())
                    if pyr_i == 1:
                        tmp_mv = (F.interpolate(mvs0[:,i,:,:,:].clone(), scale_factor=0.5, mode='bilinear', align_corners=False) / 2.0)  
                        ufs_prior = self.conv_expand_ufs(F.interpolate(ufs[:,:,i,:,:].clone(), scale_factor=0.5, mode='bilinear', align_corners=False) / 2.0)
                        rms_prior = self.conv_expand_rms(F.interpolate(rms[:,:,i,:,:].clone(), scale_factor=0.5, mode='bilinear', align_corners=False) / 2.0)
                    if pyr_i == 2:
                        tmp_mv = (F.interpolate(mvs0[:,i,:,:,:].clone(), scale_factor=0.25, mode='bilinear', align_corners=False) / 4.0)   
                        ufs_prior = self.conv_expand_ufs(F.interpolate(ufs[:,:,i,:,:].clone(), scale_factor=0.25, mode='bilinear', align_corners=False) / 4.0)
                        rms_prior = self.conv_expand_rms(F.interpolate(rms[:,:,i,:,:].clone(), scale_factor=0.25, mode='bilinear', align_corners=False) / 4.0)
                    # spatial-compensate block
                    if pyr_i != 2:
                        fea_com = fea_one_lv[:,i,:,:,:].clone() + rms_prior + self.up(aligned_fea_out[:,i,:,:,:])
                    else:
                        fea_com = fea_one_lv[:,i,:,:,:].clone() + rms_prior
                    x_n = self.RDAB(rms_prior, fea_com)  
                    # temporal-compensate alignment
                    fea_one_lv_i = self.conv_expand_fea_r(torch.cat([fea_one_lv[:,i,:,:,:].clone(), x_n],1)) # 
                    alg_nbh_fea = self.MV_deform_align(fea_one_lv[:, N//2,:,:,:].clone(), fea_one_lv_i, ufs_prior, tmp_mv) 
                    for_aligned_fea.append(alg_nbh_fea) 
                else:
                    for_aligned_fea.append(fea_one_lv[:,i,:,:,:].clone())
            for_aligned_fea = for_aligned_fea[::-1]
            
            # feat_prop = x.new_zeros(N, self.nf, H, W)
            for i in range(N):     
                if i != N // 2:
                    if pyr_i == 0:
                        tmp_mv = (mvs1[:,i,:,:,:].clone())  
                        ufs_prior = self.conv_expand_ufs(ufs[:,:,i,:,:].clone())
                        rms_prior = self.conv_expand_rms(rms[:,:,i,:,:].clone())
                    if pyr_i == 1:
                        tmp_mv = (F.interpolate(mvs1[:,i,:,:,:].clone(), scale_factor=0.5, mode='bilinear', align_corners=False) / 2.0)   
                        ufs_prior = self.conv_expand_ufs(F.interpolate(ufs[:,:,i,:,:].clone(), scale_factor=0.5, mode='bilinear', align_corners=False) / 2.0)
                        rms_prior = self.conv_expand_rms(F.interpolate(rms[:,:,i,:,:].clone(), scale_factor=0.5, mode='bilinear', align_corners=False) / 2.0)
                    if pyr_i == 2:
                        tmp_mv = (F.interpolate(mvs1[:,i,:,:,:].clone(), scale_factor=0.25, mode='bilinear', align_corners=False) / 4.0)  
                        ufs_prior = self.conv_expand_ufs(F.interpolate(ufs[:,:,i,:,:].clone(), scale_factor=0.25, mode='bilinear', align_corners=False) / 4.0)
                        rms_prior = self.conv_expand_rms(F.interpolate(rms[:,:,i,:,:].clone(), scale_factor=0.25, mode='bilinear', align_corners=False) / 4.0)
                    # spatial-compensate block   
                    if pyr_i != 2:
                        fea_com = fea_one_lv[:,i,:,:,:].clone() + rms_prior + self.up(aligned_fea_out[:,i,:,:,:])
                    else:
                        fea_com = fea_one_lv[:,i,:,:,:].clone() + rms_prior
                    _, _, c, h, w = fea_one_lv.size()
                    x_n = self.RDAB(rms_prior, fea_com) 
                    # temporal-compensate alignment
                    fea_one_lv_i = self.conv_expand_fea_r(torch.cat([fea_one_lv[:,i,:,:,:].clone(), x_n],1))
                    alg_nbh_fea = self.MV_deform_align(fea_one_lv[:, N//2,:,:,:].clone(), fea_one_lv_i, ufs_prior, tmp_mv) 
                    feat_prop = self.fb_fusion(torch.cat([for_aligned_fea[i], alg_nbh_fea],dim=1))
                    aligned_fea.append(feat_prop) 
                else:
                    aligned_fea.append(fea_one_lv[:,i,:,:,:].clone())
            # feature fusion 
            aligned_fea_out = torch.stack(aligned_fea, dim=1)   # [B, N, C, H, W]
            aligned_fea = aligned_fea_out.view(B, -1, H//(2**pyr_i), W//(2**pyr_i))
            # 3.ATFM 
            aligned_fea = self.lrelu(self.tsa_fusion((aligned_fea)))     
            fuse_fea_pyr.append(aligned_fea)

        # reconstruct CSSR 
        # 4.CSSR  cross-scale fusion module
        fuse_fea_pyr = fuse_fea_pyr[::-1]
        out = self.recon_trunk(fuse_fea_pyr)
        out_L3 = self.lrelu(self.upconv1_L3(out[2]))
        out_L3 = self.pixel_shuffle(self.pixel_shuffle(out_L3))
        out_L2 = self.lrelu(self.upconv1_L2(out[1]))
        out_L2 = self.pixel_shuffle(out_L2)
        out_fuse = torch.cat([out[0], out_L2, out_L3], 1)
        
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out_fuse)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.conv_last(out)
        # skip connection + output
        base = F.interpolate(x_center, scale_factor=4.0, mode='bilinear', align_corners=False)
        out += base
        return out, L1_fea  



class CVSR_V8(nn.Module):
    def __init__(self, nf=64, nframes=7, fea_ext_RBs=7, SCGs=4, istraining=False):
        super(CVSR_V8, self).__init__()
        self.nf = nf
        self.center = nframes // 2
        self.istraining = istraining
        self.stride = 4
        #### extraction
        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)   
        self.conv_second = nn.Conv2d(1, nf, 3, 1, 1, bias=True) 
        self.transformer_feature_extraction = PAItransformerSA_2() # PAItransformerSA_1()
        self.conv_expand_fea_r = nn.Conv2d(2*nf, nf, 3, 1, 1, bias=True)
        self.conv_expand_ufs = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.conv_expand_rms = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        #### fusion
        self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True) # 
        #### reconstruction
        self.recon_trunk = SCNet_(nf=nf, SCGroupN=7)  
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)  
        self.upconv2 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_last = nn.Conv2d(nf, 1, 3, 1, 1, bias=True)
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.MV_deform_align = MVDualAttAlignment(64, 64, 3, padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.MV_deform_align = DualAttAlignment() #  DualAttAlignment(64, 64, 3, padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.RDAB = LLongRangAttention(64) # LongRangAttention() # CrissCrossAttention() # RDAB() #  RSNLAttention()
        #### multi-scale
        self.down = Interpolate(scale_factor=0.5)
        self.up = Interpolate(scale_factor=2.0)
        #### fea pyramid fuse 
        # self.upconv1_L2 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
        # self.upconv1_L3 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
    
    def forward(self, x, mvs0, mvs1, pms, rms, ufs, pre_L1_fea=None):

        
        B, N, C, H, W = x.size()  # N video frames C=1 # mvs.shape = (b, n, 2, h, w)
        self.H = H
        self.W = W
        x_center = x[:, self.center, :, :, :].contiguous()
        # 1.GCPI
        feas_pyr = []
        # imgs -> feas # multi-scale
        if pre_L1_fea is None:
            L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
            sides_fea = self.conv_second(pms.view(-1, C, H, W))
            L1_fea = self.transformer_feature_extraction(L1_fea, sides_fea)  #   
        else:
            need_add_fea = self.lrelu(self.conv_first(x[:,-1,:,:,:]))
            need_add_sides_fea = self.conv_second(pms[:,-1,:,:,:])
            need_add_L1_fea = self.transformer_feature_extraction(need_add_fea, need_add_sides_fea)  
            need_add_L1_fea = torch.unsqueeze(need_add_L1_fea, 1)
            pre_L1_fea = pre_L1_fea.view(B, N, -1, H, W)
            L1_fea = torch.cat([pre_L1_fea[:,1:,:,:,:], need_add_L1_fea], 1)
            L1_fea = L1_fea.view(B*N, -1, H, W)
        
        feas_pyr.append(L1_fea)
        #  obtain the sequence 7 frames features
        # 2.MV-GSA
        fuse_fea_pyr = [] 
        fea_one_lv = L1_fea.view(B, N, -1, H, W)
        BB, CC, NN, HH, WW = ufs.shape
        if CC != 1:
            ufs = ufs.transpose(1, 2)
            rms = rms.transpose(1, 2)
        # local attention
        for_aligned_fea = []
        aligned_fea = []
        recon_list = []
        
        for i in range(N):     
            if i != N // 2:
                tmp_mv = (mvs1[:,i,:,:,:].clone())  
                ufs_prior = self.conv_expand_ufs(ufs[:,:,i,:,:].clone())
                rms_prior = self.conv_expand_rms(rms[:,:,i,:,:].clone())
                # spatial-compensate block   
                fea_com = fea_one_lv[:,i,:,:,:].clone() + rms_prior
                featuremap_visual(fea_com, feature_title='fea_com')
                _, _, c, h, w = fea_one_lv.size()
                x_n = self.RDAB(rms_prior, fea_com) 
                # temporal-compensate alignment
                fea_one_lv_i = self.conv_expand_fea_r(torch.cat([fea_one_lv[:,i,:,:,:].clone(), x_n],1))
                featuremap_visual(fea_one_lv_i, feature_title='align_org')
                alg_nbh_fea = self.MV_deform_align(fea_one_lv[:, N//2,:,:,:].clone(), fea_one_lv_i, ufs_prior, tmp_mv) 
                # featuremap_visual(alg_nbh_fea, feature_title='align result' )
                aligned_fea.append(alg_nbh_fea) 
            else:
                aligned_fea.append(fea_one_lv[:,i,:,:,:].clone())
        
        # feature fusion
        aligned_fea_out = torch.stack(aligned_fea, dim=1)   # [B, N, C, H, W]
        aligned_fea = aligned_fea_out.view(B, -1, H, W)
        # 3.ATFM 
        aligned_fea_ = self.lrelu(self.tsa_fusion((aligned_fea)))     
        recon_list.append(aligned_fea_)

        # reconstruct CSSR 
        # 4.CSSR  cross-scale fusion module
        out = self.recon_trunk(aligned_fea_)
        featuremap_visual(out, feature_title='out recon' )
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        featuremap_visual(out, feature_title='out final' )
        out = self.conv_last(out)
        
        # skip connection + output
        base = F.interpolate(x_center, scale_factor=4.0, mode='bilinear', align_corners=False)
        out += base
        return out, L1_fea  


class CVSR_V8_woPAB(nn.Module):
    def __init__(self, nf=64, nframes=7, fea_ext_RBs=7, SCGs=4, istraining=False):
        super(CVSR_V8_woPAB, self).__init__()
        self.nf = nf
        self.center = nframes // 2
        self.istraining = istraining
        self.stride = 4
        #### extraction
        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)   
        # self.conv_second = nn.Conv2d(1, nf, 3, 1, 1, bias=True) 
        self.transformer_feature_extraction = PAItransformerSA_woPAB() # PAItransformerSA_2()
        self.conv_expand_fea_r = nn.Conv2d(2*nf, nf, 3, 1, 1, bias=True)
        self.conv_expand_ufs = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.conv_expand_rms = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        #### fusion
        self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True) # 
        #### reconstruction
        self.recon_trunk = SCNet_(nf=nf, SCGroupN=7)  
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)  
        self.upconv2 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_last = nn.Conv2d(nf, 1, 3, 1, 1, bias=True)
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.MV_deform_align = MVDualAttAlignment(64, 64, 3, padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.MV_deform_align = DualAttAlignment() #  DualAttAlignment(64, 64, 3, padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.RDAB = LLongRangAttention(64) # LongRangAttention() # CrissCrossAttention() # RDAB() #  RSNLAttention()
        #### multi-scale
        self.down = Interpolate(scale_factor=0.5)
        self.up = Interpolate(scale_factor=2.0)
        #### fea pyramid fuse 
        # self.upconv1_L2 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
        # self.upconv1_L3 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
    
    def forward(self, x, mvs0, mvs1, pms, rms, ufs, pre_L1_fea=None):
        B, N, C, H, W = x.size()  # N video frames C=1 # mvs.shape = (b, n, 2, h, w)
        self.H = H
        self.W = W
        x_center = x[:, self.center, :, :, :].contiguous()
        # 1.GCPI
        feas_pyr = []
        # imgs -> feas # multi-scale
        if pre_L1_fea is None:
            L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
            # sides_fea = self.conv_second(pms.view(-1, C, H, W))
            L1_fea = self.transformer_feature_extraction(L1_fea)  #   
        else:
            need_add_fea = self.lrelu(self.conv_first(x[:,-1,:,:,:]))
            # need_add_sides_fea = self.conv_second(pms[:,-1,:,:,:])
            need_add_L1_fea = self.transformer_feature_extraction(need_add_fea)  
            need_add_L1_fea = torch.unsqueeze(need_add_L1_fea, 1)
            pre_L1_fea = pre_L1_fea.view(B, N, -1, H, W)
            L1_fea = torch.cat([pre_L1_fea[:,1:,:,:,:], need_add_L1_fea], 1)
            L1_fea = L1_fea.view(B*N, -1, H, W)
        
        feas_pyr.append(L1_fea)
        #  obtain the sequence 7 frames features
        # 2.MV-GSA
        fuse_fea_pyr = [] 
        fea_one_lv = L1_fea.view(B, N, -1, H, W)
        BB, CC, NN, HH, WW = ufs.shape
        if CC != 1:
            ufs = ufs.transpose(1, 2)
            rms = rms.transpose(1, 2)
        # local attention
        for_aligned_fea = []
        aligned_fea = []
        recon_list = []
        
        for i in range(N):     
            if i != N // 2:
                tmp_mv = (mvs1[:,i,:,:,:].clone())  
                ufs_prior = self.conv_expand_ufs(ufs[:,:,i,:,:].clone())
                rms_prior = self.conv_expand_rms(rms[:,:,i,:,:].clone())
                # spatial-compensate block   
                fea_com = fea_one_lv[:,i,:,:,:].clone() + rms_prior
                _, _, c, h, w = fea_one_lv.size()
                x_n = self.RDAB(rms_prior, fea_com) 
                # temporal-compensate alignment
                fea_one_lv_i = self.conv_expand_fea_r(torch.cat([fea_one_lv[:,i,:,:,:].clone(), x_n],1))
                alg_nbh_fea = self.MV_deform_align(fea_one_lv[:, N//2,:,:,:].clone(), fea_one_lv_i, ufs_prior, tmp_mv) 
                aligned_fea.append(alg_nbh_fea) 
            else:
                aligned_fea.append(fea_one_lv[:,i,:,:,:].clone())
        
        # feature fusion
        aligned_fea_out = torch.stack(aligned_fea, dim=1)   # [B, N, C, H, W]
        aligned_fea = aligned_fea_out.view(B, -1, H, W)
        # 3.ATFM 
        aligned_fea_ = self.lrelu(self.tsa_fusion((aligned_fea)))     
        recon_list.append(aligned_fea_)

        # reconstruct CSSR 
        # 4.CSSR  cross-scale fusion module
        out = self.recon_trunk(aligned_fea_)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))  # out[0]
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.conv_last(out)
        # skip connection + output
        base = F.interpolate(x_center, scale_factor=4.0, mode='bilinear', align_corners=False)
        out += base
        return out, L1_fea  



class CVSR_V8_woLA(nn.Module):
    def __init__(self, nf=64, nframes=7, fea_ext_RBs=7, SCGs=4, istraining=False):
        super(CVSR_V8_woLA, self).__init__()
        self.nf = nf
        self.center = nframes // 2
        self.istraining = istraining
        self.stride = 4
        #### extraction
        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)   
        self.conv_second = nn.Conv2d(1, nf, 3, 1, 1, bias=True) 
        self.transformer_feature_extraction = PAItransformerSA_2() 
        self.conv_expand_fea_r = nn.Conv2d(2*nf, nf, 3, 1, 1, bias=True)
        self.conv_expand_ufs = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        # self.conv_expand_rms = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        #### fusion
        self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True) 
        #### reconstruction
        self.recon_trunk = SCNet_(nf=nf, SCGroupN=7)  
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)  
        self.upconv2 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_last = nn.Conv2d(nf, 1, 3, 1, 1, bias=True)
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.MV_deform_align = MVDualAttAlignment(64, 64, 3, padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.MV_deform_align = DualAttAlignment() #  DualAttAlignment(64, 64, 3, padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.RDAB = LLongRangAttention_woLA(64) # LongRangAttention() # CrissCrossAttention() # RDAB() #  RSNLAttention()
        #### multi-scale
        self.down = Interpolate(scale_factor=0.5)
        self.up = Interpolate(scale_factor=2.0)
        #### fea pyramid fuse 
        # self.upconv1_L2 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
        # self.upconv1_L3 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
    
    def forward(self, x, mvs0, mvs1, pms, rms, ufs, pre_L1_fea=None):
        B, N, C, H, W = x.size()  # N video frames C=1 # mvs.shape = (b, n, 2, h, w)
        self.H = H
        self.W = W
        x_center = x[:, self.center, :, :, :].contiguous()
        # 1.GCPI
        feas_pyr = []
        # imgs -> feas # multi-scale
        if pre_L1_fea is None:
            L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
            sides_fea = self.conv_second(pms.view(-1, C, H, W))
            L1_fea = self.transformer_feature_extraction(L1_fea, sides_fea)  #   
        else:
            need_add_fea = self.lrelu(self.conv_first(x[:,-1,:,:,:]))
            need_add_sides_fea = self.conv_second(pms[:,-1,:,:,:])
            need_add_L1_fea = self.transformer_feature_extraction(need_add_fea, need_add_sides_fea)  
            need_add_L1_fea = torch.unsqueeze(need_add_L1_fea, 1)
            pre_L1_fea = pre_L1_fea.view(B, N, -1, H, W)
            L1_fea = torch.cat([pre_L1_fea[:,1:,:,:,:], need_add_L1_fea], 1)
            L1_fea = L1_fea.view(B*N, -1, H, W)
        
        feas_pyr.append(L1_fea)
        #  obtain the sequence 7 frames features
        # 2.MV-GSA
        fuse_fea_pyr = [] 
        fea_one_lv = L1_fea.view(B, N, -1, H, W)
        BB, CC, NN, HH, WW = ufs.shape
        if CC != 1:
            ufs = ufs.transpose(1, 2)
            rms = rms.transpose(1, 2)
        # local attention
        for_aligned_fea = []
        aligned_fea = []
        recon_list = []
        
        for i in range(N):     
            if i != N // 2:
                tmp_mv = (mvs1[:,i,:,:,:].clone())  
                ufs_prior = self.conv_expand_ufs(ufs[:,:,i,:,:].clone())
                # rms_prior = self.conv_expand_rms(rms[:,:,i,:,:].clone())
                # spatial-compensate block   
                fea_com = fea_one_lv[:,i,:,:,:].clone() # + rms_prior
                _, _, c, h, w = fea_one_lv.size()
                x_n = self.RDAB(fea_com) 
                # temporal-compensate alignment
                fea_one_lv_i = self.conv_expand_fea_r(torch.cat([fea_one_lv[:,i,:,:,:].clone(), x_n],1))
                alg_nbh_fea = self.MV_deform_align(fea_one_lv[:, N//2,:,:,:].clone(), fea_one_lv_i, ufs_prior, tmp_mv) 
                aligned_fea.append(alg_nbh_fea) 
            else:
                aligned_fea.append(fea_one_lv[:,i,:,:,:].clone())
        
        # feature fusion
        aligned_fea_out = torch.stack(aligned_fea, dim=1)   # [B, N, C, H, W]
        aligned_fea = aligned_fea_out.view(B, -1, H, W)
        # 3.ATFM 
        aligned_fea_ = self.lrelu(self.tsa_fusion((aligned_fea)))     
        recon_list.append(aligned_fea_)

        # reconstruct CSSR 
        # 4.CSSR  cross-scale fusion module
        out = self.recon_trunk(aligned_fea_)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))  # out[0]
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.conv_last(out)
        # skip connection + output
        base = F.interpolate(x_center, scale_factor=4.0, mode='bilinear', align_corners=False)
        out += base
        return out, L1_fea  



class CVSR_V8_woGA(nn.Module):
    def __init__(self, nf=64, nframes=7, fea_ext_RBs=7, SCGs=4, istraining=False):
        super(CVSR_V8_woGA, self).__init__()
        self.nf = nf
        self.center = nframes // 2
        self.istraining = istraining
        self.stride = 4
        #### extraction
        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)   
        self.conv_second = nn.Conv2d(1, nf, 3, 1, 1, bias=True) 
        self.transformer_feature_extraction = PAItransformerSA_2() # PAItransformerSA_1()
        self.conv_expand_fea_r = nn.Conv2d(2*nf, nf, 3, 1, 1, bias=True)
        self.conv_expand_ufs = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.conv_expand_rms = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        #### fusion
        self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True) # 
        #### reconstruction
        self.recon_trunk = SCNet_(nf=nf, SCGroupN=7)  
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)  
        self.upconv2 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_last = nn.Conv2d(nf, 1, 3, 1, 1, bias=True)
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.MV_deform_align = MVDualAttAlignment(64, 64, 3, padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.MV_deform_align = DualAttAlignment() #  DualAttAlignment(64, 64, 3, padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.RDAB = LLongRangAttention_woGA(64) # LongRangAttention() # CrissCrossAttention() # RDAB() #  RSNLAttention()
        #### multi-scale
        self.down = Interpolate(scale_factor=0.5)
        self.up = Interpolate(scale_factor=2.0)
        #### fea pyramid fuse 
        # self.upconv1_L2 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
        # self.upconv1_L3 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
    
    def forward(self, x, mvs0, mvs1, pms, rms, ufs, pre_L1_fea=None):
        B, N, C, H, W = x.size()  # N video frames C=1 # mvs.shape = (b, n, 2, h, w)
        self.H = H
        self.W = W
        x_center = x[:, self.center, :, :, :].contiguous()
        # 1.GCPI
        feas_pyr = []
        # imgs -> feas # multi-scale
        if pre_L1_fea is None:
            L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
            sides_fea = self.conv_second(pms.view(-1, C, H, W))
            L1_fea = self.transformer_feature_extraction(L1_fea, sides_fea)  #   
        else:
            need_add_fea = self.lrelu(self.conv_first(x[:,-1,:,:,:]))
            need_add_sides_fea = self.conv_second(pms[:,-1,:,:,:])
            need_add_L1_fea = self.transformer_feature_extraction(need_add_fea, need_add_sides_fea)  
            need_add_L1_fea = torch.unsqueeze(need_add_L1_fea, 1)
            pre_L1_fea = pre_L1_fea.view(B, N, -1, H, W)
            L1_fea = torch.cat([pre_L1_fea[:,1:,:,:,:], need_add_L1_fea], 1)
            L1_fea = L1_fea.view(B*N, -1, H, W)
        
        feas_pyr.append(L1_fea)
        #  obtain the sequence 7 frames features
        # 2.MV-GSA
        fuse_fea_pyr = [] 
        fea_one_lv = L1_fea.view(B, N, -1, H, W)
        BB, CC, NN, HH, WW = ufs.shape
        if CC != 1:
            ufs = ufs.transpose(1, 2)
            rms = rms.transpose(1, 2)
        # local attention
        for_aligned_fea = []
        aligned_fea = []
        recon_list = []
        
        for i in range(N):     
            if i != N // 2:
                tmp_mv = (mvs1[:,i,:,:,:].clone())  
                ufs_prior = self.conv_expand_ufs(ufs[:,:,i,:,:].clone())
                rms_prior = self.conv_expand_rms(rms[:,:,i,:,:].clone())
                # spatial-compensate block   
                fea_com = fea_one_lv[:,i,:,:,:].clone() + rms_prior
                _, _, c, h, w = fea_one_lv.size()
                x_n = self.RDAB(rms_prior, fea_com) 
                # temporal-compensate alignment
                fea_one_lv_i = self.conv_expand_fea_r(torch.cat([fea_one_lv[:,i,:,:,:].clone(), x_n],1))
                alg_nbh_fea = self.MV_deform_align(fea_one_lv[:, N//2,:,:,:].clone(), fea_one_lv_i, ufs_prior, tmp_mv) 
                aligned_fea.append(alg_nbh_fea) 
            else:
                aligned_fea.append(fea_one_lv[:,i,:,:,:].clone())
        
        # feature fusion
        aligned_fea_out = torch.stack(aligned_fea, dim=1)   # [B, N, C, H, W]
        aligned_fea = aligned_fea_out.view(B, -1, H, W)
        # 3.ATFM 
        aligned_fea_ = self.lrelu(self.tsa_fusion((aligned_fea)))     
        recon_list.append(aligned_fea_)

        # reconstruct CSSR 
        # 4.CSSR  cross-scale fusion module
        out = self.recon_trunk(aligned_fea_)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))  # out[0]
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.conv_last(out)
        # skip connection + output
        base = F.interpolate(x_center, scale_factor=4.0, mode='bilinear', align_corners=False)
        out += base
        return out, L1_fea  




class CVSR_V8_woMV(nn.Module):
    def __init__(self, nf=64, nframes=7, fea_ext_RBs=7, SCGs=4, istraining=False):
        super(CVSR_V8_woMV, self).__init__()
        self.nf = nf
        self.center = nframes // 2
        self.istraining = istraining
        self.stride = 4
        #### extraction
        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)   
        self.conv_second = nn.Conv2d(1, nf, 3, 1, 1, bias=True) 
        self.transformer_feature_extraction = PAItransformerSA_2() # PAItransformerSA_1()
        self.conv_expand_fea_r = nn.Conv2d(2*nf, nf, 3, 1, 1, bias=True)
        self.conv_expand_ufs = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.conv_expand_rms = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        #### fusion
        self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True) # 
        #### reconstruction
        self.recon_trunk = SCNet_(nf=nf, SCGroupN=7)  
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)  
        self.upconv2 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_last = nn.Conv2d(nf, 1, 3, 1, 1, bias=True)
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.MV_deform_align = MVDualAttAlignment(64, 64, 3, padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.MV_deform_align = DualAttAlignment_woMV() #  DualAttAlignment(64, 64, 3, padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.RDAB = LLongRangAttention(64) # LongRangAttention() # CrissCrossAttention() # RDAB() #  RSNLAttention()
        #### multi-scale
        self.down = Interpolate(scale_factor=0.5)
        self.up = Interpolate(scale_factor=2.0)
        #### fea pyramid fuse 
        # self.upconv1_L2 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
        # self.upconv1_L3 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
    
    def forward(self, x, mvs0, mvs1, pms, rms, ufs, pre_L1_fea=None):
        B, N, C, H, W = x.size()  # N video frames C=1 # mvs.shape = (b, n, 2, h, w)
        self.H = H
        self.W = W
        x_center = x[:, self.center, :, :, :].contiguous()
        # 1.GCPI
        feas_pyr = []
        # imgs -> feas # multi-scale
        if pre_L1_fea is None:
            L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
            sides_fea = self.conv_second(pms.view(-1, C, H, W))
            L1_fea = self.transformer_feature_extraction(L1_fea, sides_fea)  #   
        else:
            need_add_fea = self.lrelu(self.conv_first(x[:,-1,:,:,:]))
            need_add_sides_fea = self.conv_second(pms[:,-1,:,:,:])
            need_add_L1_fea = self.transformer_feature_extraction(need_add_fea, need_add_sides_fea)  
            need_add_L1_fea = torch.unsqueeze(need_add_L1_fea, 1)
            pre_L1_fea = pre_L1_fea.view(B, N, -1, H, W)
            L1_fea = torch.cat([pre_L1_fea[:,1:,:,:,:], need_add_L1_fea], 1)
            L1_fea = L1_fea.view(B*N, -1, H, W)
        
        feas_pyr.append(L1_fea)
        #  obtain the sequence 7 frames features
        # 2.MV-GSA
        fuse_fea_pyr = [] 
        fea_one_lv = L1_fea.view(B, N, -1, H, W)
        BB, CC, NN, HH, WW = ufs.shape
        if CC != 1:
            ufs = ufs.transpose(1, 2)
            rms = rms.transpose(1, 2)
        # local attention
        for_aligned_fea = []
        aligned_fea = []
        recon_list = []
        
        for i in range(N):     
            if i != N // 2:
                # tmp_mv = (mvs1[:,i,:,:,:].clone())  
                ufs_prior = self.conv_expand_ufs(ufs[:,:,i,:,:].clone())
                rms_prior = self.conv_expand_rms(rms[:,:,i,:,:].clone())
                # spatial-compensate block   
                fea_com = fea_one_lv[:,i,:,:,:].clone() + rms_prior
                _, _, c, h, w = fea_one_lv.size()
                x_n = self.RDAB(rms_prior, fea_com) 
                # temporal-compensate alignment
                fea_one_lv_i = self.conv_expand_fea_r(torch.cat([fea_one_lv[:,i,:,:,:].clone(), x_n],1))
                alg_nbh_fea = self.MV_deform_align(fea_one_lv[:, N//2,:,:,:].clone(), fea_one_lv_i, ufs_prior) 
                aligned_fea.append(alg_nbh_fea) 
            else:
                aligned_fea.append(fea_one_lv[:,i,:,:,:].clone())
        
        # feature fusion
        aligned_fea_out = torch.stack(aligned_fea, dim=1)   # [B, N, C, H, W]
        aligned_fea = aligned_fea_out.view(B, -1, H, W)
        # 3.ATFM 
        aligned_fea_ = self.lrelu(self.tsa_fusion((aligned_fea)))     
        recon_list.append(aligned_fea_)

        # reconstruct CSSR 
        # 4.CSSR  cross-scale fusion module
        out = self.recon_trunk(aligned_fea_)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))  # out[0]
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.conv_last(out)
        # skip connection + output
        base = F.interpolate(x_center, scale_factor=4.0, mode='bilinear', align_corners=False)
        out += base
        return out, L1_fea  



class CVSR_V8_woPd(nn.Module):
    def __init__(self, nf=64, nframes=7, fea_ext_RBs=7, SCGs=4, istraining=False):
        super(CVSR_V8_woPd, self).__init__()
        self.nf = nf
        self.center = nframes // 2
        self.istraining = istraining
        self.stride = 4
        #### extraction
        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)   
        self.conv_second = nn.Conv2d(1, nf, 3, 1, 1, bias=True) 
        self.transformer_feature_extraction = PAItransformerSA_2() # PAItransformerSA_1()
        self.conv_expand_fea_r = nn.Conv2d(2*nf, nf, 3, 1, 1, bias=True)
        # self.conv_expand_ufs = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.conv_expand_rms = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        #### fusion
        self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True) # 
        #### reconstruction
        self.recon_trunk = SCNet_(nf=nf, SCGroupN=7)  
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)  
        self.upconv2 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_last = nn.Conv2d(nf, 1, 3, 1, 1, bias=True)
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.MV_deform_align = MVDualAttAlignment(64, 64, 3, padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.MV_deform_align = DualAttAlignment_woPd() #  DualAttAlignment(64, 64, 3, padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.RDAB = LLongRangAttention(64) # LongRangAttention() # CrissCrossAttention() # RDAB() #  RSNLAttention()
        #### multi-scale
        self.down = Interpolate(scale_factor=0.5)
        self.up = Interpolate(scale_factor=2.0)
        #### fea pyramid fuse 
        # self.upconv1_L2 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
        # self.upconv1_L3 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
    
    def forward(self, x, mvs0, mvs1, pms, rms, ufs, pre_L1_fea=None):
        B, N, C, H, W = x.size()  # N video frames C=1 # mvs.shape = (b, n, 2, h, w)
        self.H = H
        self.W = W
        x_center = x[:, self.center, :, :, :].contiguous()
        # 1.GCPI
        feas_pyr = []
        # imgs -> feas # multi-scale
        if pre_L1_fea is None:
            L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
            sides_fea = self.conv_second(pms.view(-1, C, H, W))
            L1_fea = self.transformer_feature_extraction(L1_fea, sides_fea)  #   
        else:
            need_add_fea = self.lrelu(self.conv_first(x[:,-1,:,:,:]))
            need_add_sides_fea = self.conv_second(pms[:,-1,:,:,:])
            need_add_L1_fea = self.transformer_feature_extraction(need_add_fea, need_add_sides_fea)  
            need_add_L1_fea = torch.unsqueeze(need_add_L1_fea, 1)
            pre_L1_fea = pre_L1_fea.view(B, N, -1, H, W)
            L1_fea = torch.cat([pre_L1_fea[:,1:,:,:,:], need_add_L1_fea], 1)
            L1_fea = L1_fea.view(B*N, -1, H, W)
        
        feas_pyr.append(L1_fea)
        #  obtain the sequence 7 frames features
        # 2.MV-GSA
        fuse_fea_pyr = [] 
        fea_one_lv = L1_fea.view(B, N, -1, H, W)
        BB, CC, NN, HH, WW = ufs.shape
        if CC != 1:
            ufs = ufs.transpose(1, 2)
            rms = rms.transpose(1, 2)
        # local attention
        for_aligned_fea = []
        aligned_fea = []
        recon_list = []
        
        for i in range(N):     
            if i != N // 2:
                tmp_mv = (mvs1[:,i,:,:,:].clone())  
                # ufs_prior = self.conv_expand_ufs(ufs[:,:,i,:,:].clone())
                rms_prior = self.conv_expand_rms(rms[:,:,i,:,:].clone())
                # spatial-compensate block   
                fea_com = fea_one_lv[:,i,:,:,:].clone() + rms_prior
                _, _, c, h, w = fea_one_lv.size()
                x_n = self.RDAB(rms_prior, fea_com) 
                # temporal-compensate alignment
                fea_one_lv_i = self.conv_expand_fea_r(torch.cat([fea_one_lv[:,i,:,:,:].clone(), x_n],1))
                alg_nbh_fea = self.MV_deform_align(fea_one_lv[:, N//2,:,:,:].clone(), fea_one_lv_i, tmp_mv) 
                aligned_fea.append(alg_nbh_fea) 
            else:
                aligned_fea.append(fea_one_lv[:,i,:,:,:].clone())
        
        # feature fusion
        aligned_fea_out = torch.stack(aligned_fea, dim=1)   # [B, N, C, H, W]
        aligned_fea = aligned_fea_out.view(B, -1, H, W)
        # 3.ATFM 
        aligned_fea_ = self.lrelu(self.tsa_fusion((aligned_fea)))     
        recon_list.append(aligned_fea_)

        # reconstruct CSSR 
        # 4.CSSR  cross-scale fusion module
        out = self.recon_trunk(aligned_fea_)
        
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))  # out[0]
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.conv_last(out)
        # skip connection + output
        base = F.interpolate(x_center, scale_factor=4.0, mode='bilinear', align_corners=False)
        out += base
        return out, L1_fea  






class CVSR_V9(nn.Module):
    def __init__(self, nf=64, nframes=7, fea_ext_RBs=7, SCGs=4, istraining=False):
        super(CVSR_V9, self).__init__()
        self.nf = nf
        self.center = nframes // 2
        self.istraining = istraining
        self.stride = 4
        #### extraction
        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)   
        self.conv_second = nn.Conv2d(1, nf, 3, 1, 1, bias=True) 
        self.transformer_feature_extraction = PAItransformerSA_2() # PAItransformerSA()
        self.conv_expand_fea_r = nn.Conv2d(2*nf, nf, 3, 1, 1, bias=True)
        self.conv_expand_ufs = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.conv_expand_rms = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        #### fusion
        self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True) # 
        #### reconstruction
        self.recon_trunk = SCNet_(nf=nf, SCGroupN=7)  # SCNet(nf=nf, SCGroupN=7) 
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)  
        self.upconv2 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_last = nn.Conv2d(nf, 1, 3, 1, 1, bias=True)
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.MV_deform_align = MVDualAttAlignment(64, 64, 3, padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.MV_deform_align = DualAttAlignment() #  DualAttAlignment(64, 64, 3, padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.RDAB = LLongRangAttention_1(64) # LongRangAttention() # CrissCrossAttention() # RDAB() #  RSNLAttention()
        #### multi-scale
        self.down = Interpolate(scale_factor=0.5)
        self.up = Interpolate(scale_factor=2.0)
        #### fea pyramid fuse 
        # self.upconv1_L2 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
        # self.upconv1_L3 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
    
    def forward(self, x, mvs0, mvs1, pms, rms, ufs, pre_L1_fea=None):
        B, N, C, H, W = x.size()  # N video frames C=1 # mvs.shape = (b, n, 2, h, w)
        self.H = H
        self.W = W
        x_center = x[:, self.center, :, :, :].contiguous()
        # 1.GCPI
        feas_pyr = []
        # print('[pms]',x.shape, pms.shape,rms.shape, ufs.shape)
        # imgs -> feas # multi-scale
        if pre_L1_fea is None:
            L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
            sides_fea = self.conv_second(pms.view(-1, C, H, W))
            # featuremap_visual(L1_fea, feature_title='Decoded_1st' )
            # sides_fea = pms.view(-1, C, H, W) # pms.view(-1, C, H, W)
            L1_fea = self.transformer_feature_extraction(L1_fea, sides_fea)  #   
        else:
            need_add_fea = self.lrelu(self.conv_first(x[:,-1,:,:,:]))
            need_add_sides_fea = self.conv_second(pms[:,-1,:,:,:])
            # need_add_sides_fea = pms[:,-1,:,:,:]
            need_add_L1_fea = self.transformer_feature_extraction(need_add_fea, need_add_sides_fea)  
            need_add_L1_fea = torch.unsqueeze(need_add_L1_fea, 1)
            pre_L1_fea = pre_L1_fea.view(B, N, -1, H, W)
            L1_fea = torch.cat([pre_L1_fea[:,1:,:,:,:], need_add_L1_fea], 1)
            L1_fea = L1_fea.view(B*N, -1, H, W)
        
        feas_pyr.append(L1_fea)
        #  obtain the sequence 7 frames features
        # 2.MV-GSA
        fuse_fea_pyr = [] 
        fea_one_lv = L1_fea.view(B, N, -1, H, W)
        BB, CC, NN, HH, WW = ufs.shape
        if CC != 1:
            ufs = ufs.transpose(1, 2)
            rms = rms.transpose(1, 2)
        # local attention
        for_aligned_fea = []
        aligned_fea = []
        recon_list = []
        
        for i in range(N):     
            if i != N // 2:
                tmp_mv = (mvs1[:,i,:,:,:].clone())  
                ufs_prior = self.conv_expand_ufs(ufs[:,:,i,:,:].clone())
                rms_prior = self.conv_expand_rms(rms[:,:,i,:,:].clone())
                # spatial-compensate block   
                fea_com = fea_one_lv[:,i,:,:,:].clone() + rms_prior
                _, _, c, h, w = fea_one_lv.size()
                x_n = self.RDAB(rms_prior, fea_com) 
                # temporal-compensate alignment
                fea_one_lv_i = self.conv_expand_fea_r(torch.cat([fea_one_lv[:,i,:,:,:].clone(), x_n],1))
                alg_nbh_fea = self.MV_deform_align(fea_one_lv[:, N//2,:,:,:].clone(), fea_one_lv_i, ufs_prior, tmp_mv) 
                aligned_fea.append(alg_nbh_fea) 
            else:
                aligned_fea.append(fea_one_lv[:,i,:,:,:].clone())
        
        # feature fusion
        aligned_fea_out = torch.stack(aligned_fea, dim=1)   # [B, N, C, H, W]
        aligned_fea = aligned_fea_out.view(B, -1, H, W)
        # 3.ATFM 
        aligned_fea_ = self.lrelu(self.tsa_fusion((aligned_fea)))     
        recon_list.append(aligned_fea_)

        # reconstruct CSSR 
        # 4.CSSR  cross-scale fusion module
        out = self.recon_trunk(aligned_fea_)
        # featuremap_visual(out, feature_title='out recon' )
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))  # out[0]
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.conv_last(out)
        # skip connection + output
        base = F.interpolate(x_center, scale_factor=4.0, mode='bilinear', align_corners=False)
        out += base
        return out, L1_fea  








class CVSR_V8_flops(nn.Module):
    def __init__(self, nf=64, nframes=7, fea_ext_RBs=7, SCGs=4, istraining=False):
        super(CVSR_V8_flops, self).__init__()
        self.nf = nf
        self.center = nframes // 2
        self.istraining = istraining
        self.stride = 4
        #### extraction
        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)   
        self.conv_second = nn.Conv2d(1, nf, 3, 1, 1, bias=True) 
        self.transformer_feature_extraction = PAItransformerSA_2() # PAItransformerSA_1()
        self.conv_expand_fea_r = nn.Conv2d(2*nf, nf, 3, 1, 1, bias=True)
        self.conv_expand_ufs = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.conv_expand_rms = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        #### fusion
        self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True) # 
        #### reconstruction
        self.recon_trunk = SCNet_(nf=nf, SCGroupN=7)  
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)  
        self.upconv2 = nn.Conv2d(nf, nf * 4, 1, 1, 0, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_last = nn.Conv2d(nf, 1, 3, 1, 1, bias=True)
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.MV_deform_align = MVDualAttAlignment(64, 64, 3, padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.MV_deform_align = DualAttAlignment() #  DualAttAlignment(64, 64, 3, padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.RDAB = LLongRangAttention(64) # LongRangAttention() # CrissCrossAttention() # RDAB() #  RSNLAttention()
        #### multi-scale
        self.down = Interpolate(scale_factor=0.5)
        self.up = Interpolate(scale_factor=2.0)
        #### fea pyramid fuse 
        # self.upconv1_L2 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
        # self.upconv1_L3 = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
    
    def forward(self, XX, pre_L1_fea=None):

        x = XX[:,:7,:,:,:]
        pms = XX[:,7:14,:,:,:]
        rms = XX[:,14:21,:,:,:]
        ufs = XX[:,21:,:,:,:]
        B, N, C, H, W = x.size()  # N video frames C=1 # mvs.shape = (b, n, 2, h, w)
        mvs1 = torch.rand((B, N, 2, H, W)).cuda() # XX[:,7:14,:,:,:]

        
        # B, N, C, H, W = x.size()  # N video frames C=1 # mvs.shape = (b, n, 2, h, w)
        self.H = H
        self.W = W
        x_center = x[:, self.center, :, :, :].contiguous()
        # 1.GCPI
        feas_pyr = []
        # imgs -> feas # multi-scale
        if pre_L1_fea is None:
            L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
            sides_fea = self.conv_second(pms.view(-1, C, H, W))
            L1_fea = self.transformer_feature_extraction(L1_fea, sides_fea)  #   
        else:
            need_add_fea = self.lrelu(self.conv_first(x[:,-1,:,:,:]))
            need_add_sides_fea = self.conv_second(pms[:,-1,:,:,:])
            need_add_L1_fea = self.transformer_feature_extraction(need_add_fea, need_add_sides_fea)  
            need_add_L1_fea = torch.unsqueeze(need_add_L1_fea, 1)
            pre_L1_fea = pre_L1_fea.view(B, N, -1, H, W)
            L1_fea = torch.cat([pre_L1_fea[:,1:,:,:,:], need_add_L1_fea], 1)
            L1_fea = L1_fea.view(B*N, -1, H, W)
        
        feas_pyr.append(L1_fea)
        #  obtain the sequence 7 frames features
        # 2.MV-GSA
        fuse_fea_pyr = [] 
        fea_one_lv = L1_fea.view(B, N, -1, H, W)
        BB, CC, NN, HH, WW = ufs.shape
        if CC != 1:
            ufs = ufs.transpose(1, 2)
            rms = rms.transpose(1, 2)
        # local attention
        for_aligned_fea = []
        aligned_fea = []
        recon_list = []
        
        for i in range(N):     
            if i != N // 2:
                tmp_mv = (mvs1[:,i,:,:,:].clone())  
                ufs_prior = self.conv_expand_ufs(ufs[:,:,i,:,:].clone())
                rms_prior = self.conv_expand_rms(rms[:,:,i,:,:].clone())
                # spatial-compensate block   
                fea_com = fea_one_lv[:,i,:,:,:].clone() + rms_prior
                # featuremap_visual(fea_com, feature_title='fea_com')
                _, _, c, h, w = fea_one_lv.size()
                x_n = self.RDAB(rms_prior, fea_com) 
                # temporal-compensate alignment
                fea_one_lv_i = self.conv_expand_fea_r(torch.cat([fea_one_lv[:,i,:,:,:].clone(), x_n],1))
                # featuremap_visual(fea_one_lv_i, feature_title='align_org')
                alg_nbh_fea = self.MV_deform_align(fea_one_lv[:, N//2,:,:,:].clone(), fea_one_lv_i, ufs_prior, tmp_mv) 
                # featuremap_visual(alg_nbh_fea, feature_title='align result' )
                aligned_fea.append(alg_nbh_fea) 
            else:
                aligned_fea.append(fea_one_lv[:,i,:,:,:].clone())
        
        # feature fusion
        aligned_fea_out = torch.stack(aligned_fea, dim=1)   # [B, N, C, H, W]
        aligned_fea = aligned_fea_out.view(B, -1, H, W)
        # 3.ATFM 
        aligned_fea_ = self.lrelu(self.tsa_fusion((aligned_fea)))     
        recon_list.append(aligned_fea_)

        # reconstruct CSSR 
        # 4.CSSR  cross-scale fusion module
        out = self.recon_trunk(aligned_fea_)
        # featuremap_visual(out, feature_title='out recon' )
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        # featuremap_visual(out, feature_title='out final' )
        out = self.conv_last(out)
        
        # skip connection + output
        base = F.interpolate(x_center, scale_factor=4.0, mode='bilinear', align_corners=False)
        out += base
        return out, L1_fea  

