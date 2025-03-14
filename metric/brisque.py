r"""BRISQUE Metric
Created by: https://github.com/photosynthesis-team/piq/blob/master/piq/brisque.py
Modified by: Jiadi Mo (https://github.com/JiadiMo)
Reference:
    MATLAB codes: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm BRISQUE;
    Pretrained model from: https://github.com/photosynthesis-team/piq/releases/download/v0.4.0/brisque_svm_weights.pt
"""
from typing import Tuple
import torch
import numpy as np
from PIL import Image
import collections.abc
from itertools import repeat
from torch import nn as nn
import time
# from pyiqa.utils.color_util import to_y_channel
# from pyiqa.matlab_utils import imresize
# from .func_util import estimate_ggd_param, estimate_aggd_param, normalize_img_with_guass
# from pyiqa.utils.download_util import load_file_from_url

default_model_urls = {'url': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/brisque_svm_weights.pth'}

def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def exact_padding_2d(x, kernel, stride=1, dilation=1, mode='same'):
    assert len(x.shape) == 4, f'Only support 4D tensor input, but got {x.shape}'
    kernel = to_2tuple(kernel)
    stride = to_2tuple(stride)
    dilation = to_2tuple(dilation)
    b, c, h, w = x.shape
    h2 = math.ceil(h / stride[0])
    w2 = math.ceil(w / stride[1])
    pad_row = (h2 - 1) * stride[0] + (kernel[0] - 1) * dilation[0] + 1 - h
    pad_col = (w2 - 1) * stride[1] + (kernel[1] - 1) * dilation[1] + 1 - w
    pad_l, pad_r, pad_t, pad_b = (pad_col // 2, pad_col - pad_col // 2, pad_row // 2, pad_row - pad_row // 2)

    mode = mode if mode != 'same' else 'constant'
    if mode != 'symmetric':
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)
    elif mode == 'symmetric':
        x = symm_pad(x, (pad_l, pad_r, pad_t, pad_b))

    return x



def safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    r"""Safe sqrt with EPS to ensure numeric stability.

    Args:
        x (torch.Tensor): should be non-negative
    """
    EPS = torch.finfo(x.dtype).eps
    return torch.sqrt(x + EPS)


class ExactPadding2d(nn.Module):
    r"""This function calculate exact padding values for 4D tensor inputs,
    and support the same padding mode as tensorflow.
    Args:
        kernel (int or tuple): kernel size.
        stride (int or tuple): stride size.
        dilation (int or tuple): dilation size, default with 1.
        mode (srt): padding mode can be ('same', 'symmetric', 'replicate', 'circular')
    """

    def __init__(self, kernel, stride=1, dilation=1, mode='same'):
        super().__init__()
        self.kernel = to_2tuple(kernel)
        self.stride = to_2tuple(stride)
        self.dilation = to_2tuple(dilation)
        self.mode = mode

    def forward(self, x):
        return exact_padding_2d(x, self.kernel, self.stride, self.dilation, self.mode)


def imfilter(input, weight, bias=None, stride=1, padding='same', dilation=1, groups=1):
    """imfilter same as matlab.
    Args:
        input (tensor): (b, c, h, w) tensor to be filtered
        weight (tensor): (out_ch, in_ch, kh, kw) filter kernel
        padding (str): padding mode
        dilation (int): dilation of conv
        groups (int): groups of conv
    """
    kernel_size = weight.shape[2:]
    pad_func = ExactPadding2d(kernel_size, stride, dilation, mode=padding)

    return F.conv2d(pad_func(input), weight, bias, stride, dilation=dilation, groups=groups)


def fspecial(size=None, sigma=None, channels=1, filter_type='gaussian'):
    r""" Function same as 'fspecial' in MATLAB, only support gaussian now.
    Args:
        size (int or tuple): size of window
        sigma (float): sigma of gaussian
        channels (int): channels of output
    """
    if filter_type == 'gaussian':
        shape = to_2tuple(size)
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        h = torch.from_numpy(h).float().repeat(channels, 1, 1, 1)
        return h
    else:
        raise NotImplementedError(f'Only support gaussian filter now, got {filter_type}')

def normalize_img_with_guass(
    img: torch.Tensor,
    kernel_size: int = 7,
    sigma: float = 7.0 / 6,
    C: int = 1,
    padding: str = "same",
):
    kernel = fspecial(kernel_size, sigma, 1).to(img)
    mu = imfilter(img, kernel, padding=padding)
    std = imfilter(img**2, kernel, padding=padding)
    sigma = safe_sqrt((std - mu**2).abs())
    img_normalized = (img - mu) / (sigma + C)
    return img_normalized

def estimate_ggd_param(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate general gaussian distribution.

    Args:
        x (Tensor): shape (b, 1, h, w)
    """
    gamma = torch.arange(0.2, 10 + 0.001, 0.001).to(x)
    r_table = (
        torch.lgamma(1.0 / gamma)
        + torch.lgamma(3.0 / gamma)
        - 2 * torch.lgamma(2.0 / gamma)
    ).exp()
    r_table = r_table.repeat(x.size(0), 1)

    sigma_sq = x.pow(2).mean(dim=(-1, -2))
    sigma = sigma_sq.sqrt().squeeze(dim=-1)

    assert not torch.isclose(
        sigma, torch.zeros_like(sigma)
    ).all(), "Expected image with non zero variance of pixel values"

    E = x.abs().mean(dim=(-1, -2))
    rho = sigma_sq / E**2

    indexes = (rho - r_table).abs().argmin(dim=-1)
    solution = gamma[indexes]
    return solution, sigma


def estimate_aggd_param(
    block: torch.Tensor, return_sigma=False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.
    Args:
        block (Tensor): Image block with shape (b, 1, h, w).
    Returns:
        Tensor: alpha, beta_l and beta_r for the AGGD distribution
        (Estimating the parames in Equation 7 in the paper).
    """
    gam = torch.arange(0.2, 10 + 0.001, 0.001).to(block)
    r_gam = (
        2 * torch.lgamma(2.0 / gam)
        - (torch.lgamma(1.0 / gam) + torch.lgamma(3.0 / gam))
    ).exp()
    r_gam = r_gam.repeat(block.shape[0], 1)

    mask_left = block < 0
    mask_right = block > 0
    count_left = mask_left.sum(dim=(-1, -2), dtype=torch.float32)
    count_right = mask_right.sum(dim=(-1, -2), dtype=torch.float32)

    left_std = torch.sqrt((block * mask_left).pow(2).sum(dim=(-1, -2)) / (count_left))
    right_std = torch.sqrt(
        (block * mask_right).pow(2).sum(dim=(-1, -2)) / (count_right)
    )

    gammahat = left_std / right_std
    rhat = block.abs().mean(dim=(-1, -2)).pow(2) / block.pow(2).mean(dim=(-1, -2))
    rhatnorm = (rhat * (gammahat.pow(3) + 1) * (gammahat + 1)) / (
        gammahat.pow(2) + 1
    ).pow(2)
    array_position = (r_gam - rhatnorm).abs().argmin(dim=-1)

    alpha = gam[array_position]
    beta_l = (
        left_std.squeeze(-1)
        * (torch.lgamma(1 / alpha) - torch.lgamma(3 / alpha)).exp().sqrt()
    )
    beta_r = (
        right_std.squeeze(-1)
        * (torch.lgamma(1 / alpha) - torch.lgamma(3 / alpha)).exp().sqrt()
    )

    if return_sigma:
        return alpha, left_std.squeeze(-1), right_std.squeeze(-1)
    else:
        return alpha, beta_l, beta_r

from typing import Union, Dict
import torch

import math
import typing

import torch
from torch.nn import functional as F

__all__ = ['imresize']

_I = typing.Optional[int]
_D = typing.Optional[torch.dtype]


def nearest_contribution(x: torch.Tensor) -> torch.Tensor:
    range_around_0 = torch.logical_and(x.gt(-0.5), x.le(0.5))
    cont = range_around_0.to(dtype=x.dtype)
    return cont


def linear_contribution(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs()
    range_01 = ax.le(1)
    cont = (1 - ax) * range_01.to(dtype=x.dtype)
    return cont


def cubic_contribution(x: torch.Tensor, a: float = -0.5) -> torch.Tensor:
    ax = x.abs()
    ax2 = ax * ax
    ax3 = ax * ax2

    range_01 = ax.le(1)
    range_12 = torch.logical_and(ax.gt(1), ax.le(2))

    cont_01 = (a + 2) * ax3 - (a + 3) * ax2 + 1
    cont_01 = cont_01 * range_01.to(dtype=x.dtype)

    cont_12 = (a * ax3) - (5 * a * ax2) + (8 * a * ax) - (4 * a)
    cont_12 = cont_12 * range_12.to(dtype=x.dtype)

    cont = cont_01 + cont_12
    return cont


def gaussian_contribution(x: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    range_3sigma = (x.abs() <= 3 * sigma + 1)
    # Normalization will be done after
    cont = torch.exp(-x.pow(2) / (2 * sigma**2))
    cont = cont * range_3sigma.to(dtype=x.dtype)
    return cont


def discrete_kernel(kernel: str, scale: float, antialiasing: bool = True) -> torch.Tensor:
    '''
    For downsampling with integer scale only.
    '''
    downsampling_factor = int(1 / scale)
    if kernel == 'cubic':
        kernel_size_orig = 4
    else:
        raise ValueError('Pass!')

    if antialiasing:
        kernel_size = kernel_size_orig * downsampling_factor
    else:
        kernel_size = kernel_size_orig

    if downsampling_factor % 2 == 0:
        a = kernel_size_orig * (0.5 - 1 / (2 * kernel_size))
    else:
        kernel_size -= 1
        a = kernel_size_orig * (0.5 - 1 / (kernel_size + 1))

    with torch.no_grad():
        r = torch.linspace(-a, a, steps=kernel_size)
        k = cubic_contribution(r).view(-1, 1)
        k = torch.matmul(k, k.t())
        k /= k.sum()

    return k


def reflect_padding(x: torch.Tensor, dim: int, pad_pre: int, pad_post: int) -> torch.Tensor:
    '''
    Apply reflect padding to the given Tensor.
    Note that it is slightly different from the PyTorch functional.pad,
    where boundary elements are used only once.
    Instead, we follow the MATLAB implementation
    which uses boundary elements twice.
    For example,
    [a, b, c, d] would become [b, a, b, c, d, c] with the PyTorch implementation,
    while our implementation yields [a, a, b, c, d, d].
    '''
    b, c, h, w = x.size()
    if dim == 2 or dim == -2:
        padding_buffer = x.new_zeros(b, c, h + pad_pre + pad_post, w)
        padding_buffer[..., pad_pre:(h + pad_pre), :].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1, :].copy_(x[..., p, :])
        for p in range(pad_post):
            padding_buffer[..., h + pad_pre + p, :].copy_(x[..., -(p + 1), :])
    else:
        padding_buffer = x.new_zeros(b, c, h, w + pad_pre + pad_post)
        padding_buffer[..., pad_pre:(w + pad_pre)].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1].copy_(x[..., p])
        for p in range(pad_post):
            padding_buffer[..., w + pad_pre + p].copy_(x[..., -(p + 1)])

    return padding_buffer


def padding(x: torch.Tensor,
            dim: int,
            pad_pre: int,
            pad_post: int,
            padding_type: typing.Optional[str] = 'reflect') -> torch.Tensor:
    if padding_type is None:
        return x
    elif padding_type == 'reflect':
        x_pad = reflect_padding(x, dim, pad_pre, pad_post)
    else:
        raise ValueError('{} padding is not supported!'.format(padding_type))

    return x_pad


def get_padding(base: torch.Tensor, kernel_size: int, x_size: int) -> typing.Tuple[int, int, torch.Tensor]:
    base = base.long()
    r_min = base.min()
    r_max = base.max() + kernel_size - 1

    if r_min <= 0:
        pad_pre = -r_min
        pad_pre = pad_pre.item()
        base += pad_pre
    else:
        pad_pre = 0

    if r_max >= x_size:
        pad_post = r_max - x_size + 1
        pad_post = pad_post.item()
    else:
        pad_post = 0

    return pad_pre, pad_post, base


def get_weight(dist: torch.Tensor,
               kernel_size: int,
               kernel: str = 'cubic',
               sigma: float = 2.0,
               antialiasing_factor: float = 1) -> torch.Tensor:
    buffer_pos = dist.new_zeros(kernel_size, len(dist))
    for idx, buffer_sub in enumerate(buffer_pos):
        buffer_sub.copy_(dist - idx)

    # Expand (downsampling) / Shrink (upsampling) the receptive field.
    buffer_pos *= antialiasing_factor
    if kernel == 'cubic':
        weight = cubic_contribution(buffer_pos)
    elif kernel == 'gaussian':
        weight = gaussian_contribution(buffer_pos, sigma=sigma)
    else:
        raise ValueError('{} kernel is not supported!'.format(kernel))

    weight /= weight.sum(dim=0, keepdim=True)
    return weight


def reshape_tensor(x: torch.Tensor, dim: int, kernel_size: int) -> torch.Tensor:
    # Resize height
    if dim == 2 or dim == -2:
        k = (kernel_size, 1)
        h_out = x.size(-2) - kernel_size + 1
        w_out = x.size(-1)
    # Resize width
    else:
        k = (1, kernel_size)
        h_out = x.size(-2)
        w_out = x.size(-1) - kernel_size + 1

    unfold = F.unfold(x, k)
    unfold = unfold.view(unfold.size(0), -1, h_out, w_out)
    return unfold


def reshape_input(x: torch.Tensor) -> typing.Tuple[torch.Tensor, _I, _I, int, int]:
    if x.dim() == 4:
        b, c, h, w = x.size()
    elif x.dim() == 3:
        c, h, w = x.size()
        b = None
    elif x.dim() == 2:
        h, w = x.size()
        b = c = None
    else:
        raise ValueError('{}-dim Tensor is not supported!'.format(x.dim()))

    x = x.view(-1, 1, h, w)
    return x, b, c, h, w


def reshape_output(x: torch.Tensor, b: _I, c: _I) -> torch.Tensor:
    rh = x.size(-2)
    rw = x.size(-1)
    # Back to the original dimension
    if b is not None:
        x = x.view(b, c, rh, rw)  # 4-dim
    else:
        if c is not None:
            x = x.view(c, rh, rw)  # 3-dim
        else:
            x = x.view(rh, rw)  # 2-dim

    return x


def cast_input(x: torch.Tensor) -> typing.Tuple[torch.Tensor, _D]:
    if x.dtype != torch.float32 or x.dtype != torch.float64:
        dtype = x.dtype
        x = x.float()
    else:
        dtype = None

    return x, dtype


def cast_output(x: torch.Tensor, dtype: _D) -> torch.Tensor:
    if dtype is not None:
        if not dtype.is_floating_point:
            x = x - x.detach() + x.round()
        # To prevent over/underflow when converting types
        if dtype is torch.uint8:
            x = x.clamp(0, 255)

        x = x.to(dtype=dtype)

    return x


def resize_1d(x: torch.Tensor,
              dim: int,
              size: int,
              scale: float,
              kernel: str = 'cubic',
              sigma: float = 2.0,
              padding_type: str = 'reflect',
              antialiasing: bool = True) -> torch.Tensor:
    '''
    Args:
        x (torch.Tensor): A torch.Tensor of dimension (B x C, 1, H, W).
        dim (int):
        scale (float):
        size (int):
    Return:
    '''
    # Identity case
    if scale == 1:
        return x

    # Default bicubic kernel with antialiasing (only when downsampling)
    if kernel == 'cubic':
        kernel_size = 4
    else:
        kernel_size = math.floor(6 * sigma)

    if antialiasing and (scale < 1):
        antialiasing_factor = scale
        kernel_size = math.ceil(kernel_size / antialiasing_factor)
    else:
        antialiasing_factor = 1

    # We allow margin to both sizes
    kernel_size += 2

    # Weights only depend on the shape of input and output,
    # so we do not calculate gradients here.
    with torch.no_grad():
        pos = torch.linspace(
            0,
            size - 1,
            steps=size,
            dtype=x.dtype,
            device=x.device,
        )
        pos = (pos + 0.5) / scale - 0.5
        base = pos.floor() - (kernel_size // 2) + 1
        dist = pos - base
        weight = get_weight(
            dist,
            kernel_size,
            kernel=kernel,
            sigma=sigma,
            antialiasing_factor=antialiasing_factor,
        )
        pad_pre, pad_post, base = get_padding(base, kernel_size, x.size(dim))

    # To backpropagate through x
    x_pad = padding(x, dim, pad_pre, pad_post, padding_type=padding_type)
    unfold = reshape_tensor(x_pad, dim, kernel_size)
    # Subsampling first
    if dim == 2 or dim == -2:
        sample = unfold[..., base, :]
        weight = weight.view(1, kernel_size, sample.size(2), 1)
    else:
        sample = unfold[..., base]
        weight = weight.view(1, kernel_size, 1, sample.size(3))

    # Apply the kernel
    x = sample * weight
    x = x.sum(dim=1, keepdim=True)
    return x


def downsampling_2d(x: torch.Tensor, k: torch.Tensor, scale: int, padding_type: str = 'reflect') -> torch.Tensor:
    c = x.size(1)
    k_h = k.size(-2)
    k_w = k.size(-1)

    k = k.to(dtype=x.dtype, device=x.device)
    k = k.view(1, 1, k_h, k_w)
    k = k.repeat(c, c, 1, 1)
    e = torch.eye(c, dtype=k.dtype, device=k.device, requires_grad=False)
    e = e.view(c, c, 1, 1)
    k = k * e

    pad_h = (k_h - scale) // 2
    pad_w = (k_w - scale) // 2
    x = padding(x, -2, pad_h, pad_h, padding_type=padding_type)
    x = padding(x, -1, pad_w, pad_w, padding_type=padding_type)
    y = F.conv2d(x, k, padding=0, stride=scale)
    return y


def imresize(x: torch.Tensor,
             scale: typing.Optional[float] = None,
             sizes: typing.Optional[typing.Tuple[int, int]] = None,
             kernel: typing.Union[str, torch.Tensor] = 'cubic',
             sigma: float = 2,
             rotation_degree: float = 0,
             padding_type: str = 'reflect',
             antialiasing: bool = True) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor):
        scale (float):
        sizes (tuple(int, int)):
        kernel (str, default='cubic'):
        sigma (float, default=2):
        rotation_degree (float, default=0):
        padding_type (str, default='reflect'):
        antialiasing (bool, default=True):
    Return:
        torch.Tensor:
    """
    if scale is None and sizes is None:
        raise ValueError('One of scale or sizes must be specified!')
    if scale is not None and sizes is not None:
        raise ValueError('Please specify scale or sizes to avoid conflict!')

    x, b, c, h, w = reshape_input(x)

    if sizes is None and scale is not None:
        '''
        # Check if we can apply the convolution algorithm
        scale_inv = 1 / scale
        if isinstance(kernel, str) and scale_inv.is_integer():
            kernel = discrete_kernel(kernel, scale, antialiasing=antialiasing)
        elif isinstance(kernel, torch.Tensor) and not scale_inv.is_integer():
            raise ValueError(
                'An integer downsampling factor '
                'should be used with a predefined kernel!'
            )
        '''
        # Determine output size
        sizes = (math.ceil(h * scale), math.ceil(w * scale))
        scales = (scale, scale)

    if scale is None and sizes is not None:
        scales = (sizes[0] / h, sizes[1] / w)

    x, dtype = cast_input(x)

    if isinstance(kernel, str) and sizes is not None:
        # Core resizing module
        x = resize_1d(
            x,
            -2,
            size=sizes[0],
            scale=scales[0],
            kernel=kernel,
            sigma=sigma,
            padding_type=padding_type,
            antialiasing=antialiasing)
        x = resize_1d(
            x,
            -1,
            size=sizes[1],
            scale=scales[1],
            kernel=kernel,
            sigma=sigma,
            padding_type=padding_type,
            antialiasing=antialiasing)
    elif isinstance(kernel, torch.Tensor) and scale is not None:
        x = downsampling_2d(x, kernel, scale=int(1 / scale))

    x = reshape_output(x, b, c)
    x = cast_output(x, dtype)
    return x


def safe_frac_pow(x: torch.Tensor, p) -> torch.Tensor:
    EPS = torch.finfo(x.dtype).eps
    return torch.sign(x) * torch.abs(x + EPS).pow(p)


def to_y_channel(img: torch.Tensor, out_data_range: float = 1., color_space: str = 'yiq') -> torch.Tensor:
    r"""Change to Y channel
    Args:
        image tensor: tensor with shape (N, 3, H, W) in range [0, 1].
    Returns:
        image tensor: Y channel of the input tensor
    """
    assert img.ndim == 4 and img.shape[1] == 3, 'input image tensor should be RGB image batches with shape (N, 3, H, W)'
    color_space = color_space.lower()
    if color_space == 'yiq':
        img = rgb2yiq(img)
    elif color_space == 'ycbcr':
        img = rgb2ycbcr(img)
    elif color_space == 'lhm':
        img = rgb2lhm(img)
    out_img = img[:, [0], :, :] * out_data_range
    if out_data_range >= 255:
        # differentiable round with pytorch
        out_img = out_img - out_img.detach() + out_img.round()
    return out_img


def rgb2ycbcr(x: torch.Tensor) -> torch.Tensor:
    r"""Convert a batch of RGB images to a batch of YCbCr images

    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB color space, range [0, 1].

    Returns:
        Batch of images with shape (N, 3, H, W). YCbCr color space.
    """
    weights_rgb_to_ycbcr = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                         [24.966, 112.0, -18.214]]).to(x)
    bias_rgb_to_ycbcr = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(x)
    x_ycbcr = torch.matmul(x.permute(0, 2, 3, 1), weights_rgb_to_ycbcr).permute(0, 3, 1, 2) \
            + bias_rgb_to_ycbcr
    x_ycbcr = x_ycbcr / 255.
    return x_ycbcr


def ycbcr2rgb(x: torch.Tensor) -> torch.Tensor:
    r"""Convert a batch of YCbCr images to a batch of RGB images

    It implements the inversion of the above rgb2ycbcr function.

    Args:
        x: Batch of images with shape (N, 3, H, W). YCbCr color space, range [0, 1].

    Returns:
        Batch of images with shape (N, 3, H, W). RGB color space.
    """
    x = x * 255.
    weights_ycbcr_to_rgb = 255. * torch.tensor([[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                                                [0.00625893, -0.00318811, 0]]).to(x)
    bias_ycbcr_to_rgb = torch.tensor([-222.921, 135.576, -276.836]).view(1, 3, 1, 1).to(x)
    x_rgb = torch.matmul(x.permute(0, 2, 3, 1), weights_ycbcr_to_rgb).permute(0, 3, 1, 2) \
            + bias_ycbcr_to_rgb
    x_rgb = x_rgb / 255.
    return x_rgb


def rgb2lmn(x: torch.Tensor) -> torch.Tensor:
    r"""Convert a batch of RGB images to a batch of LMN images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). LMN colour space.
    """
    weights_rgb_to_lmn = torch.tensor([[0.06, 0.63, 0.27], [0.30, 0.04, -0.35], [0.34, -0.6, 0.17]]).t().to(x)
    x_lmn = torch.matmul(x.permute(0, 2, 3, 1), weights_rgb_to_lmn).permute(0, 3, 1, 2)
    return x_lmn


def rgb2xyz(x: torch.Tensor) -> torch.Tensor:
    r"""Convert a batch of RGB images to a batch of XYZ images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). XYZ colour space.
    """
    mask_below = (x <= 0.04045).to(x)
    mask_above = (x > 0.04045).to(x)

    tmp = x / 12.92 * mask_below + torch.pow((x + 0.055) / 1.055, 2.4) * mask_above

    weights_rgb_to_xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                                       [0.0193339, 0.1191920, 0.9503041]]).to(x)

    x_xyz = torch.matmul(tmp.permute(0, 2, 3, 1), weights_rgb_to_xyz.t()).permute(0, 3, 1, 2)
    return x_xyz


def xyz2lab(x: torch.Tensor, illuminant: str = 'D50', observer: str = '2') -> torch.Tensor:
    r"""Convert a batch of XYZ images to a batch of LAB images

    Args:
        x: Batch of images with shape (N, 3, H, W). XYZ colour space.
        illuminant: {“A”, “D50”, “D55”, “D65”, “D75”, “E”}, optional. The name of the illuminant.
        observer: {“2”, “10”}, optional. The aperture angle of the observer.

    Returns:
        Batch of images with shape (N, 3, H, W). LAB colour space.
    """
    epsilon = 0.008856
    kappa = 903.3
    illuminants: Dict[str, Dict] = \
        {'A': {'2': (1.098466069456375, 1, 0.3558228003436005),
               '10': (1.111420406956693, 1, 0.3519978321919493)},
         'D50': {'2': (0.9642119944211994, 1, 0.8251882845188288),
                 '10': (0.9672062750333777, 1, 0.8142801513128616)},
         'D55': {'2': (0.956797052643698, 1, 0.9214805860173273),
                 '10': (0.9579665682254781, 1, 0.9092525159847462)},
         'D65': {'2': (0.95047, 1., 1.08883),  # This was: `lab_ref_white`
                 '10': (0.94809667673716, 1, 1.0730513595166162)},
         'D75': {'2': (0.9497220898840717, 1, 1.226393520724154),
                 '10': (0.9441713925645873, 1, 1.2064272211720228)},
         'E': {'2': (1.0, 1.0, 1.0),
               '10': (1.0, 1.0, 1.0)}}

    illuminants_to_use = torch.tensor(illuminants[illuminant][observer]).to(x).view(1, 3, 1, 1)

    tmp = x / illuminants_to_use

    mask_below = tmp <= epsilon
    mask_above = tmp > epsilon
    tmp = safe_frac_pow(tmp, 1. / 3.) * mask_above + (kappa * tmp + 16.) / 116. * mask_below

    weights_xyz_to_lab = torch.tensor([[0, 116., 0], [500., -500., 0], [0, 200., -200.]]).to(x)
    bias_xyz_to_lab = torch.tensor([-16., 0., 0.]).to(x).view(1, 3, 1, 1)

    x_lab = torch.matmul(tmp.permute(0, 2, 3, 1), weights_xyz_to_lab.t()).permute(0, 3, 1, 2) + bias_xyz_to_lab
    return x_lab


def rgb2lab(x: torch.Tensor, data_range: Union[int, float] = 255) -> torch.Tensor:
    r"""Convert a batch of RGB images to a batch of LAB images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.
        data_range: dynamic range of the input image.

    Returns:
        Batch of images with shape (N, 3, H, W). LAB colour space.
    """
    return xyz2lab(rgb2xyz(x / float(data_range)))


def rgb2yiq(x: torch.Tensor) -> torch.Tensor:
    r"""Convert a batch of RGB images to a batch of YIQ images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). YIQ colour space.
    """
    yiq_weights = torch.tensor([[0.299, 0.587, 0.114], [0.5959, -0.2746, -0.3213], [0.2115, -0.5227, 0.3112]]).t().to(x)
    x_yiq = torch.matmul(x.permute(0, 2, 3, 1), yiq_weights).permute(0, 3, 1, 2)
    return x_yiq


def rgb2lhm(x: torch.Tensor) -> torch.Tensor:
    r"""Convert a batch of RGB images to a batch of LHM images

    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.

    Returns:
        Batch of images with shape (N, 3, H, W). LHM colour space.

    Reference:
        https://arxiv.org/pdf/1608.07433.pdf
    """
    lhm_weights = torch.tensor([[0.2989, 0.587, 0.114], [0.3, 0.04, -0.35], [0.34, -0.6, 0.17]]).t().to(x)
    x_lhm = torch.matmul(x.permute(0, 2, 3, 1), lhm_weights).permute(0, 3, 1, 2)
    return x_lhm

def brisque(x: torch.Tensor,
            kernel_size: int = 7,
            kernel_sigma: float = 7 / 6,
            test_y_channel: bool = True,
            pretrained_model_path: str = None) -> torch.Tensor:
    r"""Interface of BRISQUE index.

    Args:
        - x: An input tensor. Shape :math:`(N, C, H, W)`.
        - kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        - kernel_sigma: Sigma of normal distribution.
        - data_range: Maximum value range of images (usually 1.0 or 255).
        - to_y_channel: Whether use the y-channel of YCBCR.
        pretrained_model_path: The model path.

    Returns:
        Value of BRISQUE index.

    References:
        Mittal, Anish, Anush Krishna Moorthy, and Alan Conrad Bovik.
        "No-reference image quality assessment in the spatial domain."
        IEEE Transactions on image processing 21, no. 12 (2012): 4695-4708.

    """
    if test_y_channel and x.shape[1] == 3:
        x = to_y_channel(x, 255.)
    else:
        x = x * 255

    features = []
    num_of_scales = 2
    for _ in range(num_of_scales):
        features.append(natural_scene_statistics(x, kernel_size, kernel_sigma))
        x = imresize(x, scale=0.5, antialiasing=True)

    features = torch.cat(features, dim=-1)
    scaled_features = scale_features(features)

    if pretrained_model_path:
        sv_coef, sv = torch.load(pretrained_model_path)
        sv_coef = sv_coef.to(x)
        sv = sv.to(x)

    # gamma and rho are SVM model parameters taken from official implementation of BRISQUE on MATLAB
    # Source: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm
    gamma = 0.05
    rho = -153.591
    sv.t_()
    kernel_features = rbf_kernel(features=scaled_features, sv=sv, gamma=gamma)
    score = kernel_features @ sv_coef
    return score - rho


def natural_scene_statistics(luma: torch.Tensor, kernel_size: int = 7, sigma: float = 7. / 6) -> torch.Tensor:

    luma_nrmlzd = normalize_img_with_guass(luma, kernel_size, sigma, padding='same')
    alpha, sigma = estimate_ggd_param(luma_nrmlzd)
    features = [alpha, sigma.pow(2)]

    shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]

    for shift in shifts:
        shifted_luma_nrmlzd = torch.roll(luma_nrmlzd, shifts=shift, dims=(-2, -1))
        alpha, sigma_l, sigma_r = estimate_aggd_param(luma_nrmlzd * shifted_luma_nrmlzd, return_sigma=True)
        eta = (sigma_r - sigma_l
               ) * torch.exp(torch.lgamma(2. / alpha) - (torch.lgamma(1. / alpha) + torch.lgamma(3. / alpha)) / 2)
        features.extend((alpha, eta, sigma_l.pow(2), sigma_r.pow(2)))

    return torch.stack(features, dim=-1)


def scale_features(features: torch.Tensor) -> torch.Tensor:
    lower_bound = -1
    upper_bound = 1
    # Feature range is taken from official implementation of BRISQUE on MATLAB.
    # Source: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm
    feature_ranges = torch.tensor([[0.338, 10], [0.017204, 0.806612], [0.236, 1.642], [-0.123884, 0.20293],
                                   [0.000155, 0.712298], [0.001122, 0.470257], [0.244, 1.641], [-0.123586, 0.179083],
                                   [0.000152, 0.710456], [0.000975, 0.470984], [0.249, 1.555], [-0.135687, 0.100858],
                                   [0.000174, 0.684173], [0.000913, 0.534174], [0.258, 1.561], [-0.143408, 0.100486],
                                   [0.000179, 0.685696], [0.000888, 0.536508], [0.471, 3.264], [0.012809, 0.703171],
                                   [0.218, 1.046], [-0.094876, 0.187459], [1.5e-005, 0.442057], [0.001272, 0.40803],
                                   [0.222, 1.042], [-0.115772, 0.162604], [1.6e-005, 0.444362], [0.001374, 0.40243],
                                   [0.227, 0.996], [-0.117188, 0.09832299999999999], [3e-005, 0.531903],
                                   [0.001122, 0.369589], [0.228, 0.99], [-0.12243, 0.098658], [2.8e-005, 0.530092],
                                   [0.001118, 0.370399]]).to(features)

    scaled_features = lower_bound + (upper_bound - lower_bound) * (features - feature_ranges[..., 0]) / (
        feature_ranges[..., 1] - feature_ranges[..., 0])

    return scaled_features


def rbf_kernel(features: torch.Tensor, sv: torch.Tensor, gamma: float = 0.05) -> torch.Tensor:
    dist = (features.unsqueeze(dim=-1) - sv.unsqueeze(dim=0)).pow(2).sum(dim=1)
    return torch.exp(-dist * gamma)


class BRISQUE(torch.nn.Module):
    r"""Creates a criterion that measures the BRISQUE score.
    Args:
        - kernel_size (int): By default, the mean and covariance of a pixel is obtained
        by convolution with given filter_size. Must be an odd value.
        - kernel_sigma (float): Standard deviation for Gaussian kernel.
        - to_y_channel (bool): Whether use the y-channel of YCBCR.
        - pretrained_model_path (str): The model path.
    """

    def __init__(self,
                 kernel_size: int = 7,
                 kernel_sigma: float = 7 / 6,
                 test_y_channel: bool = True,
                 pretrained_model_path: str = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        # This check might look redundant because kernel size is checked within the brisque function anyway.
        # However, this check allows to fail fast when the loss is being initialised and training has not been started.
        assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'
        assert test_y_channel, f'Only [test_y_channel=True] is supported for current BRISQUE model, which is taken directly from official codes: https://github.com/utlive/BRISQUE.'
        self.kernel_sigma = kernel_sigma
        self.test_y_channel = test_y_channel
        if pretrained_model_path is not None:
            self.pretrained_model_path = pretrained_model_path
        else:
            self.pretrained_model_path =  '/share3/home/zqiang/CVSR_train/metric/weight/brisque_svm_weights.pth' #  load_file_from_url(default_model_urls['url'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computation of BRISQUE score as a loss function.
        Args:
            x: An input tensor with (N, C, H, W) shape. RGB channel order for colour images.
        Returns:
            Value of BRISQUE metric.
        """
        return brisque(
            x,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            test_y_channel=self.test_y_channel,
            pretrained_model_path=self.pretrained_model_path)


if __name__ == "__main__":
    
    ref = torch.Tensor(np.array(Image.open('/share3/home/zqiang/CVSR_train/metric/test_imgs/bikes.bmp')).transpose(2,0,1)).unsqueeze(0) # .convert('LA'))[:,:,0] # ref
    dis = torch.Tensor(np.array(Image.open('/share3/home/zqiang/CVSR_train/metric/test_imgs/bikes_distorted.bmp')).transpose(2,0,1)).unsqueeze(0) # .convert('LA'))[:,:,0] # dis
    brisque_model = BRISQUE()
    strT = time.time()
    print('BRISQUE of ref bikes image is: %0.4f'% brisque_model(ref))
    Sumtime = time.time()-strT
    print('BRISQUE of ref bikes image time: %0.4f'% Sumtime)
    print('BRISQUE of dis bikes image is: %0.4f'% brisque_model(dis))

    ref = torch.Tensor(np.array(Image.open('/share3/home/zqiang/CVSR_train/metric/test_imgs/parrots.bmp')).transpose(2,0,1)).unsqueeze(0) # .convert('LA'))[:,:,0] # ref
    dis = torch.Tensor(np.array(Image.open('/share3/home//zqiang/CVSR_train/metric/test_imgs/parrots_distorted.bmp')).transpose(2,0,1)).unsqueeze(0)
   
    print('BRISQUE of ref parrot image is: %0.4f'% brisque_model(ref))
    print('BRISQUE of dis parrot image is: %0.4f'% brisque_model(dis))