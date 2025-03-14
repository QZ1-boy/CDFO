r"""NIQE and ILNIQE Metrics
NIQE Metric
    Created by: https://github.com/xinntao/BasicSR/blob/5668ba75eb8a77e8d2dd46746a36fee0fbb0fdcd/basicsr/metrics/niqe.py
    Modified by: Jiadi Mo (https://github.com/JiadiMo)
    Reference:
        MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

ILNIQE Metric
    Created by: Chaofeng Chen (https://github.com/chaofengc)
    Reference:
        - Python codes: https://github.com/IceClear/IL-NIQE/blob/master/IL-NIQE.py
        - Matlab codes: https://www4.comp.polyu.edu.hk/~cslzhang/IQA/ILNIQE/Files/ILNIQE.zip
"""

import math
import numpy as np
import scipy
import scipy.io
import torch
import typing
from typing import Tuple
from typing import Union, Dict
import numpy as np
from PIL import Image
import collections.abc
from itertools import repeat
from torch import nn as nn
import torch.nn.functional as F
import time

__all__ = ['imresize']

_I = typing.Optional[int]
_D = typing.Optional[torch.dtype]
# from pyiqa.matlab_utils import imresize, fspecial, conv2d, imfilter, fitweibull, nancov, nanmean, blockproc
# from .func_util import estimate_aggd_param, normalize_img_with_guass, diff_round
# from pyiqa.archs.fsim_arch import _construct_filters

default_model_urls = {
    'url': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/niqe_modelparameters.mat',
    'niqe': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/niqe_modelparameters.mat',
    'ilniqe': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/ILNIQE_templateModel.mat',
}


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


def cast_input(x: torch.Tensor) -> typing.Tuple[torch.Tensor, _D]:
    if x.dtype != torch.float32 or x.dtype != torch.float64:
        dtype = x.dtype
        x = x.float()
    else:
        dtype = None

    return x, dtype


def diff_round(x: torch.Tensor) -> torch.Tensor:
    r"""Differentiable round."""
    return x - x.detach() + x.round()



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

def fitweibull(x, iters=50, eps=1e-2):
    """Simulate wblfit function in matlab.

    ref: https://github.com/mlosch/python-weibullfit/blob/master/weibull/backend_pytorch.py

    Fits a 2-parameter Weibull distribution to the given data using maximum-likelihood estimation.
    :param x (tensor): (B, N), batch of samples from an (unknown) distribution. Each value must satisfy x > 0.
    :param iters: Maximum number of iterations
    :param eps: Stopping criterion. Fit is stopped ff the change within two iterations is smaller than eps.
    :param use_cuda: Use gpu
    :return: Tuple (Shape, Scale) which can be (NaN, NaN) if a fit is impossible.
        Impossible fits may be due to 0-values in x.
    """
    ln_x = torch.log(x)
    k = 1.2 / torch.std(ln_x, dim=1, keepdim=True)
    k_t_1 = k

    for t in range(iters):
        # Partial derivative df/dk
        x_k = x**k.repeat(1, x.shape[1])
        x_k_ln_x = x_k * ln_x
        ff = torch.sum(x_k_ln_x, dim=-1, keepdim=True)
        fg = torch.sum(x_k, dim=-1, keepdim=True)
        f1 = torch.mean(ln_x, dim=-1, keepdim=True)
        f = ff / fg - f1 - (1.0 / k)

        ff_prime = torch.sum(x_k_ln_x * ln_x, dim=-1, keepdim=True)
        fg_prime = ff
        f_prime = (ff_prime / fg - (ff / fg * fg_prime / fg)) + (1. / (k * k))

        # Newton-Raphson method k = k - f(k;x)/f'(k;x)
        k = k - f / f_prime
        error = torch.abs(k - k_t_1).max().item()
        if error < eps:
            break
        k_t_1 = k

    # Lambda (scale) can be calculated directly
    lam = torch.mean(x**k.repeat(1, x.shape[1]), dim=-1, keepdim=True)**(1.0 / k)

    return torch.cat((k, lam), dim=1)  # Shape (SC), Scale (FE)

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



def _construct_filters(x: torch.Tensor,
                       scales: int = 4,
                       orientations: int = 4,
                       min_length: int = 6,
                       mult: int = 2,
                       sigma_f: float = 0.55,
                       delta_theta: float = 1.2,
                       k: float = 2.0,
                       use_lowpass_filter=True):
    """Creates a stack of filters used for computation of phase congruensy maps

    Args:
        - x: Tensor. Shape :math:`(N, 1, H, W)`.
        - scales: Number of wavelets
        - orientations: Number of filter orientations
        - min_length: Wavelength of smallest scale filter
        - mult: Scaling factor between successive filters
        - sigma_f: Ratio of the standard deviation of the Gaussian
        describing the log Gabor filter's transfer function
        in the frequency domain to the filter center frequency.
        - delta_theta: Ratio of angular interval between filter orientations
        and the standard deviation of the angular Gaussian function
        used to construct filters in the freq. plane.
        - k: No of standard deviations of the noise energy beyond the mean
            at which we set the noise threshold point, below which phase
            congruency values get penalized.
        """
    N, _, H, W = x.shape

    # Calculate the standard deviation of the angular Gaussian function
    # used to construct filters in the freq. plane.
    theta_sigma = math.pi / (orientations * delta_theta)

    # Pre-compute some stuff to speed up filter construction
    grid_x, grid_y = get_meshgrid((H, W))
    radius = torch.sqrt(grid_x**2 + grid_y**2)
    theta = torch.atan2(-grid_y, grid_x)

    # Quadrant shift radius and theta so that filters are constructed with 0 frequency at the corners.
    # Get rid of the 0 radius value at the 0 frequency point (now at top-left corner)
    # so that taking the log of the radius will not cause trouble.
    radius = ifftshift(radius)
    theta = ifftshift(theta)
    radius[0, 0] = 1

    sintheta = torch.sin(theta)
    costheta = torch.cos(theta)

    # Filters are constructed in terms of two components.
    # 1) The radial component, which controls the frequency band that the filter responds to
    # 2) The angular component, which controls the orientation that the filter responds to.
    # The two components are multiplied together to construct the overall filter.

    # First construct a low-pass filter that is as large as possible, yet falls
    # away to zero at the boundaries.  All log Gabor filters are multiplied by
    # this to ensure no extra frequencies at the 'corners' of the FFT are
    # incorporated as this seems to upset the normalisation process when
    lp = _lowpassfilter(size=(H, W), cutoff=.45, n=15)

    # Construct the radial filter components...
    log_gabor = []
    for s in range(scales):
        wavelength = min_length * mult**s
        omega_0 = 1.0 / wavelength
        gabor_filter = torch.exp((-torch.log(radius / omega_0)**2) / (2 * math.log(sigma_f)**2))
        if use_lowpass_filter:
            gabor_filter = gabor_filter * lp
        gabor_filter[0, 0] = 0
        log_gabor.append(gabor_filter)

    # Then construct the angular filter components...
    spread = []
    for o in range(orientations):
        angl = o * math.pi / orientations

        # For each point in the filter matrix calculate the angular distance from
        # the specified filter orientation.  To overcome the angular wrap-around
        # problem sine difference and cosine difference values are first computed
        # and then the atan2 function is used to determine angular distance.
        ds = sintheta * math.cos(angl) - costheta * math.sin(angl)  # Difference in sine.
        dc = costheta * math.cos(angl) + sintheta * math.sin(angl)  # Difference in cosine.
        dtheta = torch.abs(torch.atan2(ds, dc))
        spread.append(torch.exp((-dtheta**2) / (2 * theta_sigma**2)))

    spread = torch.stack(spread)
    log_gabor = torch.stack(log_gabor)

    # Multiply, add batch dimension and transfer to correct device.
    filters = (spread.repeat_interleave(scales, dim=0) * log_gabor.repeat(orientations, 1, 1)).unsqueeze(0).to(x)
    return filters



def blockproc(x, kernel, fun, border_size=None, pad_partial=False, pad_method='zero', **func_args):
    r"""blockproc function like matlab

    Difference:
        - Partial blocks is discarded (if exist) for fast GPU process.

    Args:
        x (tensor): shape (b, c, h, w)
        kernel (int or tuple): block size
        func (function): function to process each block
        border_size (int or tuple): border pixels to each block
        pad_partial: pad partial blocks to make them full-sized, default False
        pad_method: [zero, replicate, symmetric] how to pad partial block when pad_partial is set True

    Return:
        results (tensor): concatenated results of each block
    """
    assert len(x.shape) == 4, f'Shape of input has to be (b, c, h, w) but got {x.shape}'
    kernel = to_2tuple(kernel)
    if pad_partial:
        b, c, h, w = x.shape
        stride = kernel
        h2 = math.ceil(h / stride[0])
        w2 = math.ceil(w / stride[1])
        pad_row = (h2 - 1) * stride[0] + kernel[0] - h
        pad_col = (w2 - 1) * stride[1] + kernel[1] - w
        padding = (0, pad_col, 0, pad_row)
        if pad_method == 'zero':
            x = F.pad(x, padding, mode='constant')
        elif pad_method == 'symmetric':
            x = symm_pad(x, padding)
        else:
            x = F.pad(x, padding, mode=pad_method)

    if border_size is not None:
        raise NotImplementedError('Blockproc with border is not implemented yet')
    else:
        b, c, h, w = x.shape
        block_size_h, block_size_w = kernel
        num_block_h = math.floor(h / block_size_h)
        num_block_w = math.floor(w / block_size_w)

        # extract blocks in (row, column) manner, i.e., stored with column first
        blocks = F.unfold(x, kernel, stride=kernel)
        blocks = blocks.reshape(b, c, *kernel, num_block_h, num_block_w)
        blocks = blocks.permute(5, 4, 0, 1, 2, 3).reshape(num_block_h * num_block_w * b, c, *kernel)

        results = fun(blocks, func_args)
        results = results.reshape(num_block_h * num_block_w, b, *results.shape[1:]).transpose(0, 1)
        return results



def nanmean(v, *args, inplace=False, **kwargs):
    r"""nanmean same as matlab function: calculate mean values by removing all nan.
    """
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)



def nancov(x):
    r"""Calculate nancov for batched tensor, rows that contains nan value 
    will be removed.

    Args:
        x (tensor): (B, row_num, feat_dim)  

    Return:
        cov (tensor): (B, feat_dim, feat_dim)
    """
    assert len(x.shape) == 3, f'Shape of input should be (batch_size, row_num, feat_dim), but got {x.shape}'
    b, rownum, feat_dim = x.shape
    nan_mask = torch.isnan(x).any(dim=2, keepdim=True)
    cov_x = []
    for i in range(b):
        x_no_nan = x[i].masked_select(~nan_mask[i]).reshape(-1, feat_dim)
        cov_x.append(cov(x_no_nan, rowvar=False))
    return torch.stack(cov_x)


def cov(tensor, rowvar=True, bias=False):
    r"""Estimate a covariance matrix (np.cov)
    Ref: https://gist.github.com/ModarTensai/5ab449acba9df1a26c12060240773110
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    correction = int(not bias) if tensor.shape[-1] > 1 else 0
    return torch.cov(tensor, correction=correction)



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



def conv2d(input, weight, bias=None, stride=1, padding='same', dilation=1, groups=1):
    """Matlab like conv2, weights needs to be flipped.
    Args:
        input (tensor): (b, c, h, w)
        weight (tensor): (out_ch, in_ch, kh, kw), conv weight
        bias (bool or None): bias
        stride (int or tuple): conv stride
        padding (str): padding mode
        dilation (int): conv dilation
    """
    kernel_size = weight.shape[2:]
    pad_func = ExactPadding2d(kernel_size, stride, dilation, mode=padding)
    weight = torch.flip(weight, dims=(-1, -2))
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






def compute_feature(
    block: torch.Tensor,
    ilniqe: bool = False,
) -> torch.Tensor:
    """Compute features.
    Args:
        block (Tensor): Image block in shape (b, c, h, w).
    Returns:
        list: Features with length of 18.
    """
    bsz = block.shape[0]
    aggd_block = block[:, [0]]
    alpha, beta_l, beta_r = estimate_aggd_param(aggd_block)
    feat = [alpha, (beta_l + beta_r) / 2]

    # distortions disturb the fairly regular structure of natural images.
    # This deviation can be captured by analyzing the sample distribution of
    # the products of pairs of adjacent coefficients computed along
    # horizontal, vertical and diagonal orientations.
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = torch.roll(aggd_block, shifts[i], dims=(2, 3))
        alpha, beta_l, beta_r = estimate_aggd_param(aggd_block * shifted_block)
        # Eq. 8
        mean = (beta_r - beta_l) * (torch.lgamma(2 / alpha) - torch.lgamma(1 / alpha)).exp()
        feat.extend((alpha, mean, beta_l, beta_r))
    feat = [x.reshape(bsz, 1) for x in feat]

    if ilniqe:
        tmp_block = block[:, 1:4]
        channels = 4 - 1
        shape_scale = fitweibull(tmp_block.reshape(bsz * channels, -1))
        scale_shape = shape_scale[:, [1, 0]].reshape(bsz, -1)
        feat.append(scale_shape)

        mu = torch.mean(block[:, 4:7], dim=(2, 3))
        sigmaSquare = torch.var(block[:, 4:7], dim=(2, 3))
        mu_sigma = torch.stack((mu, sigmaSquare), dim=-1).reshape(bsz, -1)
        feat.append(mu_sigma)

        channels = 85 - 7
        tmp_block = block[:, 7:85].reshape(bsz * channels, 1, *block.shape[2:])
        alpha_data, beta_l_data, beta_r_data = estimate_aggd_param(tmp_block)
        alpha_data = alpha_data.reshape(bsz, channels)
        beta_l_data = beta_l_data.reshape(bsz, channels)
        beta_r_data = beta_r_data.reshape(bsz, channels)
        alpha_beta = torch.stack([alpha_data, (beta_l_data + beta_r_data) / 2], dim=-1).reshape(bsz, -1)
        feat.append(alpha_beta)

        tmp_block = block[:, 85:109]
        channels = 109 - 85
        shape_scale = fitweibull(tmp_block.reshape(bsz * channels, -1))
        scale_shape = shape_scale[:, [1, 0]].reshape(bsz, -1)
        feat.append(scale_shape)

    feat = torch.cat(feat, dim=-1)
    return feat


def niqe(img: torch.Tensor,
         mu_pris_param: torch.Tensor,
         cov_pris_param: torch.Tensor,
         block_size_h: int = 96,
         block_size_w: int = 96) -> torch.Tensor:
    """Calculate NIQE (Natural Image Quality Evaluator) metric.
    Args:
        img (Tensor): Input image.
        mu_pris_param (Tensor): Mean of a pre-defined multivariate Gaussian
            model calculated on the pristine dataset.
        cov_pris_param (Tensor): Covariance of a pre-defined multivariate
            Gaussian model calculated on the pristine dataset.
        gaussian_window (Tensor): A 7x7 Gaussian window used for smoothing the image.
        block_size_h (int): Height of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
        block_size_w (int): Width of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
    """
    assert img.ndim == 4, ('Input image must be a gray or Y (of YCbCr) image with shape (b, c, h, w).')
    # crop image
    b, c, h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[..., 0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        img_normalized = normalize_img_with_guass(img, padding='replicate')

        distparam.append(blockproc(img_normalized, [block_size_h // scale, block_size_w // scale], fun=compute_feature))

        if scale == 1:
            img = imresize(img / 255., scale=0.5, antialiasing=True)
            img = img * 255.

    distparam = torch.cat(distparam, -1)

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    mu_distparam = nanmean(distparam, dim=1)
    cov_distparam = nancov(distparam)

    # compute niqe quality, Eq. 10 in the paper
    invcov_param = torch.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    diff = (mu_pris_param - mu_distparam).unsqueeze(1)
    quality = torch.bmm(torch.bmm(diff, invcov_param), diff.transpose(1, 2)).squeeze()

    quality = torch.sqrt(quality)
    return quality


def calculate_niqe(img: torch.Tensor,
                   crop_border: int = 0,
                   test_y_channel: bool = True,
                   pretrained_model_path: str = None,
                   color_space: str = 'yiq',
                   **kwargs) -> torch.Tensor:
    """Calculate NIQE (Natural Image Quality Evaluator) metric.
    Args:
        img (Tensor): Input image whose quality needs to be computed.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        test_y_channel (Bool): Whether converted to 'y' (of MATLAB YCbCr) or 'gray'.
        pretrained_model_path (str): The pretrained model path.
    Returns:
        Tensor: NIQE result.
    """

    params = scipy.io.loadmat(pretrained_model_path)
    mu_pris_param = np.ravel(params['mu_prisparam'])
    cov_pris_param = params['cov_prisparam']
    mu_pris_param = torch.from_numpy(mu_pris_param).to(img)
    cov_pris_param = torch.from_numpy(cov_pris_param).to(img)

    mu_pris_param = mu_pris_param.repeat(img.size(0), 1)
    cov_pris_param = cov_pris_param.repeat(img.size(0), 1, 1)

    # NIQE only support gray image 
    if img.shape[1] == 3:
        img = to_y_channel(img, 255, color_space)
    elif img.shape[1] == 1:
        img = img * 255

    img = diff_round(img)
    img = img.to(torch.float64)

    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]

    niqe_result = niqe(img, mu_pris_param, cov_pris_param)

    return niqe_result


def gauDerivative(sigma, in_ch=1, out_ch=1, device=None):
    halfLength = math.ceil(3 * sigma)

    x, y = np.meshgrid(
        np.linspace(-halfLength, halfLength, 2 * halfLength + 1),
        np.linspace(-halfLength, halfLength, 2 * halfLength + 1))

    gauDerX = x * np.exp(-(x**2 + y**2) / 2 / sigma / sigma)
    gauDerY = y * np.exp(-(x**2 + y**2) / 2 / sigma / sigma)

    dx = torch.from_numpy(gauDerX).to(device)
    dy = torch.from_numpy(gauDerY).to(device)
    dx = dx.repeat(out_ch, in_ch, 1, 1)
    dy = dy.repeat(out_ch, in_ch, 1, 1)

    return dx, dy


def ilniqe(img: torch.Tensor,
           mu_pris_param: torch.Tensor,
           cov_pris_param: torch.Tensor,
           principleVectors: torch.Tensor,
           meanOfSampleData: torch.Tensor,
           resize: bool = True,
           block_size_h: int = 84,
           block_size_w: int = 84) -> torch.Tensor:
    """Calculate IL-NIQE (Integrated Local Natural Image Quality Evaluator) metric.
    Args:
        img (Tensor): Input image.
        mu_pris_param (Tensor): Mean of a pre-defined multivariate Gaussian
            model calculated on the pristine dataset.
        cov_pris_param (Tensor): Covariance of a pre-defined multivariate
            Gaussian model calculated on the pristine dataset.
        principleVectors (Tensor): Features from official .mat file.
        meanOfSampleData (Tensor): Features from official .mat file.
        resize (Bloolean): resize image. Default: True.
        block_size_h (int): Height of the blocks in to which image is divided.
            Default: 84 (the official recommended value).
        block_size_w (int): Width of the blocks in to which image is divided.
            Default: 84 (the official recommended value).
    """
    assert img.ndim == 4, ('Input image must be a gray or Y (of YCbCr) image with shape (b, c, h, w).')

    sigmaForGauDerivative = 1.66
    KforLog = 0.00001
    normalizedWidth = 524
    minWaveLength = 2.4
    sigmaOnf = 0.55
    mult = 1.31
    dThetaOnSigma = 1.10
    scaleFactorForLoG = 0.87
    scaleFactorForGaussianDer = 0.28
    sigmaForDownsample = 0.9

    EPS = 1e-8
    scales = 3
    orientations = 4
    infConst = 10000
    # nanConst = 2000

    if resize:
        img = imresize(img, sizes=(normalizedWidth, normalizedWidth))
        img = img.clamp(0.0, 255.0)

    # crop image
    b, c, h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[..., 0:num_block_h * block_size_h, 0:num_block_w * block_size_w]
    ospace_weight = torch.tensor([
        [0.3, 0.04, -0.35],
        [0.34, -0.6, 0.17],
        [0.06, 0.63, 0.27],
    ]).to(img)

    O_img = img.permute(0, 2, 3, 1) @ ospace_weight.T
    O_img = O_img.permute(0, 3, 1, 2)

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        struct_dis = normalize_img_with_guass(O_img[:, [2]], kernel_size=5, sigma=5. / 6, padding='replicate')

        dx, dy = gauDerivative(sigmaForGauDerivative / (scale**scaleFactorForGaussianDer), device=img)

        Ix = conv2d(O_img, dx.repeat(3, 1, 1, 1), groups=3)
        Iy = conv2d(O_img, dy.repeat(3, 1, 1, 1), groups=3)
        GM = torch.sqrt(Ix**2 + Iy**2 + EPS)
        Ixy = torch.stack((Ix, Iy), dim=2).reshape(Ix.shape[0], Ix.shape[1] * 2,
                                                   *Ix.shape[2:])  # reshape to (IxO1, IxO1, IxO2, IyO2, IxO3, IyO3)

        logRGB = torch.log(img + KforLog)
        logRGBMS = logRGB - logRGB.mean(dim=(2, 3), keepdim=True)

        Intensity = logRGBMS.sum(dim=1, keepdim=True) / np.sqrt(3)
        BY = (logRGBMS[:, [0]] + logRGBMS[:, [1]] - 2 * logRGBMS[:, [2]]) / np.sqrt(6)
        RG = (logRGBMS[:, [0]] - logRGBMS[:, [1]]) / np.sqrt(2)

        compositeMat = torch.cat([struct_dis, GM, Intensity, BY, RG, Ixy], dim=1)

        O3 = O_img[:, [2]]
        # gabor filter in shape (b, ori * scale, h, w)
        LGFilters = _construct_filters(
            O3,
            scales=scales,
            orientations=orientations,
            min_length=minWaveLength / (scale**scaleFactorForLoG),
            sigma_f=sigmaOnf,
            mult=mult,
            delta_theta=dThetaOnSigma,
            use_lowpass_filter=False)
        # reformat to scale * ori
        b, _, h, w = LGFilters.shape
        LGFilters = LGFilters.reshape(b, orientations, scales, h, w).transpose(1, 2).reshape(b, -1, h, w)
        # TODO: current filters needs to be transposed to get same results as matlab, find the bug
        LGFilters = LGFilters.transpose(-1, -2)
        fftIm = torch.fft.fft2(O3)

        logResponse = []
        partialDer = []
        GM = []
        for index in range(LGFilters.shape[1]):
            filter = LGFilters[:, [index]]
            response = torch.fft.ifft2(filter * fftIm)
            realRes = torch.real(response)
            imagRes = torch.imag(response)

            partialXReal = conv2d(realRes, dx)
            partialYReal = conv2d(realRes, dy)
            realGM = torch.sqrt(partialXReal**2 + partialYReal**2 + EPS)

            partialXImag = conv2d(imagRes, dx)
            partialYImag = conv2d(imagRes, dy)
            imagGM = torch.sqrt(partialXImag**2 + partialYImag**2 + EPS)

            logResponse.append(realRes)
            logResponse.append(imagRes)
            partialDer.append(partialXReal)
            partialDer.append(partialYReal)
            partialDer.append(partialXImag)
            partialDer.append(partialYImag)
            GM.append(realGM)
            GM.append(imagGM)
        logResponse = torch.cat(logResponse, dim=1)
        partialDer = torch.cat(partialDer, dim=1)
        GM = torch.cat(GM, dim=1)
        compositeMat = torch.cat((compositeMat, logResponse, partialDer, GM), dim=1)

        distparam.append(blockproc(compositeMat, [block_size_h // scale,
                         block_size_w // scale], fun=compute_feature, ilniqe=True))

        gauForDS = fspecial(math.ceil(6 * sigmaForDownsample), sigmaForDownsample).to(img)
        filterResult = imfilter(O_img, gauForDS.repeat(3, 1, 1, 1), padding='replicate', groups=3)
        O_img = filterResult[..., ::2, ::2]
        filterResult = imfilter(img, gauForDS.repeat(3, 1, 1, 1), padding='replicate', groups=3)
        img = filterResult[..., ::2, ::2]

    distparam = torch.cat(distparam, dim=-1)  # b, block_num, feature_num
    distparam[distparam > infConst] = infConst

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    coefficientsViaPCA = torch.bmm(
        principleVectors.transpose(1, 2), (distparam - meanOfSampleData.unsqueeze(1)).transpose(1, 2))
    final_features = coefficientsViaPCA.transpose(1, 2)
    b, blk_num, feat_num = final_features.shape

    # remove block features with nan and compute nonan cov
    cov_distparam = nancov(final_features)

    # replace nan in final features with mu
    mu_final_features = nanmean(final_features, dim=1, keepdim=True)
    final_features_withmu = torch.where(torch.isnan(final_features), mu_final_features, final_features)

    # compute ilniqe quality
    invcov_param = torch.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    diff = final_features_withmu - mu_pris_param.unsqueeze(1)
    quality = (torch.bmm(diff, invcov_param) * diff).sum(dim=-1)
    quality = torch.sqrt(quality).mean(dim=1)

    return quality


def calculate_ilniqe(img: torch.Tensor,
                     crop_border: int = 0,
                     pretrained_model_path: str = None,
                     **kwargs) -> torch.Tensor:
    """Calculate IL-NIQE metric.
    Args:
        img (Tensor): Input image whose quality needs to be computed.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        pretrained_model_path (str): The pretrained model path.
    Returns:
        Tensor: IL-NIQE result.
    """

    params = scipy.io.loadmat(pretrained_model_path)
    img = img * 255.
    img = diff_round(img)
    # float64 precision is critical to be consistent with matlab codes
    img = img.to(torch.float64)

    mu_pris_param = np.ravel(params['templateModel'][0][0])
    cov_pris_param = params['templateModel'][0][1]
    meanOfSampleData = np.ravel(params['templateModel'][0][2])
    principleVectors = params['templateModel'][0][3]

    mu_pris_param = torch.from_numpy(mu_pris_param).to(img)
    cov_pris_param = torch.from_numpy(cov_pris_param).to(img)
    meanOfSampleData = torch.from_numpy(meanOfSampleData).to(img)
    principleVectors = torch.from_numpy(principleVectors).to(img)

    mu_pris_param = mu_pris_param.repeat(img.size(0), 1)
    cov_pris_param = cov_pris_param.repeat(img.size(0), 1, 1)
    meanOfSampleData = meanOfSampleData.repeat(img.size(0), 1)
    principleVectors = principleVectors.repeat(img.size(0), 1, 1)

    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]

    ilniqe_result = ilniqe(img, mu_pris_param, cov_pris_param, principleVectors, meanOfSampleData)

    return ilniqe_result


class NIQE(torch.nn.Module):
    r"""Args:
        - channels (int): Number of processed channel.
        - test_y_channel (bool): whether to use y channel on ycbcr.
        - crop_border (int): Cropped pixels in each edge of an image. These
        pixels are not involved in the metric calculation.
        - pretrained_model_path (str): The pretrained model path.
    References:
        Mittal, Anish, Rajiv Soundararajan, and Alan C. Bovik.
        "Making a “completely blind” image quality analyzer."
        IEEE Signal Processing Letters (SPL) 20.3 (2012): 209-212.
    """

    def __init__(self,
                 channels: int = 1,
                 test_y_channel: bool = True,
                 color_space: str = 'yiq',
                 crop_border: int = 0,
                 pretrained_model_path: str = None) -> None:

        super(NIQE, self).__init__()
        self.channels = channels
        self.test_y_channel = test_y_channel
        self.color_space = color_space
        self.crop_border = crop_border
        if pretrained_model_path is not None:
            self.pretrained_model_path = pretrained_model_path
        else:
            self.pretrained_model_path = '/share3/home/zqiang/CVSR_train/metric/weight/niqe_modelparameters.mat' # load_file_from_url(default_model_urls['url'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computation of NIQE metric.
        Input:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
        Output:
            score (tensor): results of ilniqe metric, should be a positive real number. Shape :math:`(N, 1)`.
        """
        score = calculate_niqe(x, self.crop_border, self.test_y_channel, self.pretrained_model_path, self.color_space)
        return score


class ILNIQE(torch.nn.Module):
    r"""Args:
        - channels (int): Number of processed channel.
        - test_y_channel (bool): whether to use y channel on ycbcr.
        - crop_border (int): Cropped pixels in each edge of an image. These
        pixels are not involved in the metric calculation.
        - pretrained_model_path (str): The pretrained model path.
    References:
        Zhang, Lin, Lei Zhang, and Alan C. Bovik. "A feature-enriched
        completely blind image quality evaluator." IEEE Transactions
        on Image Processing 24.8 (2015): 2579-2591.
    """

    def __init__(self, channels: int = 3, crop_border: int = 0, pretrained_model_path: str = None) -> None:

        super(ILNIQE, self).__init__()
        self.channels = channels
        self.crop_border = crop_border
        if pretrained_model_path is not None:
            self.pretrained_model_path = pretrained_model_path
        else:
            self.pretrained_model_path = load_file_from_url(default_model_urls['ilniqe'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computation of NIQE metric.
        Input:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
        Output:
            score (tensor): results of ilniqe metric, should be a positive real number. Shape :math:`(N, 1)`.
        """
        assert x.shape[1] == 3, 'ILNIQE only support input image with 3 channels'
        score = calculate_ilniqe(x, self.crop_border, self.pretrained_model_path)
        return score



if __name__ == "__main__":
    
    ref = torch.Tensor(np.array(Image.open('/share3/home/zqiang/CVSR_train/metric/test_imgs/bikes.bmp')).transpose(2,0,1)).unsqueeze(0) # .convert('LA'))[:,:,0] # ref
    dis = torch.Tensor(np.array(Image.open('/share3/home/zqiang/CVSR_train/metric/test_imgs/bikes_distorted.bmp')).transpose(2,0,1)).unsqueeze(0) # .convert('LA'))[:,:,0] # dis
    # print('ref',ref.shape,ref.shape[1])
    niqe_model = NIQE()
    strT = time.time()
    print('NIQE of ref bikes image is: %0.4f'% niqe_model(ref))
    Sumtime = time.time()-strT
    print('NIQE of ref bikes image time: %0.4f'% Sumtime)

    print('NIQE of dis bikes image is: %0.4f'% niqe_model(dis))

    ref = torch.Tensor(np.array(Image.open('/share3/home/zqiang/CVSR_train/metric/test_imgs/parrots.bmp')).transpose(2,0,1)).unsqueeze(0) # .convert('LA'))[:,:,0] # ref
    dis = torch.Tensor(np.array(Image.open('/share3/home//zqiang/CVSR_train/metric/test_imgs/parrots_distorted.bmp')).transpose(2,0,1)).unsqueeze(0)
   
    print('NIQE of ref parrot image is: %0.4f'% niqe_model(ref))
    print('NIQE of dis parrot image is: %0.4f'% niqe_model(dis))