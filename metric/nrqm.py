r"""NRQM Metric, proposed in

Chao Ma, Chih-Yuan Yang, Xiaokang Yang, Ming-Hsuan Yang
"Learning a No-Reference Quality Metric for Single-Image Super-Resolution"
Computer Vision and Image Understanding (CVIU), 2017

Matlab reference: https://github.com/chaoma99/sr-metric
This PyTorch implementation by: Chaofeng Chen (https://github.com/chaofengc)

"""
import math
import scipy.io
import torch
from torch import Tensor
import torch.nn.functional as F
import collections.abc
from itertools import repeat
from metric.niqe import NIQE
from warnings import warn
import numpy as np
import torch
from scipy.special import factorial
from torch import nn as nn
from typing import Tuple
from typing import Union, Dict
from PIL import Image
import typing
import collections.abc
import time
__all__ = ['imresize']
_I = typing.Optional[int]
_D = typing.Optional[torch.dtype]

default_model_urls = {'url': 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/NRQM_model.mat'}


# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function



################################################################################
################################################################################
import numpy as np
import torch


def abs(x):
    return torch.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-12)


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def batch_fftshift2d(x):
    '''Args:
        x: An complex tensor. Shape :math:`(N, C, H, W)`.
        Pytroch version >= 1.8.0
    '''
    real, imag = x.real, x.imag
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim) // 2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


def batch_ifftshift2d(x):
    '''Args:
        x: An input tensor. Shape :math:`(N, C, H, W, 2)`.
    Return:
        An complex tensor. Shape :math:`(N, C, H, W)`.
    '''
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim) // 2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim) // 2)
    return torch.complex(real, imag)  # convert to complex (real&imag)


def prepare_grid(m, n):
    x = np.linspace(-(m // 2) / (m / 2), (m // 2) / (m / 2) - (1 - m % 2) * 2 / m, num=m)
    y = np.linspace(-(n // 2) / (n / 2), (n // 2) / (n / 2) - (1 - n % 2) * 2 / n, num=n)
    xv, yv = np.meshgrid(y, x)
    angle = np.arctan2(yv, xv)
    rad = np.sqrt(xv**2 + yv**2)
    rad[m // 2][n // 2] = rad[m // 2][n // 2 - 1]
    log_rad = np.log2(rad)
    return log_rad, angle


def rcosFn(width, position):
    N = 256  # abritrary
    X = np.pi * np.array(range(-N - 1, 2)) / 2 / N
    Y = np.cos(X)**2
    Y[0] = Y[1]
    Y[N + 2] = Y[N + 1]
    X = position + 2 * width / np.pi * (X + np.pi / 4)
    return X, Y


def pointOp(im, Y, X):
    out = np.interp(im.flatten(), X, Y)
    return np.reshape(out, im.shape)


def getlist(coeff):
    straight = [bands for scale in coeff[1:-1] for bands in scale]
    straight = [coeff[0]] + straight + [coeff[-1]]
    return straight

def ssim_func(X,
         Y,
         win=None,
         get_ssim_map=False,
         get_cs=False,
         get_weight=False,
         downsample=False,
         data_range=1.,
         ):
    if win is None:
        win = fspecial(11, 1.5, X.shape[1]).to(X)
    
    C1 = (0.01 * data_range)**2
    C2 = (0.03 * data_range)**2

    # Averagepool image if the size is large enough
    f = max(1, round(min(X.size()[-2:]) / 256))
    # Downsample operation is used in official matlab code
    if (f > 1) and downsample:
        X = F.avg_pool2d(X, kernel_size=f)
        Y = F.avg_pool2d(Y, kernel_size=f)

    mu1 = filter2(X, win, 'valid')
    mu2 = filter2(Y, win, 'valid')
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(X * X, win, 'valid') - mu1_sq
    sigma2_sq = filter2(Y * Y, win, 'valid') - mu2_sq
    sigma12 = filter2(X * Y, win, 'valid') - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    cs_map = F.relu(cs_map)  # force the ssim response to be nonnegative to avoid negative results.
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    ssim_val = ssim_map.mean([1, 2, 3])

    if get_weight:
        weights = torch.log((1 + sigma1_sq / C2) * (1 + sigma2_sq / C2))
        return ssim_map, weights

    if get_ssim_map:
        return ssim_map

    if get_cs:
        return ssim_val, cs_map.mean([1, 2, 3])

    return ssim_val




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


def filter2(input, weight, shape='same'):
    if shape == 'same':
        return imfilter(input, weight, groups=input.shape[1])
    elif shape == 'valid':
        return F.conv2d(input, weight, stride=1, padding=0, groups=input.shape[1])
    else:
        raise NotImplementedError(f'Shape type {shape} is not implemented.')



def extract_2d_patches(x, kernel, stride=1, dilation=1, padding="same"):
    """
    Extracts 2D patches from a 4D tensor.

    Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        - kernel (int): Size of the kernel to be used for patch extraction.
        - stride (int): Stride of the kernel. Default is 1.
        - dilation (int): Dilation rate of the kernel. Default is 1.
        - padding (str): Type of padding to be applied. Can be "same" or "none". Default is "same".

    Returns:
        torch.Tensor: Extracted patches tensor of shape (batch_size, num_patches, channels, kernel, kernel).
    """
    b, c, h, w = x.shape
    if padding != "none":
        x = exact_padding_2d(x, kernel, stride, dilation, mode=padding)

    # Extract patches
    patches = F.unfold(x, kernel, dilation, stride=stride)
    b, _, pnum = patches.shape
    patches = patches.transpose(1, 2).reshape(b, pnum, c, kernel, kernel)
    return patches




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


def im2col(x, kernel, mode='sliding'):
    r"""simple im2col as matlab

    Args:
        x (Tensor): shape (b, c, h, w)
        kernel (int): kernel size
        mode (string): 
            - sliding (default): rearranges sliding image neighborhoods of kernel size into columns with no zero-padding
            - distinct: rearranges discrete image blocks of kernel size into columns, zero pad right and bottom if necessary
    Return:
        flatten patch (Tensor): (b, h * w / kernel **2, kernel * kernel)
    """
    b, c, h, w = x.shape
    kernel = to_2tuple(kernel)

    if mode == 'sliding':
        stride = 1
    elif mode == 'distinct':
        stride = kernel
        h2 = math.ceil(h / stride[0])
        w2 = math.ceil(w / stride[1])
        pad_row = (h2 - 1) * stride[0] + kernel[0] - h
        pad_col = (w2 - 1) * stride[1] + kernel[1] - w
        x = F.pad(x, (0, pad_col, 0, pad_row))
    else:
        raise NotImplementedError(f'Type {mode} is not implemented yet.')

    patches = F.unfold(x, kernel, dilation=1, stride=stride)
    b, _, pnum = patches.shape
    patches = patches.transpose(1, 2).reshape(b, pnum, -1)
    return patches

import numpy as np
import torch


def abs(x):
    return torch.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-12)


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def batch_fftshift2d(x):
    '''Args:
        x: An complex tensor. Shape :math:`(N, C, H, W)`.
        Pytroch version >= 1.8.0
    '''
    real, imag = x.real, x.imag
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim) // 2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


def batch_ifftshift2d(x):
    '''Args:
        x: An input tensor. Shape :math:`(N, C, H, W, 2)`.
    Return:
        An complex tensor. Shape :math:`(N, C, H, W)`.
    '''
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim) // 2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim) // 2)
    return torch.complex(real, imag)  # convert to complex (real&imag)


def prepare_grid(m, n):
    x = np.linspace(-(m // 2) / (m / 2), (m // 2) / (m / 2) - (1 - m % 2) * 2 / m, num=m)
    y = np.linspace(-(n // 2) / (n / 2), (n // 2) / (n / 2) - (1 - n % 2) * 2 / n, num=n)
    xv, yv = np.meshgrid(y, x)
    angle = np.arctan2(yv, xv)
    rad = np.sqrt(xv**2 + yv**2)
    rad[m // 2][n // 2] = rad[m // 2][n // 2 - 1]
    log_rad = np.log2(rad)
    return log_rad, angle


def rcosFn(width, position):
    N = 256  # abritrary
    X = np.pi * np.array(range(-N - 1, 2)) / 2 / N
    Y = np.cos(X)**2
    Y[0] = Y[1]
    Y[N + 2] = Y[N + 1]
    X = position + 2 * width / np.pi * (X + np.pi / 4)
    return X, Y


def pointOp(im, Y, X):
    out = np.interp(im.flatten(), X, Y)
    return np.reshape(out, im.shape)


def getlist(coeff):
    straight = [bands for scale in coeff[1:-1] for bands in scale]
    straight = [coeff[0]] + straight + [coeff[-1]]
    return straight

def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    Args:
        x: the input signal
        norm: the normalization, None or 'ortho'
    Return:
        the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=-1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=-1))

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def dct2d(x, norm='ortho'):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)




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




class SCFpyr_PyTorch(object):
    '''
    This is a modified version of buildSFpyr, that constructs a
    complex-valued steerable pyramid  using Hilbert-transform pairs
    of filters. Note that the imaginary parts will *not* be steerable.
    Pytorch version >= 1.8.0

    '''

    def __init__(self, height=5, nbands=4, scale_factor=2, device=None):
        self.height = height  # including low-pass and high-pass
        self.nbands = nbands  # number of orientation bands
        self.scale_factor = scale_factor
        self.device = torch.device('cpu') if device is None else device

        # Cache constants
        self.lutsize = 1024
        self.Xcosn = np.pi * np.array(range(-(2 * self.lutsize + 1), (self.lutsize + 2))) / self.lutsize
        self.alpha = (self.Xcosn + np.pi) % (2 * np.pi) - np.pi
        self.complex_fact_construct = np.power(complex(0, -1), self.nbands - 1)
        self.complex_fact_reconstruct = np.power(complex(0, 1), self.nbands - 1)

    ################################################################################
    # Construction of Steerable Pyramid

    def build(self, im_batch):
        ''' Decomposes a batch of images into a complex steerable pyramid.
        The pyramid typically has ~4 levels and 4-8 orientations.

        Args:
            im_batch (torch.Tensor): Batch of images of shape [N,C,H,W]

        Returns:
            pyramid: list containing torch.Tensor objects storing the pyramid
        '''

        assert im_batch.device == self.device, 'Devices invalid (pyr = {}, batch = {})'.format(
            self.device, im_batch.device)
        # assert im_batch.dtype == torch.float32, 'Image batch must be torch.float32'
        assert im_batch.dim() == 4, 'Image batch must be of shape [N,C,H,W]'
        assert im_batch.shape[1] == 1, 'Second dimension must be 1 encoding grayscale image'

        im_batch = im_batch.squeeze(1)  # flatten channels dim
        height, width = im_batch.shape[1], im_batch.shape[2]

        # Check whether image size is sufficient for number of levels
        if self.height > int(np.floor(np.log2(min(width, height))) - 2):
            raise RuntimeError('Cannot build {} levels, image too small.'.format(self.height))

        # Prepare a grid
        log_rad, angle = prepare_grid(height, width)

        # Radial transition function (a raised cosine in log-frequency):
        Xrcos, Yrcos = rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)

        YIrcos = np.sqrt(1 - Yrcos**2)

        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)

        # Note that we expand dims to support broadcasting later
        lo0mask = torch.from_numpy(lo0mask).float()[None, :, :, None].to(self.device)
        hi0mask = torch.from_numpy(hi0mask).float()[None, :, :, None].to(self.device)

        # Fourier transform (2D) and shifting
        batch_dft = torch.fft.fft2(im_batch)
        batch_dft = batch_fftshift2d(batch_dft)

        # Low-pass
        lo0dft = batch_dft * lo0mask

        # Start recursively building the pyramids
        coeff = self._build_levels(lo0dft, log_rad, angle, Xrcos, Yrcos, self.height)

        # High-pass
        hi0dft = batch_dft * hi0mask
        hi0 = batch_ifftshift2d(hi0dft)
        hi0 = torch.fft.ifft2(hi0)
        hi0_real = hi0.real
        coeff.insert(0, hi0_real)
        return coeff

    def _build_levels(self, lodft, log_rad, angle, Xrcos, Yrcos, height):

        if height <= 0:

            # Low-pass
            lo0 = batch_ifftshift2d(lodft)
            lo0 = torch.fft.ifft2(lo0)
            lo0_real = lo0.real
            coeff = [lo0_real]

        else:

            Xrcos = Xrcos - np.log2(self.scale_factor)

            ####################################################################
            ####################### Orientation bandpass #######################
            ####################################################################

            himask = pointOp(log_rad, Yrcos, Xrcos)
            himask = torch.from_numpy(himask[None, :, :, None]).float().to(self.device)

            order = self.nbands - 1
            const = np.power(2, 2 * order) * np.square(factorial(order)) / (self.nbands * factorial(2 * order))
            Ycosn = 2 * np.sqrt(const) * np.power(np.cos(self.Xcosn), order) * (np.abs(self.alpha) < np.pi / 2)  # [n,]

            # Loop through all orientation bands
            orientations = []
            for b in range(self.nbands):

                anglemask = pointOp(angle, Ycosn, self.Xcosn + np.pi * b / self.nbands)
                anglemask = anglemask[None, :, :, None]  # for broadcasting
                anglemask = torch.from_numpy(anglemask).float().to(self.device)

                # Bandpass filtering
                banddft = lodft * anglemask * himask

                # Now multiply with complex number
                # (x+yi)(u+vi) = (xu-yv) + (xv+yu)i
                banddft = torch.unbind(banddft, -1)
                banddft_real = self.complex_fact_construct.real * banddft[
                    0] - self.complex_fact_construct.imag * banddft[1]
                banddft_imag = self.complex_fact_construct.real * banddft[
                    1] + self.complex_fact_construct.imag * banddft[0]
                banddft = torch.stack((banddft_real, banddft_imag), -1)

                band = batch_ifftshift2d(banddft)
                band = torch.fft.ifft2(band)
                orientations.append(torch.stack((band.real, band.imag), -1))

            ####################################################################
            ######################## Subsample lowpass #########################
            ####################################################################

            # Don't consider batch_size and imag/real dim
            dims = np.array(lodft.shape[1:3])

            # Both are tuples of size 2
            low_ind_start = (np.ceil((dims + 0.5) / 2) - np.ceil((np.ceil((dims - 0.5) / 2) + 0.5) / 2)).astype(int)
            low_ind_end = (low_ind_start + np.ceil((dims - 0.5) / 2)).astype(int)

            # Subsampling indices
            log_rad = log_rad[low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]
            angle = angle[low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1]]

            # Actual subsampling
            lodft = lodft[:, low_ind_start[0]:low_ind_end[0], low_ind_start[1]:low_ind_end[1], :]

            # Filtering
            YIrcos = np.abs(np.sqrt(1 - Yrcos**2))
            lomask = pointOp(log_rad, YIrcos, Xrcos)
            lomask = torch.from_numpy(lomask[None, :, :, None]).float()
            lomask = lomask.to(self.device)

            # Convolution in spatial domain
            lodft = lomask * lodft

            ####################################################################
            ####################### Recursion next level #######################
            ####################################################################

            coeff = self._build_levels(lodft, log_rad, angle, Xrcos, Yrcos, height - 1)
            coeff.insert(0, orientations)

        return coeff




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






def get_guass_pyramid(x: Tensor, scale: int = 2):
    r"""Get gaussian pyramid images with gaussian kernel.
    """
    pyr = [x]
    kernel = fspecial(3, 0.5, x.shape[1]).to(x)
    pad_func = ExactPadding2d(3, stride=1, mode='same')
    for i in range(scale):
        x = F.conv2d(pad_func(x), kernel, groups=x.shape[1])
        x = x[:, :, 1::2, 1::2]
        pyr.append(x)

    return pyr


def get_var_gen_gauss(x, eps=1e-7):
    r"""Get mean and variance of input local patch.
    """
    std = x.abs().std(dim=-1, unbiased=True)
    mean = x.abs().mean(dim=-1)
    rho = std / (mean + eps)
    return rho


def gamma_gen_gauss(x: Tensor, block_seg=1e4):
    r"""General gaussian distribution estimation.

    Args:
        block_seg: maximum number of blocks in parallel to avoid OOM
    """
    pshape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    eps = 1e-7
    gamma = torch.arange(0.03, 10 + 0.001, 0.001).to(x)
    r_table = (torch.lgamma(1. / gamma) + torch.lgamma(3. / gamma) - 2 * torch.lgamma(2. / gamma)).exp()
    r_table = r_table.unsqueeze(0)

    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=True)
    mean_abs = (x - mean).abs().mean(dim=-1, keepdim=True)**2

    rho = var / (mean_abs + eps)

    if rho.shape[0] > block_seg:
        rho_seg = rho.chunk(int(rho.shape[0] // block_seg))
        indexes = []
        for r in rho_seg:
            tmp_idx = (r - r_table).abs().argmin(dim=-1)
            indexes.append(tmp_idx)
        indexes = torch.cat(indexes)
    else:
        indexes = (rho - r_table).abs().argmin(dim=-1)

    solution = gamma[indexes].reshape(*pshape)
    return solution


def gamma_dct(dct_img_block: torch.Tensor):
    r"""Generalized gaussian distribution features
    """
    b, _, _, h, w = dct_img_block.shape
    dct_flatten = dct_img_block.reshape(b, -1, h * w)[:, :, 1:]
    g = gamma_gen_gauss(dct_flatten)
    g = torch.sort(g, dim=-1)[0]
    return g


def coeff_var_dct(dct_img_block: torch.Tensor):
    r"""Gaussian var, mean features
    """
    b, _, _, h, w = dct_img_block.shape
    dct_flatten = dct_img_block.reshape(b, -1, h * w)[:, :, 1:]
    rho = get_var_gen_gauss(dct_flatten)
    rho = torch.sort(rho, dim=-1)[0]
    return rho


def oriented_dct_rho(dct_img_block: torch.Tensor):
    r"""Oriented frequency features
    """
    eps = 1e-8

    # oriented 1
    feat1 = torch.cat([
        dct_img_block[..., 0, 1:],
        dct_img_block[..., 1, 2:],
        dct_img_block[..., 2, 4:],
        dct_img_block[..., 3, 5:],
    ],
        dim=-1).squeeze(-2)
    g1 = get_var_gen_gauss(feat1, eps)

    # oriented 2
    feat2 = torch.cat([
        dct_img_block[..., 1, [1]],
        dct_img_block[..., 2, 2:4],
        dct_img_block[..., 3, 2:5],
        dct_img_block[..., 4, 3:],
        dct_img_block[..., 5, 4:],
        dct_img_block[..., 6, 4:],
    ],
        dim=-1).squeeze(-2)
    g2 = get_var_gen_gauss(feat2, eps)

    # oriented 3
    feat3 = torch.cat([
        dct_img_block[..., 1:, 0],
        dct_img_block[..., 2:, 1],
        dct_img_block[..., 4:, 2],
        dct_img_block[..., 5:, 3],
    ],
        dim=-1).squeeze(-2)
    g3 = get_var_gen_gauss(feat3, eps)

    rho = torch.stack([g1, g2, g3], dim=-1).var(dim=-1)
    rho = torch.sort(rho, dim=-1)[0]
    return rho


def block_dct(img: Tensor):
    r"""Get local frequency features
    """
    img_blocks = extract_2d_patches(img, 3 + 2 * 2, 3)
    dct_img_blocks = dct2d(img_blocks)

    features = []
    # general gaussian distribution features
    gamma_L1 = gamma_dct(dct_img_blocks)
    p10_gamma_L1 = gamma_L1[:, :math.ceil(0.1 * gamma_L1.shape[-1]) + 1].mean(dim=-1)
    p100_gamma_L1 = gamma_L1.mean(dim=-1)
    features += [p10_gamma_L1, p100_gamma_L1]

    # coefficient variation estimation
    coeff_var_L1 = coeff_var_dct(dct_img_blocks)
    p10_last_cv_L1 = coeff_var_L1[:, math.floor(0.9 * coeff_var_L1.shape[-1]):].mean(dim=-1)
    p100_cv_L1 = coeff_var_L1.mean(dim=-1)
    features += [p10_last_cv_L1, p100_cv_L1]

    # oriented dct features
    ori_dct_feat = oriented_dct_rho(dct_img_blocks)
    p10_last_orientation_L1 = ori_dct_feat[:, math.floor(0.9 * ori_dct_feat.shape[-1]):].mean(dim=-1)
    p100_orientation_L1 = ori_dct_feat.mean(dim=-1)
    features += [p10_last_orientation_L1, p100_orientation_L1]

    dct_feat = torch.stack(features, dim=1)
    return dct_feat


def norm_sender_normalized(pyr, num_scale=2, num_bands=6, blksz=3, eps=1e-12):
    r"""Normalize pyramid with local spatial neighbor and band neighbor
    """
    border = blksz // 2
    guardband = 16
    subbands = []
    for si in range(num_scale):
        for bi in range(num_bands):
            idx = si * num_bands + bi
            current_band = pyr[idx]

            N = blksz**2

            # 3x3 window pixels
            tmp = F.unfold(current_band.unsqueeze(1), 3, stride=1)
            tmp = tmp.transpose(1, 2)
            b, hw = tmp.shape[:2]
            # parent pixels
            parent_idx = idx + num_bands
            if parent_idx < len(pyr):
                tmp_parent = pyr[parent_idx]
                tmp_parent = imresize(tmp_parent, sizes=current_band.shape[-2:])
                tmp_parent = tmp_parent[:, border:-border, border:-border].reshape(b, hw, 1)
                tmp = torch.cat((tmp, tmp_parent), dim=-1)
                N += 1
            # neighbor band pixels
            for ni in range(num_bands):
                if ni != bi:
                    ni_idx = si * num_bands + ni
                    tmp_nei = pyr[ni_idx]
                    tmp_nei = tmp_nei[:, border:-border, border:-border].reshape(b, hw, 1)
                    tmp = torch.cat((tmp, tmp_nei), dim=-1)
            C_x = tmp.transpose(1, 2) @ tmp / tmp.shape[1]
            # correct possible negative eigenvalue
            L, Q = torch.linalg.eigh(C_x)
            L_pos = L * (L > 0)
            L_pos_sum = L_pos.sum(dim=1, keepdim=True)
            L = L_pos * L.sum(dim=1, keepdim=True) / (L_pos_sum + (L_pos_sum == 0).to(L.dtype))
            C_x = Q @ torch.diag_embed(L) @ Q.transpose(1, 2)

            o_c = current_band[:, border:-border, border:-border]
            b, h, w = o_c.shape
            o_c = o_c.reshape(b, hw)
            o_c = o_c - o_c.mean(dim=1, keepdim=True)

            if tmp.shape[1] >= 2e5: # To avoid out of GPU memory
                C_x = C_x.cpu()
                tmp = tmp.cpu()
            if hasattr(torch.linalg, 'lstsq'):
                tmp_y = torch.linalg.lstsq(C_x.transpose(1, 2), tmp.transpose(1, 2)).solution.transpose(1, 2) * tmp / N
            else:
                warn(
                    "For numerical stability, we use torch.linal.lstsq to calculate matrix inverse for PyTorch > 1.9.0. The results might be slightly different if you use older version of PyTorch.")
                tmp_y = (tmp @ torch.linalg.pinv(C_x)) * tmp / N
            tmp_y = tmp_y.to(o_c)

            z = tmp_y.sum(dim=2).sqrt()
            mask = z != 0
            g_c = o_c * mask / (z * mask + eps)
            g_c = g_c.reshape(b, h, w)

            gb = int(guardband / (2**(si)))
            g_c = g_c[:, gb:-gb, gb:-gb]
            g_c = g_c - g_c.mean(dim=(1, 2), keepdim=True)
            subbands.append(g_c)

    return subbands


def global_gsm(img: Tensor):
    """Global feature from gassian scale mixture model
    """
    batch_size = img.shape[0]
    num_bands = 6
    pyr = SCFpyr_PyTorch(height=2, nbands=num_bands, device=img.device).build(img)
    lp_bands = [x[..., 0] for x in pyr[1]] \
        + [x[..., 0] for x in pyr[2]]
    subbands = norm_sender_normalized(lp_bands)

    feat = []
    # gamma
    for sb in subbands:
        feat.append(gamma_gen_gauss(sb.reshape(batch_size, -1)))

    # gamma cross scale
    for i in range(num_bands):
        sb1 = subbands[i].reshape(batch_size, -1)
        sb2 = subbands[i + num_bands].reshape(batch_size, -1)
        gs = gamma_gen_gauss(torch.cat((sb1, sb2), dim=1))
        feat.append(gs)

    # structure correlation between scales
    hp_band = pyr[0]
    for sb in lp_bands:
        curr_band = imresize(sb, sizes=hp_band.shape[1:]).unsqueeze(1)
        _, tmpscore = ssim_func(curr_band, hp_band.unsqueeze(1), get_cs=True, data_range=255)
        feat.append(tmpscore)

    # structure correlation between orientations
    for i in range(num_bands):
        for j in range(i + 1, num_bands):
            _, tmpscore = ssim_func(subbands[i].unsqueeze(1), subbands[j].unsqueeze(1), get_cs=True, data_range=255)
            feat.append(tmpscore)

    feat = torch.stack(feat, dim=1)
    return feat


def tree_regression(feat, ldau, rdau, threshold_value, pred_value, best_attri):
    r"""Simple decision tree regression.
    """
    prev_k = k = 0
    for i in range(ldau.shape[0]):
        best_col = best_attri[k] - 1
        threshold = threshold_value[k]
        key_value = feat[best_col]
        prev_k = k
        k = ldau[k] - 1 if key_value <= threshold else rdau[k] - 1
        if k == -1:
            break
    y_pred = pred_value[prev_k]
    return y_pred


def random_forest_regression(feat, ldau, rdau, threshold_value, pred_value, best_attri):
    r"""Simple random forest regression.
    Note: currently, this is non-differentiable and only support CPU.
    """
    feat = feat.cpu().data.numpy()
    b, dim = feat.shape
    node_num, tree_num = ldau.shape

    pred = []
    for i in range(b):
        tmp_feat = feat[i]
        tmp_pred = []
        for i in range(tree_num):
            tmp_result = tree_regression(tmp_feat, ldau[:, i], rdau[:, i], threshold_value[:, i], pred_value[:, i],
                                         best_attri[:, i])
            tmp_pred.append(tmp_result)
        pred.append(tmp_pred)
    pred = torch.tensor(pred)
    return pred.mean(dim=1, keepdim=True)


def nrqm(
    img: Tensor,
    linear_param,
    rf_param,
) -> Tensor:
    """Calculate NRQM
    Args:
        img (Tensor): Input image.
        linear_param (np.array): (4, 1) linear regression params
        rf_param: params of 3 random forest for 3 kinds of features
    """
    assert img.ndim == 4, ('Input image must be a gray or Y (of YCbCr) image with shape (b, c, h, w).')

    # crop image
    b, c, h, w = img.shape
    img = img.double()
    img_pyr = get_guass_pyramid(img / 255.)

    # DCT features
    f1 = []
    for im in img_pyr:
        f1.append(block_dct(im))
    f1 = torch.cat(f1, dim=1)

    # gsm features
    f2 = global_gsm(img)

    # svd features
    f3 = []
    for im in img_pyr:
        col = im2col(im, 5, 'distinct')
        _, s, _ = torch.linalg.svd(col, full_matrices=False)
        f3.append(s)
    f3 = torch.cat(f3, dim=1)

    # Random forest regression. Currently not differentiable and only support CPU
    preds = torch.ones(b, 1)
    for feat, rf in zip([f1, f2, f3], rf_param):
        tmp_pred = random_forest_regression(feat, *rf)
        preds = torch.cat((preds, tmp_pred), dim=1)
    quality = preds @ torch.tensor(linear_param)

    return quality.squeeze()


def calculate_nrqm(img: torch.Tensor,
                   crop_border: int = 0,
                   test_y_channel: bool = True,
                   pretrained_model_path: str = None,
                   color_space: str = 'yiq',
                   **kwargs) -> torch.Tensor:
    """Calculate NRQM
    Args:
        img (Tensor): Input image whose quality needs to be computed.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        test_y_channel (Bool): Whether converted to 'y' (of MATLAB YCbCr) or 'gray'.
        pretrained_model_path (String): The pretrained model path.
    Returns:
        Tensor: NIQE result.
    """

    params = scipy.io.loadmat(pretrained_model_path)['model']
    linear_param = params['linear'][0, 0]
    rf_params_list = []
    for i in range(3):
        tmp_list = []
        tmp_param = params['rf'][0, 0][0, i][0, 0]
        tmp_list.append(tmp_param[0])  # ldau
        tmp_list.append(tmp_param[1])  # rdau
        tmp_list.append(tmp_param[4])  # threshold value
        tmp_list.append(tmp_param[5])  # pred value
        tmp_list.append(tmp_param[6])  # best attribute index
        rf_params_list.append(tmp_list)

    if test_y_channel and img.shape[1] == 3:
        img = to_y_channel(img, 255, color_space)
    
    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]

    nrqm_result = nrqm(img, linear_param, rf_params_list)

    return nrqm_result.to(img)


class NRQM(torch.nn.Module):
    r""" NRQM metric
    Ma, Chao, Chih-Yuan Yang, Xiaokang Yang, and Ming-Hsuan Yang.
    "Learning a no-reference quality metric for single-image super-resolution."
    Computer Vision and Image Understanding 158 (2017): 1-16.
    Args:
        - channels (int): Number of processed channel.
        - test_y_channel (Boolean): whether to use y channel on ycbcr.
        - crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        - pretrained_model_path (String): The pretrained model path.
    """

    def __init__(self,
                 test_y_channel: bool = True,
                 color_space: str = 'yiq',
                 crop_border: int = 0,
                 pretrained_model_path: str = None) -> None:

        super(NRQM, self).__init__()
        self.test_y_channel = test_y_channel
        self.crop_border = crop_border
        self.color_space = color_space

        if pretrained_model_path is not None:
            self.pretrained_model_path = pretrained_model_path
        else:
            self.pretrained_model_path = '/share3/home/zqiang/CVSR_train/metric/weight/NRQM_model.mat' # load_file_from_url(default_model_urls['url'])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Computation of NRQM metric.
        Args:
            X: An input tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of nrqm metric.
        """
        score = calculate_nrqm(X, self.crop_border, self.test_y_channel, self.pretrained_model_path, self.color_space)
        return score


class PI(torch.nn.Module):
    r""" Perceptual Index (PI), introduced by
    Blau, Yochai, Roey Mechrez, Radu Timofte, Tomer Michaeli, and Lihi Zelnik-Manor.
    "The 2018 pirm challenge on perceptual image super-resolution."
    In Proceedings of the European Conference on Computer Vision (ECCV) Workshops, pp. 0-0. 2018.
    Ref url: https://github.com/roimehrez/PIRM2018
    It is a combination of NIQE and NRQM: 1/2 * ((10 - NRQM) + NIQE)
    Args:
        - color_space (str): color space of y channel, default ycbcr.
        - crop_border (int): Cropped pixels in each edge of an image, default 4.
    """

    def __init__(self, crop_border=4, color_space='ycbcr'):
        super(PI, self).__init__()
        self.nrqm = NRQM(crop_border=crop_border, color_space=color_space)
        self.niqe = NIQE(crop_border=crop_border, color_space=color_space)

    def forward(self, X: Tensor) -> Tensor:
        r"""Computation of PI metric.
        Args:
            X: An input tensor. Shape :math:`(N, C, H, W)`.
        Returns:
            Value of PI metric.
        """
        nrqm_score = self.nrqm(X)
        niqe_score = self.niqe(X)
        score = 1 / 2 * (10 - nrqm_score + niqe_score)
        return score



if __name__ == "__main__":
    
    ref = torch.Tensor(np.array(Image.open('/share3/home/zqiang/CVSR_train/metric/test_imgs/bikes.bmp')).transpose(2,0,1)).unsqueeze(0) # .convert('LA'))[:,:,0] # ref
    dis = torch.Tensor(np.array(Image.open('/share3/home/zqiang/CVSR_train/metric/test_imgs/bikes_distorted.bmp')).transpose(2,0,1)).unsqueeze(0) # .convert('LA'))[:,:,0] # dis
    nrqm_model = NRQM()
    pi_model = PI()
    strT = time.time()
    print('NRQM of ref bikes image is: %0.4f'% nrqm_model(ref))
    Sumtime = time.time()-strT
    print('NRQM of ref bikes image time: %0.4f'% Sumtime)
    print('NRQM  of ref bikes image is: %0.4f'% nrqm_model(ref))
    print('NRQM  of dis bikes image is: %0.4f'% nrqm_model(dis))

    strT = time.time()
    print('PI of ref bikes image is: %0.4f'% pi_model(ref))
    Sumtime = time.time()-strT
    print('PI of ref bikes image time: %0.4f'% Sumtime)

    print('PI of ref bikes image is: %0.4f'% pi_model(ref))
    print('PI of dis bikes image is: %0.4f'% pi_model(dis))

    ref = torch.Tensor(np.array(Image.open('/share3/home/zqiang/CVSR_train/metric/test_imgs/parrots.bmp')).transpose(2,0,1)).unsqueeze(0) # .convert('LA'))[:,:,0] # ref
    dis = torch.Tensor(np.array(Image.open('/share3/home//zqiang/CVSR_train/metric/test_imgs/parrots_distorted.bmp')).transpose(2,0,1)).unsqueeze(0)
   
    print('NRQM  of ref parrot image is: %0.4f'% nrqm_model(ref))
    print('NRQM  of dis parrot image is: %0.4f'% nrqm_model(dis))
    print('PI of ref parrot image is: %0.4f'% pi_model(ref))
    print('PI of dis parrot image is: %0.4f'% pi_model(dis))