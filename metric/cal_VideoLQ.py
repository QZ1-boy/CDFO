import cv2
import numpy as np
from PIL import Image
import sys
from metric.niqe import NIQE
from metric.nrqm import NRQM
from metric.nrqm import PI
from metric.brisque import BRISQUE
import torch

def rgb2ycbcr(img, y_only=False):
    """Convert a RGB image to YCbCr image.

    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    else:
        out_img = np.matmul(
            img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2rgb(img):
    """Convert a YCbCr image to RGB image.

    This function produces the same results as Matlab's ycbcr2rgb function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `YCrCb <-> RGB`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        ndarray: The converted RGB image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img) * 255
    out_img = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                              [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]  # noqa: E126
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2bgr(img):
    """Convert a YCbCr image to BGR image.

    The bgr version of ycbcr2rgb.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `YCrCb <-> BGR`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        ndarray: The converted BGR image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img) * 255
    out_img = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0.00791071, -0.00153632, 0],
                              [0, -0.00318811, 0.00625893]]) * 255.0 + [-276.836, 135.576, -222.921]  # noqa: E126
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img





def rgb2ycbcr(img, y_only=False):
    """Convert a RGB image to YCbCr image.

    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    else:
        out_img = np.matmul(
            img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.




def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)




def calculate_psnr(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    # img1 = reorder_image(img1, input_order=input_order)
    # img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate SSIM (structural similarity).
    Ref:
    Image quality assessment: From error visibility to structural similarity
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    # img1 = reorder_image(img1, input_order=input_order)
    # img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()




def calculate_tOF(true_img_cur, pred_img_cur, true_img_pre, pred_img_pre):

    true_img_cur = cv2.cvtColor(true_img_cur, cv2.COLOR_RGB2GRAY)
    pred_img_cur = cv2.cvtColor(pred_img_cur, cv2.COLOR_RGB2GRAY)
    true_img_pre = cv2.cvtColor(true_img_pre, cv2.COLOR_RGB2GRAY)
    pred_img_pre = cv2.cvtColor(pred_img_pre, cv2.COLOR_RGB2GRAY)

    # forward flow
    true_OF = cv2.calcOpticalFlowFarneback(
        true_img_pre, true_img_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    pred_OF = cv2.calcOpticalFlowFarneback(
        pred_img_pre, pred_img_cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # EPE
    diff_OF = true_OF - pred_OF
    tOF = np.mean(np.sqrt(np.sum(diff_OF**2, axis=-1)))

    return tOF







def main():
    # res_vid_name = [
    #         'BasketballDrive_fps50_480x272_500F.yuv', 
    #         'Kimono1_fps24_480x272_240F.yuv', 
    #         'BQTerrace_fps60_480x272_600F.yuv', 
    #         'ParkScene_fps24_480x272_240F.yuv'
    #         ]
    # gt_vid_name = [
    #            'BasketballDrive_1920x1080_50_500F.yuv', 
    #            'Kimono1_1920x1080_24_240F.yuv', 
    #            'BQTerrace_1920x1080_60_600F.yuv', 
    #            'ParkScene_1920x1080_24_240F.yuv'
    #            ]
    # cal_psnr_ssim('/data/cpl/testing_results/train_1126_LD_QP32_J_EDVR_M/', res_vid_name, gt_vid_name)
    pass


def cal_psnr_ssim(save_path, res_vid_name, gt_vid_name, gt_path, one_video=True):
    my_filepaths = gt_path
    seq_ave_psnr = 0
    seq_ave_ssim = 0
    for res_f, gt_f in zip(res_vid_name, gt_vid_name):
        frames = int(res_f[-8:-5])
        psnr = 0
        ssim = 0

        for fm_i in range(frames):
            idx ="%05d" % fm_i
            res_img_name = save_path + res_f + '/' + idx + '.png' 
            gt_img_name = my_filepaths + gt_f + '/' + idx + '.png' 

            res_img = Image.open(res_img_name)
            gt_img = Image.open(gt_img_name)
            min_width = min(res_img.width, gt_img.width)
            min_height = min(res_img.height, gt_img.height)
            
            res_img = np.array(res_img)
            res_img = res_img[:min_height,:min_width,np.newaxis].astype(np.float64)
            gt_img = np.array(gt_img)
            gt_img = gt_img[:min_height,:min_width,np.newaxis].astype(np.float64)

            f_psnr = calculate_psnr(res_img, gt_img, 4, test_y_channel=True)
            f_ssim = calculate_ssim(res_img, gt_img, 4, test_y_channel=True)
            psnr += f_psnr
            ssim += f_ssim

            print('seq: %s (%d/%d) ... ' % (res_f, fm_i, frames), end='\r')
        
        seq_ave_psnr += psnr / frames
        seq_ave_ssim += ssim / frames
        psnr_ = '%.3f' % (psnr / frames)
        ssim_ = '%.5f' % (ssim / frames)
        msg = '%s Average PSNR/SSIM: %.3f/%.5f' % (gt_f, psnr / frames, ssim / frames)
        print(msg)
        if one_video:
            return psnr_, ssim_, seq_ave_psnr, seq_ave_ssim
        
    # seq_ave_msg = 'Sequence Average PSNR/SSIM: %.3f/%.5f' % (seq_ave_psnr / len(gt_vid_name), seq_ave_ssim / len(gt_vid_name))



def get_Real_world(output):
    # cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
    # print('output',output.shape)
    # torch.Tensor(np.array(output).transpose(2,0,1)).unsqueeze(0)
    model_NIQE = NIQE()
    # model_NRQM = NRQM()
    # model_PI = PI()
    model_BRISQUE = BRISQUE()
    niqe = model_NIQE(torch.Tensor(np.array(output).transpose(2,0,1)).unsqueeze(0))
    # nrqm = model_NRQM(torch.Tensor(np.array(output).transpose(2,0,1)).unsqueeze(0))
    # pi = model_PI(torch.Tensor(np.array(output).transpose(2,0,1)).unsqueeze(0))
    # torch.Tensor(output.transpose(2,0,1)).unsqueeze(0)
    # print('output',torch.Tensor(output.transpose(2,0,1)).unsqueeze(0).shape, type(output))
    brisque = model_BRISQUE(torch.Tensor(np.array(output).transpose(2,0,1)).unsqueeze(0))
    nrqm = 0.0 
    pi = 0.0 
    # brisque = 0.0
    return np.array(niqe), np.array(nrqm), np.array(pi), np.array(brisque[0])


def numpy2tensor(input_seq, rgb_range=1.):
    tensor_list = []
    for img in input_seq:
        img = np.array(img).astype('float64')
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
        tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
        tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
        tensor_list.append(tensor)
    stacked = torch.stack(tensor_list).unsqueeze(0)
    return stacked


def tensor2numpy( tensor, rgb_range=1.):
    rgb_coefficient = 255 / rgb_range
    img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
    img = img[0].data
    img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    return img


def cal_niqe_nrqm_pi_brisque_REDS(save_path, res_vid_name, one_video=True):
    seq_ave_niqe = 0
    seq_ave_nrqm = 0
    seq_ave_pi = 0
    seq_ave_brisque = 0
    frames = 100
    niqe = 0
    nrqm = 0
    pi = 0
    brisque = 0
    if res_vid_name == '030':
        frames = 40
    elif res_vid_name == '031':
        frames = 34
    elif res_vid_name == '032':
        frames = 48
    elif res_vid_name == '033':
        frames = 47
    else:
        frames = 100

    for fm_i in range(frames):
        idx ="%08d" % fm_i
        res_img_name = save_path + '/' + idx + '.png' 
        res_img = Image.open(res_img_name)       
        res_tensor = np.array(res_img)
        f_niqe, f_nrqm, f_pi, f_brisque = get_Real_world(res_tensor)
        niqe += f_niqe
        nrqm += f_nrqm
        pi += f_pi
        brisque += f_brisque

        print('seq: %s (%d/%d) ... ' % (res_vid_name, fm_i, frames), end='\r')
        
    seq_ave_niqe += niqe / frames
    seq_ave_nrqm += nrqm / frames
    seq_ave_pi += pi / frames
    seq_ave_brisque += brisque / frames
    niqe_ = '%.5f' % (niqe / frames)
    nrqm_ = '%.5f' % (nrqm / frames)
    pi_ = '%.5f' % (pi / frames)
    brisque_ = '%.5f' % (brisque / frames)
    msg = '%s Average NIQE/NRQM/PI/BRISQUE: %.5f/%.5f/%.5f/%.5f' % (res_vid_name, seq_ave_niqe, seq_ave_nrqm, seq_ave_pi, seq_ave_brisque)
    print(msg)
    if one_video:
        return niqe_, nrqm_, pi_, brisque_
        



def cal_psnr_ssim_tOF_Vid4(save_path, res_vid_name, gt_vid_name, gt_path, one_video=True):
    my_filepaths = gt_path
    seq_ave_psnr = 0
    seq_ave_ssim = 0
    seq_ave_tOF = 0
    for res_f, gt_f in zip(res_vid_name, gt_vid_name):
        if res_f == 'calendar':
            frames = 41
        elif res_f == 'city':
            frames = 34
        elif res_f == 'foliage':
            frames = 49
        elif res_f == 'walk':
            frames = 47
        else:
            frames = 100

        psnr = 0
        ssim = 0
        tOF = 0

        for fm_i in range(frames):
            idx ="%08d" % fm_i
            res_img_name = save_path + res_f + '/' + idx + '.png' 
            gt_img_name = my_filepaths + gt_f + '/' + idx + '.png' 

            res_img = Image.open(res_img_name)
            gt_img = Image.open(gt_img_name)
           
            res_img = np.array(res_img)# .astype(np.float64)
            gt_img = np.array(gt_img)# .astype(np.float64)
            # print('res_img',res_img.shape, gt_img.shape)
            if fm_i == 0:
                gt_img_pre = gt_img
                res_img_pre = res_img

            f_psnr = calculate_psnr(res_img, gt_img, 4, test_y_channel=True)
            f_ssim = calculate_ssim(res_img, gt_img, 4, test_y_channel=True)
            f_tOF = calculate_tOF(gt_img, res_img, gt_img_pre, res_img_pre)
            gt_img_pre = gt_img
            res_img_pre = res_img
            psnr += f_psnr
            ssim += f_ssim
            tOF += f_tOF

            print('seq: %s (%d/%d) ... ' % (res_f, fm_i, frames), end='\r')
        
        seq_ave_psnr += psnr / frames
        seq_ave_ssim += ssim / frames
        seq_ave_tOF += ssim / frames
        psnr_ = '%.3f' % (psnr / frames)
        ssim_ = '%.5f' % (ssim / frames)
        tOF_ = '%.5f' % (tOF / frames)
        msg = '%s Average PSNR/SSIM/tOF: %.3f/%.5f/%.3f' % (gt_f, psnr / frames, ssim / frames, tOF / frames)
        print(msg)
        if one_video:
            return psnr_, ssim_, tOF_, seq_ave_psnr, seq_ave_ssim, seq_ave_tOF
        
    # seq_ave_msg = 'Sequence Average PSNR/SSIM: %.3f/%.5f' % (seq_ave_psnr / len(gt_vid_name), seq_ave_ssim / len(gt_vid_name))



if __name__ == '__main__':
    main()
