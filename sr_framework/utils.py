import logging
import torch
import cv2
import math
import os
import os.path as osp
import glob
import os.path as osp
import sys
import random
from shutil import get_terminal_size
import time
from datetime import datetime
import numpy as np
#from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
#plt.style.use(['science', 'no-latex'])
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def logger(name, filepath, pretrained=None):
    dir_path = osp.dirname(filepath)
    if not osp.exists(dir_path):
        os.mkdir(dir_path)
    if osp.exists(filepath) and pretrained is None:
        os.remove(filepath)

    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(filename)s[%(lineno)d] | %(message)s', datefmt='%H:%M:%S')
    stream_hd = logging.StreamHandler()
    stream_hd.setFormatter(formatter)
    lg.addHandler(stream_hd)

    file_hd = logging.FileHandler(filepath)
    file_hd.setFormatter(formatter)
    lg.addHandler(file_hd)
    
    return lg    


def generate_train_val(dataroot, prefix, random_sample=False, num=0):
    img_list = sorted(os.listdir(dataroot))
    train_file_path = osp.join('data', prefix+'_train.txt')
    val_file_path = osp.join('data', prefix+'_val.txt')
    train_list = []
    val_list = []

    if random_sample == False:
        train_list = img_list[:len(img_list)-num]
        val_list = img_list[len(img_list)-num:]
    else:
        val_list = sorted(random.sample(img_list, num))
        train_list = [x for x in img_list if x not in val_list]

    with open(train_file_path, 'w') as fp:
        for i, name in enumerate(train_list):
            fp.write(name)
            if i < len(train_list) - 1:
                fp.write('\n')

    with open(val_file_path, 'w') as fp:
        for i, name in enumerate(val_list):
            fp.write(name)
            if i < len(val_list) - 1:
                fp.write('\n')


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def calc_metrics(img1, img2, crop_border, test_Y=False):
    #
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2
    height, width = img1.shape[:2]
    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    psnr = calc_psnr(cropped_im1 * 255, cropped_im2 * 255)
    ssim = calc_ssim((cropped_im1 * 255).astype(np.uint8), (cropped_im2 * 255).astype(np.uint8))
    return psnr, ssim


class ProgressBar(object):
    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(' ' * self.bar_width, self.task_num, 'Prepare...'))
        else:
            sys.stdout.write('completed:0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='Validation...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')
            sys.stdout.write('\033[J')
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s elapsed: {}s, ETA: {:5}s\n{}\n'.format(bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()


def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

'''
def ssim(img1, img2, multichannel=True):
    assert img1.dtype == img2.dtype == np.uint8, 'np.uint8 is supposed.'
    s = compare_ssim(img1, img2, multichannel=multichannel)
    return s
'''

def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):

    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(img1.squeeze(), img2.squeeze())
    else:
        raise ValueError('Wrong input dims in calc_ssim')


def savefig(x_list, y_list, xlabel, ylabel, color_list, label_list, save_path):
    
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i in range(len(x_list)):
        plt.plot(x_list[i], y_list[i], color=color_list[i], label=label_list[i])

    plt.legend(loc='best', fontsize='small', frameon=True)
    plt.savefig(save_path)
    plt.close()


def Frequency_analysis(image):

    # Fourier Transform
    fft_res = np.abs(np.fft.fft2(image))

    # Center
    fft_res = np.fft.fftshift(fft_res)
    
    h, w = fft_res.shape
    fft_res = np.pad(fft_res, pad_width=((0, (h+1)%2), (0, (w+1)%2)), mode='constant', constant_values=0)
    h, w = fft_res.shape
    if h > w:
        pad = h - w
        fft_res = np.pad(fft_res, pad_width=((0, 0), (pad//2, pad//2)), mode='constant', constant_values=0)
    elif w > h:
        pad = w - h
        fft_res = np.pad(fft_res, pad_width=((pad//2, pad//2), (0, 0)), mode='constant', constant_values=0)

    h, w = fft_res.shape
    if h!= w:
        raise ValueError('')

    max_range = h // 2
    cy, cx = h//2, w//2
    x, y = [], []
    for r in range(max_range):
        x.append(r)
        f = 0.0
        if r == 0:
            f = fft_res[cy, cx]
        else:
            f += sum(fft_res[cy-r, cx-r:cx+r])
            f += sum(fft_res[cy-r:cy+r, cx+r])
            f += sum(fft_res[cy+r, cx+r:cx-r:-1])
            f += sum(fft_res[cy+r:cy-r:-1, cx-r])
        y.append(np.log(1+f))

    # Normalize frequency to [0, 1]
    max_freq = np.max(x)
    x = x / max_freq
    
    return x, y


def calc_cuda_time(val_loader, model):
    test_times = 0.0
    model.eval()

    # start testing time
    test_times = 0

    pbar_time = ProgressBar(5)
    for i in range(5):
        test_time = sr_forward_time(val_loader, model)
        test_times += test_time
        pbar_time.update('')
    avg_test_time = (test_times / 5) / len(val_loader)
    return avg_test_time


def sr_forward_time(dataloader, model):
    cuda_time = 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i, data in enumerate(dataloader):
        lr_imgs = data['LR'].to(device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            sr_imgs = model(lr_imgs)
        end_event.record()

        torch.cuda.synchronize(device)

        cuda_time += start_event.elapsed_time(end_event)

    return cuda_time


if __name__ == '__main__':
    # modify dataroot according to yout path
    dataroot = '/data/dzc/SISRDataset/DIV2K/DIV2K_train_HR'
    prefix = 'DIV2K'
    generate_train_val(dataroot, prefix, False, num=100)
    
