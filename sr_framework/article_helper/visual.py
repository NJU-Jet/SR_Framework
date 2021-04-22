import numpy as np
import cv2
import imageio
import os
import os.path as osp
import sys
sys.path.append('../')
from utils import calc_metrics
from .show import *
from tqdm import tqdm, trange


def get_direction(x, y, w, h):
    left, right, top, bottom = 0, 0, 0, 0
    half_w, half_h = w // 2, h // 2

    if x < half_w:
        if x >= 256:
            left = x - 256
            right = x + 256
        else:
            left = 0
            right = 512
    else:
        if (w-x-1) >= 256:
            right = x + 256
            left = x - 256
        else:
            right = w - 1
            left = w - 513

    if y < half_h:
        if y >= 256:
            top = y - 256
            bottom = y + 256
        else:
            top = 0
            bottom = 512
    else:
        if (h-y-1) >= 256:
            bottom = y+256
            top = y-256
        else:
            bottom = h - 1
            top = h - 513
    
    return left, right, top, bottom
            

def generate_best(dir_list, border=4):
    img_list_dict = dict()

    # get sorted img names in corresponding model
    for dir_name in dir_list:
        img_list_dict[dir_name] = sorted(os.listdir(dir_name))
    
    length = len(dir_list)
    for i in range(0, 100):    # for every Urban img
        # load hr img
        hr_basename = img_list_dict[dir_list[0]][i]
        hr_path = osp.join(dir_list[0], hr_basename)
        hr_img = imageio.imread(hr_path, pilmode='RGB')
        h, w = hr_img.shape[:-1]
        h_step, w_step = h // 20, w // 20
        img_psnrs, img_ssims = [], []

        # get metrics of different models for this img
        for k in range(length-1):
            basename = img_list_dict[dir_list[k+1]][i]
            path = osp.join(dir_list[k+1], basename)
            img = imageio.imread(path, pilmode='RGB')
            if dir_list[k+1] == 'IDN':
                img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_REPLICATE)
            psnr, ssim = calc_metrics(hr_img, img, crop_border=border, test_Y=True)
            img_psnrs.append(psnr)
            img_ssims.append(ssim)
        str_img_psnrs = ['{:.2f}'.format(x) for x in img_psnrs]
        print('full img[{:03d}] | {}'.format((i+1), str_img_psnrs))

        # whether best is ours
        if np.argmax(np.array(img_psnrs)) < length-2 or np.argmax(np.array(img_ssims)) < length-2:
            continue

        # fixed stride for different location, get 64*64*3 patch
        for y in range(0, h-64, h_step):
            for x in range(0, w-64, w_step):
                imgs, psnrs, ssims = [], [], []
                # plot rectangle on hr img
                hr_img1 = hr_img.copy()
                cv2.rectangle(hr_img1, (x, y), (x+63, y+63), (255, 0, 0), 2)
                left, right, top, bottom = get_direction(x+32, y+32, w, h)
                imgs.append(hr_img1[top:bottom+1, left:right+1, :]) # 513 * 513 * 3
                hr_patch = hr_img[y:y+64, x:x+64, :]
                imgs.append(hr_patch)
                
                # for different model, get corresponding patch
                for k in range(length-1):
                    basename = img_list_dict[dir_list[k+1]][i]
                    path = osp.join(dir_list[k+1], basename)
                    img = imageio.imread(path, pilmode='RGB')
                    if dir_list[k+1] == 'IDN':
                        img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_REPLICATE)
                    img_patch = img[y:y+64, x:x+64, :]
                    imgs.append(img_patch)

                    # calculate psnr and ssim
                    psnr, ssim = calc_metrics(hr_patch, img_patch)
                    psnrs.append(psnr)
                    ssims.append(ssim)
                    str_psnrs = ['{:.2f}'.format(psnr) for psnr in psnrs]
                print('[{:03d}] | ({}/{}, {}/{}) | {}'.format((i+1), y, h, x, w, str_psnrs))
                if np.argmax(np.array(psnrs)) == length-2 and np.argmax(np.array(ssims)) == length-2:
                    print('Saving...')
                    plot_compare(imgs, img_psnrs, img_ssims, i+1, '{}_{}'.format(y, x), dir_list)



def plot_compare(imgs, psnrs, ssims, index, save_basename, labels):
    dirname1 = osp.join('best', 'png_{:03d}'.format(index))
    dirname2 = osp.join('best', 'pdf_{:03d}'.format(index))
    
    if not osp.exists(dirname1):
        os.makedirs(dirname1)
    if not osp.exists(dirname2):
        os.makedirs(dirname2)

     save_path1 = osp.join(dirname1, save_basename+'.png')
    #save_path2 = osp.join(dirname2, save_basename+'.pdf')
    
    img_h, img_w = imgs[0].shape[:-1]
    assert img_h == 513, '{}'.format(index)
    assert img_w == 513, '{}'.format(index)

    # create figure
    s = 10
    t1 = 5
    t2 = 1
    b = 2
    bls = 2

    lh = 2*s + t1
    lw = img_w * lh / img_h
    h = 2*(s + t1 + b)    
    w = 5*s + lw + bls + 4*t2 + 2*b
    size = 100

    fig = plt.figure(figsize=(w, h))
    

    rec0 = [b/w, (b+t1)/h, lw/w, lh/h]
    ax0 = fig.add_axes(rec0)
    default_ax(ax0)
    ax0.imshow(imgs[0])
    
    rec0_txt = [b/w, (b+0.3*t1)/h, lw/w, 0.7*t1/h]
    ax0_txt = fig.add_axes(rec0_txt)
    default_ax(ax0_txt)
    ax0_txt.set_title(label='img{:03d} from Urban100'.format(index), y=0, fontdict={'fontsize': size})

    rec1 = [(b+lw+bls)/w, (2*t1+s+b)/h, s/w, s/h]
    ax1 = fig.add_axes(rec1)
    default_ax(ax1)
    ax1.imshow(imgs[1])

    rec1_txt = [(b+lw+bls)/w, (s+b+t1+t1*0.3)/h, s/w, 0.7*t1/h]
    ax1_txt = fig.add_axes(rec1_txt)
    default_ax(ax1_txt)

if __name__ == '__main__':
    dir_list = ['HR', 'BICUBIC', 'FSRCNN', 'VDSR', 'DRRN', 'LapSRN', 'IDN', 'CARN', 'IMDN', 'XXX']
    scale = 4
    generate_best(dir_list, border=scale)
