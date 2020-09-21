import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import math

#plt.style.use(['science', 'no-latex'])
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def default_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def show_feature_map(img, save_path):
    plt.figure()
    
    '''axis'''
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()


def show_relation(t1, t2, t3, c4, save_path):
    plt.figure()
    x = np.arange(4)
    total_width, n = 0.8, 4
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.bar(x, t1, color='lime', width=width, label='$F_{NLT}^0$')
    plt.bar(x + width, t2, color='orangered', width=width, label='$F_{NLT}^1$')
    plt.bar(x + 2*width, t3, color='darkmagenta', width=width, label='$F_{NLT}^2$')
    plt.bar(x + 3*width, c4, color='blue', width=width, label='$F_B^4$')

    plt.legend(loc='best', fontsize='x-small', frameon=True)
    group_names = ['FTB1', 'FTB2', 'FTB3', 'FTB4']
    plt.xticks([0, 1, 2, 3], group_names)
    plt.ylabel('Average absolute filter weights')
    plt.savefig(save_path)
    plt.close()


def show_frequency(x, y, save_path):
    plt.figure()
    plt.xlabel('Low Frequency  ->  High Frequency')
    plt.ylabel('Spectral densities')
    plt.plot(x, y, marker = 'o', markerfacecolor='r', markersize=3.0)
    plt.savefig(save_path)
    plt.close()

 
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
    ax1_txt.set_title(label='{}\nPSNR/SSIM'.format(labels[0]), y=0, fontdict={'fontsize': size})

    rec2 = [(b+lw+bls+s+t2)/w, (2*t1+s+b)/h, s/w, s/h]
    ax2 = fig.add_axes(rec2)
    default_ax(ax2)
    ax2.imshow(imgs[2])

    rec2_txt = [(b+lw+bls+s+t2)/w, (s+b+t1+t1*0.3)/h, s/w, 0.7*t1/h]
    ax2_txt = fig.add_axes(rec2_txt)
    default_ax(ax2_txt)
    ax2_txt.set_title(label='{}\n{:.2f}/{:.4f}'.format(labels[1], psnrs[0], ssims[0]), y=0, fontdict={'fontsize': size})

    rec3 = [(b+lw+bls+2*s+2*t2)/w, (2*t1+s+b)/h, s/w, s/h]
    ax3 = fig.add_axes(rec3)
    default_ax(ax3)
    ax3.imshow(imgs[3])

    rec3_txt = [(b+lw+bls+2*s+2*t2)/w, (s+b+t1+t1*0.3)/h, s/w, 0.7*t1/h]
    ax3_txt = fig.add_axes(rec3_txt)
    default_ax(ax3_txt)
    ax3_txt.set_title(label='{}\n{:.2f}/{:.4f}'.format(labels[2], psnrs[1], ssims[1]), y=0, fontdict={'fontsize': size})
    
    rec4 = [(b+lw+bls+3*s+3*t2)/w, (2*t1+s+b)/h, s/w, s/h]
    ax4 = fig.add_axes(rec4)
    default_ax(ax4)
    ax4.imshow(imgs[4])

    rec4_txt = [(b+lw+bls+3*s+3*t2)/w, (s+b+t1+t1*0.3)/h, s/w, 0.7*t1/h]
    ax4_txt = fig.add_axes(rec4_txt)
    default_ax(ax4_txt)
    ax4_txt.set_title(label='{}\n{:.2f}/{:.4f}'.format(labels[3], psnrs[2], ssims[2]), y=0, fontdict={'fontsize': size})
    
    rec5 = [(b+lw+bls+4*s+4*t2)/w, (2*t1+s+b)/h, s/w, s/h]
    ax5 = fig.add_axes(rec5)
    default_ax(ax5)
    ax5.imshow(imgs[5])

    rec5_txt = [(b+lw+bls+4*s+4*t2)/w, (s+b+t1+t1*0.3)/h, s/w, 0.7*t1/h]
    ax5_txt = fig.add_axes(rec5_txt)
    default_ax(ax5_txt)
    ax5_txt.set_title(label='{}\n{:.2f}/{:.4f}'.format(labels[4], psnrs[3], ssims[3]), y=0, fontdict={'fontsize': size})

    rec6 = [(b+lw+bls)/w, (t1+b)/h, s/w, s/h]
    ax6 = fig.add_axes(rec6)
    default_ax(ax6)
    ax6.imshow(imgs[6])

    rec6_txt = [(b+lw+bls)/w, (b+t1*0.3)/h, s/w, 0.7*t1/h]
    ax6_txt = fig.add_axes(rec6_txt)
    default_ax(ax6_txt)
    ax6_txt.set_title(label='{}\n{:.2f}/{:.4f}'.format(labels[5], psnrs[4], ssims[4]), y=0, fontdict={'fontsize': size})
    
    rec7 = [(b+lw+bls+s+t2)/w, (t1+b)/h, s/w, s/h]
    ax7 = fig.add_axes(rec7)
    default_ax(ax7)
    ax7.imshow(imgs[7])

    rec7_txt = [(b+lw+bls+s+t2)/w, (b+t1*0.3)/h, s/w, 0.7*t1/h]
    ax7_txt = fig.add_axes(rec7_txt)
    default_ax(ax7_txt)
    ax7_txt.set_title(label='{}\n{:.2f}/{:.4f}'.format(labels[6], psnrs[5], ssims[5]), y=0, fontdict={'fontsize': size})
    
    rec8 = [(b+lw+bls+2*s+2*t2)/w, (t1+b)/h, s/w, s/h]
    ax8 = fig.add_axes(rec8)
    default_ax(ax8)
    ax8.imshow(imgs[8])

    rec8_txt = [(b+lw+bls+2*s+2*t2)/w, (b+t1*0.3)/h, s/w, 0.7*t1/h]
    ax8_txt = fig.add_axes(rec8_txt)
    default_ax(ax8_txt)
    ax8_txt.set_title(label='{}\n{:.2f}/{:.4f}'.format(labels[7], psnrs[6], ssims[6]), y=0, fontdict={'fontsize': size})

    rec9 = [(b+lw+bls+3*s+3*t2)/w, (t1+b)/h, s/w, s/h]
    ax9 = fig.add_axes(rec9)
    default_ax(ax9)
    ax9.imshow(imgs[9])

    rec9_txt = [(b+lw+bls+3*s+3*t2)/w, (b+t1*0.3)/h, s/w, 0.7*t1/h]
    ax9_txt = fig.add_axes(rec9_txt)
    default_ax(ax9_txt)
    ax9_txt.set_title(label='{}\n{:.2f}/{:.4f}'.format(labels[8], psnrs[7], ssims[7]), y=0, fontdict={'fontsize': size})

    rec10 = [(b+lw+bls+4*s+4*t2)/w, (t1+b)/h, s/w, s/h]
    ax10 = fig.add_axes(rec10)
    default_ax(ax10)
    ax10.imshow(imgs[10])

    rec10_txt = [(b+lw+bls+4*s+4*t2)/w, (b+t1*0.3)/h, s/w, 0.7*t1/h]
    ax10_txt = fig.add_axes(rec10_txt)
    default_ax(ax10_txt)
    ax10_txt.set_title(label='{}(ours)\n{:.2f}/{:.4f}'.format(labels[9], psnrs[8], ssims[8]), y=0, fontdict={'fontsize': size, 'fontweight': 700})

    plt.savefig(save_path1)
    #plt.savefig(save_path2)

    plt.close()
