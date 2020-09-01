from .BaseDataset import Base
import numpy as np
import os
import os.path as osp
import torchvision.transforms as T
from PIL import Image
import logging
import sys
sys.path.append('../')
from utils import rgb2ycbcr

class DIV2KDataset(Base):
    def __init__(self, opt):
        super(DIV2KDataset, self).__init__(opt)
        self.dataroot_hr = opt['dataroot_HR']   
        self.dataroot_lr = opt['dataroot_LR']
        self.filename_path = opt['filename_path']
        self.use_flip = opt['use_flip']
        self.use_rot = opt['use_rot']
        self.ps = opt['patch_size']
        self.scale = opt['scale']
        self.split = opt['split']
        self.noise = opt['noise']
        self.train_Y = opt['train_Y']
        self.enlarge_times = opt['enlarge_times']

        self.img_list = []        
        with open(self.filename_path, 'r') as f:
            filenames = f.readlines()
        for line in filenames:
            self.img_list.append(line.strip())

        
    def __len__(self):
        return len(self.img_list) * self.enlarge_times


    def __getitem__(self, idx):
        idx = idx % len(self.img_list)
        hr_path = osp.join(self.dataroot_hr, self.img_list[idx])
        base, ext = osp.splitext(self.img_list[idx])
        lr_basename = base + 'x{}'.format(self.scale) + ext
        lr_path = osp.join(self.dataroot_lr, lr_basename)
        hr = np.array(Image.open(hr_path))
        lr = np.array(Image.open(lr_path))

        if self.noise is not None:
            lr = self.add_noise(lr, self.noise['type'], self.noise['value'])
        if self.train_Y:
            lr = rgb2ycbcr(lr)[:, :, np.newaxis]
            hr = rgb2ycbcr(hr)[:, :, np.newaxis]

        data = {}
        if self.split == 'train':
            lr_patch, hr_patch = self.get_patch(lr, hr, self.ps, self.scale)
            lr, hr = self.augment(lr_patch, hr_patch, self.use_flip, self.use_rot)
        lr ,hr = self.np2tensor(lr), self.np2tensor(hr)
        
        data['LR'] = lr
        data['HR'] = hr
        return data
