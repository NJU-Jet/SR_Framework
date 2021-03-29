from .BaseDataset import Base
import numpy as np
import os
import os.path as osp
from PIL import Image
import logging
import imageio

class TestLRHR(Base):
    def __init__(self, opt):
        super(TestLRHR, self).__init__(opt)
        self.dataroot_hr = opt['dataroot_HR']   
        self.dataroot_lr = opt['dataroot_LR']
        self.scale = opt['scale']

        self.lr_img_list = sorted(os.listdir(self.dataroot_lr))
        self.hr_img_list = sorted(os.listdir(self.dataroot_hr))
        assert len(self.lr_img_list) == len(self.hr_img_list)
        
        
    def __len__(self):
        return len(self.lr_img_list)

    def __getitem__(self, idx):
        hr_path = osp.join(self.dataroot_hr, self.hr_img_list[idx])
        lr_path = osp.join(self.dataroot_lr, self.lr_img_list[idx])
        hr = self.load_img(hr_path)
        lr = self.load_img(lr_path)

        data = {}
        lr ,hr = self.np2tensor(lr), self.np2tensor(hr)
        
        data['LR'] = lr
        data['HR'] = hr
        data['filename'] = self.lr_img_list[idx]
        return data
