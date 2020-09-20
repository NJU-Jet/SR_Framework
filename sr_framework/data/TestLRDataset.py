import os
import os.path as osp
import torch
from .BaseDataset import Base
from PIL import Image

class TestLR(Base):
    def __init__(self, opt):
        super(TestLR, self).__init__(opt)
        self.dataroot_lr = opt['dataroot_LR']
        self.img_list = sorted(os.listdir(self.dataroot_lr))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        lr_path = osp.join(self.dataroot_lr, self.img_list[idx])
        lr = np.array(Image.open(lr_path))
        lr = self.np2tensor(lr)

        return lr        
