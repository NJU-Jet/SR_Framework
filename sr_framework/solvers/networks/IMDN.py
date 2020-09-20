import torch
import torch.nn as nn
from .blocks import MeanShift, IMDN_Module

class IMDN(nn.Module):
    def __init__(self, upscale_factor=4, in_channels=3, num_fea=64, out_channels=3, imdn_blocks=6):
        super(IMDN, self).__init__()
        
        #self.sub_mean = MeanShift()
        #self.add_mean = MeanShift(sign=1)

        # extract features
        self.fea_conv = nn.Conv2d(in_channels, num_fea, 3, 1, 1)
    
        # map
        self.IMDN1 = IMDN_Module(num_fea)
        self.IMDN2 = IMDN_Module(num_fea)
        self.IMDN3 = IMDN_Module(num_fea)
        self.IMDN4 = IMDN_Module(num_fea)
        self.IMDN5 = IMDN_Module(num_fea)
        self.IMDN6 = IMDN_Module(num_fea)

        self.fuse = nn.Sequential(
            nn.Conv2d(num_fea * imdn_blocks, num_fea, 1, 1, 0),
            nn.LeakyReLU(0.05)
        )
        self.LR_conv = nn.Conv2d(num_fea, num_fea, 3, 1, 1)

        # reconstruct
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_fea, out_channels * (upscale_factor ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale_factor)
        )
   
    def forward(self, x):
        #x = self.sub_mean(x)

        # extract features
        x = self.fea_conv(x)

        # body map
        out1 = self.IMDN1(x)
        out2 = self.IMDN2(out1)
        out3 = self.IMDN3(out2)
        out4 = self.IMDN4(out3)
        out5 = self.IMDN5(out4)
        out6 = self.IMDN6(out5)

        out = self.LR_conv(self.fuse(torch.cat([out1, out2, out3, out4, out5, out6], dim=1))) + x

        # reconstruct
        out = self.upsampler(out)
        #out = self.add_mean(out)

        return out
