import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import MeanShift, UpSampler

class Block(nn.Module):
    def __init__(self, num_fea):
        super(Block, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_fea, num_fea, 3, 1, 1)
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(num_fea * 2, num_fea, 1, 1, 0),
            nn.ReLU(True)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_fea, num_fea, 3, 1, 1)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(num_fea * 3, num_fea, 1, 1, 0),
            nn.ReLU(True)
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True), 
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(num_fea * 4, num_fea, 1, 1, 0),
            nn.ReLU(True)
        )

        self.act = nn.ReLU(True)

    def forward(self, x):
        b1 = self.act(self.b1(x) + x)
        c1 = torch.cat([x, b1], dim=1) # num_fea * 2
        o1 = self.c1(c1)

        b2 = self.act(self.b2(o1) + o1)
        c2 = torch.cat([c1, b2], dim=1) # num_fea * 3
        o2 = self.c2(c2)

        b3 = self.act(self.b3(o2) + o2)
        c3 = torch.cat([c2, b3], dim=1) # num_fea * 4
        o3 = self.c3(c3)

        return o3

class CARN(nn.Module):
    def __init__(self, upscale_factor, in_channels, num_fea, out_channels, use_skip=False):
        super(CARN, self).__init__()
        self.use_skip = use_skip
        self.upscale_factor = upscale_factor
        self.sub_mean = MeanShift()
        self.add_mean = MeanShift(sign=1)

        # extract features
        self.fea_in = nn.Conv2d(in_channels, num_fea, 3, 1, 1)

        # CARN body
        self.b1 = Block(num_fea)
        self.c1 = nn.Sequential(
            nn.Conv2d(num_fea * 2, num_fea, 1, 1, 0),
            nn.ReLU(True)
        )

        self.b2 = Block(num_fea)
        self.c2 = nn.Sequential(
            nn.Conv2d(num_fea * 3, num_fea, 1, 1, 0),
            nn.ReLU(True)
        )

        self.b3 = Block(num_fea)
        self.c3 = nn.Sequential(
            nn.Conv2d(num_fea * 4, num_fea, 1, 1, 0),
            nn.ReLU(True)
        )

        # Reconstruct
        self.upsampler = UpSampler(upscale_factor, num_fea)
        self.last_conv = nn.Conv2d(num_fea, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.sub_mean(x)

        # feature extraction
        x = self.fea_in(x)
        if self.use_skip:
            inter_res = F.interpolate(x, scale_factor= self.upscale_factor, mode='bicubic', align_corners=False)        

        # body
        b1 = self.b1(x)
        c1 = torch.cat([b1, x], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)   

        # Reconstruct
        out = self.upsampler(o3)
        if self.use_skip:
            out += inter_res
        out = self.last_conv(out)

        out = self.add_mean(out)

        return out         
