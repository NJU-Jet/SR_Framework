import torch
import torch.nn as nn
from .blocks import MeanShift, UpSampler
from torchsummaryX import summary
import numpy as np 

class CA(nn.Module):
    def __init__(self, num_fea=64, reduction=16):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(num_fea, num_fea//reduction, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(num_fea//reduction, num_fea, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)

        return torch.mul(x, y)


class RCAB(nn.Module):
    def __init__(self, num_fea=64):
        super(RCAB, self).__init__()

        self.res_conv = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            CA(num_fea=num_fea)
        )

    def forward(self, x):
        res = self.res_conv(x)
        out = res + x

        return out


class RG(nn.Module):
    def __init__(self, num_fea=64, num_RCABs=10):
        super(RG, self).__init__()
        
        RCABs = []
        for i in range(num_RCABs):
            RCABs.append(RCAB(num_fea=num_fea))
        RCABs.append(nn.Conv2d(num_fea, num_fea, 3, 1, 1))
        self.RCABs = nn.Sequential(*RCABs)
        
    def forward(self, x):
        res = self.RCABs(x)
        out = res + x

        return out


class RCAN(nn.Module):
    def __init__(self, upscale_factor=2, in_channels=3, num_fea=64, out_channels=3, num_RGs=10, num_RCABs=20):
        super(RCAN, self).__init__()
        self.num_RGs = num_RGs
        self.sub_mean = MeanShift()
        self.add_mean = MeanShift(sign=1)

        # shallow feature extraction
        self.fea_conv = nn.Conv2d(in_channels, num_fea, 3, 1, 1)

        # RGs
        RGs = []
        for i in range(num_RGs):
            RGs.append(RG(num_fea=num_fea, num_RCABs=num_RCABs))
        self.LR_conv = nn.Conv2d(num_fea, num_fea, 3, 1, 1)
        self.RGs = nn.ModuleList(RGs)

        # reconstruction
        self.upsampler = nn.Sequential(
            UpSampler(upscale_factor=upscale_factor, num_fea=num_fea),
            nn.Conv2d(num_fea, out_channels, 3, 1, 1)
        )

    def forward(self, x):
        x = self.sub_mean(x)

        # shallow feature extraction
        fea = self.fea_conv(x)

        # RGs
        res = fea
        for i in range(self.num_RGs):
            res = self.RGs[i](res)
        res = self.LR_conv(res)
        out = res + fea

        # reconstruction
        out = self.upsampler(out)

        out = self.add_mean(out)

        return out

if __name__ == '__main__':
    # 15.592M, 916.9G
    s = 4
    model = RCAN(upscale_factor=s).to('cuda:0')
    in_ = torch.randn(1, 3, round(720/s), round(1280/s)).to('cuda:0')
    summary(model ,in_)
