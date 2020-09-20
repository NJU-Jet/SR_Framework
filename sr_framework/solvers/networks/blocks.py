import torch
import torch.nn as nn
import sys
import math

class MeanShift(nn.Conv2d):
    def __init__(self, mean=[0.4488, 0.4371, 0.4040], std=[1.0, 1.0, 1.0], sign=-1):
        super(MeanShift, self).__init__(3, 3, 1)
        std = torch.Tensor(std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * torch.Tensor(mean)
        self.bias.data.div_(std)
        for p in self.parameters():
            p.requires_grad = False


class CA(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(CA, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels//ratio, 1, 1, 0, bias=False)
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels, 1, 1, 0, bias=False)
        
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_value = self.avg(x)
        max_value = self.max(x)

        avg_out = self.fc2(self.act(self.fc1(avg_value)))
        max_out = self.fc2(self.act(self.fc1(max_value)))
        
        out = avg_out + max_out
        
        return self.sigmoid(out)

class ResBlock(nn.Module):
    def __init__(self, num_fea=64):
        super(ResBlock, self).__init__()
        self.res_conv = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_fea, num_fea, 3, 1, 1)
        )

    def forward(self, x):
        out = self.res_conv(x)

        return out + x

class UpSampler(nn.Module):
    def __init__(self, upscale_factor=2, num_fea=64):
        super(UpSampler, self).__init__()
        if (upscale_factor & (upscale_factor-1)) == 0: # upscale_factor = 2^n
            m = []
            for i in range(int(math.log(upscale_factor, 2))):
                m.append(nn.Conv2d(num_fea, num_fea * 4, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
            self.upsample = nn.Sequential(*m)

        elif upscale_factor == 3:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_fea, num_fea * 9, 3, 1, 1),
                nn.PixelShuffle(3)
            )
        else:
            raise NotImplementedError('Error upscale_factor in Upsampler')

    def forward(self, x):
        return self.upsample(x)
        

def mean_channels(x):
    assert(x.dim() == 4)
    spatial_sum = x.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (x.shape[2] * x.shape[3])

def std(x):
    assert(x.dim() == 4)
    x_mean = mean_channels(x)
    x_var = (x - x_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (x.shape[2] * x.shape[3])
    return x_var.pow(0.5)

class CCA(nn.Module):
    def __init__(self, num_fea, reduction=16):
        super(CCA, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.std = std
        
        self.atten_conv = nn.Sequential(
            nn.Conv2d(num_fea, num_fea // reduction, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(num_fea // reduction, num_fea, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        atten = self.avg(x) + self.std(x)
        atten = self.atten_conv(atten)

        return x * atten

class IMDN_Module(nn.Module):
    def __init__(self, num_fea, distill_ratio=0.25):
        super(IMDN_Module, self).__init__()
        self.distilled_channels = int(num_fea * distill_ratio)
        self.remain_channels = int(num_fea - self.distilled_channels)
        self.conv1 = nn.Conv2d(num_fea, num_fea, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.remain_channels, num_fea, 3, 1, 1)
        self.conv3 = nn.Conv2d(self.remain_channels, num_fea, 3, 1, 1)
        self.conv4 = nn.Conv2d(self.remain_channels, self.distilled_channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.05)
        self.fuse = nn.Conv2d(num_fea, num_fea, 1, 1, 0)
        self.cca = CCA(num_fea)

    def forward(self, x):
        out1 = self.act(self.conv1(x))
        d1, r1 = torch.split(out1, (self.distilled_channels, self.remain_channels), dim=1)
        out2 = self.act(self.conv2(r1))
        d2, r2 = torch.split(out2, (self.distilled_channels, self.remain_channels), dim=1)
        out3 = self.act(self.conv3(r2))
        d3, r3 = torch.split(out3, (self.distilled_channels, self.remain_channels), dim=1)
        d4 = self.act(self.conv4(r3))
        out = torch.cat([d1, d2, d3, d4], dim=1)
        out = self.cca(torch.cat([d1, d2, d3, d4], dim=1))
        out = self.fuse(out)

        return out + x
