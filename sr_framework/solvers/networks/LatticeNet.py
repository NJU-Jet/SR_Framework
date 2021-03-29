import torch
import torch.nn as nn
from .blocks import MeanShift, std
import torch.nn.functional as F

class CoffConv(nn.Module):
    def __init__(self, num_fea):
        super(CoffConv, self).__init__()
        self.upper_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_fea, num_fea // 16, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_fea // 16, num_fea, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        
        self.std = std
        self.lower_branch = nn.Sequential(
            nn.Conv2d(num_fea, num_fea // 16, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_fea // 16, num_fea, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, fea):
        upper = self.upper_branch(fea)
        lower = self.std(fea)
        lower = self.lower_branch(lower)

        out = torch.add(upper, lower) / 2
        
        return out


class LBlock(nn.Module):
    def __init__(self, num_fea):
        super(LBlock, self).__init__()
        self.H_conv = nn.Sequential(
            nn.Conv2d(num_fea, 48, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(48, num_fea, 3, 1, 1),
            nn.LeakyReLU(0.05),
        )

        self.A1_coff_conv = CoffConv(num_fea)
        self.B1_coff_conv = CoffConv(num_fea)

        self.G_conv = nn.Sequential(
            nn.Conv2d(num_fea, 48, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(48, num_fea, 3, 1, 1),
            nn.LeakyReLU(0.05),
        )

        self.A2_coff_conv = CoffConv(num_fea)
        self.B2_coff_conv = CoffConv(num_fea)

        self.fuse = nn.Conv2d(num_fea*2, num_fea, 1, 1, 0)


    def forward(self, x):
        H = self.H_conv(x)
        A1 = self.A1_coff_conv(H)
        P1 = x + A1*H
        B1 = self.B1_coff_conv(x)
        Q1 = H + B1*x

        G = self.G_conv(P1)
        B2 = self.B2_coff_conv(G)
        Q2 = Q1 + B2*G
        A2 = self.A2_coff_conv(Q1)
        P2 = G + Q1*A2

        out = self.fuse(torch.cat([P2, Q2], dim=1))

        return out


class BFModule(nn.Module):
    def __init__(self, num_fea):
        super(BFModule, self).__init__()
        self.conv4 = nn.Conv2d(num_fea, num_fea//2, 1, 1, 0)
        self.conv3 = nn.Conv2d(num_fea, num_fea//2, 1, 1, 0)
        self.fuse43 = nn.Conv2d(num_fea, num_fea//2, 1, 1, 0)
        self.conv2 = nn.Conv2d(num_fea, num_fea//2, 1, 1,0)        
        self.fuse32 = nn.Conv2d(num_fea, num_fea//2, 1, 1, 0)
        self.conv1 = nn.Conv2d(num_fea, num_fea//2, 1, 1, 0)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x_list):
        H4 = self.act(self.conv4(x_list[3]))
        H3_half = self.act(self.conv3(x_list[2]))
        H3 = self.fuse43(torch.cat([H4, H3_half], dim=1))      
        H2_half = self.act(self.conv2(x_list[1]))
        H2 = self.fuse32(torch.cat([H3, H2_half], dim=1))
        H1_half = self.act(self.conv1(x_list[0]))
        H1 = torch.cat([H2, H1_half], dim=1)

        return H1
        

class LatticeNet(nn.Module):
    def __init__(self, upscale_factor=2, in_channels=3, num_fea=64, out_channels=3, num_LBs=4):
        super(LatticeNet, self).__init__()
        self.sub_mean = MeanShift()
        self.add_mean = MeanShift(sign=1)
        self.num_LBs = num_LBs

        # feature extraction
        self.fea_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_fea, 3, 1, 1),
            nn.Conv2d(num_fea, num_fea, 3, 1, 1)
        )

        # LBlocks
        LBs = []
        for i in range(num_LBs):
            LBs.append(LBlock(num_fea))
        self.LBs = nn.ModuleList(LBs)

        # BFModule
        self.BFM = BFModule(num_fea)

        # Reconstruction
        self.upsample = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.Conv2d(num_fea, out_channels * (upscale_factor**2), 3, 1, 1),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        x = self.sub_mean(x)

        # feature extraction
        fea = self.fea_conv(x)    

        # LBlocks
        outs = []
        temp = fea
        for i in range(self.num_LBs):
            temp = self.LBs[i](temp)
            outs.append(temp)

        # BFM
        H = self.BFM(outs)

        # reconstruct
        out = self.upsample(H + fea)

        out = self.add_mean(out)

        return out
