import torch
import torch.nn as nn
import torch.nn.functional as F

class DRCN(nn.Module):
    def __init__(self, upscale_factor=2, in_channels=3, num_fea=256, out_channels=3, recurisive_times=9):
        super(DRCN, self).__init__()
        self.upscale_factor = upscale_factor
        self.recurisive_times = recurisive_times

        # feature extraction
        self.fea_in = nn.Sequential(
            nn.Conv2d(in_channels, num_fea, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True)
        )

        # body recurisive
        self.body = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True)
        )

        # reconstruction
        self.reconstruct = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_fea, out_channels, 3, 1, 1),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        
        # feature extraction
        out = self.fea_in(x)

        # body recurisive
        H = []
        h = out
        for i in range(self.recurisive_times):
            h = self.body(h)
            H.append(h)

        # reconstruction
        recon = []
        for i in range(self.recurisive_times):
            recon_h = self.reconstruct(H[i])
            recon.append(recon_h)

        return sum(recon) / self.recurisive_times + x 
