import torch
import torch.nn as nn
import torch.nn.functional as F

class DRRN(nn.Module):
    def __init__(self, upscale_factor=2, in_channels=3, num_fea=128, out_channels=3, num_U=9):
        super(DRRN, self).__init__()
        self.upscale_factor = upscale_factor
        self.num_U = num_U

        # feature extraction
        self.fea_in = nn.Sequential(
            nn.Conv2d(in_channels, num_fea, 3, 1, 1),
            nn.ReLU(True)
        )

        # body: recurisive conv
        self.body = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True)
        )

        # reconstruction
        self.reconstruct = nn.Sequential(
            nn.Conv2d(num_fea, out_channels, 3, 1, 1),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        # feature extraction
        x = self.fea_in(x)

        # recurisive conv
        out = x
        for i in range(self.num_U):
            out = self.body(out)
            out += x
        
        # reconstruction
        out = self.reconstruct(out)

        return out
