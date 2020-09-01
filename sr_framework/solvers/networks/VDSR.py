import torch
import torch.nn as nn
import torch.nn.functional as F

class VDSR(nn.Module):
    def __init__(upscale_factor, in_channels=3, num_fea=64, out_channels=3):
    super(VDSR, self).__init__()
    self.upscale_factor = upscale_factor
    # feature extraction
    self.fea_in = nn.Sequential(
        nn.Conv2d(in_channels, num_fea, 3, 1, 1),
        nn.ReLU(True)
    )

    # body
    m = []
    for i in range(18):
        layer = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.ReLU(True)
        )
        m.append(layer)
    self.body = nn.Sequential(*m)

    # reconstruction
    self.reconstruct_conv = nn.Conv2d(num_fea, out_channels, 3, 1, 1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        
        # feature extraction
        out = self.fea_in(x)

        # body conv
        out = self.body(out)

        # reconstruct
        out = self.reconstruct_conv(out)

        return out + x
